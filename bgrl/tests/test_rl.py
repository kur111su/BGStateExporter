"""B2/B3 冒烟测试（CI 用，CPU 即可）：模型形状、动作屏蔽、动作头顺序、参数量、GAE。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import taverngym  # noqa: E402
import taverngym._core as core  # noqa: E402
from rl.model import build_model, n_params  # noqa: E402
from rl.ppo import Rollout  # noqa: E402


def main() -> int:
    # 动作头顺序和 TavernGym 一致
    names = [core.action_name(a) for a in range(core.N_ACTIONS)]
    assert names[4] == "HeroPower(0)" and names[6] == "DarkGift" and names[7] == "Cancel"
    assert names[46 + 3 * 8 + 5] == "Play { hand: 3, pos: 5 }"
    assert names[126 + 4 * 7 + 2] == "Magnetize { hand: 4, board: 2 }"
    assert names[196 + 5 * 7 + 1] == "Move { from: 5, to: 1 }"

    from rl.model import SIZES
    for name, kw in SIZES.items():
        print(f"{name}: {n_params(build_model(**kw)) / 1e6:.2f}M")
    m = build_model(**SIZES["small"])
    print(("::notice::" if os.environ.get("GITHUB_ACTIONS") else "") + f"默认模型参数 {n_params(m) / 1e6:.2f}M")
    assert 1.5e6 < n_params(m) < 6e6

    env = taverngym.BatchEnv(n_games=8, seed=1, controlled=range(8), threads=2)
    small = build_model(64, 2, 2).eval()
    steps = 0
    for _ in range(300):
        o = env.observe()
        f, c, mk = (torch.as_tensor(o["features"]), torch.as_tensor(o["cards"]).long(), torch.as_tensor(o["action_mask"]))
        logits, v, pl = small(f, c, mk)
        assert logits.shape == (env.n_slots, core.N_ACTIONS) and v.shape == (env.n_slots,) and pl.shape[1] == 8
        a, logp, _ = small.act(f, c, mk)
        ready = o["ready"]
        assert all(o["action_mask"][i, a[i]] for i in np.flatnonzero(ready)), "选了不合法的动作"
        _, _, info = env.step(a.numpy())
        assert not info["rejected"][ready].any()
        steps += int(ready.sum())
    print(f"随机初始化模型走了 {steps} 个决策，没有不合法动作")

    # 优化后的前向（合并嵌入、只算有内容的词、压缩输入）和原来的写法结果一致
    torch.manual_seed(0)
    ref_m = build_model(64, 2, 2, 32).eval()
    o = env.observe()
    f = torch.as_tensor(o["features"]); c = torch.as_tensor(o["cards"]).long(); mk = torch.as_tensor(o["action_mask"])
    f = f.clone(); f[:, :500] *= 50.0                                   # 带一些大数值
    with torch.no_grad():
        ref_m.pack = False
        l0, v0, p0 = ref_m(f, c, mk, slow=True)
        ref_m.pack = True
        l1, v1, p1 = ref_m(f, c, mk)
        from rl.model import soft_log
        l2, v2, p2 = ref_m(soft_log(f), c, mk, compressed=True)
    legal = mk
    for a_, b_ in ((l0, l1), (l0, l2)):
        assert torch.allclose(a_[legal], b_[legal], atol=2e-4, rtol=1e-4), (a_[legal] - b_[legal]).abs().max()
    assert torch.allclose(v0, v1, atol=1e-4) and torch.allclose(p0, p2, atol=2e-4)
    print("优化后的前向和原写法一致（最大差 %.2e）" % float((l0[legal] - l1[legal]).abs().max()))

    # 体系感知（第 15300 轮起）：旧检查点没有 comp_* 参数也能读，读进来结果和原来一样；模型自己算的阵容得分和 rl/comp 的一样
    from rl.comp import CompTables, load_target_comps
    from rl.model import COMP_F, K_MAX
    from tg.catalog import base_of_index, card_id, n_cards
    from tg.obs import C_BOARD, F_LOBBY, lobby_bits
    old = {k: v for k, v in ref_m.state_dict().items() if not k.startswith(("comp_", "syn_"))}
    m2 = build_model(64, 2, 2, 32).eval()
    m2.load_state_dict(old)
    with torch.no_grad():
        l3, v3, _ = m2(f, c, mk)
    assert torch.allclose(l1[legal], l3[legal], atol=1e-5) and torch.allclose(v1, v3, atol=1e-6), "旧检查点读进来结果变了"
    comps = load_target_comps()
    idx = {card_id(i): i for i in range(1, n_cards())}
    apm = next(x for x in comps if x["name"] == "机械 APM")
    cc = c.clone()
    cc[:, C_BOARD:C_BOARD + 2] = torch.tensor([idx[x] for x in sorted(apm["core"])])
    ff = f.clone()
    ff[:, list(F_LOBBY.values())] = 0.0
    ff[:, F_LOBBY["MECHANICAL"]] = 1.0
    with torch.no_grad():
        feats = ref_m.comp(soft_log(ff), cc)
    assert feats.shape == (len(cc), K_MAX, COMP_F)
    tab = CompTables(comps, [base_of_index(i) if i else "" for i in range(n_cards())])
    want = tab.scores(cc.numpy(), lobby_bits(ff.numpy()))
    got = feats[:, :len(comps), 5].numpy()
    assert np.allclose(got, want, atol=1e-5), np.abs(got - want).max()
    k_apm = [x["name"] for x in comps].index("机械 APM")
    assert (feats[:, k_apm, 6] == 1).all(), "机械 APM 应该是主选"
    # 体系感知的输入层不是 0 以后：优化前向和原写法仍然一致，梯度能传到 comp_proj
    torch.nn.init.normal_(ref_m.comp_proj.weight, std=0.05)
    with torch.no_grad():
        ref_m.pack = False
        l4, _, _ = ref_m(ff, cc, mk, slow=True)
        ref_m.pack = True
        l5, _, _ = ref_m(ff, cc, mk)
    assert torch.allclose(l4[legal], l5[legal], atol=2e-4, rtol=1e-4)
    ref_m.zero_grad()
    lg, vv, _ = ref_m(ff, cc, mk)
    (vv.sum() + ref_m.last_comp_logits.sum()).backward()
    assert ref_m.comp_proj.weight.grad.abs().sum() > 0 and ref_m.comp_head.weight.grad.abs().sum() > 0
    torch.nn.init.zeros_(ref_m.comp_proj.weight)
    # 旧检查点的优化器状态（少了 comp_proj / comp_head 这 4 个参数）也能接上
    from rl.ppo import grow_opt_state
    opt_new = torch.optim.Adam(m2.parameters(), lr=1e-4)
    names = [k for k, _ in m2.named_parameters()]
    assert names[-6:] == ["comp_proj.weight", "comp_proj.bias", "comp_head.weight", "comp_head.bias",
                          "syn_proj.weight", "syn_proj.bias"], names[-6:]
    opt_old = torch.optim.Adam([p_ for k, p_ in m2.named_parameters() if not k.startswith(("comp_", "syn_"))], lr=1e-4)
    m2.zero_grad()
    m2(f, c, mk)[1].sum().backward()
    opt_old.step()
    od = opt_old.state_dict()
    opt_new.load_state_dict(grow_opt_state(opt_new, od))
    assert len(opt_new.state) == len(od["state"]) and torch.equal(opt_new.state[next(m2.parameters())]["exp_avg"],
                                                                   od["state"][0]["exp_avg"])
    opt_new.step()                                     # 新参数第一次更新时自己建 Adam 状态
    # 联动感知（第 15313 轮起）：每张卡和体系 / 引擎 / 三连 / 经济的关系
    from rl.comp import ENGINE_RULES, is_econ
    from rl.model import SYN_F
    from tg.catalog import catalog
    from tg.obs import C_HAND, C_SHOP
    det = catalog()
    brann = "BG_LOE_077"
    bc = next(x for x in sorted(idx) if x != brann and det.get(x, {}).get("kind") == "minion" and det[x].get("in_pool")
              and "Battlecry" in str(det[x].get("text")) and x == base_of_index(idx[x]))
    econ = next(x for x in sorted(idx) if is_econ(det.get(x) or {}) and det[x].get("kind") == "minion" and det[x].get("in_pool"))
    core = sorted(apm["core"])
    c2 = torch.zeros_like(c[:1])
    c2[0, C_BOARD:C_BOARD + 3] = torch.tensor([idx[core[0]], idx[core[1]], idx[brann]])
    c2[0, C_HAND:C_HAND + 2] = torch.tensor([idx[bc], idx[core[0]]])
    c2[0, C_SHOP:C_SHOP + 2] = torch.tensor([idx[econ], idx[core[1]]])
    with torch.no_grad():
        _, syn = ref_m.comp.sense(soft_log(ff[:1]), c2)
    assert syn.shape[-1] == SYN_F and brann in ENGINE_RULES
    nb, nh = C_HAND - C_BOARD, C_SHOP - C_HAND
    b0, b2, h0, h1, s0, s1 = syn[0, 0], syn[0, 2], syn[0, nb], syn[0, nb + 1], syn[0, nb + nh], syn[0, nb + nh + 1]
    assert b0[0] == 1 and s1[0] == 1 and b2[0] == 0, "机械 APM 的核心卡（场上 / 酒馆）应该标成主选核心卡"
    assert h0[6] == 1, "手里的战吼随从是场上布莱恩的燃料"
    from rl.comp import engine_score, engine_tables
    etab = engine_tables([base_of_index(i) if i else "" for i in range(n_cards())])
    cn = c2[0].numpy()
    phi, _ = engine_score(cn[C_BOARD:C_HAND], np.concatenate([cn[C_BOARD:C_HAND], cn[C_HAND:C_SHOP]]), etab)
    assert abs(float(b2[7]) - min(1.0, phi)) < 1e-6 and phi > 0, (float(b2[7]), phi)   # 布莱恩：燃料几成
    assert h1[8] == 0.5 and b0[8] == 0.5 and s1[8] == 0.5, "同名卡已有张数（三连进度）"
    assert s0[9] == 1, "倒转经济的卡"
    assert float(syn[0, nb + 2:nb + nh].abs().sum()) == 0, "空格子全是 0"
    torch.nn.init.normal_(ref_m.syn_proj.weight, std=0.05)
    with torch.no_grad():
        ref_m.pack = False
        l6, _, _ = ref_m(ff[:1], c2, mk[:1], slow=True)
        ref_m.pack = True
        l7, _, _ = ref_m(ff[:1], c2, mk[:1])
    lg1 = mk[:1].bool()
    assert torch.allclose(l6[lg1], l7[lg1], atol=2e-4, rtol=1e-4), "联动感知：优化前向和原写法不一致"
    torch.nn.init.zeros_(ref_m.syn_proj.weight)
    print(f"体系感知：旧检查点兼容（模型和优化器），阵容得分和 rl/comp 一致，{len(comps)} 套阵容；联动感知标记正确"
          f"（燃料 {det[bc].get('name_zh')}、经济 {det[econ].get('name_zh')}）")

    # 训练数据按压缩后的半精度存：极大的数值也不会变成 inf
    from rl.ppo import Policy
    pol = Policy(ref_m, torch.device("cpu"))
    huge = o["features"].copy(); huge[:, :100] = 1e7
    res = pol(huge, o["cards"], o["action_mask"], want_obs=True)
    assert np.isfinite(res[3]).all() and res[3].dtype == np.float16
    print("压缩存储没有溢出")

    # GAE：γ=1、λ=1 时优势 = 最终奖励 - 价值；另一个槽位没打完，用自举价值
    r = Rollout(2)
    for i in range(3):
        r.add([0, 1], np.zeros((2, 3), np.float32), np.zeros((2, 2), np.int32), np.zeros((2, 8), bool),
              np.zeros(2, np.int64), np.zeros(2, np.float32), np.full(2, 0.1 * i, np.float32))
    r.reward(0, 1.0, True, 1, comp=4)
    b = r.collect(1.0, 1.0, np.asarray([0.0, 0.5], np.float32))
    s0, s1 = b["slot"] == 0, b["slot"] == 1
    assert np.allclose(b["adv"][s0], [1.0, 0.9, 0.8]) and np.allclose(b["ret"][s0], 1.0), b
    assert np.allclose(b["ret"][s1], 0.5) and list(b["place"][s0]) == [1, 1, 1] and list(b["place"][s1]) == [0, 0, 0]
    assert list(b["comp"][s0]) == [4, 4, 4] and list(b["comp"][s1]) == [-1, -1, -1]          # 局末成了哪一套（体系感知辅助头）
    print("GAE 检查通过")

    # 混合精度：GPU 上用 float16，CPU 上用 bfloat16 走一遍前向和反向（之前 float16 下屏蔽值溢出）
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float16 if dev == "cuda" else torch.bfloat16
    small = small.to(dev).train()
    o = env.observe()
    f, c, mk = (torch.as_tensor(o["features"], device=dev), torch.as_tensor(o["cards"], device=dev).long(),
                torch.as_tensor(o["action_mask"], device=dev))
    with torch.autocast(dev, dtype=dt):
        logits, v, pl = small(f, c, mk)
    loss = torch.nn.functional.log_softmax(logits, -1).logsumexp(-1).mean() + v.mean() + pl.mean()
    loss.backward()
    assert torch.isfinite(loss), loss
    print(f"混合精度（{dev} {dt}）前向反向通过")

    # 后期局面数值很大（攻防上万）：输入压缩后半精度下也不能出 NaN
    with torch.no_grad(), torch.autocast(dev, dtype=dt):
        big = f.clone()
        big[:, :] = big * 1000.0 + 5000.0 * (big != 0)
        lg, vv, pp = small(big, c, mk)
    assert torch.isfinite(lg).all() and torch.isfinite(vv).all() and torch.isfinite(pp).all()
    a2, _, _ = small.act(big, c, mk)
    print("大数值输入没有 NaN")
    # 健康检查和回退：参数被弄成 NaN 后能发现并回到上一次健康的状态
    import tempfile
    from rl.ppo import Config, Trainer
    with tempfile.TemporaryDirectory() as d:
        cfg = Config(run_dir=d, device="cpu", games_bot=2, games_league=1, steps_per_iter=8, d=32, layers=1, heads=2,
                     dc=16, snapshot_every=0, eval_every=0, save_every=0, threads=1, minibatch=64, micro_batch=32)
        tr = Trainer(cfg)
        tr.train(1)
        tr.mark_good()
        lr0 = tr.opt.param_groups[0]["lr"]
        with torch.no_grad():
            next(tr.model.parameters()).fill_(float("nan"))
        prob = tr.health({})
        assert prob, "没发现 NaN 参数"
        tr.rollback(prob)
        assert all(torch.isfinite(p).all() for p in tr.model.parameters())
        assert abs(tr.opt.param_groups[0]["lr"] - lr0 / 2) < 1e-12
        tr.train(1)
        assert not tr.health({})
        # 引擎内部出错（Rust panic → PanicException）：这组对局作废、换新环境，训练接着跑
        PanicException = type("PanicException", (BaseException,), {})
        name, env, off = tr.envs[0]
        real = env.step
        calls = {"n": 0}

        def boom(a):
            calls["n"] += 1
            if calls["n"] == 3:
                raise PanicException("index out of bounds: the len is 5 but the index is 6")
            return real(a)
        env.step = boom
        tr.train(1)
        assert getattr(tr, "env_rebuilds", 0) == 1 and tr.envs[0][1] is not env, "引擎出错后没有换环境"
        tr.train(1)
        # 考试守护：凑满窗口创新低 → best.pt；退步 → 换回 best.pt、系数缩放；latest.pt 带着守护状态
        tr.cfg.guard, tr.cfg.guard_window, tr.cfg.guard_patience, tr.cfg.guard_margin = True, 2, 1, 0.3
        tr.guard_base = {"pro_coef": 0.1}
        assert "guard_avg" not in tr._guard(4.0)
        assert tr._guard(4.0)["guard_best"] == 4.0 and (Path(d) / "best.pt").exists()
        ref = [p.detach().clone() for p in tr.model.parameters()]
        with torch.no_grad():
            for p in tr.model.parameters():
                p.add_(1.0)
        out = tr._guard(5.0)
        assert out.get("guard_fix"), out
        assert all(torch.equal(a, b) for a, b in zip(ref, tr.model.parameters())), "没换回 best.pt"
        assert abs(tr.cfg.pro_coef - 0.07) < 1e-9 and tr.guard_st["n"] == 1
        tr.train(1)                                                    # 换回以后接着练（新优化器）
        tr.save()
        tr.guard_st, tr.cfg.pro_coef = tr._guard_new(), 0.1
        tr.load(Path(d) / "latest.pt")
        assert tr.guard_st["n"] == 1 and abs(tr.cfg.pro_coef - 0.07) < 1e-9
        tr.log_f.close()
    print("健康检查和回退通过；引擎出错能换环境接着练；考试守护能存最好的、退步时换回来")
    print("TEST OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
