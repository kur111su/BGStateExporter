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
    r.reward(0, 1.0, True, 1)
    b = r.collect(1.0, 1.0, np.asarray([0.0, 0.5], np.float32))
    s0, s1 = b["slot"] == 0, b["slot"] == 1
    assert np.allclose(b["adv"][s0], [1.0, 0.9, 0.8]) and np.allclose(b["ret"][s0], 1.0), b
    assert np.allclose(b["ret"][s1], 0.5) and list(b["place"][s0]) == [1, 1, 1] and list(b["place"][s1]) == [0, 0, 0]
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
        tr.log_f.close()
    print("健康检查和回退通过")
    print("TEST OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
