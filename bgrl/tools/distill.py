"""离线向老师学（python bg.py distill）：先把老师（外部模型）的打法整个搬过来，再回去接着 PPO。

为什么：PPO 里向老师学（teacher_coef 0.05）每轮只在一半的局面、老师有把握的那一步上推一下，
和老师一致 0.43 → 0.73 用了 1600 多轮（约 30 小时），这期间半像老师、半像自己，考试反而从 4.7 变差到 5.1。
这里每个局面都拿老师的整个概率分布当答案（KL），一轮几十秒，一两个小时能追到老师附近。

做法（DAgger）：
  1. 打 --games 局（对手和考试一样：第 5 回合起全换成高分阵容），学习者座位每一回合按概率 β 交给老师打（取最大概率），
     否则模型自己打（按概率抽）。β 从 --beta 慢慢降到 0.2：先看老师怎么走，再在自己会走到的局面上学老师怎么接。
  2. 每个学习者的局面都记下老师的答案（不许挪随从的那个分布；挪随从还是模型自己的）。
  3. 最近 3 轮的局面放在一起，按 KL(老师 ‖ 模型) 训练 --epochs 遍（只算挪随从以外的动作）。
  4. 每 --eval-every 轮考一次（128 局、和训练里的考试一样），记下最好的一版。
结束时把考试最好的一版（包括开始前那版）写回 runs/v3/latest.pt，优化器重新开始、学习率慢慢爬上来；
开始前的 latest.pt 留在 runs/v3/pre_distill.pt。面板的“停止”：这一轮学完、存盘退出。
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.8")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.6")

import argparse
import json
import shutil
import sys
import time
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

A_MOVE = 196


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recipe", default="configs/v3.json")
    ap.add_argument("--rounds", type=int, default=120)
    ap.add_argument("--games", type=int, default=128, help="每轮打几局")
    ap.add_argument("--epochs", type=float, default=2.0, help="每轮在最近几轮的局面上学几遍（按这一轮的局面数算）")
    ap.add_argument("--keep", type=int, default=3, help="最近几轮的局面放在一起学")
    ap.add_argument("--beta", type=float, default=0.7, help="一开始每个回合交给老师打的概率（慢慢降到 --beta-end）")
    ap.add_argument("--beta-end", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--temp", type=float, default=1.0, help="老师分布的温度（<1 更尖）")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-games", type=int, default=128)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--device", default="")
    ap.add_argument("--teacher", default="", help="老师模型文件（默认用配方里的 teacher）")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--smoke", action="store_true", help="极小规模跑通（CI）：不读 / 不写 runs/v3")
    return ap.parse_args(argv)


def build_trainer(a):
    import torch
    from tools.train import parse as tparse
    argv = ["--recipe", a.recipe] + (["--threads", str(a.threads)] if a.threads else []) + \
        (["--device", a.device] if a.device else []) + (["--teacher", a.teacher] if a.teacher else [])
    cfg, ta = tparse(argv)
    root = HERE.parent
    if a.smoke:
        cfg.d, cfg.layers, cfg.heads, cfg.dc = 64, 2, 2, 32
        cfg.run_dir = str(root / "runs" / "_distill_smoke")
        cfg.threads = min(cfg.threads or 2, os.cpu_count() or 1)
    else:
        cfg.run_dir = ta.run
    from rl.ppo import Trainer
    tr = Trainer(cfg)
    latest = Path(cfg.run_dir) / "latest.pt"
    if not a.smoke:
        if not latest.exists():
            raise SystemExit(f"没有 {latest}：先训练过才有东西可以学")
        tr.load(latest)
    if getattr(tr, "teacher", None) is None:
        if a.smoke:                                     # CI 没有别人的权重：拿一份随机的同结构模型当老师（只验证流程）
            from rl.model import build_model
            tr.teacher = build_model(**tr.mkw).to(tr.dev).eval()
            for q in tr.teacher.parameters():
                q.requires_grad_(False)
        else:
            raise SystemExit(f"没有老师（配方里 teacher = {cfg.teacher!r}，文件要在 bgrl/{cfg.teacher}）")
    torch.manual_seed(a.seed)
    return tr, latest


class Buffer:
    """最近几轮的局面：压缩后的特征（float16）、卡牌、合法动作（按位压缩）、老师的对数概率（float16）。"""

    def __init__(self, keep: int):
        self.rounds = deque(maxlen=keep)

    def add(self, rows: dict):
        if len(rows["f"]):
            self.rounds.append(rows)

    def __len__(self):
        return sum(len(r["f"]) for r in self.rounds)

    def sample(self, rng, n):
        import numpy as np
        sizes = np.array([len(r["f"]) for r in self.rounds])
        pick = rng.integers(0, sizes.sum(), n)
        edges = np.cumsum(sizes)
        out = {k: [] for k in ("f", "c", "m", "t")}
        for ri, r in enumerate(self.rounds):
            lo = edges[ri] - sizes[ri]
            j = pick[(pick >= lo) & (pick < edges[ri])] - lo
            for k in out:
                out[k].append(r[k][j])
        return {k: np.concatenate(v) for k, v in out.items()}


def teacher_logp(tr, f, c, m, temp):
    """老师在原始特征上的对数概率（不许挪随从——还有别的动作时；外部模型自己会去掉）。MPS：行数凑成 64 的倍数。"""
    import numpy as np
    import torch
    import torch.nn.functional as F
    n = len(f)
    if tr.dev.type == "mps" and n % 64:
        pad = 64 - n % 64
        rep = lambda x: np.concatenate([x, np.repeat(np.asarray(x)[-1:], pad, 0)])   # noqa: E731
        f, c, m = rep(f), rep(c), rep(m)
    ft = torch.as_tensor(np.ascontiguousarray(f), dtype=torch.float32).to(tr.dev)
    ct = torch.as_tensor(np.ascontiguousarray(c), dtype=torch.long).to(tr.dev)
    mt = torch.as_tensor(np.ascontiguousarray(m), dtype=torch.bool).to(tr.dev)
    other = mt[:, :A_MOVE].any(-1, keepdim=True)
    mt = mt & ~(other & (torch.arange(mt.shape[1], device=mt.device) >= A_MOVE))
    with torch.inference_mode(), torch.autocast(tr.dev.type, dtype=torch.bfloat16, enabled=tr.amp):
        lg = tr.teacher(ft, ct, mt)[0].float()
    lg = lg.masked_fill(~mt, -1e9) / max(1e-3, temp)
    return F.log_softmax(lg, -1).cpu().numpy()[:n]


def collect(tr, a, rnd, beta, rng):
    """打一轮：学习者座位每回合按 β 交给老师（取最大概率），否则模型自己（按概率抽）；每个局面记老师的答案。"""
    import numpy as np
    from taverngym import BatchEnv

    from tg.actions import is_panic
    seed = a.seed * 1000 + rnd
    env = BatchEnv(a.games, seed=seed, controlled=(0,), threads=tr.cfg.threads, config=tr.cfg.rules())
    booster = tr._make_booster(1.0, seed)
    if booster is not None:
        booster.strength = 1.0
    obs = env.observe()
    who = {}                                          # (局, 回合) → 这回合交给老师？
    places, rows = {}, {"f": [], "c": [], "m": [], "t": []}
    n_teacher = n_dec = 0
    tr.model.eval()
    for _ in range(30000):
        if len(places) >= a.games:
            break
        ready = np.flatnonzero(obs["ready"])
        actions = np.zeros(env.n_slots, dtype=np.int64)
        if len(ready):
            f, c, m = obs["features"][ready], obs["cards"][ready], obs["action_mask"][ready]
            sa, _, _, fz = tr.learner(f, c, m, want_obs=True)
            tl = teacher_logp(tr, f, c, m, a.temp)
            ta = tl.argmax(-1)
            turn = np.rint(np.asarray(f[:, 0], dtype=np.float32) * 20).astype(int)
            use_t = np.array([who.setdefault((int(i), int(t)), bool(rng.random() < beta)) for i, t in zip(ready, turn)])
            actions[ready] = np.where(use_t, ta, sa)
            n_teacher += int(use_t.sum())
            n_dec += len(ready)
            keep = m[:, :A_MOVE].any(-1) & np.array([int(i) not in places for i in ready])   # 只有挪随从可做的局面不学
            if keep.any():
                rows["f"].append(fz[keep])
                rows["c"].append(np.asarray(c[keep], dtype=np.int16))
                rows["m"].append(np.packbits(np.asarray(m[keep], dtype=bool), axis=1))
                rows["t"].append(np.maximum(tl[keep], -30.0).astype(np.float16))
        try:
            _, dones, info = env.step(actions)
        except BaseException as ex:                    # noqa: BLE001
            if not is_panic(ex):
                raise
            print(f"  TavernGym 引擎出错（{str(ex)[:100]}），这一轮按已经打完的 {len(places)} 局算", flush=True)
            break
        for i in np.flatnonzero(dones):
            if int(i) not in places:
                places[int(i)] = int(info["placement"][i])
        obs = env.observe()
        if booster is not None and booster.boost(env, obs["features"]):
            obs = env.observe()
    out = {k: (np.concatenate(v) if v else np.zeros((0,))) for k, v in rows.items()}
    ps = [p for p in places.values() if p > 0]
    return out, ps, n_teacher / max(1, n_dec)


def train_round(tr, opt, buf, n_steps, a, rng):
    import numpy as np
    import torch
    import torch.nn.functional as F
    A_N = tr.model.L.actions
    st = {"kl": [], "agree": [], "ent_s": [], "ent_t": []}
    tr.model.train()
    for _ in range(n_steps):
        b = buf.sample(rng, a.batch)
        f = torch.as_tensor(b["f"], dtype=torch.float32).to(tr.dev)
        c = torch.as_tensor(b["c"].astype(np.int64)).to(tr.dev)
        m = torch.as_tensor(np.unpackbits(b["m"], axis=1)[:, :A_N].astype(bool)).to(tr.dev)
        tl = torch.as_tensor(b["t"], dtype=torch.float32).to(tr.dev)
        nm = m & (torch.arange(A_N, device=m.device) < A_MOVE)
        with torch.autocast(tr.dev.type, dtype=torch.bfloat16, enabled=tr.amp):
            lg = tr.model(f, c, m, compressed=True)[0]
        lq = F.log_softmax(lg.float().masked_fill(~nm, -1e9), -1)
        pt = tl.exp() * nm
        kl = (pt * (tl - lq)).masked_fill(~nm, 0).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        kl.backward()
        torch.nn.utils.clip_grad_norm_(tr.model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            st["kl"].append(float(kl))
            st["agree"].append(float((lq.argmax(-1) == tl.argmax(-1)).float().mean()))
            st["ent_s"].append(float(-(lq.exp() * lq).masked_fill(~nm, 0).sum(-1).mean()))
            st["ent_t"].append(float(-(pt * tl).masked_fill(~nm, 0).sum(-1).mean()))
        del f, c, m, tl, lg, lq, pt, kl
    tr.model._ver = getattr(tr.model, "_ver", 0) + 1       # 采样用的副本下次出动作前同步
    tr.model.eval()
    if tr.dev.type == "mps":
        torch.mps.empty_cache()
    return {k: round(float(np.mean(v)), 3) for k, v in st.items() if v}


def exam(tr, games):
    import numpy as np

    from rl.rollout import place_score_equiv
    place, top1 = tr.evaluate(games=games, seed=1999, boosted=True)
    ps = tr.last_eval_places
    return {"place": place, "top1": top1, "top4": round(float(np.mean([p <= 4 for p in ps])), 3) if ps else None,
            "score": round(place_score_equiv(ps, tr.place_tab), 3) if ps else 99.0}


def main(argv=None) -> int:
    a = parse(argv)
    if a.smoke:
        a.rounds, a.games, a.eval_every, a.eval_games, a.batch = 2, 2, 2, 2, 64
    import numpy as np
    import torch
    tr, latest = build_trainer(a)
    run = Path(tr.cfg.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    stop = run / "STOP"
    stop.unlink(missing_ok=True)
    log = (run / "distill.jsonl").open("a", encoding="utf-8")
    if not a.smoke:
        shutil.copyfile(latest, run / "pre_distill.pt")
        print(f"开始前的模型留在 {run / 'pre_distill.pt'}", flush=True)
    rng = np.random.default_rng(a.seed)
    opt = torch.optim.Adam(tr.model.parameters(), lr=a.lr, eps=1e-5)
    buf = Buffer(a.keep)

    t0 = time.time()
    e0 = exam(tr, a.eval_games)
    print(f"开始前考试（{a.eval_games} 局，全强度高分阵容）：平均 {e0['place']} 名、吃鸡 {e0['top1']:.1%}、前四 {e0['top4']:.0%}、"
          f"等效 {e0['score']}（{time.time() - t0:.0f} 秒）", flush=True)
    best = {"score": e0["score"], "round": 0, "state": {k: v.detach().cpu().clone() for k, v in tr.model.state_dict().items()}}
    log.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "round": 0, "iter": tr.iter, "exam": e0}) + "\n")
    log.flush()
    stopped = False
    for rnd in range(1, a.rounds + 1):
        beta = a.beta + (a.beta_end - a.beta) * min(1.0, (rnd - 1) / max(1, a.rounds * 0.6))
        t1 = time.time()
        rows, ps, frac_t = collect(tr, a, rnd, beta, rng)
        buf.add(rows)
        t2 = time.time()
        n_steps = max(1, int(round(a.epochs * len(rows["f"]) / a.batch)))
        st = train_round(tr, opt, buf, n_steps, a, rng) if len(buf) >= a.batch or a.smoke and len(buf) else {}
        t3 = time.time()
        rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "round": rnd, "beta": round(beta, 3), "teacher_frac": round(frac_t, 3),
               "rows": int(len(rows["f"])), "buffer": len(buf), "place": round(float(np.mean(ps)), 3) if ps else None,
               "steps": n_steps, **st, "t_collect": round(t2 - t1, 1), "t_train": round(t3 - t2, 1)}
        print(f"第 {rnd}/{a.rounds} 轮：老师打 {frac_t:.0%}、这轮平均 {rec['place']} 名、{rec['rows']} 个局面；"
              f"和老师一致 {st.get('agree', 0):.2f}、KL {st.get('kl', 0):.3f}、熵 模型 {st.get('ent_s', 0):.2f} / 老师 {st.get('ent_t', 0):.2f}"
              f"（打 {t2 - t1:.0f} 秒、学 {t3 - t2:.0f} 秒）", flush=True)
        stopped = stop.exists()
        if rnd % a.eval_every == 0 or rnd == a.rounds or stopped:
            e = exam(tr, a.eval_games)
            rec["exam"] = e
            mark = ""
            if e["score"] < best["score"]:
                best = {"score": e["score"], "round": rnd,
                        "state": {k: v.detach().cpu().clone() for k, v in tr.model.state_dict().items()}}
                mark = "  ← 目前最好"
            print(f"  考试：平均 {e['place']} 名、吃鸡 {e['top1']:.1%}、前四 {e['top4']:.0%}、等效 {e['score']}"
                  f"（开始前 {e0['score']}）{mark}", flush=True)
        log.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log.flush()
        if stopped:
            stop.unlink(missing_ok=True)
            print("收到停止请求：这一轮学完，存盘退出", flush=True)
            break

    tr.model.load_state_dict({k: v.to(next(tr.model.parameters()).device) for k, v in best["state"].items()})
    tr.model._ver = getattr(tr.model, "_ver", 0) + 1
    if best["round"] == 0:
        print(f"学完以后考试没有比开始前好（最好 {best['score']}），模型保持开始前那版", flush=True)
    else:
        print(f"用第 {best['round']} 轮那版（等效 {best['score']}，开始前 {e0['score']}）", flush=True)
    if not a.smoke:
        tr.opt = torch.optim.Adam(tr.model.parameters(), lr=tr.cfg.lr, eps=1e-5)   # 回到 PPO：优化器重新开始、学习率慢慢爬上来
        tr._start_warmup()
        tr._reset_drift()
        tr.save()
        print(f"→ {latest}（接着点“训练”就从这版接着练）", flush=True)
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
