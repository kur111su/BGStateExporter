"""把你的对局记录变成“示范”数据：每一步操作前的局面（TavernGym 观察）+ 你实际做的动作。

训练时加一点模仿损失（rl/ppo.py 的 bc_coef）：模型在 v3 里怎么都学不会按时升本（第 13 回合平均 3–4 本，
你约 5.8 本；TavernBench 的回合计划题里该升本的 19 题它只选对 1 题），奖励信号太远，探索也碰不到。
你的操作直接告诉它“这种局面下升本”。另外留出每 5 局里的 1 局不参与训练，用来测模型和你的操作对得上多少
（日志里的 human_top1 / human_upgrade / human_end），这比影子模式评估快，也不用开游戏。

每个回合除了你的各个操作，还在回合最后一步之后加一条“结束回合”（不然模型只见过“有事可做”的局面）。
局面重建不了、动作在重建的局面里不合法的跳过。

用法：bg.py demos              读插件的 games 目录，写 runs/demos.npz
      bg.py demos --if-stale   有新对局记录才重新生成（train.bat 每次启动时用）
      bg.py demos --fixture tests/fixtures/replay.jsonl.gz   用 CI 样本
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paths import RUNS, games_dir  # noqa: E402
from tg.actions import A_END  # noqa: E402

MAX_CANDS = 8
KINDS = ("buy", "sell", "play", "cast", "magnetize", "upgrade", "refresh", "hero_power", "move", "end")


def _obs(g):
    f, c = g.observe(0)
    m = np.frombuffer(g.legal_mask(0), dtype=np.uint8).astype(bool)
    return np.frombuffer(f, "<f4").copy(), np.frombuffer(c, "<i4").copy(), m


def samples_from_actions(acts, split: int, seed0: int = 0):
    """acts：同一局的动作（按时间顺序）。返回 [(f, c, m, 候选动作, 类别, 回合, split)] 和跳过原因计数。"""
    from tg.bridge import build_game
    from tg.replay import translate
    out, skip = [], Counter()
    for i, a in enumerate(acts):
        nxt = acts[i + 1] if i + 1 < len(acts) else None
        jobs = []
        if a.kind != "multi":
            jobs.append((a.before, a.kind, a))
        if nxt is None or nxt.turn != a.turn:                     # 这一回合的最后一步：之后是“结束回合”
            jobs.append(((a.afters or [a.after])[-1], "end", None))
        for state, kind, act in jobs:
            try:
                g, _ = build_game(state, seed=seed0 + i)
                f, c, m = _obs(g)
            except Exception as ex:                                # noqa: BLE001
                skip[f"重建失败 {type(ex).__name__}"] += 1
                continue
            if kind == "end":
                cands = [A_END]
            else:
                cands = translate(act)
            cands = [x for x in cands if 0 <= x < len(m) and m[x]][:MAX_CANDS]
            if not cands:
                skip[f"{kind} 不合法或翻译不了"] += 1
                continue
            out.append((f, c, m, cands, kind, int(a.turn or 0), split))
    return out, skip


def _game_files(paths):
    from tools.tg_replay import schema_of
    files = []
    for p in paths:
        files += sorted(glob.glob(os.path.join(p, "*.jsonl"))) if os.path.isdir(p) else [p]
    return [f for f in files if schema_of(f) == 2]


def save(rows, out: Path, meta: dict):
    n = len(rows)
    cand = np.full((n, MAX_CANDS), -1, dtype=np.int16)
    for i, r in enumerate(rows):
        cand[i, :len(r[3])] = r[3]
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, f=np.stack([r[0] for r in rows]).astype(np.float32),
                        c=np.stack([r[1] for r in rows]).astype(np.int32),
                        m=np.packbits(np.stack([r[2] for r in rows]), axis=1), cand=cand,
                        kind=np.array([KINDS.index(r[4]) for r in rows], dtype=np.int8),
                        turn=np.array([r[5] for r in rows], dtype=np.int16),
                        split=np.array([r[6] for r in rows], dtype=np.int8),
                        n_actions=np.int32(len(rows[0][2])), meta=np.array(json.dumps(meta, ensure_ascii=False)))
    os.replace(tmp, out)


def load(path) -> dict:
    """训练用：返回各数组（m 已解包成 bool）。"""
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["m"] = np.unpackbits(d["m"], axis=1)[:, :int(d["n_actions"])].astype(bool)
    d["meta"] = json.loads(str(d["meta"]))
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--fixture", default="")
    ap.add_argument("--out", default=str(RUNS / "demos.npz"))
    ap.add_argument("--if-stale", action="store_true", help="没有比 demos.npz 更新的对局记录就不重新生成")
    ap.add_argument("--val-every", type=int, default=5, help="每几局留一局做验证（不参与训练）")
    a = ap.parse_args()
    out = Path(a.out)
    t0 = time.time()
    rows, skip = [], Counter()
    if a.fixture:
        from tools.tg_replay import actions_from_fixture
        acts = actions_from_fixture(a.fixture)
        # 样本是从多局里抽的，按 25 个一组当成“一局”（这样也有验证组）
        sources = [(f"fixture#{i // 25}", acts[i:i + 25]) for i in range(0, len(acts), 25)]
    else:
        files = _game_files(a.paths or [str(games_dir())])
        if not files:
            print(f"没有可用的对局记录（{a.paths or games_dir()}）")
            return 0
        if a.if_stale and out.exists() and max(os.path.getmtime(f) for f in files) <= out.stat().st_mtime:
            print(f"示范数据是最新的（{out}）")
            return 0
        sources = [(f, None) for f in files]
    from tg.logs import load_game, segment
    for k, (name, acts) in enumerate(sources):
        if acts is None:
            try:
                acts = segment(load_game(name))
            except Exception as ex:                               # noqa: BLE001
                print(f"[{k + 1}/{len(sources)}] {os.path.basename(name)}：读不了（{type(ex).__name__}），跳过", flush=True)
                continue
        split = 1 if a.val_every and k % a.val_every == a.val_every - 1 else 0
        r, s = samples_from_actions(acts, split, seed0=k * 1000)
        rows += r
        skip += s
        print(f"[{k + 1}/{len(sources)}] {os.path.basename(str(name))}：{len(r)} 条{'（验证）' if split else ''}", flush=True)
    if not rows:
        print("一条示范都没生成", dict(skip))
        return 1
    kinds = Counter(r[4] for r in rows)
    meta = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "games": len(sources), "samples": len(rows),
            "val": sum(r[6] for r in rows), "kinds": dict(kinds), "skipped": dict(skip)}
    save(rows, out, meta)
    print(f"示范数据：{len(sources)} 局，{len(rows)} 条（验证 {meta['val']}），按类别 {dict(kinds)}；"
          f"跳过 {dict(skip)}；{time.time() - t0:.0f} 秒 → {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
