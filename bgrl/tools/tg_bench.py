"""A1：检查 TavernGym 能否在这台电脑上运行，并测速。

用法：python tg_bench.py [--quick]
结果写到 bgrl/reports/tg_bench_时间.txt / .json（--quick 只跑几局、不写报告，供 CI 冒烟测试）。

测四件事：
  1. 一局完整 8 人对局（全部用 TavernGym 自带的脚本机器人），单线程每秒几局
  2. 批量环境 BatchEnv：1 个座位随机合法动作 + 7 个机器人，不同线程数下每秒多少个决策、多少局
  3. 批量环境 8 个座位都由“我们”控制（自我对弈的用法）
  4. 精确存档 / 读档往返
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

REPORT_DIR = Path(__file__).resolve().parent.parent / "reports"


def log(lines: list[str], msg: str) -> None:
    print(msg, flush=True)
    lines.append(msg)


def bot_games(tg, n: int) -> tuple[float, list]:
    t0 = time.perf_counter()
    last = []
    for seed in range(n):
        g = tg.Game(seed=seed)
        while not g.is_over():
            for s in range(g.num_players):
                if g.needs_action(s):
                    g.step(s, g.bot_action(s))
        last = list(g.placements())
    return n / (time.perf_counter() - t0), last


def batch_random(tg, np, n_games: int, threads: int, controlled, seconds: float) -> dict:
    env = tg.BatchEnv(n_games=n_games, seed=0, controlled=controlled, threads=threads)
    rng = np.random.default_rng(0)
    obs = env.observe()
    decisions = finished = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        mask = obs["action_mask"]
        # 每个槽位在合法动作里均匀随机选一个（不合法的槽位给 0，没准备好的会被忽略）
        noise = rng.random(mask.shape) * mask
        actions = noise.argmax(axis=1).astype("int64")
        ready = obs["ready"]
        _, dones, _ = env.step(actions)
        decisions += int(ready.sum())
        finished += int(np.asarray(dones).sum())
        obs = env.observe()
    dt = time.perf_counter() - t0
    feat = obs["features"]
    return {"games": n_games, "threads": threads, "controlled": len(list(controlled)),
            "decisionsPerSec": round(decisions / dt), "finishedEpisodesPerSec": round(finished / dt, 1),
            "obsFloats": int(feat.shape[1]), "obsCards": int(obs["cards"].shape[1]),
            "actions": int(obs["action_mask"].shape[1])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    lines: list[str] = []
    data: dict = {"time": datetime.now().isoformat(timespec="seconds"), "python": sys.version.split()[0],
                  "platform": platform.platform(), "cpus": os.cpu_count(), "processor": platform.processor()}
    log(lines, f"=== A1：TavernGym 检查 === {data['time']}；Python {data['python']}；{data['platform']}；"
               f"CPU 逻辑核数 {data['cpus']}；{data['processor']}")
    import numpy as np
    import taverngym as tg
    info = {}
    for name in ("build_info", "build_is_current"):
        fn = getattr(tg, name, None)
        if fn:
            try:
                info[name] = fn() if not callable(fn) else fn()
            except Exception as ex:          # noqa: BLE001
                info[name] = f"error: {ex}"
    data["build"] = {k: (v if isinstance(v, (str, int, float, bool, dict, list)) else str(v)) for k, v in info.items()}
    log(lines, f"taverngym 已加载：{data['build']}")

    n = 5 if args.quick else 200
    gps, last = bot_games(tg, n)
    data["botGamesPerSecSingleThread"] = round(gps, 1)
    log(lines, f"1. 机器人对局（单线程，{n} 局）：每秒 {gps:,.1f} 局；最后一局名次 {last}")
    if args.quick:
        return 0

    cpus = os.cpu_count() or 1
    runs = []
    for threads in sorted({1, max(1, cpus // 2), cpus}):
        r = batch_random(tg, np, n_games=max(64, threads * 16), threads=threads, controlled=(0,), seconds=8)
        runs.append(r)
        log(lines, f"2. 批量环境（1 个随机座位 + 7 个机器人，{threads} 线程，{r['games']} 局并行）："
                   f"每秒 {r['decisionsPerSec']:,} 个决策、{r['finishedEpisodesPerSec']} 局结束")
    r = batch_random(tg, np, n_games=max(64, cpus * 8), threads=cpus, controlled=range(8), seconds=8)
    runs.append(r)
    log(lines, f"3. 批量环境（8 个座位都随机，{cpus} 线程，{r['games']} 局并行）："
               f"每秒 {r['decisionsPerSec']:,} 个决策、{r['finishedEpisodesPerSec']} 局结束")
    log(lines, f"   观察：{r['obsFloats']} 个数值 + {r['obsCards']} 个卡牌编号；动作 {r['actions']} 个")
    data["batch"] = runs

    g = tg.Game(seed=7)
    for _ in range(300):
        if g.is_over():
            break
        for s in range(g.num_players):
            if g.needs_action(s):
                g.step(s, g.bot_action(s))
    saved = g.state_json()
    g2 = tg.Game.from_json(saved)
    same = g2.state_json() == saved
    data["saveRoundTrip"] = same
    log(lines, f"4. 存档读档往返：{'一致' if same else '不一致'}（存档 {len(saved) / 1024:.0f} KB）")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt = REPORT_DIR / f"tg_bench_{stamp}.txt"
    txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (REPORT_DIR / f"tg_bench_{stamp}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n报告已保存：{txt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
