"""B3：训练入口。

用法：
  python train.py --smoke                    几轮极小规模训练，检查能跑通（CI / 装好后先跑这个）
  python train.py --bench                    测速：采样每秒多少决策、每轮更新多久（不存检查点）
  python train.py --run runs/v1 [--iters 2000] [--resume]   正式训练
其它参数见 --help（和 bgrl/rl/ppo.py 的 Config 一一对应）。
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def main() -> int:
    from rl.ppo import Config, Trainer
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--run", default="")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stage", action="store_true", help="新阶段：以当前策略为锚加 KL 约束")
    for f in dataclasses.fields(Config):
        if f.name == "run_dir":
            continue
        ap.add_argument("--" + f.name.replace("_", "-"), type=type(f.default) if not isinstance(f.default, bool) else
                        (lambda s: s.lower() in ("1", "true", "yes")), default=None)
    a = ap.parse_args()
    cfg = Config()
    for f in dataclasses.fields(Config):
        v = getattr(a, f.name, None)
        if v is not None:
            setattr(cfg, f.name, v)
    root = HERE.parent
    if a.smoke:
        cfg.games_bot, cfg.games_league, cfg.steps_per_iter = 4, 2, 64
        cfg.d, cfg.layers, cfg.heads = 64, 2, 2
        cfg.snapshot_every, cfg.eval_every, cfg.save_every = 2, 3, 2
        cfg.profile_games, cfg.eval_games, cfg.minibatch = 1, 2, 256
        cfg.iters = a.iters or 4
        cfg.threads = min(cfg.threads, os.cpu_count() or 1)
        cfg.run_dir = str(root / "runs" / "_smoke")
    elif a.bench:
        cfg.snapshot_every = cfg.eval_every = cfg.save_every = 0
        cfg.iters = a.iters or 3
        cfg.run_dir = str(root / "runs" / "_bench")
    else:
        cfg.run_dir = a.run or str(root / "runs" / time.strftime("%Y%m%d_%H%M%S"))
    tr = Trainer(cfg)
    latest = Path(cfg.run_dir) / "latest.pt"
    if a.resume and latest.exists():
        tr.load(latest)
        print(f"从 {latest} 继续（第 {tr.iter} 轮）")
    if a.stage:
        tr.start_stage()

    def show(rec):
        keys = ("iter", "decisions", "dps", "t_collect", "t_update", "kl", "ent", "vf", "place_bot", "place_league",
                "eval_place", "eval_top1", "snapshot", "snapshot_dist", "early_stop")
        line = json.dumps({k: rec[k] for k in keys if k in rec}, ensure_ascii=False)
        print(("::notice::" if os.environ.get("GITHUB_ACTIONS") else "") + line, flush=True)
    tr.train(cfg.iters, on_log=show)
    if a.smoke:
        print("SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
