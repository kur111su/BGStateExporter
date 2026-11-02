"""B3：训练入口。

用法：
  python train.py --smoke                    几轮极小规模训练，检查能跑通（CI / 装好后先跑这个）
  python train.py --bench [--size small|medium|large]   测速：每秒多少决策、采样/更新各多久、显存（不存检查点）
  python train.py --run runs/v1 [--iters 2000] [--resume]   正式训练
  python train.py --recipe configs/v3.json --resume        按配方文件训练（train.bat 就是这样）
其它参数见 --help（和 bgrl/rl/ppo.py 的 Config 一一对应）。
优先级：Config 默认值 < 配方文件（--recipe）的 config < 命令行参数。配方里还可以写 run（训练目录）。
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


def load_recipe(path: str) -> dict:
    """读配方文件：先按当前目录找，找不到再按 bgrl 目录找。config 里的每一项必须是 Config 的字段，类型按默认值转换。"""
    from rl.ppo import Config
    p = Path(path.replace("\\", "/"))                  # train.bat 写的是 configs\v3.json
    if not p.exists() and not p.is_absolute():
        p = HERE.parent / path
    if not p.exists():
        raise SystemExit(f"配方文件不存在：{path}")
    r = json.loads(p.read_text(encoding="utf-8"))
    fields = {f.name: f for f in dataclasses.fields(Config)}
    out = {}
    for k, v in (r.get("config") or {}).items():
        if k not in fields or k == "run_dir":
            raise SystemExit(f"配方 {p}：不认识的参数 {k}")
        d = fields[k].default
        out[k] = v if isinstance(d, bool) or d is None else type(d)(v)
    r["config"] = out
    return r


def parse(argv=None):
    """命令行 → (Config, 参数)。Config 默认值 < 配方 < 命令行；--smoke / --bench 的固定设置在 main 里再改。"""
    from rl.ppo import Config
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--run", default="")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stage", action="store_true", help="新阶段：以当前策略为锚加 KL 约束")
    ap.add_argument("--init", default="", help="新训练从这个检查点 / 快照的参数开始（已有 latest.pt 时忽略）")
    ap.add_argument("--recipe", default="", help="配方文件（JSON，例如 configs/v3.json）：训练目录和参数，命令行参数优先")
    for f in dataclasses.fields(Config):
        if f.name == "run_dir":
            continue
        ap.add_argument("--" + f.name.replace("_", "-"), type=type(f.default) if not isinstance(f.default, bool) else
                        (lambda s: s.lower() in ("1", "true", "yes")), default=None)
    a = ap.parse_args(argv)
    cfg = Config()
    recipe = load_recipe(a.recipe) if a.recipe else {}
    for k, v in (recipe.get("config") or {}).items():
        setattr(cfg, k, v)
    if not a.run and recipe.get("run"):
        a.run = recipe["run"]
    for f in dataclasses.fields(Config):
        v = getattr(a, f.name, None)
        if v is not None:
            setattr(cfg, f.name, v)
    return cfg, a


def main() -> int:
    from rl.ppo import Trainer
    cfg, a = parse()
    root = HERE.parent
    if a.smoke:
        cfg.games_bot, cfg.games_league, cfg.steps_per_iter = 4, 2, 64
        cfg.d, cfg.layers, cfg.heads, cfg.dc = 64, 2, 2, 32
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
    elif a.init:
        tr.init_from(a.init)
    if a.stage:
        tr.start_stage()

    def show(rec):
        if "rollback" in rec:
            print(json.dumps(rec, ensure_ascii=False), flush=True)
            return
        keys = ("iter", "decisions", "games_done", "dps", "t_collect", "t_env", "t_net", "t_update", "gpu_mem_gb", "kl", "ent", "vf", "place_bot", "place_league",
                "eval_place", "eval_top1", "eval_boost_place", "eval_boost_top1", "snapshot", "snapshot_dist", "snapshot_tier",
                "bc", "bc_scale", "pro_bc", "pro_top1", "pro_top1_train", "pro_upgrade", "pro_top1_hi", "pro_upgrade_hi", "pro_end", "pro_nll", "human_top1", "human_top1_train", "human_nll", "human_upgrade", "human_end", "human_sell", "up_turn", "ref_turn", "buy_turn", "sell_turn", "ref_waste", "tempo_bonus", "tier_gap", "up_afford", "up_t2", "starve_turn", "end_rich", "unspent", "pro_scale", "env_rebuild", "comp_q", "comp_coh", "comp_syn", "comp_key", "comp_golden", "boost_strength", "pro_injected", "pro_failed", "place_boost", "boost_place_avg", "boosts", "t_boost", "early_stop", "bad_rows", "skipped", "noop_frac")
        line = json.dumps({k: rec[k] for k in keys if k in rec}, ensure_ascii=False)
        print(("::notice::" if os.environ.get("GITHUB_ACTIONS") else "") + line, flush=True)
    tr.train(cfg.iters, on_log=show)
    if a.smoke:
        print("SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
