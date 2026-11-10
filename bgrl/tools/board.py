"""训练看板（像 wandb 那样看曲线）：读 runs/<run>/log.jsonl，写 runs/<run>/dashboard.md + board/*.svg。画图在 rl/board.py。

  bg.py board                     画一次（默认 runs/v3，最近 3000 轮）
  bg.py board --window 1000       只看最近 1000 轮
  bg.py board --run runs/v3 --open  画完用默认程序打开
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paths import RUNS  # noqa: E402
from rl.board import render  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(RUNS / "v3"))
    ap.add_argument("--window", type=int, default=3000, help="除了考试那张图，其余看最近多少轮（默认 3000）")
    ap.add_argument("--open", action="store_true", help="画完用默认程序打开 dashboard.md")
    a = ap.parse_args(argv)
    run = Path(a.run)
    p = render(run, a.window)
    if p is None:
        print(f"{run / 'log.jsonl'} 里还没有训练记录", flush=True)
        return 1
    print(f"看板 → {p}（图在 {run / 'board'}）", flush=True)
    if a.open:
        import os
        if hasattr(os, "startfile"):
            os.startfile(str(p))                           # noqa: S606 —— Windows 用默认程序打开
    return 0


if __name__ == "__main__":
    sys.exit(main())
