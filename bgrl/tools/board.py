"""训练看板：读 runs/<run>/log.jsonl，写 runs/<run>/dashboard.md（关键指标、阵容测试、课程的表、事件）。内容在 rl/board.py。

  bg.py board                     写一次（默认 runs/v3）
  bg.py board --window 1000       表只算最近 1000 轮
  bg.py board --run runs/v3 --open  写完用默认程序打开
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
    ap.add_argument("--window", type=int, default=3000, help="送整套阵容 / 成型阵容课程的表、回退次数算最近多少轮（默认 3000）")
    ap.add_argument("--open", action="store_true", help="写完用默认程序打开 dashboard.md")
    a = ap.parse_args(argv)
    run = Path(a.run)
    p = render(run, a.window)
    if p is None:
        print(f"{run / 'log.jsonl'} 里还没有训练记录", flush=True)
        return 1
    print(f"看板 → {p}", flush=True)
    if a.open:
        import subprocess
        import sys
        if sys.platform == "darwin":
            subprocess.run(["open", str(p)], check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
