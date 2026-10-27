"""B1：现实对手画像 vs TavernGym 脚本机器人画像。

用法：
  python opp_profile.py [对局记录目录或文件 ...] [--games 60] [--save]
    不给对局记录：用 bgrl/rl/real_profile.json（仓库里存的汇总画像）
    --save：把这次从对局记录算出的现实画像写回 bgrl/rl/real_profile.json（只有每回合的汇总数字）
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from rl.league import REAL_PROFILE, distance, load_real_profile, real_profile, sim_profile  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--annotate", action="store_true")
    a = ap.parse_args()
    files: list[str] = []
    for p in a.paths:
        files += sorted(glob.glob(os.path.join(p, "*.jsonl"))) if os.path.isdir(p) else [p]
    if files:
        real = real_profile(files)
        if a.save:
            REAL_PROFILE.write_text(json.dumps(real, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"现实画像已保存：{REAL_PROFILE}")
    else:
        real = load_real_profile()
    bot = sim_profile(None, games=a.games)
    lines = [f"现实：{real['games']} 局" + (f"，你的平均名次 {real['avgPlace']:.2f}" if real.get("avgPlace") else "") +
             f"；机器人：{bot['games']} 局自我对弈；距离 {distance(bot, real):.2f}",
             "回合 | 现实对手阵容总攻防 中位数(四分位) 等级 血量 | 机器人 中位数 等级 血量"]
    for t in range(1, 16):
        r, b = real["turns"].get(str(t)), bot["turns"].get(str(t))
        if not r or not b:
            continue
        lines.append(f"{t:>2} | {r['boardMedian']:>7.0f} ({r['boardQ1']:.0f}–{r['boardQ3']:.0f}) {r['tier'] or 0:.1f} "
                     f"{r['hp'] or 0:.0f} | {b['boardMedian']:>6.0f} {b['tier'] or 0:.1f} {b['hp'] or 0:.0f}")
    print("\n".join(lines))
    if a.annotate:
        print("::notice::" + " ".join(lines[:1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
