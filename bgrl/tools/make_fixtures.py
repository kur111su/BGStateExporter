"""从对局记录生成 A4 / A5 检查用的局面文件。

用法：python make_fixtures.py 对局记录.jsonl ... --out tests/fixtures/recruit_states_v3.jsonl
每回合取一条局面稳定的招募记录（带 _seen：对手最近一次交手的阵容），去掉面板用的字段。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from tg.logs import load_game, recruit_snapshots  # noqa: E402

KEEP = ("seq", "turn", "phase", "availableRaces", "anomalyDbfId", "player", "tavern", "leaderboard",
        "nextOpponentPlayerId", "offered", "damageCap", "damageCapEnabled", "gameTags", "_file", "_seen")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    n = 0
    with open(a.out, "w", encoding="utf-8") as fh:
        for f in a.files:
            for s in recruit_snapshots(load_game(f)):
                fh.write(json.dumps({k: s.get(k) for k in KEEP if k in s}, ensure_ascii=False) + "\n")
                n += 1
    print(f"{n} 个局面 → {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
