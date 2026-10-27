"""从对局记录生成 A5 回放检查用的精简文件（CI 用）。

用法：python make_replay_fixture.py 对局记录.jsonl ... --out tests/fixtures/replay_v3.jsonl.gz
每个动作一行：动作 + 动作前局面（重建用，去掉用不到的标签）+ 动作后各候选局面（只留比较用的字段）。
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from tg.logs import load_game, segment  # noqa: E402

KEEP = ("seq", "turn", "phase", "availableRaces", "player", "tavern", "leaderboard", "nextOpponentPlayerId",
        "offered", "_file", "_seen")
CARD_DROP = ("tags", "dbfId", "name")


def _slim_cards(cards):
    return [{k: v for k, v in c.items() if k not in CARD_DROP} for c in cards or []]


def slim_before(s: dict) -> dict:
    s = {k: s.get(k) for k in KEEP if k in s}
    p = dict(s["player"])
    p.pop("heroTags", None)
    p["playerTags"] = {k: v for k, v in (p.get("playerTags") or {}).items() if not k.isdigit()}
    for z in ("board", "hand", "trinkets"):
        p[z] = _slim_cards(p.get(z))
    s["player"] = p
    s["tavern"] = _slim_cards(s.get("tavern"))
    s["_seen"] = {h: {"turn": v["turn"], "board": _slim_cards(v["board"])} for h, v in (s.get("_seen") or {}).items()}
    return s


def slim_after(s: dict) -> dict:
    p = s["player"]
    keep = ("cardId", "position", "attack", "health", "golden", "taunt", "divineShield", "poisonous", "venomous",
            "reborn", "windfury", "stealth", "type", "entityId")
    cards = lambda lst: [{k: c.get(k) for k in keep if k in c} for c in lst or []]   # noqa: E731
    return {"turn": s.get("turn"), "offered": s.get("offered"), "tavern": cards(s.get("tavern")),
            "player": {"gold": p.get("gold"), "tavernTier": p.get("tavernTier"), "board": cards(p.get("board")),
                       "hand": cards(p.get("hand"))}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    n = 0
    with gzip.open(a.out, "wt", encoding="utf-8") as fh:
        for f in a.files:
            for x in segment(load_game(f)):
                fh.write(json.dumps({**x.to_json(), "before": slim_before(x.before),
                                     "afters": [slim_after(y) for y in x.afters]}, ensure_ascii=False) + "\n")
                n += 1
    print(f"{n} 个动作 → {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
