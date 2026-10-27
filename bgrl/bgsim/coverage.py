"""卡牌实现覆盖率报告：python -m bgsim.coverage [--replay 回放报告.json]"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

from . import cards as cardsmod
from . import game  # noqa: F401  （注册实现）
from .effects.core import IMPLS


def report(db: dict, replay: dict | None = None) -> dict:
    cards = db["cards"]
    out = {}
    for cat in ("pool", "spell", "hero", "trinket", "anomaly", "quest"):
        ids = [c["id"] for c in cards.values() if c["category"] == cat and not c.get("duosOnly")]
        if cat == "hero":
            ids_impl = [cards[i].get("heroPower") for i in ids]
            st = Counter(IMPLS[h].status if h in IMPLS else "missing" for h in ids_impl)
        else:
            st = Counter(IMPLS[i].status if i in IMPLS else "missing" for i in ids)
        out[cat] = {"total": len(ids), **dict(st)}
    if replay:
        rows = replay["summary"]["cards"]
        out["replay"] = {"actions": replay["summary"]["actions"], "byResult": replay["summary"]["byResult"],
                         "cardsSeen": len(rows),
                         "cardsPass80": sum(1 for r in rows if r["passRate"] >= 0.8)}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", default=str(cardsmod.DEFAULT_OUT))
    ap.add_argument("--replay", default="")
    a = ap.parse_args(argv)
    db = cardsmod.load(a.cards)
    rep = json.load(open(a.replay, encoding="utf-8")) if a.replay else None
    r = report(db, rep)
    names = {"pool": "随从池", "spell": "酒馆法术", "hero": "英雄技能", "trinket": "饰品", "anomaly": "畸变", "quest": "任务"}
    for k, v in r.items():
        if k in names:
            print(f"{names[k]}：共 {v['total']}，已实现 {v.get('done', 0)}，近似 {v.get('approx', 0)}，"
                  f"只有战斗效果 {v.get('combat', 0)}，白板 {v.get('vanilla', 0)}，未实现 {v.get('missing', 0)}")
    if "replay" in r:
        print("回放：", r["replay"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
