"""最终阵容的统计和打分：主种族、常见搭配、关键卡。

  comp_stats      一批最终阵容的汇总（主种族分布、常用卡），写进 rl/h13_profile.json 的 final
  synergy         经常一起上场的两张卡（提升度）
  board_scores    一个阵容的配合程度：主种族纯度 coh、常见搭配对数 syn、关键卡张数 key
  load_comp_refs  训练和统计用的参照：rl/h13_profile.json 的搭配 + rl/card_value.json 的关键卡

训练里的阵容成型奖励（rl/ppo.py Trainer._comp_shape）、goldflow / comptest 的阵容对比、
native_demos 生成万三画像都用这一份。卡牌 ID 都是去掉金色 _G 之后的。
"""
from __future__ import annotations

import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

from tg.catalog import catalog

REFS = Path(__file__).resolve().parent          # rl/：h13_profile.json、card_value.json 在这里
ZH_RACE = {"QUILBOAR": "野猪人", "UNDEAD": "亡灵", "BEAST": "野兽", "MURLOC": "鱼人", "DEMON": "恶魔", "MECHANICAL": "机械",
           "ELEMENTAL": "元素", "PIRATE": "海盗", "DRAGON": "龙", "NAGA": "纳迦", "ABERRATION": "畸变", "混合": "混合"}


def comp_stats(boards: list[list[str]], goldens: list[int], tiers: list[int], det: dict | None = None) -> dict:
    """最终阵容汇总：boards = 每局最后的场面（卡牌 ID，金色已去掉 _G）。主种族：同一种族（含全种族）≥ 4 个。"""
    det = det or catalog()
    main, cards = Counter(), Counter()
    for ids in boards:
        for x in set(ids):
            cards[x] += 1
        rc = Counter(r for x in ids for r in (det.get(x) or {}).get("races", []) if r != "ALL")
        allr = sum(1 for x in ids if "ALL" in (det.get(x) or {}).get("races", []))
        top = rc.most_common(1)
        main[top[0][0] if top and top[0][1] + allr >= 4 else "混合"] += 1
    n = max(1, len(boards))
    return {"n": len(boards), "tier": round(float(np.mean(tiers)), 2) if tiers else None,
            "golden": round(float(np.mean(goldens)), 2) if goldens else None,
            "main": [[k, round(v / n, 3)] for k, v in main.most_common()],
            "cards": [[x, (det.get(x) or {}).get("name_zh", x), round(v / n, 3)] for x, v in cards.most_common(40)]}


def synergy(boards: list[list[str]], det: dict, min_games: int = 12, top: int = 40) -> list:
    """经常一起上场的两张卡（提升度 = 一起出现的局数 / 各自独立时的期望）。"""
    one, two = Counter(), Counter()
    for ids in boards:
        u = sorted(set(ids))
        one.update(u)
        two.update(combinations(u, 2))
    n = len(boards)
    out = [[round(k * n / (one[x] * one[y]), 2), k, (det.get(x) or {}).get("name_zh", x), (det.get(y) or {}).get("name_zh", y), x, y]
           for (x, y), k in two.items() if k >= min_games]
    return sorted(out, reverse=True)[:top]


def board_scores(ids: list[str], det: dict, pairs: set, key: set) -> dict:
    """一个最终阵容的配合程度：
      coh   主种族纯度 = 最多的那个种族（全种族随从都算）的随从数 / 场上随从数
      syn   场上两两组合里，属于高手常见搭配（synergy 里提升度 ≥ 3）的有几对
      key   场上关键卡（rl/card_value.json 的 key）有几张"""
    races = Counter()
    for x in ids:
        rs = (det.get(x) or {}).get("races", [])
        for r in ({r for r in rs if r != "ALL"} if "ALL" not in rs else set(ZH_RACE) - {"混合"}):
            races[r] += 1
    u = sorted(set(ids))
    return {"coh": round(max(races.values()) / len(ids), 3) if races and ids else 0.0,
            "syn": sum(1 for x, y in combinations(u, 2) if (x, y) in pairs),
            "key": sum(1 for x in ids if x in key)}


def load_comp_refs(root: Path = REFS) -> tuple[set, set]:
    """(高手常见搭配 {(id, id)}, 关键卡 {id})：rl/h13_profile.json 和 rl/card_value.json。"""
    pairs, key = set(), set()
    hp, cv = Path(root) / "h13_profile.json", Path(root) / "card_value.json"
    if hp.exists():
        for row in json.loads(hp.read_text(encoding="utf-8")).get("synergy", []):
            if len(row) >= 6 and row[0] >= 3:
                pairs.add(tuple(sorted((row[4], row[5]))))
    if cv.exists():
        key = set(json.loads(cv.read_text(encoding="utf-8")).get("key", []))
    return pairs, key
