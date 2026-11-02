"""TavernGym 卡牌词表（进程内缓存，只读）。所有模块读卡牌信息都走这里，不再各自解析 catalog_json()。

  catalog()           {卡牌 ID: 详情}（name_zh、kind、tier、races、text_zh …）
  card_id(i)          观察里的卡牌编号 → 卡牌 ID（金色是 xxx_G）
  base_id(cid)        金色 ID → 普通 ID（xxx_G → xxx）；None / 空串原样返回
  is_golden_id(cid)   是不是金色 ID
  base_of_index(i)    卡牌编号 → 普通 ID
  n_cards()           词表大小（编号 0 = 空）

只在第一次调用时 import taverngym（arena 连接器的环境里没有它，所以不要在模块顶层 import）。
"""
from __future__ import annotations

import json
from functools import lru_cache


def base_id(card_id):
    return card_id[:-2] if card_id and card_id.endswith("_G") else card_id


def is_golden_id(card_id) -> bool:
    return bool(card_id) and card_id.endswith("_G")


@lru_cache(maxsize=1)
def catalog() -> dict[str, dict]:
    import taverngym
    return {c["id"]: c for c in json.loads(taverngym.catalog_json())}


@lru_cache(maxsize=1)
def n_cards() -> int:
    import taverngym
    return int(taverngym.N_CARDS)


@lru_cache(maxsize=None)
def card_id(index: int) -> str:
    import taverngym
    return taverngym.card_info(int(index))[0]


def base_of_index(index: int) -> str:
    return base_id(card_id(index))


def details(card_id_or_base: str) -> dict:
    """卡牌详情（金色 ID 按普通 ID 查）；查不到返回空字典。"""
    cat = catalog()
    return cat.get(card_id_or_base) or cat.get(base_id(card_id_or_base)) or {}
