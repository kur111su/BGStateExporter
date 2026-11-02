"""TavernGym 观察（v1：3227 个浮点 + 124 个卡牌编号）里各项数据的位置。所有模块按这里取，不再各写一遍下标。

不 import taverngym / torch（arena 连接器的环境里也能用）。和 TavernGym 实际布局是否一致由 tests/test_supermask.py 核对
（按 rl/model.Layout 从 taverngym.layout() 算出的切片比）。

浮点（features）：
  F_TURN    回合 / TURN_SCALE          F_GOLD    金币 / GOLD_SCALE
  F_TIER    酒馆等级 / TIER_SCALE      F_UPCOST  升本费 / GOLD_SCALE（满级 0；费用可能被减到 0）
  F_GIFT    黑暗之赐现在能不能买（> 0 能）
  F_BOARD   场上随从起点，每格 MINION_F 个；格内 MF_ATK / MF_HP = 攻 / 血 ÷ 10，MF_GOLDEN = 金色
  F_CHOICE  待选项起点，每格 ITEM_F 个
卡牌编号（cards）：
  C_POWERS 英雄技能（2 格）、C_BOARD 场上（N_BOARD 格）、C_HAND 手牌（N_HAND）、C_SHOP 酒馆（N_SHOP）、C_CHOICE 待选（N_CHOICE）
"""
from __future__ import annotations

import numpy as np

TURN_SCALE, GOLD_SCALE, TIER_SCALE = 20, 10, 6
F_TURN, F_GOLD, F_TIER, F_UPCOST, F_GIFT = 0, 7, 10, 11, 22
MINION_F, ITEM_F = 31, 40
F_BOARD = 68
MF_ATK, MF_HP, MF_GOLDEN = 1, 2, 5
F_CHOICE = 1005

C_POWERS, C_BOARD, C_HAND, C_SHOP, C_CHOICE = 1, 19, 26, 36, 44
N_BOARD, N_HAND, N_SHOP, N_CHOICE, N_POS = 7, 10, 8, 16, 8


def _scaled(f, i, scale):
    return np.rint(np.asarray(f)[..., i] * scale).astype(int)


def turn(f):
    return _scaled(f, F_TURN, TURN_SCALE)


def gold(f):
    return _scaled(f, F_GOLD, GOLD_SCALE)


def tier(f):
    return _scaled(f, F_TIER, TIER_SCALE)


def up_cost(f):
    return _scaled(f, F_UPCOST, GOLD_SCALE)


def board_cards(c):
    """cards[..., 场上 7 格]"""
    return np.asarray(c)[..., C_BOARD:C_BOARD + N_BOARD]


def shop_cards(c):
    return np.asarray(c)[..., C_SHOP:C_SHOP + N_SHOP]


def board_stats(f, slot: int) -> tuple[int, int]:
    """(攻, 血)：场上第 slot 格随从。"""
    base = F_BOARD + MINION_F * slot
    return int(round(float(f[base + MF_ATK]) * 10)), int(round(float(f[base + MF_HP]) * 10))
