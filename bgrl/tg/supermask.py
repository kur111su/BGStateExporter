"""只有观察（没有合法动作表）的示范数据用：从观察推一个“可能合法”的动作集合（合法动作的超集）。

模型算动作概率时只在这个集合里归一；集合里多出来的不合法动作，训练时只会把它们的分数往下压，
实战时它们本来就被真正的合法动作表屏蔽掉，所以不影响。规则（卡牌编号槽位见 rl/model.py 的 cslices）：
  结束、刷新、冻结、取消：总在；升本：没满 6 本（升本费可能被减到 0，不能用它判断）；黑暗之赐：观察第 22 个数 > 0
  英雄技能：技能槽有卡；买：酒馆第 i 格有卡；卖 / 发动：场上第 i 格有随从
  选择：待选第 i 格有内容（浮点特征不全是 0；有的选项没有卡牌编号）（有待选项时只剩选择和取消）
  打出：手牌第 h 张有卡、位置 ≤ 场上随从数；磁力：手牌 h、场上 b 都有；换位：两个位置都在场上、不同
"""
from __future__ import annotations

import numpy as np

from tg.actions import (A_ACTIVATE, A_BUY, A_CANCEL, A_CHOOSE, A_DARK_GIFT, A_END, A_FREEZE, A_HERO_POWER,
                        A_MAGNETIZE, A_MOVE, A_PLAY, A_REFRESH, A_SELL, A_UPGRADE, N_ACTIONS)

C_POWERS, C_BOARD, C_HAND, C_SHOP, C_CHOICE = 1, 19, 26, 36, 44
NB, NH, NS, NC, NPOS = 7, 10, 8, 16, 8
F_TIER, F_GIFT = 10, 22
F_CHOICE, IF = 1005, 40                                  # 待选项的浮点特征从第 1005 个开始，每格 40 个


def super_mask(f: np.ndarray, c: np.ndarray) -> np.ndarray:
    """f [N,3227]、c [N,124] → bool [N,245]。"""
    n = len(f)
    m = np.zeros((n, N_ACTIONS), dtype=bool)
    board = c[:, C_BOARD:C_BOARD + NB] > 0
    hand = c[:, C_HAND:C_HAND + NH] > 0
    shop = c[:, C_SHOP:C_SHOP + NS] > 0
    choice = np.abs(f[:, F_CHOICE:F_CHOICE + NC * IF]).reshape(n, NC, IF).max(-1) > 0
    nb = board.sum(1)
    pend = choice.any(1)
    m[:, [A_END, A_REFRESH, A_FREEZE, A_CANCEL]] = True
    m[:, A_UPGRADE] = np.rint(f[:, F_TIER] * 6) < 6
    m[:, A_DARK_GIFT] = f[:, F_GIFT] > 0
    m[:, A_HERO_POWER:A_HERO_POWER + 2] = c[:, C_POWERS:C_POWERS + 2] > 0
    m[:, A_BUY:A_BUY + NS] = shop
    m[:, A_SELL:A_SELL + NB] = board
    m[:, A_ACTIVATE:A_ACTIVATE + NB] = board
    for h in range(NH):
        for p in range(NPOS):
            m[:, A_PLAY + h * NPOS + p] = hand[:, h] & (p <= nb)
        for b in range(NB):
            m[:, A_MAGNETIZE + h * NB + b] = hand[:, h] & board[:, b]
    for a in range(NB):
        for b in range(NB):
            m[:, A_MOVE + a * NB + b] = board[:, a] & board[:, b] & (a != b)
    m[pend] = False                                         # 有待选项：只剩选择和取消
    m[:, A_CHOOSE:A_CHOOSE + NC] = choice
    m[pend, A_CANCEL] = True
    return m
