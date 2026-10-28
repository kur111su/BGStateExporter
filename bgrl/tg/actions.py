"""TavernGym 的动作编号（245 个，所有模块共用这一份）。

  0 结束回合  1 刷新  2 冻结  3 升级  4–5 英雄技能  6 黑暗之赐  7 取消
  8–15   买酒馆第 i 个                 16–22  卖场上第 i 个
  23–29  发动场上第 i 个               30–45  选择第 i 个选项（发现 / 抉择 / 目标）
  46–125 打出手牌 h 放到位置 p：46 + h*8 + p
  126–195 把手牌 h 磁力吸附到场上 b：126 + h*7 + b
  196–244 把场上 f 移到 t：196 + f*7 + t
"""
from __future__ import annotations

A_END, A_REFRESH, A_FREEZE, A_UPGRADE, A_HERO_POWER, A_DARK_GIFT, A_CANCEL = 0, 1, 2, 3, 4, 6, 7
A_BUY, A_SELL, A_ACTIVATE, A_CHOOSE, A_PLAY, A_MAGNETIZE, A_MOVE = 8, 16, 23, 30, 46, 126, 196
N_ACTIONS = 245
N_POS, N_BOARD, N_HAND, N_SHOP, N_CHOICE = 8, 7, 10, 8, 16


def is_noop(a):
    """“不改变局面”的动作：冻结、调站位（训练里有小惩罚，影子模式里不列成步骤）。
    a 可以是整数，也可以是 numpy 数组（逐个判断）。"""
    return (a == A_FREEZE) | (a >= A_MOVE)
