"""塑形审计（C3，第 30060 轮起）：每一项奖励塑形实际给了学习者多少分，按项、按动作分开记。

为什么：奖励塑形有十来项（升本节奏、钱够不升本、剩钱、阵容强度、养场面、目标阵容、引擎联动、经济、卖该留的牌、
无效刷新……），每项是针对一个问题加的，互相作用说不清。C1 / C2 两个超参实验下来名次没动，和高手差距最大的还是
第 6–8 回合升本少（34–39%，高手 54–63%）、刷新多。先量清楚每项塑形在第 6–8 回合往哪个方向推，再一项一项关掉做对比。

记两样（只记不改训练）：
  · 每轮日志 sh_<项>：这一轮每个学习者局（打完的）平均从这一项拿了多少分（名次奖励 sh_place 一起记，好比大小）；
  · 考试那轮 shape_mid：第 6–8 回合里，每种动作（升本 / 买 / 刷新 / 卖 / 结束回合 / 其它）每做一次平均从各项拿多少分
    （上次考试以来累计）。比如“升本 -0.01、买 +0.02”就说明这一项在把模型从升本推向买牌。
势能型的几项（目标阵容、引擎联动、阵容强度、养场面）分数挂在结束回合那一步上，按动作看主要落在 end 上；它们推动作要经过
价值头传回去，看每局合计（sh_<项>）更准。
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

KINDS = ("upgrade", "buy", "refresh", "sell", "end", "other")
MID = (6, 8)                                      # 第几回合到第几回合按动作细分


def kind_of(a: int) -> str:
    from tg.actions import A_BUY, A_END, A_REFRESH, A_SELL, A_UPGRADE
    a = int(a)
    if a == A_UPGRADE:
        return "upgrade"
    if a == A_REFRESH:
        return "refresh"
    if a == A_END:
        return "end"
    if A_BUY <= a < A_SELL:
        return "buy"
    if A_SELL <= a < A_SELL + 7:
        return "sell"
    return "other"


class ShapeAudit:
    def __init__(self):
        self.tot: dict[str, float] = defaultdict(float)          # 这一轮：项 → 合计
        self.n_ep = 0                                            # 这一轮打完的学习者局数
        self.mid: dict[tuple, float] = defaultdict(float)        # 上次考试以来：(项, 动作) → 合计（第 6–8 回合）
        self.mid_n: dict[str, int] = defaultdict(int)            # 上次考试以来：动作 → 次数（第 6–8 回合）

    @staticmethod
    def _mid(turn) -> bool:
        return MID[0] <= int(turn) <= MID[1]

    def count(self, ak, turns):
        """一批学习者决策（每批调一次）：数第 6–8 回合每种动作做了几次。"""
        for a, t in zip(ak, turns):
            if self._mid(t):
                self.mid_n[kind_of(a)] += 1

    def add(self, term: str, d, ak, turns):
        """一批决策上这一项给的分（d 和 ak 一样长）。"""
        d = np.asarray(d, dtype=np.float64)
        if not d.size:
            return
        s = float(d.sum())
        if s == 0.0 and not d.any():
            return
        self.tot[term] += s
        for j in np.flatnonzero(d):
            if self._mid(turns[j]):
                self.mid[(term, kind_of(ak[j]))] += float(d[j])

    def add_one(self, term: str, v: float, a=None, turn=None):
        """单个分数（改已经存下的上一步、局末的分）。a / turn 给了且在第 6–8 回合就算进按动作的表。"""
        v = float(v)
        if v == 0.0:
            return
        self.tot[term] += v
        if a is not None and turn is not None and self._mid(turn):
            self.mid[(term, kind_of(a))] += v

    def episode(self):
        self.n_ep += 1

    def take(self) -> dict:
        """每轮：sh_<项> = 每局平均（这一轮没打完的局就不记）。"""
        out = {}
        if self.n_ep:
            out = {f"sh_{k}": round(v / self.n_ep, 4) for k, v in sorted(self.tot.items())}
            out["sh_eps"] = self.n_ep
        self.tot, self.n_ep = defaultdict(float), 0
        return out

    def take_mid(self) -> dict:
        """考试那轮：{项: {动作: 每做一次平均拿多少}}，只列有分的。"""
        out: dict[str, dict] = {}
        for (term, k), v in sorted(self.mid.items()):
            n = self.mid_n.get(k, 0)
            if n:
                out.setdefault(term, {})[k] = round(v / n, 5)
        if out:
            out["_n"] = {k: self.mid_n[k] for k in KINDS if self.mid_n.get(k)}
        self.mid, self.mid_n = defaultdict(float), defaultdict(int)
        return out
