"""写死规则的龙虾流机器人：拿来量“龙虾流在 TavernGym 里、对高分阵容到底能打多好”，打得好就当老师教模型。

为什么要它（第 1900 轮）：强制定向局练了约 1400 轮，模型拿着龙虾流还是 5.6 名、提图斯 3%、陆行鸟 12%、
龙虾加成第 12 回合才 2.8。先用规则把这套打法写出来，看看上限；上限高，就让模型模仿它（老师）。

打法（按高手的思路）：
  · 升本：第 2 回合 2 本、第 5 回合 3 本、第 7 回合 4 本、第 8 回合 5 本、第 10 回合 6 本（钱够就升）
  · 买：逐亡陆行鸟 > 提图斯 > 美味龙虾 > 猎头狮鹫 > 囤食土狼 > 深潜狮子鱼 > 尖利的鲨鱼 > 其它野兽；
        前 4 回合没有野兽时随便买身材；不买别的种族
  · 打出来：进击随从（狮鹫 / 土狼）放最左，美味龙虾紧跟在进击随从后面（最左边的亡语），其它放右边
  · 招募阶段的连招：进击随从在最左时，发动深潜狮子鱼、卖尖利的鲨鱼（让最左边的野兽打鱼饵 → 进击 →
        陆行鸟触发最左边的亡语 = 龙虾，提图斯翻倍；龙虾每触发一次，之后的龙虾更强）
  · 站位不对就挪：进击随从挪到最左，龙虾挪到进击随从后面
  · 场满了、酒馆里有更好的：卖掉分最低的（不卖龙虾流的牌）
  · 5 本以后还缺陆行鸟 / 提图斯、钱够：刷新找
发现 / 抉择：挑分最高的。

用法：bg.py goldflow --policy lobster --boosted --games 128 --lobby BEAST
"""
from __future__ import annotations

import numpy as np

from tg.actions import (A_ACTIVATE, A_BUY, A_CHOOSE, A_END, A_MOVE, A_PLAY, A_REFRESH, A_SELL, A_UPGRADE,
                        N_BOARD, N_CHOICE, N_HAND, N_POS, N_SHOP)
from tg.obs import C_BOARD, C_CHOICE, C_HAND, C_SHOP, gold, tier, turn

LOBSTER, GRIFFIN, ROADRUNNER, TITUS = "BG36_202", "BG36_204", "BG36_208", "BG25_354"
LIONFISH, SHARK, HYENA = "BG36_201", "BG36_206", "BG36_210"
RALLY = (GRIFFIN, HYENA)
CORE = {ROADRUNNER: (100, 2), TITUS: (95, 1), LOBSTER: (85, 3), GRIFFIN: (80, 2), HYENA: (75, 2),
        LIONFISH: (70, 2), SHARK: (50, 1)}                   # 分数、最多留几张（多了只当普通野兽）
TIER_PLAN = {2: 2, 3: 5, 4: 7, 5: 8, 6: 10}


class LobsterBot:
    def __init__(self):
        from tg.catalog import base_id, card_id, catalog, n_cards
        det = catalog()
        n = n_cards()
        self.ids = [base_id(card_id(i)) if i else "" for i in range(n)]
        self.beast = np.array([bool(i) and "BEAST" in ((det.get(self.ids[i]) or {}).get("races") or ())
                               for i in range(n)])
        self.minion = np.array([bool(i) and (det.get(self.ids[i]) or {}).get("kind") == "minion" for i in range(n)])
        self.tier_of = np.array([int((det.get(self.ids[i]) or {}).get("tier") or 0) for i in range(n)])
        self.dr = np.array([bool(i) and "Deathrattle" in str((det.get(self.ids[i]) or {}).get("text") or "")
                            for i in range(n)])

    # ---------------------------------------------------------------- 打分
    def score(self, idx: int, have: dict, t: int) -> float:
        cid = self.ids[idx]
        if cid in CORE:
            s, cap = CORE[cid]
            return s if have.get(cid, 0) < cap else 30
        if self.beast[idx]:
            return 10 + 3 * self.tier_of[idx]
        if self.minion[idx]:
            return 5 + self.tier_of[idx] if t <= 4 else 0
        return 0                                            # 法术、饰品：不买

    def _have(self, board, hand) -> dict:
        h: dict = {}
        for x in list(board) + list(hand):
            if x:
                h[self.ids[x]] = h.get(self.ids[x], 0) + 1
        return h

    # ---------------------------------------------------------------- 一行（一个座位）
    def act_one(self, f, c, m) -> int:
        legal = np.flatnonzero(m)
        t, g, tr = int(turn(f)), int(gold(f)), int(tier(f))
        board = [int(x) for x in c[C_BOARD:C_BOARD + N_BOARD]]
        hand = [int(x) for x in c[C_HAND:C_HAND + N_HAND]]
        shop = [int(x) for x in c[C_SHOP:C_SHOP + N_SHOP]]
        have = self._have(board, hand)
        nb = sum(1 for x in board if x)

        # 1. 发现 / 抉择 / 选目标
        ch = [a for a in legal if A_CHOOSE <= a < A_CHOOSE + N_CHOICE]
        if ch:
            opts = [int(x) for x in c[C_CHOICE:C_CHOICE + N_CHOICE]]
            return int(max(ch, key=lambda a: self.score(opts[a - A_CHOOSE], have, t) if opts[a - A_CHOOSE] else -1))

        # 2. 升本
        want = max([k for k, tt in TIER_PLAN.items() if t >= tt], default=1)
        if tr < want and m[A_UPGRADE]:
            return A_UPGRADE

        # 3. 打出手牌：进击随从放最左，龙虾放在进击随从后面，其它放最右
        plays = []
        for h, x in enumerate(hand):
            if not x:
                continue
            cid = self.ids[x]
            if cid in RALLY:
                pos = 0
            elif cid == LOBSTER:
                pos = sum(1 for b in board if b and self.ids[b] in RALLY)
            else:
                pos = nb
            cand = [A_PLAY + h * N_POS + p for p in (pos, nb, 0) if p <= N_POS - 1]
            for a in cand:
                if m[a]:
                    plays.append((self.score(x, {}, t), a))
                    break
            else:
                for p in range(N_POS):                      # 法术等：随便一个能打的位置
                    if m[A_PLAY + h * N_POS + p]:
                        plays.append((self.score(x, {}, t), A_PLAY + h * N_POS + p))
                        break
        if plays:
            return int(max(plays)[1])

        # 4. 站位：进击随从挪到最左、龙虾挪到进击随从后面
        ids_b = [self.ids[x] if x else "" for x in board]
        if nb >= 2 and ids_b[0] not in RALLY:
            for i, cid in enumerate(ids_b):
                if cid in RALLY and m[A_MOVE + i * 7 + 0]:
                    return A_MOVE + i * 7 + 0
        if ids_b[0] in RALLY:
            k = 1
            while k < nb and ids_b[k] in RALLY:
                k += 1
            first_dr = next((i for i, x in enumerate(board) if x and self.dr[x]), None)
            if first_dr is not None and ids_b[first_dr] != LOBSTER:
                for i, cid in enumerate(ids_b):
                    if cid == LOBSTER and i != k and k < 7 and m[A_MOVE + i * 7 + k]:
                        return A_MOVE + i * 7 + k

        # 5. 招募阶段连招：进击随从在最左时发动狮子鱼、卖鲨鱼
        if ids_b[0] in RALLY:
            for i, cid in enumerate(ids_b):
                if cid == LIONFISH and m[A_ACTIVATE + i]:
                    return A_ACTIVATE + i
            for i, cid in enumerate(ids_b):
                if cid == SHARK and m[A_SELL + i]:
                    return A_SELL + i

        # 6. 买
        buys = [(self.score(x, have, t), A_BUY + s) for s, x in enumerate(shop) if x and m[A_BUY + s]]
        best = max(buys) if buys else (0, None)
        if best[1] is not None and best[0] > 0:
            hand_n = sum(1 for x in hand if x)
            if nb < 7 or hand_n < 9:
                return int(best[1])
        # 场满了、有更好的：卖掉分最低的（不卖龙虾流的牌）
        if nb >= 7 and best[0] >= 50:
            sells = [(self.score(x, {}, t), A_SELL + i) for i, x in enumerate(board)
                     if x and self.ids[x] not in CORE and m[A_SELL + i]]
            if sells and min(sells)[0] < best[0]:
                return int(min(sells)[1])

        # 7. 刷新：钱够、酒馆里没想要的
        missing = (tr >= 5 and (have.get(ROADRUNNER, 0) < 2 or have.get(TITUS, 0) < 1))
        if m[A_REFRESH] and (g >= 4 or (missing and g >= 1)) and best[0] < 40:
            return A_REFRESH
        return A_END if m[A_END] else int(legal[0])

    def __call__(self, f, c, m):
        m = np.asarray(m).astype(bool)
        return np.array([self.act_one(f[i], c[i], m[i]) for i in range(len(m))], dtype=np.int64)


def lobster_policy():
    """给 goldflow.play_model 的 policy=(act, info)。"""
    return LobsterBot(), {"iter": None, "name": "龙虾流规则机器人"}
