"""送牌 / 送整套阵容（课程之一，原 rl/curricula/gifts.CardGifts / CompGifts）：机器人局里往学习者手里放关键卡，或开局定一套本局凑得成的阵容按回合送。"""
from __future__ import annotations

import json
import random

import numpy as np

from tg.catalog import base_id

from ._json import _dumps, _loads


class CardGifts:
    """送牌课程（第 20800 轮起）：frac 的机器人局里，turns 每回合招募开始时按 p 的机会往学习者手里放一张“高手抢着买、
    模型从来不买”的牌：rl/card_value.json 里高手挑中倍率 ≥ min_lift 的随从和法术（发现、免费刷新、鲜血宝石、抉择……；
    纯加金币的经济法术不送——那些模型自己会买，倍率比高手还高）。酒馆等级够得着、本局开了那个种族、手牌没满才送。
    场面、金币、酒馆不动：打不打、怎么选、留不留全看模型。

    为什么：模型买这些牌的倍率是 0.0–0.3（高手 2–4.5）——无畏的食客、刷新畸体、尖利的鲨鱼、圈套陷阱师、竞技表演者、
    热气球测绘员……高手靠它们每回合多拿宝石、刷新、发现、金色随从，把场面养大（第 9 回合卖牌收入 8 块，模型 2 块）。
    模型从来没拿过这些牌，价值网络不知道“手里有一张无畏的食客”值多少，所以买它永远不划算。送到手里以后它会打出来、
    做抉择、用宝石，价值网络学到这些局面值多少，酒馆里再看到这张牌时“买”的优势就是正的——不用改奖励、不逼动作。
    试过按规则给建议动作逼它照着凑（第 20350–20800 轮），带坏了所有局；这里只是让它见过。"""

    def __init__(self, frac: float = 0.25, turns=(4, 5, 6, 7, 8, 9, 10, 11, 12), p: float = 0.6,
                 min_lift: float = 2.0, seed: int = 0, cards: list | None = None):
        import re
        from pathlib import Path
        from tg.catalog import catalog
        self.det = catalog()
        self.by_cid = {int(v["cid"]): k for k, v in self.det.items() if "cid" in v}
        self.frac, self.turns, self.p = float(frac), set(int(t) for t in turns), float(p)
        self.rng = random.Random(seed + 7717)
        if cards is None:
            cv = Path(__file__).resolve().parent.parent / "card_value.json"     # rl/card_value.json
            ref = json.loads(cv.read_text(encoding="utf-8")).get("cards", {}) if cv.exists() else {}
            cards = [k for k, v in ref.items() if float(v.get("lift", 0)) >= min_lift]
        self.pool = []                                 # [(卡 id, 种类, 等级, 种族)]
        for k in cards:
            d = self.det.get(k) or {}
            kind = d.get("kind")
            if not d.get("in_pool") or kind not in ("minion", "tavern_spell"):
                continue
            if kind == "tavern_spell" and re.search(r"\bGold\b", d.get("text") or ""):
                continue
            self.pool.append((k, kind, int(d.get("tier") or 1), list(d.get("races") or [])))
        self.plan: dict[int, list] = {}                # 对局 → [第几局, 送不送, 最近送过的回合]
        self.n_given, self.n_full = 0, 0
        self.given: dict[str, int] = {}

    def is_active(self, g: int) -> bool:
        v = self.plan.get(g)
        return v is not None and bool(v[1])

    def active(self) -> set:
        return {g for g, v in self.plan.items() if v[1]}

    def options(self, tier: int, lob) -> list:
        return [x for x in self.pool if x[2] <= tier and
                (not lob or not x[3] or any(r in lob or r == "ALL" for r in x[3]))]

    def apply(self, env, features: np.ndarray, episodes=None, skip=None) -> bool:
        """每步 observe 之后调用：新的一局先决定送不送（skip 里的对局不送：送整套阵容的局）；送的局在 turns 里每回合
        第一次看到招募阶段时按 p 送一张。返回有没有改（改了要重新 observe）。"""
        from tg.obs import lobby
        if self.frac <= 0 or not self.pool:
            return False
        from taverngym import _core
        turns = np.rint(features[:, 0] * 20).astype(int)
        changed = False
        for g in range(env.n_games):
            ep = int(episodes[g]) if episodes is not None else env.episode_index(g)
            cur = self.plan.get(g)
            if cur is None or cur[0] != ep:
                cur = self.plan[g] = [ep, self.rng.random() < self.frac and not (skip and g in skip), -1]
            t = int(turns[g])
            if not cur[1] or t not in self.turns or cur[2] == t:
                continue
            s = _loads(env.save(g))
            game = s["game"]
            if game.get("turn") != t or str(game.get("phase")) != "Recruit":
                continue
            cur[2] = t
            if self.rng.random() >= self.p:
                continue
            me = game["players"][0]
            if len(me.get("hand") or []) >= 10:
                self.n_full += 1
                continue
            opts = self.options(int(me.get("tier") or 1), lobby(features[g]))
            if not opts:
                continue
            k, kind = self.rng.choice(opts)[:2]
            try:
                G = _core.Game.from_json(_dumps(game))
                G.debug(_dumps({"cmd": "add_minion" if kind == "minion" else "add_spell", "p": 0, "zone": "hand",
                                "card": k}))
            except BaseException as ex:                            # noqa: BLE001 —— 引擎内部出错：这回合不送
                if isinstance(ex, (KeyboardInterrupt, SystemExit)):
                    raise
                continue
            s["game"] = _loads(G.state_json())
            env.load(g, _dumps(s))
            self.n_given += 1
            self.given[k] = self.given.get(k, 0) + 1
            changed = True
        return changed

    def finish(self, g: int) -> bool:
        """对局 g 打完了：返回这局是不是送牌局。"""
        cur = self.plan.pop(g, None)
        return bool(cur is not None and cur[1])

    def take_stats(self) -> dict:
        out = {"gifts": self.n_given, "gift_hand_full": self.n_full}
        self.n_given, self.n_full = 0, 0
        return out


class CompGifts:
    """送整套阵容（第 21800 轮起）：frac 的机器人局里，开局给学习者定一套本局凑得成的目标阵容（rl/target_comps.json，
    S / A 权重 1、B 0.5），turns 每回合招募开始时按 p 的机会把这套还没有的核心卡送一张到手里（先送等级低的；
    只送酒馆等级 + 2 以内的；核心卡齐了送这套的配套卡）。场面、金币、酒馆不动：打不打、留不留、怎么围着它做全看模型。

    为什么：送牌课程（CardGifts）是唯一让模型开始买以前不买的牌的办法——牌到了手里，它打出来、尝到好处，价值网络才知道
    这张牌值多少。这里送的是一整套的核心卡：每套阵容（包括高手样本里很少见的）都轮得到，它才见得到“凑齐以后”是什么样。
    试过把酒馆里的牌换成这套的（第 20020–20800 轮的阵容练习局）：牌刷到了也不买，做成的不到 1%。"""

    def __init__(self, frac: float = 0.2, turns=(5, 6, 7, 8, 9, 10, 11), p: float = 0.7, seed: int = 0,
                 comps: list | None = None):
        from tg.catalog import catalog
        from ..comp import load_target_comps, reachable
        self.det = catalog()
        self.frac, self.turns, self.p = float(frac), set(int(t) for t in turns), float(p)
        self.rng = random.Random(seed + 9311)
        self.by_cid = {int(v["cid"]): k for k, v in self.det.items() if "cid" in v}
        self.comps = load_target_comps() if comps is None else comps
        self.w = [1.0 if c["tier"] in "SA" else 0.5 for c in self.comps]
        self._reachable = reachable
        self.plan: dict[int, list] = {}                # 对局 → [第几局, 送不送, 最近送过的回合, 阵容（None = 还没定）]
        self.n_given, self.n_full, self.n_none = 0, 0, 0

    def is_active(self, g: int) -> bool:
        v = self.plan.get(g)
        return v is not None and bool(v[1])

    def active(self) -> set:
        return {g for g, v in self.plan.items() if v[1]}

    def _tier(self, cid: str) -> int:
        return int((self.det.get(cid) or {}).get("tier") or 1)

    def _owned(self, me: dict) -> set:
        """场上 + 手里的随从（存档里的卡牌编号 → 卡牌 ID，金色算普通的）。"""
        out = set()
        for x in me.get("board") or []:
            out.add(base_id(self.by_cid.get(int(x.get("card", -1)), "")))
        for h in me.get("hand") or []:
            m = (h.get("item") or {}).get("Minion") or {}
            if "card" in m:
                out.add(base_id(self.by_cid.get(int(m["card"]), "")))
        out.discard("")
        return out

    def next_card(self, comp: dict, own: set, tier: int):
        """这套还没有的核心卡里等级最低的一张（等级 ≤ 酒馆等级 + 2）；核心卡齐了 → 一张还没有的配套卡（等级 ≤ 酒馆等级）。"""
        miss = sorted((self._tier(k), k) for k in comp["core"] if k not in own and (self.det.get(k) or {}).get("in_pool"))
        if miss:
            return miss[0][1] if miss[0][0] <= tier + 2 else None
        sup = sorted(k for k in (comp.get("keep_support") or comp.get("support") or ())
                     if k not in own and self._tier(k) <= tier and (self.det.get(k) or {}).get("kind") == "minion")
        return self.rng.choice(sup) if sup else None

    def apply(self, env, features: np.ndarray, episodes=None, skip=None) -> bool:
        """每步 observe 之后调用：新的一局先决定送不送（skip 里的对局不送）；送的局在 turns 里每回合第一次看到招募阶段时
        按 p 送一张。返回有没有改（改了要重新 observe）。"""
        from tg.obs import lobby
        if self.frac <= 0 or not self.comps:
            return False
        from taverngym import _core
        turns = np.rint(features[:, 0] * 20).astype(int)
        changed = False
        for g in range(env.n_games):
            ep = int(episodes[g]) if episodes is not None else env.episode_index(g)
            cur = self.plan.get(g)
            if cur is None or cur[0] != ep:
                cur = self.plan[g] = [ep, self.rng.random() < self.frac and not (skip and g in skip), -1, None]
            t = int(turns[g])
            if not cur[1] or t not in self.turns or cur[2] == t:
                continue
            s = _loads(env.save(g))
            game = s["game"]
            if game.get("turn") != t or str(game.get("phase")) != "Recruit":
                continue
            cur[2] = t
            if cur[3] is None:                         # 第一次送：定下这局的阵容（本局凑得成的里面按权重抽）
                lob = lobby(features[g])
                ok = [k for k, c in enumerate(self.comps) if self._reachable(c, lob)]
                if not ok:
                    cur[1] = False
                    self.n_none += 1
                    continue
                cur[3] = self.comps[self.rng.choices(ok, weights=[self.w[k] for k in ok])[0]]
            if self.rng.random() >= self.p:
                continue
            me = game["players"][0]
            if len(me.get("hand") or []) >= 10:
                self.n_full += 1
                continue
            own = self._owned(me)
            k = self.next_card(cur[3], own, int(me.get("tier") or 1))
            if not k:
                continue
            try:
                G = _core.Game.from_json(_dumps(game))
                G.debug(_dumps({"cmd": "add_minion", "p": 0, "zone": "hand", "card": k}))
            except BaseException as ex:                            # noqa: BLE001 —— 引擎内部出错：这回合不送
                if isinstance(ex, (KeyboardInterrupt, SystemExit)):
                    raise
                continue
            s["game"] = _loads(G.state_json())
            env.load(g, _dumps(s))
            self.n_given += 1
            changed = True
        return changed

    def finish(self, g: int) -> str:
        """对局 g 打完了：送过整套阵容的局返回那套的名字（还没定下来 = "?"），否则空字符串。"""
        cur = self.plan.pop(g, None)
        if cur is None or not cur[1]:
            return ""
        return cur[3]["name"] if cur[3] else "?"

    def take_stats(self) -> dict:
        out = {"cgifts": self.n_given, "cgift_hand_full": self.n_full, "cgift_none": self.n_none}
        self.n_given, self.n_full, self.n_none = 0, 0, 0
        return out
