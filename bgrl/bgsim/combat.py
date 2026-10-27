"""战斗后端。

BobsBuddyCombat：正式用，调用 Bob's Buddy（Windows + HDT，见 bobsbuddy.py）。一回合的所有对战一次批量模拟。
SimpleCombat：只用于没有 Bob's Buddy 的环境（开发机、单元测试）。只实现关键词，不实现任何随从效果。
"""
from __future__ import annotations

import random

from .model import Minion, Player


def minion_in(m: Minion) -> dict:
    d = {"id": m.card_id, "atk": max(0, m.atk), "hp": max(1, m.hp), "tier": m.tier, "golden": m.golden,
         "taunt": "taunt" in m.kw, "ds": "divineShield" in m.kw, "poison": "poisonous" in m.kw,
         "venom": "venomous" in m.kw, "wf": "windfury" in m.kw, "mwf": "megaWindfury" in m.kw,
         "stealth": "stealth" in m.kw, "reborn": "reborn" in m.kw}
    if any(m.sd):
        d["sd"] = list(m.sd)
    return d


def side_in(game, p: Player, friendly: bool) -> dict:
    hero = game.card(game.skins.get(p.hero_id, p.hero_id))
    s = {"health": max(1, p.total_health), "tier": p.tier, "board": [minion_in(m) for m in p.board]}
    if hero.get("heroPower"):
        s["heroPowers"] = [{"id": hero["heroPower"], "used": p.hero_power_used}]
    if p.trinkets:
        s["trinkets"] = list(p.trinkets)
    return s


def combat_in(game, a: Player, b: Player) -> dict:
    return {"turn": game.turn, "races": list(game.races), "damageCap": game.rules.cap_for_turn(game.turn),
            "me": side_in(game, a, True), "opp": side_in(game, b, False)}


class BobsBuddyCombat:
    def __init__(self, bb) -> None:
        self.bb = bb                      # bobsbuddy.BobsBuddy 实例
        self.errors = 0

    def fight(self, game, pairs: list[tuple[Player, Player]]) -> list[int]:
        res = self.bb.simulate([combat_in(game, a, b) for a, b in pairs], iterations=1)
        out = []
        for r in res:
            if r.get("err") or not r.get("dmg"):
                self.errors += 1
                out.append(0)
            else:
                out.append(int(r["dmg"][0]))
        return out


class SimpleCombat:
    """简化战斗：攻击、嘲讽、圣盾、剧毒/烈毒、风怒、复生；伤害 = 酒馆等级 + 存活随从星级之和。"""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def fight(self, game, pairs: list[tuple[Player, Player]]) -> list[int]:
        return [self._one(a, b) for a, b in pairs]

    def _one(self, a: Player, b: Player) -> int:
        sides = [[self._unit(m) for m in a.board], [self._unit(m) for m in b.board]]
        turn = 0 if len(sides[0]) > len(sides[1]) else 1 if len(sides[1]) > len(sides[0]) else self.rng.randint(0, 1)
        nxt = [0, 0]
        for _ in range(200):
            if not sides[0] or not sides[1]:
                break
            att_side, def_side = sides[turn], sides[1 - turn]
            i = nxt[turn] % len(att_side)
            att = att_side[i]
            for _ in range(2 if att["wf"] else 1):
                if not def_side or att["hp"] <= 0:
                    break
                taunts = [d for d in def_side if d["taunt"]]
                tgt = self.rng.choice(taunts or def_side)
                self._hit(tgt, att)
                self._hit(att, tgt)
                for side in (att_side, def_side):
                    for d in [x for x in side if x["hp"] <= 0]:
                        idx = side.index(d)
                        side.remove(d)
                        if d["reborn"]:
                            side.insert(idx, dict(d, hp=1, reborn=False))
            if att in att_side:
                nxt[turn] = att_side.index(att) + 1
            turn = 1 - turn
        if sides[0] and not sides[1]:
            return a.tier + sum(x["tier"] for x in sides[0])
        if sides[1] and not sides[0]:
            return -(b.tier + sum(x["tier"] for x in sides[1]))
        return 0

    @staticmethod
    def _unit(m: Minion) -> dict:
        return {"atk": max(0, m.atk), "hp": max(1, m.hp), "tier": m.tier, "taunt": "taunt" in m.kw,
                "ds": "divineShield" in m.kw, "poison": "poisonous" in m.kw or "venomous" in m.kw,
                "wf": "windfury" in m.kw or "megaWindfury" in m.kw, "reborn": "reborn" in m.kw}

    @staticmethod
    def _hit(target: dict, src: dict) -> None:
        if src["atk"] <= 0:
            return
        if target["ds"]:
            target["ds"] = False
            return
        target["hp"] -= src["atk"]
        if src["poison"]:
            target["hp"] = 0
