"""HSReplay XML（Firestone / HSReplay 的回放，Power.log 规范化后的格式）→ 和插件一样的对局记录（schema 2）。

插件的 StateReader.cs 从 HDT 的实体（每个实体一组标签）读局面；回放 XML 里是同一套实体和标签的变化流。
这里按顺序重放这些变化（FullEntity / ShowEntity / ChangeEntity / HideEntity / TagChange），
在每个 <Options>（主角要做决定的时刻）和每场战斗开始时，按 StateReader.cs 的规则生成一条记录：
玩家（英雄、血甲、等级、金币、英雄技能、场面、手牌、饰品、任务、标签）、酒馆（招募阶段鲍勃的场面）、
待选项、排行榜、游戏标签；战斗记录带对手阵容（GameHistory 用它记“见过的阵容”）。

之后和插件的记录走同一条路：tg.logs 切成动作 → tools/make_demos.py 生成示范。

和 HDT 不同的地方：
  - 回放里没有可选种族（HDT 从内存读）：用本局酒馆里出现过的随从种族里最多的 5 个；
  - 主角 = <Player isMainPlayer="true">（上传回放的人）；回合 = (游戏实体 TURN + 1) // 2，TURN 为奇数是招募阶段。
"""
from __future__ import annotations

import io
import json
import zipfile
from collections import Counter
from functools import lru_cache
from pathlib import Path
from xml.etree.ElementTree import iterparse

HERE = Path(__file__).resolve().parent

T_ZONE, T_CONTROLLER, T_CARDTYPE, T_ZONE_POS = 49, 50, 202, 263
T_ATK, T_HEALTH, T_DAMAGE, T_ARMOR, T_COST, T_PREMIUM = 47, 45, 44, 292, 48, 12
T_RES, T_RES_USED, T_TEMP_RES, T_TURN, T_PLAYER_ID = 26, 25, 295, 20, 30
T_TECH, T_HERO_ENTITY, T_ATTACHED, T_TECH_LEVEL = 1377, 27, 40, 1440
T_NEXT_OPP, T_PLACE, T_EXHAUSTED = 1360, 1373, 43
T_EXTRA_GOLD, T_FREE_REFRESH, T_DARK_GIFT = 4286, 2452, 4865
T_TAUNT, T_DS, T_POISON, T_VENOM, T_REBORN, T_WF, T_MEGA_WF = 190, 194, 363, 2853, 1085, 189, 1207
T_STEALTH, T_DEATHRATTLE, T_FROZEN = 191, 217, 260
Z_PLAY, Z_HAND, Z_SECRET = 1, 3, 7
CT_HERO, CT_MINION, CT_SPELL, CT_ENCH, CT_HP, CT_BGQR, CT_BGSPELL, CT_TRINKET = 3, 4, 5, 6, 10, 40, 42, 44


@lru_cache(maxsize=1)
def _enums():
    d = json.loads((HERE / "gametags.json").read_text(encoding="utf-8"))
    tags = {int(k): v for k, v in d["GameTag"].items()}
    return tags, dict(d.get("GameTagNames") or {v: k for k, v in tags.items()})


def tag_name(t: int) -> str:
    return _enums()[0].get(t, str(t))


def tag_id(name: str) -> int:
    return _enums()[1][name]


@lru_cache(maxsize=1)
def _catalog():
    import taverngym
    return {c["id"]: c for c in json.loads(taverngym.catalog_json())}


def _useful(name: str) -> bool:      # StateReader.IsUsefulTag
    return (name.startswith("BACON") or name.startswith("MODULAR_ENTITY_PART") or name.startswith("TAG_SCRIPT_DATA_ENT")
            or name in ("COPIED_FROM_ENTITY_ID", "DARK_GIFT_ENTITY", "CREATOR_DBID", "AVENGE", "SPELLCRAFT", "DORMANT",
                        "IMMUNE", "MEGA_WINDFURY", "TAG_LAST_KNOWN_COST_IN_HAND"))


class Ent:
    __slots__ = ("id", "card", "tags")

    def __init__(self, eid: int, card: str | None = None):
        self.id, self.card, self.tags = eid, card or None, {}

    def g(self, t: int) -> int:
        return self.tags.get(t, 0)


class Replay:
    """按顺序重放一份回放，产出记录。"""

    def __init__(self):
        self.ents: dict[int, Ent] = {}
        self.game_id = None
        self.main_pid = None              # 主角的 PLAYER_ID（= 他的随从的 CONTROLLER）
        self.main_ent = None              # 主角的玩家实体 id
        self.opp_pid = None
        self.offered: list[int] = []
        self.records: list[dict] = []
        self.races = Counter()
        self.seq = 0
        self._combat_turn_done = set()
        self._script_tags = [tag_id(f"TAG_SCRIPT_DATA_NUM_{i}") for i in (1, 2, 3, 4)]
        self._quest, self._sidequest, self._objective = tag_id("QUEST"), tag_id("SIDE_QUEST"), tag_id("OBJECTIVE")
        self._trinket_tag = tag_id("BACON_TRINKET")

    # ---------------- 实体变化 ----------------

    def _ent(self, eid: int, card=None) -> Ent:
        e = self.ents.get(eid)
        if e is None:
            e = self.ents[eid] = Ent(eid, card)
        elif card:
            e.card = card
        return e

    @staticmethod
    def _tags(el) -> dict[int, int]:
        out = {}
        for t in el:
            if t.tag == "Tag":
                try:
                    out[int(t.get("tag"))] = int(t.get("value"))
                except (TypeError, ValueError):
                    pass
        return out

    def _game_entity(self, el) -> None:
        """第一次：游戏实体。之后再出现 = 断线重连后整局重新下发：清掉所有旧实体（不在新状态里的就是已经没了），
        而且这时两个玩家实体的标签被并进了同一个 GameEntity 元素：从第二个 CONTROLLER 标签起，按 ENTITY_ID 分给各玩家。"""
        gid = int(el.get("id"))
        pairs = []
        for t in el:
            if t.tag == "Tag":
                try:
                    pairs.append((int(t.get("tag")), int(t.get("value"))))
                except (TypeError, ValueError):
                    pass
        segs, cur = [], []
        for t, v in pairs:
            if t == T_CONTROLLER and segs is not None and cur and any(tt == T_CARDTYPE for tt, _ in cur):
                segs.append(cur)
                cur = []
            cur.append((t, v))
        segs.append(cur)
        if self.game_id is not None:           # 重连：整局重新下发
            self.ents = {}
            self.offered = []
        self.game_id = gid
        self._ent(gid).tags.update(dict(segs[0]))
        for seg in segs[1:]:
            d = dict(seg)
            eid = d.get(53)                   # ENTITY_ID
            if not eid:
                continue
            e = self._ent(eid)
            e.tags.update(d)
            if self.main_pid is not None and d.get(T_PLAYER_ID) == self.main_pid:
                self.main_ent = eid

    def feed(self, ev: str, el) -> None:
        name = el.tag
        if ev == "start":
            if name == "Block" and el.get("type") == "1":           # ATTACK：战斗开始后第一下攻击前记一条战斗记录
                self._combat_snapshot()
            return
        if name == "GameEntity":
            self._game_entity(el)
        elif name == "Player":
            e = self._ent(int(el.get("id")))
            e.tags.update(self._tags(el))
            if el.get("isMainPlayer") == "true":
                self.main_ent, self.main_pid = e.id, int(el.get("playerID"))
            else:
                self.opp_pid = int(el.get("playerID"))
        elif name in ("FullEntity", "ShowEntity", "ChangeEntity"):
            eid = int(el.get("id") or el.get("entity"))
            e = self._ent(eid, el.get("cardID"))
            if name == "FullEntity":
                e.tags = self._tags(el)
            else:
                e.tags.update(self._tags(el))
        elif name == "HideEntity":
            e = self._ent(int(el.get("entity")))
            if el.get("zone"):
                e.tags[T_ZONE] = int(el.get("zone"))
        elif name == "TagChange":
            try:
                e = self._ent(int(el.get("entity")))
                t, v = int(el.get("tag")), int(el.get("value"))
            except (TypeError, ValueError):
                return
            e.tags[t] = v
        elif name == "Choices":
            if el.get("playerID") and self.main_ent is not None and int(el.get("playerID")) == self.main_ent:
                self.offered = [int(c.get("entity")) for c in el if c.tag == "Choice"]
                # 待选期间回放里没有 Options（选完才有），不在这里记一条的话记录里永远看不到待选项：
                # 插件那边待选时也会记一条。黑暗之赐（GameHistory 靠“待选项 + 金币少 3”认）和发现都靠这条
                self._snapshot()
        elif name == "ChosenEntities":
            self.offered = []
        elif name == "Options":
            self._snapshot()

    # ---------------- 读局面（StateReader.cs） ----------------

    def _player_ents(self, pid):
        return [e for e in self.ents.values() if e.tags.get(T_CONTROLLER) == pid]

    def _turn(self) -> tuple[int, int]:
        t = self.ents[self.game_id].g(T_TURN) if self.game_id is not None else 0
        return (t + 1) // 2, t

    def _card(self, e: Ent) -> dict:
        cat = _catalog().get(e.card or "") or {}
        base = cat
        if not base and e.card and e.card.endswith("_G"):
            base = _catalog().get(e.card[:-2]) or {}
        tier = e.g(T_TECH_LEVEL) or int(base.get("tier") or 0)
        ct = e.g(T_CARDTYPE)
        tags = {}
        for t, v in e.tags.items():
            if v:
                n = tag_name(t)
                if _useful(n):
                    tags[n] = v
        sd = [e.g(t) for t in self._script_tags]
        return {"entityId": e.id, "cardId": e.card, "dbfId": None, "name": base.get("name_zh") or base.get("name") or e.card,
                "type": "minion" if ct == CT_MINION else "spell" if ct in (CT_SPELL, CT_BGSPELL) else "other",
                "position": e.g(T_ZONE_POS), "attack": e.g(T_ATK), "health": e.g(T_HEALTH) - e.g(T_DAMAGE),
                "tier": tier, "cost": e.g(T_COST) if T_COST in e.tags else int(base.get("cost") or 0),
                "races": list(base.get("races") or []), "golden": e.g(T_PREMIUM) > 0,
                "taunt": e.g(T_TAUNT) > 0, "divineShield": e.g(T_DS) > 0, "poisonous": e.g(T_POISON) > 0,
                "venomous": e.g(T_VENOM) > 0, "reborn": e.g(T_REBORN) > 0,
                "windfury": e.g(T_WF) > 0 or e.g(T_MEGA_WF) > 0, "stealth": e.g(T_STEALTH) > 0,
                "deathrattle": e.g(T_DEATHRATTLE) > 0, "frozen": e.g(T_FROZEN) > 0,
                "scriptData": sd if any(sd) else None, "enchantments": self._ench.get(e.id), "tags": tags or None}

    def _enchantments(self):
        groups: dict[int, dict[str, list[Ent]]] = {}
        for e in self.ents.values():
            if e.g(T_CARDTYPE) == CT_ENCH and e.g(T_ZONE) == Z_PLAY and e.g(T_ATTACHED) > 0:
                groups.setdefault(e.g(T_ATTACHED), {}).setdefault(e.card or "", []).append(e)
        out = {}
        for target, by in groups.items():
            out[target] = [{"cardId": cid, "name": cid, "count": len(lst),
                            "scriptData1": max(lst, key=lambda x: x.id).g(self._script_tags[0]),
                            "scriptData2": max(lst, key=lambda x: x.id).g(self._script_tags[1])}
                           for cid, lst in by.items()]
        return out

    @staticmethod
    def _named(e: Ent | None, keep=lambda n: True):
        if e is None:
            return None
        d = {}
        for t, v in e.tags.items():
            if v:
                n = tag_name(t)
                if keep(n):
                    d[n] = v
        return d or None

    def _shop(self, mine: list[Ent], tier: int) -> dict:
        def find(cid):
            c = [e for e in mine if e.card == cid]
            return max(c, key=lambda e: e.id) if c else None
        shop = {"buyCost": 3, "buyCostKnown": False, "spellBuyCost": 0, "refreshCost": 1, "refreshCostKnown": False,
                "freeRefreshes": 0, "freezeCost": 0, "canUpgrade": False, "upgradeCost": 0}
        b = find("TB_BaconShop_DragBuy")
        if b is not None and T_COST in b.tags:
            shop.update(buyCost=b.g(T_COST), buyCostKnown=True)
        sb = find("TB_BaconShop_DragBuy_Spell")
        if sb is not None:
            shop["spellBuyCost"] = sb.g(T_COST)
        r = find("TB_BaconShop_8p_Reroll_Button")
        if r is not None:
            shop.update(refreshCost=r.g(T_COST), refreshCostKnown=True)
        fz = find("TB_BaconShopLockAll_Button")
        if fz is not None:
            shop["freezeCost"] = fz.g(T_COST)
        up = find(f"TB_BaconShopTechUp0{tier + 1}_Button")
        if up is not None:
            shop.update(upgradeCost=up.g(T_COST), canUpgrade=True)
        pe = self.ents.get(self.main_ent)
        shop["freeRefreshes"] = pe.g(T_FREE_REFRESH) if pe else 0
        return shop

    def _leaderboard(self):
        best: dict[int, Ent] = {}
        for e in self.ents.values():
            p = e.g(T_PLACE)
            if e.g(T_CARDTYPE) == CT_HERO and p > 0 and (p not in best or e.id > best[p].id):
                best[p] = e
        return [{"place": p, "playerId": e.g(T_PLAYER_ID), "heroCardId": e.card, "heroName": e.card,
                 "health": e.g(T_HEALTH) - e.g(T_DAMAGE), "armor": e.g(T_ARMOR), "tavernTier": e.g(T_TECH),
                 "isPlayer": e.tags.get(T_CONTROLLER) == self.main_pid, "dead": e.g(T_HEALTH) - e.g(T_DAMAGE) <= 0}
                for p, e in sorted(best.items())]

    def _state(self, phase_override: str | None = None) -> dict | None:
        if self.main_ent is None or self.game_id is None:
            return None
        turn, raw = self._turn()
        if turn <= 0:
            return None
        pe = self.ents[self.main_ent]
        hero = self.ents.get(pe.g(T_HERO_ENTITY))
        if hero is None or hero.tags.get(T_CONTROLLER) != self.main_pid:
            return None
        phase = phase_override or ("recruit" if raw % 2 == 1 else "combat")
        if hero.g(T_HEALTH) - hero.g(T_DAMAGE) <= 0:
            phase = "game_over"
        self._ench = self._enchantments()
        mine = self._player_ents(self.main_pid)
        board_all = [e for e in mine if e.g(T_ZONE) == Z_PLAY]
        hand = sorted((e for e in mine if e.g(T_ZONE) == Z_HAND), key=lambda e: e.g(T_ZONE_POS))
        secret = [e for e in mine if e.g(T_ZONE) == Z_SECRET]
        tier = hero.g(T_TECH)
        res, temp, used = pe.g(T_RES), pe.g(T_TEMP_RES), pe.g(T_RES_USED)
        player = {
            "heroCardId": hero.card, "heroName": hero.card, "health": hero.g(T_HEALTH) - hero.g(T_DAMAGE),
            "armor": hero.g(T_ARMOR), "tavernTier": tier, "gold": max(0, res + temp - used), "maxGold": res,
            "extraGoldNextTurn": max(pe.g(T_EXTRA_GOLD), hero.g(T_EXTRA_GOLD)), "leaderboardPlace": hero.g(T_PLACE),
            "heroPowers": [{"entityId": e.id, "cardId": e.card, "name": e.card, "cost": e.g(T_COST),
                            "used": e.g(T_EXHAUSTED) > 0,
                            "scriptData": ([e.g(t) for t in self._script_tags[:3]]
                                           if any(e.g(t) for t in self._script_tags[:3]) else None),
                            "tags": self._named(e, _useful)}
                           for e in sorted(board_all, key=lambda x: x.id) if e.g(T_CARDTYPE) == CT_HP],
            "board": [self._card(e) for e in sorted((e for e in board_all if e.g(T_CARDTYPE) == CT_MINION),
                                                    key=lambda e: e.g(T_ZONE_POS))],
            "hand": [self._card(e) for e in hand],
            "trinkets": [self._card(e) for e in board_all if e.g(self._trinket_tag) > 0],
            "questRewards": [self._card(e) for e in board_all if e.g(T_CARDTYPE) == CT_BGQR],
            "quests": [self._card(e) for e in secret if e.g(self._quest) > 0 or e.g(self._sidequest) > 0],
            "objectives": [self._card(e) for e in secret if e.g(self._objective) > 0],
            "playerTags": self._named(pe), "heroTags": self._named(hero), "shop": self._shop(mine, tier),
        }
        game = self.ents[self.game_id]
        s = {"seq": self.seq, "schema": 2, "isBattlegrounds": True, "inGame": True, "mode": "solo",
             "turn": turn, "phase": phase, "availableRaces": [r for r, _ in self.races.most_common(5)],
             "anomalyDbfId": game.g(tag_id("BACON_GLOBAL_ANOMALY_DBID")) or None,
             "player": player, "tavern": None, "opponent": None,
             "leaderboard": self._leaderboard(), "nextOpponentPlayerId": pe.g(T_NEXT_OPP),
             "damageCap": game.g(tag_id("BACON_COMBAT_DAMAGE_CAP")),
             "damageCapEnabled": game.g(tag_id("BACON_COMBAT_DAMAGE_CAP_ENABLED")) > 0,
             "gameTags": self._named(game, lambda n: n.startswith("BACON")), "offered": None}
        opp = sorted((e for e in self._player_ents(self.opp_pid)
                      if e.g(T_ZONE) == Z_PLAY and e.g(T_CARDTYPE) in (CT_MINION, CT_BGSPELL)),
                     key=lambda e: e.g(T_ZONE_POS))
        if phase == "recruit":
            s["tavern"] = [self._card(e) for e in opp]
            for e in opp:
                for r in (_catalog().get(e.card or "") or {}).get("races") or []:
                    if r not in ("ALL",):
                        self.races[r] += 1
        elif phase == "combat":
            oh = [e for e in self._player_ents(self.opp_pid) if e.g(T_ZONE) == Z_PLAY and e.g(T_CARDTYPE) == CT_HERO]
            oh = max(oh, key=lambda e: e.id) if oh else None
            s["opponent"] = {"heroCardId": oh.card if oh else None, "heroName": oh.card if oh else None,
                             "health": (oh.g(T_HEALTH) - oh.g(T_DAMAGE)) if oh else 0,
                             "armor": oh.g(T_ARMOR) if oh else 0, "tavernTier": oh.g(T_TECH) if oh else 0,
                             "board": [self._card(e) for e in opp if e.g(T_CARDTYPE) == CT_MINION]}
        if self.offered:
            cards = [self.ents[i] for i in self.offered if i in self.ents]
            if cards:
                if any(c.g(T_DARK_GIFT) > 0 for c in cards):
                    kind = "dark_gift"
                elif all(c.g(self._trinket_tag) > 0 for c in cards):
                    kind = "trinket"
                elif all(c.g(T_CARDTYPE) == CT_HP for c in cards):
                    kind = "hero_power"
                elif all(c.g(T_CARDTYPE) == CT_HERO for c in cards):
                    kind = "hero"
                elif all(c.g(T_CARDTYPE) == CT_MINION for c in cards):
                    kind = "discover"
                elif all(c.g(T_CARDTYPE) in (CT_SPELL, CT_BGSPELL) for c in cards):
                    kind = "discover_spell"
                else:
                    kind = "other"
                s["offered"] = {"kind": kind, "cards": [self._card(c) for c in cards]}
        self.seq += 1
        return s

    def _snapshot(self):
        s = self._state()
        if s is not None and s["phase"] == "recruit":
            self.records.append(s)

    def _combat_snapshot(self):
        turn, raw = self._turn()
        if raw % 2 == 1 or turn in self._combat_turn_done:
            return
        s = self._state("combat")
        if s is not None and s.get("opponent") and s["opponent"]["board"]:
            self._combat_turn_done.add(turn)
            self.records.append(s)


def parse(xml_bytes: bytes) -> list[dict]:
    """一份回放 XML → 记录列表（招募阶段每个决策点一条 + 每回合战斗开始一条）。"""
    r = Replay()
    for ev, el in iterparse(io.BytesIO(xml_bytes), events=("start", "end")):
        r.feed(ev, el)
        if ev == "end" and el.tag in ("FullEntity", "ShowEntity", "ChangeEntity", "TagChange", "HideEntity",
                                       "Options", "Choices", "ChosenEntities", "MetaData"):
            el.clear()
    return r.records


def read_xml(path) -> bytes:
    """.xml 或 .xml.zip（分享包里每局一个 zip，里面一个 replay.xml）。"""
    path = Path(path)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.endswith(".xml"))
            return z.read(name)
    return path.read_bytes()
