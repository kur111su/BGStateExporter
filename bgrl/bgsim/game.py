"""酒馆战旗招募阶段规则引擎（单人模式，8 人）。

战斗交给 CombatBackend（正式训练用 Bob's Buddy，见 combat.py）。

动作（元组，所有卡用 uid 指代）：
    ("buy", uid)                  买酒馆里的随从或法术
    ("sell", uid)                 卖场上的随从
    ("play", uid, pos, target)    打出手牌：pos = 放在场上第几个位置（0..len），target = 目标 uid 或 None
    ("move", uid, pos)            调整站位
    ("activate", uid, target)     随从的主动技能
    ("hero_power", target)        英雄技能
    ("upgrade",) ("refresh",) ("freeze",)
    ("choose", index)             发现 / 抉择
    ("end",)                      结束回合
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from . import effects  # noqa: F401  （导入时注册所有卡牌实现）
from .effects.core import IMPLS, Ctx, Impl
from .model import Choice, Minion, Player, Spell
from .rules import Rules

TRIPLE_REWARD = "TB_BaconShop_Triples_01"
VOLUMIZERS = {"BG34_170t", "BG34_170t2", "BG34_170t3"}


class IllegalAction(Exception):
    pass


@dataclass
class TurnResult:
    pid: int
    opponent: int
    damage_dealt: int
    damage_taken: int


class Game:
    def __init__(self, db: dict, rules: Rules | None = None, seed: int | None = None, combat=None,
                 heroes: list[str] | None = None, races: list[str] | None = None) -> None:
        self.db = db["cards"] if "cards" in db else db
        self.skins = db.get("skins", {}) if "cards" in db else {}
        self.rules = rules or Rules()
        self.rng = random.Random(seed)
        self.combat = combat
        self.turn = 0
        self._uid = 1000
        self.log: list[str] = []
        all_races = sorted({r for c in self.db.values() if c["category"] == "pool" for r in c["races"]} - {"ALL"})
        self.races = races or sorted(self.rng.sample(all_races, self.rules.races_per_game))
        self.pool: dict[str, int] = {}
        for c in self.db.values():
            if c["category"] != "pool" or c["duosOnly"] or c["tier"] > self.rules.max_tier:
                continue
            if c["races"] and "ALL" not in c["races"] and not any(r in self.races for r in c["races"]):
                continue
            self.pool[c["id"]] = self.rules.pool_copies.get(c["tier"], 5)
        self.spell_pool = [c["id"] for c in self.db.values() if c["category"] == "spell" and not c["duosOnly"]]
        hero_ids = heroes or self.rng.sample([c["id"] for c in self.db.values() if c["category"] == "hero"],
                                             self.rules.players)
        self.players: list[Player] = []
        for i, h in enumerate(hero_ids):
            hc = self.db.get(h, {})
            armor = hc.get("armor", self.rules.default_armor) if hc else self.rules.default_armor
            p = Player(pid=i, hero_id=h, health=hc.get("hp") or self.rules.hero_health, armor=armor,
                       upgrade_cost=self.rules.upgrade_cost[2])
            self.players.append(p)
        self.dead_order: list[int] = []
        self.last_boards: dict[int, list[Minion]] = {}

    # ---------------- 工具 ----------------

    def uid(self) -> int:
        self._uid += 1
        return self._uid

    def card(self, card_id: str) -> dict:
        return self.db.get(card_id) or {"id": card_id, "category": "unknown", "tier": 1, "atk": 0, "hp": 0,
                                        "races": [], "keywords": [], "mechanics": [], "numbers": [],
                                        "golden": card_id.endswith("_G"), "type": "minion", "cost": 0}

    @staticmethod
    def base_id(card_id: str) -> str:
        return card_id[:-2] if card_id.endswith("_G") else card_id

    def impl(self, card_id: str) -> Impl | None:
        return IMPLS.get(self.base_id(card_id)) or IMPLS.get(card_id)

    def new_minion(self, card_id: str, atk: int | None = None, hp: int | None = None) -> Minion:
        c = self.card(card_id)
        m = Minion(self.uid(), card_id, c["atk"] if atk is None else atk, c["hp"] if hp is None else hp,
                   c.get("tier") or 1, list(c.get("races", [])), bool(c.get("golden")),
                   set(k for k in c.get("keywords", []) if k != "magnetic"))
        sd = c.get("scriptData") or []
        for i, v in enumerate(sd[:4]):
            m.sd[i] = v
        return m

    def new_spell(self, card_id: str, **data) -> Spell:
        c = self.card(card_id)
        return Spell(self.uid(), card_id, cost=c.get("cost", 0), tier=c.get("tier", 0), data=dict(data))

    def alive(self) -> list[Player]:
        return [p for p in self.players if p.alive]

    @property
    def done(self) -> bool:
        return len(self.alive()) <= 1

    def ctx(self, p: Player, m: Minion | None, card_id: str, **kw) -> Ctx:
        c = self.card(card_id)
        return Ctx(self, p, m, c, bool(c.get("golden")), list(c.get("numbers", [])), **kw)

    # ---------------- 卡池 / 酒馆 ----------------

    def draw_from_pool(self, max_tier: int, k: int, exact_tier: bool = False, race: str | None = None,
                       exclude: set[str] | None = None, mechanic: str | None = None) -> list[str]:
        out: list[str] = []
        for _ in range(k):
            cands = [(cid, n) for cid, n in self.pool.items() if n > 0
                     and (self.db[cid]["tier"] == max_tier if exact_tier else self.db[cid]["tier"] <= max_tier)
                     and (race is None or race in self.db[cid]["races"] or "ALL" in self.db[cid]["races"])
                     and (mechanic is None or mechanic in self.db[cid]["mechanics"])
                     and not (exclude and cid in exclude)]
            if not cands:
                break
            total = sum(n for _, n in cands)
            r = self.rng.random() * total
            for cid, n in cands:
                r -= n
                if r < 0:
                    break
            self.pool[cid] -= 1
            out.append(cid)
        return out

    def return_to_pool(self, m: Minion) -> None:
        base = self.base_id(m.card_id)
        if base in self.pool:
            self.pool[base] += 3 if m.golden else 1

    def roll_shop(self, p: Player, keep_frozen: bool = False) -> None:
        kept = [m for m in p.shop if m.frozen] if keep_frozen else []
        for m in p.shop:
            if m not in kept:
                self.return_to_pool(m)
        kept_spells = [s for s in p.shop_spells if s.data.get("frozen")] if keep_frozen else []
        p.shop = kept
        p.shop_spells = kept_spells
        n = self.rules.shop_minions.get(p.tier, 3) - len(p.shop)
        for cid in self.draw_from_pool(p.tier, max(0, n)):
            m = self.new_minion(cid)
            self._apply_tavern_buffs(p, m)
            p.shop.append(m)
        for _ in range(max(0, self.rules.shop_spells - len(p.shop_spells))):
            cands = [s for s in self.spell_pool if (self.db[s]["tier"] or 1) <= p.tier]
            if cands:
                p.shop_spells.append(self.new_spell(self.rng.choice(cands)))
        for m in p.shop:
            m.frozen = False
        for s in p.shop_spells:
            s.data["frozen"] = False
        p.frozen = False
        for how, a, h in p.data.get("refresh_buffs", []):
            if not p.shop:
                break
            targets = p.shop if how == "all" else [self.rng.choice(p.shop)]
            for m in targets:
                m.atk += a
                m.hp += h

    def _apply_tavern_buffs(self, p: Player, m: Minion) -> None:
        m.atk += p.counters["tavern_atk"]
        m.hp += p.counters["tavern_hp"]
        for r in m.races:
            a, h = p.counters.get("tavern_" + r, (0, 0))
            m.atk += a
            m.hp += h

    # ---------------- 回合流程 ----------------

    def start_turn(self) -> None:
        self.turn += 1
        for p in self.alive():
            for m in p.board + [c for c in p.hand if isinstance(c, Minion)]:
                m.atk -= m.temp_atk
                m.hp -= m.temp_hp
                m.temp_atk = m.temp_hp = 0
                m.kw -= set(m.data.pop("temp_kw", ()))
            p.max_gold = min(self.rules.max_gold, self.rules.start_gold + self.turn - 1) + \
                p.counters.get("max_gold_bonus", 0)
            p.gold = p.max_gold + p.extra_gold_next
            p.extra_gold_next = 0
            if self.turn > 1 and p.tier < self.rules.max_tier:
                p.upgrade_cost = max(0, p.upgrade_cost - 1)
            p.hero_power_used = False
            p.hero_power_uses = 0
            p.actions_this_turn = 0
            p.counters["spells_cast_turn"] = 0
            p.counters["gold_spent_turn"] = 0
            for _ in range(p.data.pop("time_mgmt", 0)):         # 时间管理：回合开始 +2/+2
                for m in p.board:
                    m.atk += 2
                    m.hp += 2
            self.roll_shop(p, keep_frozen=p.frozen)
            self._tick_hand(p)
            self.fire(p, "start_of_turn")
            for m in list(p.board):                       # 法术迸发：每回合给一张临时法术
                sc = self.card(m.card_id).get("spellcraft")
                if sc:
                    self.add_to_hand(p, sc, temp=True, source=m.uid)

    def end_recruit(self, p: Player) -> None:
        """一名玩家结束招募阶段：回合结束效果（“直到下回合”的加成在下回合开始时清掉），丢弃临时法术。"""
        times = 1 + self.passive_count(p, "end_of_turn_extra")
        for _ in range(times):
            self.fire(p, "end_of_turn")
        p.hand = [c for c in p.hand if not (isinstance(c, Spell) and c.data.get("temp"))]
        p.pending = []

    def _tick_hand(self, p: Player) -> None:
        """回合开始时手牌里的倒计时（上锁宝箱等）。"""
        for i, c in enumerate(list(p.hand)):
            impl = self.impl(c.card_id)
            if impl and impl.in_hand_start_of_turn:
                impl.in_hand_start_of_turn(self.ctx(p, None, c.card_id, other=c))

    def pair_players(self) -> list[tuple[Player, Player | None]]:
        alive = self.alive()
        self.rng.shuffle(alive)
        pairs: list[tuple[Player, Player | None]] = []
        free = list(alive)
        while len(free) >= 2:
            a = free.pop(0)
            avoid = set(a.recent_opponents[-self.rules.recent_opponents_avoid:])
            idx = next((i for i, b in enumerate(free) if b.pid not in avoid), 0)
            b = free.pop(idx)
            pairs.append((a, b))
        if free:
            pairs.append((free[0], None))          # 打幽灵
        return pairs

    def combat_phase(self) -> list[TurnResult]:
        pairs = self.pair_players()
        jobs = []
        for a, b in pairs:
            ghost = None
            if b is None:
                ghost_pid = self.dead_order[-1] if self.dead_order else None
                ghost = self.players[ghost_pid] if ghost_pid is not None else None
            jobs.append((a, b or ghost, b is None))
        dmg = self.combat.fight(self, [(a, b) for a, b, _ in jobs]) if self.combat else [0] * len(jobs)
        for (a, b, is_ghost), d in zip(jobs, dmg):
            for pl, res in ((a, 1 if d > 0 else -1 if d < 0 else 0), (None if is_ghost else b, -1 if d > 0 else 1 if d < 0 else 0)):
                if pl is not None:
                    pl.data["last_result"] = res
                    n = pl.data.pop("overconfidence", 0)            # 自负
                    if n:
                        pl.extra_gold_next += n * (3 if res > 0 else 1 if res == 0 else 0)
                    self.fire(pl, "after_combat", amount=res)
        cap = self.rules.cap_for_turn(self.turn)
        results = []
        for (a, b, is_ghost), d in zip(jobs, dmg):
            if b is None:
                continue
            if cap:
                d = max(-cap, min(cap, d))
            a.recent_opponents.append(b.pid)
            a.seen_boards[b.pid] = (self.turn, [m.copy(0) for m in b.board])
            if not is_ghost:
                b.recent_opponents.append(a.pid)
                b.seen_boards[a.pid] = (self.turn, [m.copy(0) for m in a.board])
            if d > 0 and not is_ghost:
                self.damage(b, d)
            elif d < 0:
                self.damage(a, -d)
            results.append(TurnResult(a.pid, b.pid, max(d, 0), max(-d, 0)))
            if not is_ghost:
                results.append(TurnResult(b.pid, a.pid, max(-d, 0), max(d, 0)))
        self._eliminate()
        return results

    def damage(self, p: Player, n: int) -> None:
        a = min(p.armor, n)
        p.armor -= a
        p.health -= n - a

    def _eliminate(self) -> None:
        dying = [p for p in self.alive() if p.health <= 0]
        if not dying:
            return
        dying.sort(key=lambda p: p.health)                 # 血量越低名次越靠后
        place = len(self.alive())
        for p in dying:
            p.alive = False
            p.place = place
            p.dead_turn = self.turn
            place -= 1
            self.dead_order.append(p.pid)
            for m in p.board + [c for c in p.hand if isinstance(c, Minion)] + p.shop:
                self.return_to_pool(m)
        if len(self.alive()) == 1:
            self.alive()[0].place = 1

    def run_turn(self, policies) -> list[TurnResult]:
        """policies[pid](game, player) -> action。跑一个完整回合（招募 + 战斗）。"""
        self.start_turn()
        for p in self.alive():
            while True:
                if p.actions_this_turn >= self.rules.max_actions_per_turn and not p.pending:
                    break
                a = policies[p.pid](self, p)
                if a[0] == "end" and not p.pending:
                    break
                self.step(p, a)
            self.end_recruit(p)
        return self.combat_phase()

    # ---------------- 触发 ----------------

    def fire(self, p: Player, hook: str, counts: bool = True, **kw) -> None:
        """对玩家场上（从左到右）和英雄技能、饰品触发某个时机。"""
        for m in list(p.board):
            if m not in p.board:
                continue
            impl = self.impl(m.card_id)
            fn = getattr(impl, hook, None) if impl else None
            if fn:
                times = 1
                fn_ctx = self.ctx(p, m, m.card_id, **kw)
                for _ in range(times):
                    fn(fn_ctx)
        for src in [p.hero_id, *p.trinkets]:
            impl = self.impl(self.skins.get(src, src))
            fn = getattr(impl, hook, None) if impl else None
            if fn:
                fn(self.ctx(p, None, src, **kw))

    def passive_count(self, p: Player, key: str) -> int:
        n = 0
        for m in p.board:
            impl = self.impl(m.card_id)
            if impl and key in impl.passive:
                v = impl.passive[key]
                n += v * (2 if m.golden else 1) if isinstance(v, int) else 1
        return n

    # ---------------- 卡牌进出 ----------------

    def add_to_hand(self, p: Player, card_id: str, **data):
        if len(p.hand) >= self.rules.hand_size:
            return None
        c = self.card(card_id)
        card = self.new_minion(card_id) if c.get("type") == "minion" else self.new_spell(card_id, **data)
        if isinstance(card, Minion):
            card.data.update(data)
            self.gain_minion(p, card)
        p.hand.append(card)
        self.fire(p, "on_card_added", other=card)
        if isinstance(card, Minion):
            self.check_triple(p, card)
        return card

    def gain_minion(self, p: Player, m: Minion) -> None:
        """随从进入玩家手牌/场上时，套用“本局（无论在哪）”的加成。"""
        if m.data.get("_gained"):
            return
        m.data["_gained"] = True
        if m.is_race("UNDEAD"):
            m.atk += p.counters.get("undead_atk", 0)
        if self.base_id(m.card_id) in VOLUMIZERS:
            m.atk += p.counters.get("vol_atk", 0)
            m.hp += p.counters.get("vol_hp", 0)

    def summon(self, p: Player, card_id: str, right_of: Minion | None = None, atk: int | None = None,
               hp: int | None = None) -> Minion | None:
        if len(p.board) >= self.rules.board_size:
            return None
        m = self.new_minion(card_id, atk, hp)
        self.gain_minion(p, m)
        pos = p.board.index(right_of) + 1 if right_of in p.board else len(p.board)
        p.board.insert(pos, m)
        self.fire(p, "after_summon", other=m)
        self.check_triple(p, m)
        return m

    def check_triple(self, p: Player, m: Minion) -> None:
        if m.golden:
            return
        base = m.card_id
        golden_id = self.card(base).get("goldenId")
        if not golden_id:
            return
        copies = [x for x in p.board + p.hand if isinstance(x, Minion) and x.card_id == base and not x.golden]
        if len(copies) < 3:
            return
        use = sorted(copies, key=lambda x: x in p.board)[:3]       # 先用手牌里的
        gc = self.card(golden_id)
        bc = self.card(base)
        g = self.new_minion(golden_id)
        g.atk = gc["atk"] + sum(x.atk - bc["atk"] for x in use)
        g.hp = gc["hp"] + sum(x.hp - bc["hp"] for x in use)
        for x in use:
            g.kw |= x.kw
            (p.board if x in p.board else p.hand).remove(x)
        g.triple_reward = True
        p.hand.append(g)

    def discover(self, p: Player, tier: int | None = None, race: str | None = None, exact_tier: bool = False,
                 count: int = 3, kind: str = "discover", source: str = "", mechanic: str | None = None) -> None:
        t = tier or p.tier
        ids = self.draw_from_pool(t, count, exact_tier=exact_tier, race=race, mechanic=mechanic)
        if not ids:
            return
        options = [self.new_minion(cid) for cid in ids]

        def pick(game: "Game", pl: Player, idx: int) -> None:
            for i, o in enumerate(options):
                if i == idx:
                    if len(pl.hand) < game.rules.hand_size:
                        game.gain_minion(pl, o)
                        pl.hand.append(o)
                        game.fire(pl, "after_discover", other=o)
                        game.fire(pl, "on_card_added", other=o)
                        game.check_triple(pl, o)
                    else:
                        game.return_to_pool(o)
                else:
                    game.return_to_pool(o)
        p.pending.append(Choice(kind, options, pick, source))

    # ---------------- 动作 ----------------

    def legal_actions(self, p: Player) -> list[tuple]:
        if p.pending:
            return [("choose", i) for i in range(len(p.pending[0].options))]
        acts: list[tuple] = [("end",)]
        buy_cost = self.buy_cost(p)
        if len(p.hand) < self.rules.hand_size:
            acts += [("buy", m.uid) for m in p.shop if p.gold >= buy_cost]
            acts += [("buy", s.uid) for s in p.shop_spells if p.gold >= s.cost]
        acts += [("sell", m.uid) for m in p.board]
        for c in p.hand:
            if isinstance(c, Minion):
                magnetic = "magnetic" in self.card(c.card_id).get("keywords", [])
                if len(p.board) < self.rules.board_size:
                    for t in self.targets(p, self._bc_target(c), c) or [None]:
                        acts.append(("play", c.uid, len(p.board), t))
                if magnetic:
                    acts += [("play", c.uid, -1, x.uid) for x in p.board if x.is_race("MECHANICAL")]
            else:
                impl = self.impl(c.card_id)
                spec = impl.spell_target if impl else None
                if p.gold >= 0:
                    ts = self.targets(p, spec, None)
                    if spec and not ts:
                        continue
                    acts += [("play", c.uid, 0, t) for t in (ts or [None])]
        if len(p.board) > 1:
            acts += [("move", m.uid, i) for m in p.board for i in range(len(p.board)) if p.board.index(m) != i]
        for m in p.board:
            impl = self.impl(m.card_id)
            if impl and impl.activate and p.gold >= impl.activate_cost:
                acts += [("activate", m.uid, t) for t in (self.targets(p, impl.activate_target, m) or [None])]
        hp = self.hero_power_impl(p)
        if hp and hp.hero_power and p.hero_power_uses < hp.hp_uses_per_turn and p.gold >= self.hp_cost(p):
            acts += [("hero_power", t) for t in (self.targets(p, hp.hp_target, None) or [None])]
        if p.tier < self.rules.max_tier and p.gold >= p.upgrade_cost:
            acts.append(("upgrade",))
        if p.gold >= self.refresh_cost(p) or p.free_refreshes > 0:
            acts.append(("refresh",))
        acts.append(("freeze",))
        return acts

    def _bc_target(self, m: Minion) -> str | None:
        impl = self.impl(m.card_id)
        return impl.battlecry_target if impl else None

    def targets(self, p: Player, spec: str | None, source: Minion | None) -> list[int]:
        if not spec:
            return []
        if spec == "friendly":
            return [m.uid for m in p.board]
        if spec == "other_friendly":
            return [m.uid for m in p.board if m is not source]
        if spec.startswith("friendly:"):
            race = spec.split(":", 1)[1]
            return [m.uid for m in p.board if m.is_race(race) and m is not source]
        if spec == "tavern":
            return [m.uid for m in p.shop]
        if spec == "any":
            return [m.uid for m in p.board + p.shop]
        return []

    def buy_cost(self, p: Player) -> int:
        return max(0, self.rules.buy_cost + p.counters["buy_cost_mod"])

    def refresh_cost(self, p: Player) -> int:
        return max(0, self.rules.refresh_cost + p.counters["refresh_cost_mod"])

    def hero_power_impl(self, p: Player) -> Impl | None:
        hc = self.card(self.skins.get(p.hero_id, p.hero_id))
        return self.impl(hc.get("heroPower", "")) if hc.get("heroPower") else None

    def hp_cost(self, p: Player) -> int:
        impl = self.hero_power_impl(p)
        if impl and impl.hp_cost is not None:
            return impl.hp_cost
        hc = self.card(self.skins.get(p.hero_id, p.hero_id))
        return self.card(hc.get("heroPower", "")).get("cost", 0)

    def spend(self, p: Player, n: int) -> None:
        if n > p.gold:
            raise IllegalAction("金币不够")
        p.gold -= n
        if n > 0:
            p.counters["gold_spent_turn"] += n
            self.fire(p, "on_gold_spent", amount=n)

    def step(self, p: Player, a: tuple) -> None:
        kind = a[0]
        p.actions_this_turn += 1
        if p.pending:
            if kind != "choose":
                raise IllegalAction("有待选择的选项")
            ch = p.pending.pop(0)
            if ch.on_pick:
                ch.on_pick(self, p, int(a[1]))
            return
        if kind == "buy":
            self._buy(p, a[1])
        elif kind == "sell":
            self._sell(p, a[1])
        elif kind == "play":
            self._play(p, a[1], a[2], a[3] if len(a) > 3 else None)
        elif kind == "move":
            m = p.find(a[1])
            if m not in p.board:
                raise IllegalAction("不在场上")
            p.board.remove(m)
            p.board.insert(max(0, min(a[2], len(p.board))), m)
        elif kind == "activate":
            m = p.find(a[1])
            impl = self.impl(m.card_id) if m else None
            if not impl or not impl.activate or m not in p.board:
                raise IllegalAction("不能激活")
            self.spend(p, impl.activate_cost)
            impl.activate(self.ctx(p, m, m.card_id, target=p.find(a[2]) if len(a) > 2 and a[2] else None))
        elif kind == "hero_power":
            impl = self.hero_power_impl(p)
            if not impl or not impl.hero_power:
                raise IllegalAction("没有可用的英雄技能")
            self.spend(p, self.hp_cost(p))
            p.hero_power_uses += 1
            p.hero_power_used = True
            hc = self.card(self.skins.get(p.hero_id, p.hero_id))
            impl.hero_power(self.ctx(p, None, hc["heroPower"], target=p.find(a[1]) if len(a) > 1 and a[1] else None))
        elif kind == "upgrade":
            if p.tier >= self.rules.max_tier:
                raise IllegalAction("已满级")
            self.spend(p, p.upgrade_cost)
            p.tier += 1
            p.upgrade_cost = self.rules.upgrade_cost.get(p.tier + 1, 0)
            self.fire(p, "on_upgrade")
        elif kind == "refresh":
            if p.free_refreshes > 0:
                p.free_refreshes -= 1
            else:
                self.spend(p, self.refresh_cost(p))
            p.counters["refreshes"] += 1
            self.roll_shop(p)
            self.fire(p, "on_refresh")
        elif kind == "freeze":
            p.frozen = not p.frozen
            for m in p.shop:
                m.frozen = p.frozen
            for s in p.shop_spells:
                s.data["frozen"] = p.frozen
        elif kind == "end":
            pass
        else:
            raise IllegalAction(f"未知动作 {kind}")

    def _buy(self, p: Player, uid: int) -> None:
        if len(p.hand) >= self.rules.hand_size:
            raise IllegalAction("手牌满了")
        m = next((x for x in p.shop if x.uid == uid), None)
        if m is not None:
            self.spend(p, self.buy_cost(p))
            p.shop.remove(m)
            m.frozen = False
            self.gain_minion(p, m)
            p.hand.append(m)
            p.counters["minions_bought"] += 1
            self.fire(p, "after_buy", other=m)
            self.check_triple(p, m)
            return
        s = next((x for x in p.shop_spells if x.uid == uid), None)
        if s is None:
            raise IllegalAction("酒馆里没有这张牌")
        cost = max(0, s.cost - p.data.pop("next_spell_discount", 0))
        self.spend(p, cost)
        p.shop_spells.remove(s)
        p.hand.append(s)
        self.fire(p, "after_buy", other=s)

    def _sell(self, p: Player, uid: int) -> None:
        m = next((x for x in p.board if x.uid == uid), None)
        if m is None:
            raise IllegalAction("场上没有这个随从")
        impl = self.impl(m.card_id)
        p.board.remove(m)
        bonus = m.data.get("sell_bonus", 0)
        if impl and impl.passive.get("sell_bonus_if_lost") and p.data.get("last_result", 0) < 0:
            bonus += impl.passive["sell_bonus_if_lost"] * (2 if m.golden else 1) + (1 if m.golden else 0)
        p.gold += self.rules.sell_value + bonus
        p.counters["minions_sold"] += 1
        if impl and impl.on_sell:
            impl.on_sell(self.ctx(p, m, m.card_id))
        self.fire(p, "after_sell", other=m)
        self.return_to_pool(m)

    def _play(self, p: Player, uid: int, pos: int, target_uid: int | None) -> None:
        c = next((x for x in p.hand if x.uid == uid), None)
        if c is None:
            raise IllegalAction("手里没有这张牌")
        target = p.find(target_uid) if target_uid else None
        if isinstance(c, Spell):
            p.hand.remove(c)
            self.cast_spell(p, c, target)
            return
        impl = self.impl(c.card_id)
        if pos == -1:                                   # 磁力吸附
            if target is None or target not in p.board or not target.is_race("MECHANICAL"):
                raise IllegalAction("磁力目标必须是场上的机械")
            p.hand.remove(c)
            target.atk += c.atk
            target.hp += c.hp
            target.kw |= c.kw
            target.attached.append(c.card_id)
            if impl and impl.passive.get("on_play_or_magnetize"):
                impl.passive["on_play_or_magnetize"](self.ctx(p, c, c.card_id, target=target))
            self.fire(p, "on_magnetize", other=c, target=target)
            return
        if len(p.board) >= self.rules.board_size:
            raise IllegalAction("场上满了")
        p.hand.remove(c)
        p.board.insert(max(0, min(pos, len(p.board))), c)
        if impl and impl.passive.get("on_play_or_magnetize"):
            impl.passive["on_play_or_magnetize"](self.ctx(p, c, c.card_id, target=target))
        if impl and impl.battlecry:
            times = 1 + self.passive_count(p, "battlecry_extra")
            for _ in range(times):
                impl.battlecry(self.ctx(p, c, c.card_id, target=target))
                p.counters["battlecries"] += 1
                self.fire(p, "after_battlecry", other=c)
        if impl and impl.choose:
            self.choose_one(p, c, c.card_id, impl, target)
        if c.triple_reward:
            c.triple_reward = False
            self.add_to_hand(p, TRIPLE_REWARD, tier=min(p.tier + 1, self.rules.triple_reward_max_tier))
        if c.is_race("ELEMENTAL"):
            p.counters["elementals_played"] += 1
        self.fire(p, "after_play", other=c)
        self.fire(p, "after_summon", other=c)
        self.check_triple(p, c)

    def choose_one(self, p: Player, m: Minion | None, card_id: str, impl: Impl, target, spell=None) -> None:
        """抉择：两个选项作为待选择项；“两个效果合并”（荆棘开拓者等）时直接都执行。"""
        def run(idx: int) -> None:
            impl.choose[idx](self.ctx(p, m, card_id, target=target, spell=spell))
        if p.counters.get("choose_both", 0) > 0 or (m is not None and m.data.get("choose_both")):
            p.counters["choose_both"] = max(0, p.counters.get("choose_both", 0) - 1)
            for i in range(len(impl.choose)):
                run(i)
        else:
            opts = [Spell(self.uid(), f"{card_id}#选项{i + 1}") for i in range(len(impl.choose))]
            p.pending.append(Choice("choose_one", opts, lambda g, pl, idx: run(idx), card_id))
        self.fire(p, "after_choose_one", other=m)

    def cast_spell(self, p: Player, s: Spell, target: Minion | None) -> None:
        impl = self.impl(s.card_id)
        if s.card_id == TRIPLE_REWARD:
            self.discover(p, tier=s.data.get("tier", min(p.tier + 1, 6)), exact_tier=True, kind="triple")
        elif impl and (impl.cast or impl.choose):
            times = 1
            if target is not None and target in p.board:
                times += self.passive_count(p, "friendly_spell_extra")
            if impl.passive.get("bounty"):
                times += self.passive_count(p, "bounty_extra")
            for _ in range(times):
                if impl.cast:
                    impl.cast(self.ctx(p, None, s.card_id, target=target, spell=s))
            if impl.choose:
                self.choose_one(p, None, s.card_id, impl, target, spell=s)
            if target is not None and target in p.board:
                self.fire(p, "spell_on_minion", spell=s, target=target)
        p.counters["spells_cast"] += 1
        p.counters["spells_cast_turn"] += 1
        if s.card_id not in ("BG20_GEM", TRIPLE_REWARD) and self.card(s.card_id).get("category") == "spell":
            p.data["last_spell"] = s.card_id
        self.fire(p, "after_spell", spell=s, target=target)
