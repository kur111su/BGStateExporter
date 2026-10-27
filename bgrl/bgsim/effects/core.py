"""卡牌效果：注册表 + 常用操作（效果模板的积木）。

每张卡的实现是一个 Impl，按“触发时机”挂函数，函数参数是 Ctx：
    ctx.game / ctx.p（玩家）/ ctx.m（这张随从，法术时为 None）/ ctx.card（卡牌数据）
    ctx.g（是否金色）/ ctx.n（这张卡文本里的数字；金色卡的文本数字本来就是翻倍后的）
    ctx.target（指定目标）/ ctx.other（触发它的另一张卡）/ ctx.spell（触发它的法术）

触发时机（招募阶段）：
    battlecry        打出时（受“战吼触发两次”影响）
    on_sell          卖掉这张时
    end_of_turn      回合结束（从左到右）
    start_of_turn    回合开始
    after_play       你打出另一张随从后（other = 那张随从）
    after_spell      你施放法术后（spell = 法术牌，target = 法术目标）
    after_buy        你购买一张牌后（other）
    after_sell       你卖掉另一张随从后（other）
    after_summon     招募阶段召唤随从后（other；打出也算召唤）
    on_refresh       刷新酒馆后
    on_upgrade       升级酒馆后
    on_magnetize     有机械磁力吸附到某随从上后（other = 被吸附的，target = 宿主）
    on_discard       弃一张牌后（other）
    on_gold_spent    花钱后（n = 花了多少）
    activate         主动技能（花金币），配合 activate_cost
    cast             法术牌的效果
    hero_power       英雄技能的效果，配合 hp_cost / hp_target
    passive          被动标记（字典），引擎在相应位置读取，例如 {"battlecry_extra": 1}

target 规格（battlecry_target / spell_target / hp_target / activate_target）：
    None | "friendly" | "friendly:RACE" | "other_friendly" | "tavern" | "any"
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..model import Minion, Player, Spell


@dataclass
class Impl:
    card_id: str
    status: str = "done"                  # done 已实现 | approx 近似 | combat 只有战斗效果（交给 Bob's Buddy） | vanilla
    note: str = ""
    battlecry: Callable | None = None
    battlecry_target: str | None = None
    on_sell: Callable | None = None
    end_of_turn: Callable | None = None
    start_of_turn: Callable | None = None
    after_play: Callable | None = None
    after_spell: Callable | None = None
    after_buy: Callable | None = None
    after_sell: Callable | None = None
    after_summon: Callable | None = None
    on_refresh: Callable | None = None
    on_upgrade: Callable | None = None
    on_magnetize: Callable | None = None
    on_discard: Callable | None = None
    on_gold_spent: Callable | None = None
    activate: Callable | None = None
    activate_cost: int = 0
    activate_target: str | None = None
    cast: Callable | None = None
    spell_target: str | None = None
    hero_power: Callable | None = None
    hp_cost: int | None = None            # None = 用卡牌数据里的费用
    hp_target: str | None = None
    hp_uses_per_turn: int = 1
    passive: dict = field(default_factory=dict)
    choose: list | None = None            # 抉择的两个选项（函数列表）
    in_hand_start_of_turn: Callable | None = None   # 在手牌里时，回合开始触发（上锁宝箱倒计时等）
    after_battlecry: Callable | None = None
    after_discover: Callable | None = None
    after_choose_one: Callable | None = None
    on_card_added: Callable | None = None          # 有牌加入手牌时
    spell_on_minion: Callable | None = None        # 你对某随从施放法术后（target）
    after_combat: Callable | None = None           # 战斗结束后（近似处理战斗中产生的永久效果）
    on_gem: Callable | None = None                 # 鲜血宝石用在某随从上后（target）


IMPLS: dict[str, Impl] = {}


def card(card_id: str, **kw) -> Impl:
    """注册一张卡（普通版 id；金色版自动共用）。"""
    impl = Impl(card_id, **kw)
    IMPLS[card_id] = impl
    return impl


def combat_only(*card_ids: str, note: str = "") -> None:
    for cid in card_ids:
        IMPLS[cid] = Impl(cid, status="combat", note=note)


def vanilla(*card_ids: str) -> None:
    for cid in card_ids:
        IMPLS[cid] = Impl(cid, status="vanilla")


@dataclass
class Ctx:
    game: object
    p: Player
    m: Minion | None
    card: dict
    g: bool
    n: list[int]
    target: Minion | None = None
    other: object = None
    spell: Spell | None = None
    amount: int = 0

    def num(self, i: int, default: int = 0) -> int:
        return self.n[i] if i < len(self.n) else default


# ---------------- 积木：查找 ----------------

def friends(p: Player, race: str | None = None, exclude: Minion | None = None) -> list[Minion]:
    return [x for x in p.board if x is not exclude and (race is None or x.is_race(race))]


def random_friend(ctx: Ctx, race: str | None = None, exclude_self: bool = True) -> Minion | None:
    pool = friends(ctx.p, race, ctx.m if exclude_self else None)
    return ctx.game.rng.choice(pool) if pool else None


def adjacent(p: Player, m: Minion) -> list[Minion]:
    if m not in p.board:
        return []
    i = p.board.index(m)
    return [p.board[j] for j in (i - 1, i + 1) if 0 <= j < len(p.board)]


def leftmost(p: Player, race: str | None = None, exclude: Minion | None = None) -> Minion | None:
    xs = friends(p, race, exclude)
    return xs[0] if xs else None


def rightmost(p: Player, race: str | None = None, exclude: Minion | None = None) -> Minion | None:
    xs = friends(p, race, exclude)
    return xs[-1] if xs else None


def hand_minions(p: Player, race: str | None = None) -> list[Minion]:
    return [c for c in p.hand if isinstance(c, Minion) and (race is None or c.is_race(race))]


# ---------------- 积木：改变状态 ----------------

def buff(m: Minion | None, atk: int = 0, hp: int = 0, temp: bool = False) -> None:
    if m is None:
        return
    m.atk += atk
    m.hp += hp
    if temp:
        m.temp_atk += atk
        m.temp_hp += hp


def buff_all(ms, atk: int = 0, hp: int = 0, temp: bool = False) -> None:
    for m in list(ms):
        buff(m, atk, hp, temp)


def give(m: Minion | None, *kws: str) -> None:
    if m is not None:
        m.kw.update(kws)


def gain_gold(ctx_or_p, n: int) -> None:
    p = ctx_or_p.p if isinstance(ctx_or_p, Ctx) else ctx_or_p
    p.gold += n


def gold_next_turn(p: Player, n: int) -> None:
    p.extra_gold_next += n


def add_card(ctx: Ctx, card_id: str, **data) -> object:
    return ctx.game.add_to_hand(ctx.p, card_id, **data)


def blood_gems(ctx: Ctx, target: Minion | None, times: int = 1) -> None:
    """对目标使用鲜血宝石（每颗 +宝石攻击/+宝石生命）。"""
    if target is None:
        return
    for _ in range(times):
        buff(target, ctx.p.counters["blood_gem_atk"], ctx.p.counters["blood_gem_hp"])
        ctx.game.fire(ctx.p, "after_spell", spell=Spell(0, "BG20_GEM"), target=target, counts=False)


def gem_upgrade(p: Player, atk: int = 0, hp: int = 0) -> None:
    p.counters["blood_gem_atk"] += atk
    p.counters["blood_gem_hp"] += hp


def tavern_buff(ctx: Ctx, atk: int = 0, hp: int = 0, race: str | None = None) -> None:
    """“酒馆里的随从获得 +X/+Y”（本局永久）：已在酒馆的立即加，之后刷出来的也加。"""
    if race is None:
        ctx.p.counters["tavern_atk"] += atk
        ctx.p.counters["tavern_hp"] += hp
    else:
        key = "tavern_" + race
        a, h = ctx.p.counters.get(key, (0, 0))
        ctx.p.counters[key] = (a + atk, h + hp)
    for m in ctx.p.shop:
        if race is None or m.is_race(race):
            buff(m, atk, hp)


def summon(ctx: Ctx, card_id: str, right_of: Minion | None = None, atk: int | None = None,
           hp: int | None = None) -> Minion | None:
    return ctx.game.summon(ctx.p, card_id, right_of if right_of is not None else ctx.m, atk=atk, hp=hp)


def discover(ctx: Ctx, tier: int | None = None, race: str | None = None, exact_tier: bool = False,
             count: int = 3, kind: str = "discover", source: str = "", mechanic: str | None = None) -> None:
    ctx.game.discover(ctx.p, tier=tier, race=race, exact_tier=exact_tier, count=count, kind=kind,
                      source=source or (ctx.card.get("id") if ctx.card else ""), mechanic=mechanic)


def golden_mult(ctx: Ctx) -> int:
    return 2 if ctx.g else 1


# ---------------- 积木：酒馆法术、卡池、计数器 ----------------

GEMS = {"BG20_GEM", "BG20_GEM_Reborn", "BG20_GEM_Taunt", "BG20_GEM_DivineShield"}


def spell_stats(ctx: Ctx, m: Minion | None, atk: int = 0, hp: int = 0, temp: bool = False) -> None:
    """法术给属性：酒馆法术额外加“你的酒馆法术额外 +X/+Y”。"""
    if m is None:
        return
    if ctx.spell is not None and ctx.spell.card_id not in GEMS:
        if atk or hp:
            atk += ctx.p.counters["tavern_spell_atk"]
            hp += ctx.p.counters["tavern_spell_hp"]
    buff(m, atk, hp, temp)


def tavern_spell_bonus(p: Player, atk: int = 0, hp: int = 0) -> None:
    p.counters["tavern_spell_atk"] += atk
    p.counters["tavern_spell_hp"] += hp


def get_random_minion(ctx: Ctx, n: int = 1, tier: int | None = None, race: str | None = None,
                      exact: bool = False, golden: bool = False, mechanic: str | None = None) -> list:
    """从卡池里随机获得随从（会从卡池里拿走）。"""
    game, out = ctx.game, []
    for _ in range(n):
        ids = game.draw_from_pool(tier or ctx.p.tier, 1, exact_tier=exact, race=race,
                                  mechanic=mechanic)
        if not ids:
            break
        cid = ids[0]
        if golden and game.card(cid).get("goldenId"):
            cid = game.card(cid)["goldenId"]
        c = game.add_to_hand(ctx.p, cid)
        if c is not None and golden:
            c.triple_reward = False
        out.append(c)
    return out


def get_random_spell(ctx: Ctx, n: int = 1, cost: int | None = None, tier: int | None = None,
                     stats_only: bool = False) -> list:
    game, out = ctx.game, []
    cands = [s for s in game.spell_pool if (game.card(s).get("tier") or 1) <= (tier or ctx.p.tier)
             and (cost is None or game.card(s).get("cost") == cost)
             and (not stats_only or "/+" in game.card(s).get("textEn", "") or "+" in game.card(s).get("textEn", ""))]
    for _ in range(n):
        if cands:
            out.append(game.add_to_hand(ctx.p, game.rng.choice(cands)))
    return out


def discover_spell(ctx: Ctx, tier: int | None = None, count: int = 3) -> None:
    game = ctx.game
    cands = [s for s in game.spell_pool if (game.card(s).get("tier") or 1) <= (tier or ctx.p.tier)]
    if not cands:
        return
    picks = game.rng.sample(cands, min(count, len(cands)))
    options = [game.new_spell(s) for s in picks]

    def pick(g, pl, idx):
        if len(pl.hand) < g.rules.hand_size:
            pl.hand.append(options[idx])
            g.fire(pl, "after_discover", other=options[idx])
            g.fire(pl, "on_card_added", other=options[idx])
    from ..model import Choice
    ctx.p.pending.append(Choice("discover_spell", options, pick, ctx.card.get("id", "")))


def consume_tavern(ctx: Ctx, m: Minion | None, how: str = "random", mult: int = 1, keywords: bool = False) -> bool:
    """吞食酒馆里的一个随从，获得其属性值（×mult）。"""
    p = ctx.p
    if m is None or not p.shop:
        return False
    if how == "highest_hp":
        v = max(p.shop, key=lambda x: (x.hp, x.atk))
    elif how == "highest_atk":
        v = max(p.shop, key=lambda x: (x.atk, x.hp))
    else:
        v = ctx.game.rng.choice(p.shop)
    p.shop.remove(v)
    buff(m, v.atk * mult, v.hp * mult)
    if keywords:
        m.kw |= v.kw
    ctx.game.return_to_pool(v)
    return True


def undead_attack(ctx: Ctx, n: int) -> None:
    """“你的亡灵在本局中（无论在哪）+n 攻击力”。"""
    ctx.p.counters["undead_atk"] = ctx.p.counters.get("undead_atk", 0) + n
    for m in ctx.p.board + hand_minions(ctx.p):
        if m.is_race("UNDEAD"):
            m.atk += n


def volumizer_bonus(ctx: Ctx, atk: int = 0, hp: int = 0) -> None:
    from ..game import VOLUMIZERS
    ctx.p.counters["vol_atk"] = ctx.p.counters.get("vol_atk", 0) + atk
    ctx.p.counters["vol_hp"] = ctx.p.counters.get("vol_hp", 0) + hp
    for m in ctx.p.board + hand_minions(ctx.p):
        if ctx.game.base_id(m.card_id) in VOLUMIZERS or any(a in VOLUMIZERS for a in m.attached):
            n = 1 + sum(1 for a in m.attached if a in VOLUMIZERS) - (0 if ctx.game.base_id(m.card_id) in VOLUMIZERS else 1)
            m.atk += atk * n
            m.hp += hp * n


def deity(ctx: Ctx, atk: int, hp: int) -> None:
    """“给你的神祇 +X/+Y”：神祇是战斗中出现的随从，这里只记在计数器上（近似）。"""
    ctx.p.counters["deity_atk"] = ctx.p.counters.get("deity_atk", 0) + atk
    ctx.p.counters["deity_hp"] = ctx.p.counters.get("deity_hp", 0) + hp


def beetle_bonus(ctx: Ctx, atk: int, hp: int) -> None:
    ctx.p.counters["beetle_atk"] = ctx.p.counters.get("beetle_atk", 0) + atk
    ctx.p.counters["beetle_hp"] = ctx.p.counters.get("beetle_hp", 0) + hp


def discard(ctx: Ctx, card=None) -> object:
    """弃一张手牌（默认最左边的），触发“弃牌后”效果；被弃掉的牌有自己的弃牌效果时执行。"""
    p = ctx.p
    if card is None:
        card = p.hand[0] if p.hand else None
    if card is None or card not in p.hand:
        return None
    p.hand.remove(card)
    p.counters["discards"] += 1
    impl = ctx.game.impl(card.card_id)
    if impl and impl.passive.get("on_discard_self"):
        impl.passive["on_discard_self"](ctx.game.ctx(p, None, card.card_id, other=card))
    ctx.game.fire(p, "on_discard", other=card)
    return card


def free_refreshes(p: Player, n: int) -> None:
    p.free_refreshes += n


def random_keyword(ctx: Ctx, m: Minion | None, n: int = 1) -> None:
    if m is None:
        return
    pool = [k for k in ("taunt", "divineShield", "windfury", "reborn", "venomous", "stealth") if k not in m.kw]
    for k in ctx.game.rng.sample(pool, min(n, len(pool))):
        m.kw.add(k)


def bonus_keywords(p: Player) -> int:
    kws = set()
    for m in p.board:
        kws |= m.kw & {"taunt", "divineShield", "windfury", "reborn", "venomous", "stealth", "poisonous"}
    return len(kws)


def most_common_type(p: Player) -> str | None:
    from collections import Counter
    c = Counter(r for m in p.board for r in m.races if r != "ALL")
    return c.most_common(1)[0][0] if c else None


def make_golden(ctx: Ctx, m: Minion | None) -> None:
    """把随从变成金色：换成金色卡，属性 = 金色基础属性 + 原来多出的属性。"""
    if m is None or m.golden:
        return
    game = ctx.game
    gid = game.card(m.card_id).get("goldenId")
    if not gid:
        return
    bc, gc = game.card(m.card_id), game.card(gid)
    m.atk += gc["atk"] - bc["atk"]
    m.hp += gc["hp"] - bc["hp"]
    m.card_id = gid
    m.golden = True


def refresh_buff(p: Player, how: str, atk: int, hp: int) -> None:
    """“本局每次刷新酒馆后，给其中一个随机随从（random）/ 所有随从（all）+X/+Y”。"""
    p.data.setdefault("refresh_buffs", []).append((how, atk, hp))
