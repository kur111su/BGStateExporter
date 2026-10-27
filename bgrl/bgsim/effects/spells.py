"""酒馆法术和衍生法术（鲜血宝石、三连奖励、法术迸发……）。

status="approx" 表示做了近似（比如战斗开始时生效的法术，环境里只记录，交给 Bob's Buddy 时尚未传入）。
"""
from .core import (Impl, adjacent, beetle_bonus, blood_gems, buff, buff_all, card, consume_tavern, deity,
                   discover, discover_spell, free_refreshes, friends, gain_gold, gem_upgrade, get_random_minion,
                   get_random_spell, give, gold_next_turn, hand_minions, leftmost, make_golden, most_common_type,
                   spell_stats, tavern_buff, tavern_spell_bonus, undead_attack, bonus_keywords, refresh_buff)

F = "friendly"


def _t(fn, target=F, **kw):
    return dict(cast=fn, spell_target=target, **kw)


# ---------------- 鲜血宝石 ----------------

def _gem(ctx, kw=None):
    t = ctx.target
    if t is None:
        return
    times = 1
    if ctx.spell is not None and not ctx.spell.data.get("from_effect"):
        times += ctx.game.passive_count(ctx.p, "gem_extra")
    for _ in range(times):
        buff(t, ctx.p.counters["blood_gem_atk"], ctx.p.counters["blood_gem_hp"])
        if kw and t.is_race("QUILBOAR"):
            give(t, kw)
    ctx.game.fire(ctx.p, "on_gem", target=t)


card("BG20_GEM", **_t(lambda c: _gem(c)))
card("BG20_GEM_Reborn", **_t(lambda c: _gem(c, "reborn")))
card("BG20_GEM_Taunt", **_t(lambda c: _gem(c, "taunt")))
card("BG20_GEM_DivineShield", **_t(lambda c: _gem(c, "divineShield")))
card("BG31_893", choose=[lambda c: gem_upgrade(c.p, 1, 0), lambda c: gem_upgrade(c.p, 0, 1)])     # 宝石日

# ---------------- 1 本 ----------------
card("BG28_503", **_t(lambda c: (spell_stats(c, c.target, 0, 3), give(c.target, "taunt"))))      # 强固
card("BG28_504", cast=lambda c: get_random_minion(c, tier=1, exact=True))                       # 招募学徒
card("BG28_512", cast=lambda c: _steal_random(c))                                               # 魔法套索
card("BG28_810", cast=lambda c: gain_gold(c, 1))                                                 # 酒馆币
card("BG28_897", **_t(lambda c: spell_stats(c, c.target, 2, 2)))                                 # 香蕉果盘
card("BG28_966", cast=lambda c: tavern_buff(c, 1, 2), status="approx", note="酒馆随从只加当前这批")  # 是苹果哦
card("BG31_880", spell_target=F, choose=[lambda c: spell_stats(c, c.target, 3, 1),               # 联盟旗帜
                                          lambda c: spell_stats(c, c.target, 2, 3)], status="approx",
     note="第二个选项的数值按 +2/+3 估计")
card("BG33_101", cast=lambda c: discover(c, tier=1, exact_tier=True))                           # 新芽


def _steal_random(c):
    if c.p.shop and len(c.p.hand) < c.game.rules.hand_size:
        m = c.game.rng.choice(c.p.shop)
        c.p.shop.remove(m)
        c.game.gain_minion(c.p, m)
        c.p.hand.append(m)
        c.game.check_triple(c.p, m)


# ---------------- 2 本 ----------------
def _chefs_choice(c):
    t = c.target
    races = [r for r in (t.races if t else []) if r != "ALL"]
    if races:
        get_random_minion(c, race=c.game.rng.choice(races))


card("BG28_518", **_t(_chefs_choice))                                                           # 主厨甄选
card("BG28_571", cast=lambda c: gain_gold(c, 1), status="approx", note="用生命购买未实现")          # 拼命发掘
card("BG28_805", cast=lambda c: _strike_oil(c, 1))                                              # 钻探原油
card("BG28_827", cast=lambda c: free_refreshes(c.p, 2))                                         # 快速浏览
card("BG34_330", cast=lambda c: discover(c, tier=c.p.tier, exact_tier=True), status="approx",
     note="锁定 1 回合未实现")
card("BG35_951", cast=lambda c: [spell_stats(c, m, 1, 2) for m in
                                 c.game.rng.sample(c.p.board, min(4, len(c.p.board)))])          # 暴风城之力
card("BG36_883", **_t(lambda c: spell_stats(c, c.target, 2, 3)), status="approx",               # 胜利者的面包
     note="赢了下回合再加未实现")


def _strike_oil(c, n):
    c.p.counters["max_gold_bonus"] = c.p.counters.get("max_gold_bonus", 0) + n
    c.p.max_gold += n
    c.p.gold += n


# ---------------- 3 本 ----------------
card("BG28_168", cast=lambda c: [spell_stats(c, m, 1, 1) for m in list(c.p.board)])            # 闪亮指环


def _trousers(c):
    spell_stats(c, c.target, 1, 2)
    if c.target is not None:
        if "taunt" in c.target.kw:
            c.target.kw.discard("taunt")
        else:
            c.target.kw.add("taunt")


card("BG28_520", **_t(_trousers))                                                               # 搞怪裤
card("BG28_521", cast=lambda c: discover(c, race=most_common_type(c.p)))                        # 位面望远镜
card("BG28_800", cast=lambda c: gold_next_turn(c.p, 2))                                         # 慎重投资
card("BG28_884", cast=lambda c: c.p.data.__setitem__("overconfidence", c.p.data.get("overconfidence", 0) + 1))  # 自负
card("BG28_886", cast=lambda c: tavern_buff(c, 2, 2))                                           # 富饶之杖


def _robust_evolution(c):
    t = c.target
    if t is None:
        return
    ids = c.game.draw_from_pool(min(t.tier + 1, 6), 1, exact_tier=True)
    if ids:
        nc = c.game.card(ids[0])
        t.card_id, t.tier, t.races = ids[0], nc["tier"], list(nc["races"])


card("BG30_804", **_t(_robust_evolution))                                                       # 强健进化
card("BG31_881", choose=[lambda c: [spell_stats(c, m, 2, 2) for m in list(c.p.board)],          # 时间管理
                         lambda c: c.p.data.__setitem__("time_mgmt", c.p.data.get("time_mgmt", 0) + 2)])
card("BG32_337", **_t(lambda c: [spell_stats(c, c.target, 1, 1) for _ in range(1 + bonus_keywords(c.p))]))  # 海鲜杂烩
_FOUR = lambda c: c.game.rng.sample(c.p.board, min(4, len(c.p.board)))  # noqa: E731
card("BG33_811", cast=lambda c: [spell_stats(c, m, 0, 4) for m in _FOUR(c)], passive={"bounty": 1})   # 生命悬赏令
card("BG33_812", cast=lambda c: [spell_stats(c, m, 4, 0) for m in _FOUR(c)], passive={"bounty": 1})   # 敌对悬赏令
card("BG33_813", cast=lambda c: spell_stats(c, leftmost(c.p), 6, 6), passive={"bounty": 1})          # 谋私悬赏令
card("BG33_814", cast=lambda c: get_random_minion(c, race=most_common_type(c.p)), passive={"bounty": 1})  # 友方
card("BG33_815", cast=lambda c: gain_gold(c, 2), passive={"bounty": 1})                             # 财富悬赏令
card("BG36_624", **_t(lambda c: spell_stats(c, c.target, 4, 8)))                                   # 修理活


# ---------------- 4 本 ----------------
def _conch(c):
    got = get_random_minion(c, race="MURLOC")
    if got and got[0] is not None:
        c.game.add_to_hand(c.p, got[0].card_id)


card("BG28_601", cast=_conch)                                                                   # 克隆海螺
card("BG28_603", cast=lambda c: beetle_bonus(c, 0, 0), status="approx", note="战斗中召唤甲虫交给战斗模拟（未传入）")
card("BG28_606", cast=lambda c: [c.game.add_to_hand(c.p, sid) for sid in c.game.rng.sample(
    sorted({d["spellcraft"] for d in c.game.db.values() if d.get("spellcraft") and d["category"] == "pool"}), 3)])


def _gem_confiscation(c):
    t = c.target
    if t is None:
        return
    for _ in range(3):
        buff(t, c.p.counters["blood_gem_atk"], c.p.counters["blood_gem_hp"])


card("BG28_698", **_t(_gem_confiscation), status="approx", note="偷取邻居宝石未实现")              # 查抄宝石
card("BG28_825", **_t(lambda c: (spell_stats(c, c.target, 7, 7), give(c.target, "taunt"))))     # 防御者的仪式


def _natural_blessing(c):
    t = c.target
    if t is None:
        return
    races = set(t.races) - {"ALL"}
    for m in list(c.p.board):
        if m is t or races & set(m.races) or "ALL" in m.races:
            spell_stats(c, m, 2, 1)


card("BG28_845", **_t(_natural_blessing))                                                       # 自然祝福


def _tea_set(c):
    done = set()
    for m in list(c.p.board):
        rs = [r for r in m.races if r != "ALL" and r not in done]
        if rs:
            done.add(rs[0])
            spell_stats(c, m, 4, 4)


card("BG28_888", cast=_tea_set)                                                                 # 乱放的茶具
card("BG31_819", cast=lambda c: (c.game.add_to_hand(c.p, "BG31_816"), c.game.add_to_hand(c.p, "BG31_818")))
card("BG31_890", choose=[lambda c: discover(c, tier=c.p.tier, exact_tier=True),                # 无限潜力
                         lambda c: discover_spell(c, tier=c.p.tier)])
card("BG32_815", **_t(lambda c: [spell_stats(c, c.target, 1, 1) for _ in
                                 range(4 if c.target and c.target.is_race("NAGA") else 2)]))     # 潮汐转换
card("BG34_444", cast=lambda c: refresh_buff(c.p, "random", 9, 9))                             # 东风
card("BG34_689", cast=lambda c: refresh_buff(c.p, "all", 2, 2), status="approx", note="按每颗宝石 +1/+1 估计")
card("BG34_888", cast=lambda c: discover(c, race="UNDEAD"), status="approx", note="本回合打出即死亡未实现")
card("BG35_912", **_t(lambda c: [tavern_buff(c, 3, 3, r) for r in (c.target.races if c.target else [])
                                 if r != "ALL"][:1]))                                             # 艾欧娜尔的恩惠


def _dragonbreath(c):
    for m in list(c.p.board):
        n = 1 + (1 if m.is_race("DRAGON") else 0) + (1 if "divineShield" in m.kw else 0)
        for _ in range(n):
            spell_stats(c, m, 3, 2)


card("BG36_246", cast=_dragonbreath)                                                            # 强力龙息
card("BG36_301t", cast=lambda c: [spell_stats(c, m, 1, 1) for m in list(c.p.board)],             # 腐泥侵蚀
     passive={"on_discard_self": lambda c: [buff(m, 1, 1) for _ in range(2) for m in list(c.p.board)]})
card("BG36_880", spell_target="friendly:DEMON",
     cast=lambda c: [consume_tavern(c, c.target, keywords=True) for _ in range(2)])            # 理性癫狂
card("BG36_884", cast=lambda c: [c.game.add_to_hand(c.p, "EBG_Spell_014") for _ in range(3)])   # 武器锻造

# ---------------- 5 本 ----------------
card("BG28_500", cast=lambda c: setattr(c.p, "armor", 5))                                       # 护甲储备
card("BG28_573", cast=lambda c: None, status="approx", note="战斗开始效果，未传入战斗模拟")
card("BG28_604", spell_target="friendly:UNDEAD",
     cast=lambda c: (_destroy(c, c.target), undead_attack(c, 8)))                               # 屠宰
card("BG28_607", spell_target="friendly:DEMON",
     cast=lambda c: [consume_tavern(c, c.target) for _ in range(3)])                            # 腐化纸杯蛋糕
card("BG28_830", cast=lambda c: make_golden(c, c.game.rng.choice(c.p.shop)) if c.p.shop else None)  # 点金术
card("BG28_849", cast=lambda c: None, status="approx", note="用法术刷新酒馆未实现")
card("BG28_882", cast=lambda c: discover(c, mechanic="deathrattle"), status="approx")
card("BG28_GIL_836", cast=lambda c: discover(c, mechanic="battlecry"), status="approx")
card("BG31_886", spell_target=F, choose=[lambda c: [spell_stats(c, c.target, 6, 6) for _ in range(2)],
                                          lambda c: [spell_stats(c, m, 2, 3) for m in list(c.p.board)]],
     status="approx", note="第二个选项数值按 +2/+3 估计")
card("BG34_889", cast=lambda c: None, status="approx", note="战斗开始效果，未传入战斗模拟")
card("BG34_990", cast=lambda c: [spell_stats(c, m, 6 if m.golden else 3, 4 if m.golden else 2) for m in list(c.p.board)])
card("BG35_922", cast=lambda c: [spell_stats(c, m, 4 if m.is_race("NAGA") else 2, 4 if m.is_race("NAGA") else 2)
                                 for m in list(c.p.board)])
card("BG36_303", cast=lambda c: gain_gold(c, 2),
     passive={"on_discard_self": lambda c: _strike_oil(c, 2)})                                  # 腐化硬币
card("BG36_371", cast=lambda c: deity(c, 7, 7),
     passive={"on_discard_self": lambda c: deity(c, 14, 14)}, status="approx", note="神祇只记数")


def _channel(c):
    t = c.target
    if t is None or t not in c.p.board:
        return
    c.game.step(c.p, ("sell", t.uid))
    others = c.p.board
    if others:
        buff(c.game.rng.choice(others), t.atk, t.hp)


card("EBG_Spell_032", **_t(_channel))                                                           # 引导吞噬者
card("EBG_Spell_037", cast=lambda c: None, status="approx", note="换英雄技能未实现")


def _destroy(c, m):
    if m in c.p.board:
        c.p.board.remove(m)
        c.p.counters["friendly_deaths"] += 1


# ---------------- 6 / 7 本 ----------------
card("BG28_169", cast=lambda c: [spell_stats(c, m, 2, 2) for _ in range(2) for m in list(c.p.board)])
card("BG28_838", **_t(lambda c: (setattr(c.target, "atk", 20), setattr(c.target, "hp", 20)) if c.target else None))
card("BG31_892", cast=lambda c: discover(c, mechanic="chooseOne"), status="approx", note="效果合并未实现")
card("EBG_Spell_017", spell_target=F, cast=lambda c: make_golden(c, c.target) if c.target and c.target.tier <= 4 else None)
card("EBG_Spell_038", cast=lambda c: None, status="approx", note="按种族刷新酒馆未实现")
card("BG28_507", **_t(lambda c: give(c.target, "divineShield")))
card("BG31_889", cast=lambda c: None, status="approx", note="战斗开始效果")
card("BG31_896", cast=lambda c: discover(c, tier=7, exact_tier=True))
card("BG34_272", cast=lambda c: [spell_stats(c, m, 3, 3) for _ in range(1 + len({r for x in c.p.board for r in x.races
                                                                              if r != "ALL"})) for m in list(c.p.board)])

# ---------------- 衍生法术 ----------------
card("EBG_Spell_014", **_t(lambda c: spell_stats(c, c.target, 4, 0)))                           # 尖利箭矢
card("BG_EX1_014t", **_t(lambda c: spell_stats(c, c.target, 1, 1)))                             # 香蕉
card("BG_TB_006", **_t(lambda c: spell_stats(c, c.target, 2, 2)))                               # 大香蕉
card("BG35_910", **_t(lambda c: spell_stats(c, c.target, 4 + 2 * c.p.counters.get("elementals_turn", 0),
                                            4 + 2 * c.p.counters.get("elementals_turn", 0))), status="approx")
card("BG24_HERO_100_Buddyt", cast=lambda c: gain_gold(c, 8))


def _arcane_absorption(c):
    t = c.target
    if t is None or not c.p.shop:
        return
    v = max(c.p.shop, key=lambda x: x.hp)
    spell_stats(c, t, v.atk // 2, v.hp // 2)


card("BG35_911", spell_target="friendly:ELEMENTAL", cast=_arcane_absorption)                      # 奥术吸收


def _lockbox_tick(c):
    box = c.other
    left = box.data.get("left", 5) - 1
    box.data["left"] = left
    if left <= 0 and box in c.p.hand:
        c.p.hand.remove(box)
        races = [r for r in c.game.races]
        get_random_minion(c, race=c.game.rng.choice(races) if races else None, golden=True)


card("BG36_520t", in_hand_start_of_turn=_lockbox_tick)                                           # 上锁宝箱（不能打出）

# 法术迸发
card("BG23_000t", **_t(lambda c: buff(c.target, c.num(0, 2), 0, temp=True)))
card("BG23_004t", **_t(lambda c: (buff(c.target, 2, 6, temp=True), _temp_kw(c.target, "taunt"))))
card("BG23_007t", **_t(lambda c: (buff(c.target, 2, 2), _temp_kw(c.target, "windfury")
                                  if c.target and c.target.is_race("NAGA") else None)))
card("BG23_008t", **_t(lambda c: _temp_kw(c.target, "divineShield")))
card("BG27_514t", spell_target="tavern", cast=lambda c: c.game.add_to_hand(c.p, c.target.card_id) if c.target else None)
card("BG31_920t", cast=lambda c: get_random_minion(c, race="NAGA", tier=c.p.tier))
card("BG31_924t", **_t(lambda c: buff(c.target, 1 + c.p.counters["spells_cast"] // 3,
                                      1 + c.p.counters["spells_cast"] // 3, temp=True)))
card("BG32_835t", cast=lambda c: tavern_spell_bonus(c.p, 1, 1))
card("BG33_319t", cast=lambda c: get_random_spell(c, stats_only=True))


def _temp_kw(m, kw):
    if m is not None and kw not in m.kw:
        m.kw.add(kw)
        m.data.setdefault("temp_kw", []).append(kw)


Impl  # 保持导入
