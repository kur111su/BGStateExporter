"""随从池里的随从（招募阶段的效果）。

只有战斗中生效的效果（亡语、战斗开始、集结、复仇、攻击时……）交给 Bob's Buddy，这里标 combat_only。
战斗中产生、会留到战斗之后的效果（集结/亡语“获取一张牌”、“永久获得”等），Bob's Buddy 不返回，
这里用 after_combat 近似：集结按每场触发一次，亡语按一半概率触发，标 status="approx"。
k = 金色倍数（金色卡文本里的数字通常是普通版的 2 倍）。
"""
from .core import (adjacent, blood_gems, buff, buff_all, card, combat_only, consume_tavern, deity, discard,
                   discover, discover_spell, free_refreshes, friends, gain_gold, gem_upgrade, get_random_minion,
                   get_random_spell, give, gold_next_turn, hand_minions, leftmost, make_golden, random_keyword,
                   rightmost, spell_stats, summon, tavern_buff, tavern_spell_bonus, undead_attack, vanilla,
                   volumizer_bonus, beetle_bonus, bonus_keywords, refresh_buff)


def k(c) -> int:
    return 2 if c.g else 1


def gems(c, n):
    for _ in range(n):
        c.game.add_to_hand(c.p, "BG20_GEM")


def rally(fn, note="集结在战斗中触发，按每场一次近似"):
    return dict(after_combat=fn, status="approx", note=note)


def deathrattle(fn, note="亡语在战斗中触发，按一半概率近似"):
    def after(c):
        if c.game.rng.random() < 0.5:
            fn(c)
    return dict(after_combat=after, status="approx", note=note)


def is_self(c):
    return c.other is c.m


VOLS = ("BG34_170t", "BG34_170t2", "BG34_170t3")

# =============================== 1 本 ===============================
card("BG20_100", battlecry=lambda c: gems(c, 2 * k(c)))                                          # 剃刀沼泽地卜师
card("BG23_000")                                                                                # 迷你鱼人（法术迸发）
vanilla("BG25_001", "BGS_119")
combat_only("BG25_013", "BG28_300", "BG29_611", "BG29_888", "BG32_330", "BG31_803", "BG36_200")
card("BG26_135", battlecry=lambda c: gold_next_turn(c.p, k(c)))                                 # 南海卖艺者
card("BG26_146", end_of_turn=lambda c: buff(c.m, 0, k(c)))                                       # 摇篮机器人
card("BG31_149", battlecry=lambda c: random_keyword(c, c.m, k(c)))                               # 泡泡枪手
card("BG31_330", battlecry=lambda c: c.p.data.__setitem__("next_spell_discount",
                                                           c.p.data.get("next_spell_discount", 0) + k(c)),
     status="approx", note="折扣在下次买法术时扣（引擎里尚未减价）")
card("BG31_815", battlecry=lambda c: [tavern_buff(c, 1, 1, "ELEMENTAL") for _ in range(k(c))])   # 沙丘土著
card("BG32_236", note="永远是金色、不给三连奖励")                                                    # 夺金健将
card("BG33_886", **rally(lambda c: [buff(c.m, c.p.counters["blood_gem_atk"], c.p.counters["blood_gem_hp"])
                                    for _ in range(k(c))]))                                      # 獠牙露营者


def _scarlet(c):
    if c.m.atk >= 6:
        give(c.m, "divineShield")


card("BG35_814", end_of_turn=_scarlet, after_spell=_scarlet, status="approx", note="在回合结束和施法后检查")
card("BG36_098", on_sell=lambda c: [c.game.add_to_hand(c.p, "BG36_098t") for _ in range(k(c))],
     status="approx", note="触手 cardId 未核实")
card("BG36_110", battlecry=lambda c: deity(c, 2 * k(c), k(c)), status="approx", note="神祇只记数")
card("BG36_345", activate=lambda c: buff(c.target, 3 * k(c), 3 * k(c)), activate_cost=2,
     activate_target="other_friendly")                                                          # 可疑的监狱守卫
card("BG36_360", status="approx", note="每局不同，按卡面")
card("BG36_921", spell_on_minion=lambda c: buff(c.m, 0, k(c)) if c.target is c.m else None)      # 逃亡者
card("BGS_004", after_play=lambda c: (c.game.damage(c.p, 1), [buff(c.m, 2, 2) for _ in range(k(c))])
     if not is_self(c) and c.other.is_race("DEMON") else None)                                   # 愤怒编织者

# =============================== 2 本 ===============================
card("BG20_101", **rally(lambda c: gems(c, k(c))))                                               # 剃刀沼泽公猪
combat_only("BG21_015", "BG25_008", "BG25_022", "BG26_805", "BG29_300", "BG33_430", "BG34_140", "BGS_034",
            "BG36_116")
card("BG22_202", on_sell=lambda c: get_random_minion(c, n=k(c), race="MURLOC"))                  # 小蝌蚪
card("BG23_002", battlecry=lambda c: [c.game.add_to_hand(c.p, "BG28_810") for _ in range(k(c))])  # 收集贝壳者
card("BG23_009", status="approx", note="法术迸发永久生效未实现")


def _mind_muck(c):
    t = c.target
    if t is not None:
        consume_tavern(c, t, mult=k(c))


card("BG23_357", battlecry=_mind_muck, battlecry_target="friendly:DEMON")                        # 心灵淤泥


def _patient_scout(c):
    tier = min(6, 1 + c.m.data.get("turns_held", 0))
    for _ in range(k(c)):
        discover(c, tier=tier, exact_tier=True)


card("BG24_715", on_sell=_patient_scout, start_of_turn=lambda c: c.m.data.__setitem__(
    "turns_held", c.m.data.get("turns_held", 0) + 1))                                            # 耐心的侦察兵
card("BG25_011", battlecry=lambda c: undead_attack(c, k(c)))                                     # 蛛魔群
card("BG26_174", status="approx", note="英雄受伤回溯：战斗相关，未实现")
card("BG26_963", battlecry=lambda c: buff_all(friends(c.p, "DRAGON", c.m), k(c), k(c)),
     status="approx", note="战斗开始部分交给战斗模拟")                                              # 电子合成器
card("BG31_177", after_play=lambda c: buff(c.other, 3 * k(c), k(c)) if not is_self(c) and c.other.is_race("MECHANICAL") else None,
     on_magnetize=lambda c: buff(c.target, 3 * k(c), k(c)))                                      # 机械侏儒翻译员
card("BG31_320", choose=[lambda c: gems(c, 2 * k(c)), lambda c: [c.game.add_to_hand(c.p, "BG31_893") for _ in range(k(c))]])
card("BG31_801", battlecry=lambda c: beetle_bonus(c, 2 * k(c), 3 * k(c)), status="approx",
     note="甲虫加成只记数，召唤在战斗中")


def _baller(attr):
    def on_sell(c):
        lvl = c.p.counters.get("baller_" + attr, 0) + 1
        a = lvl * k(c) if attr in ("atk", "both") else 0
        h = lvl * k(c) if attr in ("hp", "both") else 0
        if attr == "both":
            a = h = 2 * lvl * k(c)
        buff_all(c.p.board, a, h)
        c.p.counters["baller_" + attr] = lvl
    return on_sell


card("BG31_816", on_sell=_baller("atk"), status="approx", note="“提升以后的投球手”按每次 +1 估计")  # 火焰投球手
card("BG31_818", on_sell=_baller("hp"), status="approx", note="同上")                              # 冰雪投球手
card("BG31_924")                                                                                # 奇术师（法术迸发）


def _sylvar(c):
    times = 1 + sum(1 for m in c.p.board if m.golden)
    for _ in range(times):
        for m in adjacent(c.p, c.m):
            buff(m, k(c), 0)


card("BG32_235", end_of_turn=_sylvar)                                                            # 冲浪的希尔梵
card("BG32_237", choose=[lambda c: tavern_spell_bonus(c.p, k(c), 0), lambda c: tavern_spell_bonus(c.p, 0, k(c))])


def _vol(atk, hp):
    def f(c):
        if not c.m.data.get("vol_done"):
            c.m.data["vol_done"] = True
            volumizer_bonus(c, atk * k(c), hp * k(c))
    return f


card("BG34_170t", passive={"on_play_or_magnetize": _vol(3, 0)})                                 # 红色扩音机
card("BG34_170t2", passive={"on_play_or_magnetize": _vol(0, 3)})                                # 蓝色扩音机
card("BG34_170t3", passive={"on_play_or_magnetize": _vol(1, 2)})                                # 绿色扩音机
card("BG35_150", battlecry=lambda c: c.p.data.__setitem__("fodder", c.p.data.get("fodder", 0) + 3),
     status="approx", note="饲料未实现（只记数）")
card("BG36_099", activate=lambda c: (discard(c), deity(c, 2 * k(c), 2 * k(c))), activate_cost=2, status="approx")
card("BG36_100", on_sell=lambda c: get_random_spell(c, n=2 * k(c)), status="approx", note="打出一张弃另一张未实现")
card("BG36_201", activate=lambda c: None, activate_cost=2, status="approx", note="诱饵未实现")
card("BG36_342", activate=lambda c: [discover_spell(c) for _ in range(k(c))], activate_cost=2)    # 机智的船难海盗


def _decoy(c):
    for _ in range(k(c)):
        if c.p.shop and len(c.p.hand) < c.game.rules.hand_size:
            m = max(c.p.shop, key=lambda x: x.atk)
            c.p.shop.remove(m)
            c.game.gain_minion(c.p, m)
            c.p.hand.append(m)
            c.game.check_triple(c.p, m)


card("BG36_354", activate=_decoy, activate_cost=2)                                               # 诱饵召唤师


def _lockbox(c, n):
    box = next((x for x in c.p.hand if x.card_id == "BG36_520t"), None)
    if box is not None:
        box.data["left"] = box.data.get("left", 5) - n
    else:
        b = c.game.add_to_hand(c.p, "BG36_520t")
        if b is not None:
            b.data["left"] = 5


card("BG36_520", battlecry=lambda c: _lockbox(c, k(c)))                                          # 污水越狱者
card("BGS_115", on_sell=lambda c: [c.game.add_to_hand(c.p, "BGS_115t") for _ in range(k(c))])    # 商贩元素

# =============================== 3 本 ===============================
card("BG23_004")                                                                                # 深海垂钓者（法术迸发）
card("BG23_007")                                                                                # 驭浪者（法术迸发）
combat_only("BG24_500", "BG25_010", "BG27_556", "BG28_309", "BG29_816", "BG30_125", "BG33_155", "BG34_405",
            "BGS_126", "BGS_131", "BG_BOT_911", "BG_DEEP_015", "BG36_202", "BG36_207")
card("BG26_147", start_of_turn=lambda c: gain_gold(c, k(c)))                                     # 协调者
card("BG26_524", status="approx", note="刷新改为扣血未实现")
card("BG27_000", battlecry=lambda c: [c.game.add_to_hand(c.p, "BG28_503") for _ in range(2 * k(c))])  # 钢铁场地管理员
card("BG27_005", after_spell=lambda c: [buff_all(c.p.board, 1, 0) for _ in range(k(c))]
     if c.spell is not None and c.spell.card_id not in ("BG20_GEM",) else None)                   # 时空船长钩尾
card("BG27_084", choose=[lambda c: (buff(c.target, k(c), k(c)), give(c.target, "reborn")),
                         lambda c: (buff(c.target, 2 * k(c), 0), give(c.target, "windfury"))],
     battlecry_target="friendly:BEAST", status="approx", note="选项二数值按 +2 估计")


def _graverobber(c):
    t = c.target
    if t is not None and t in c.p.board:
        c.p.board.remove(t)
        c.p.counters["friendly_deaths"] += 1
        for _ in range(k(c)):
            c.game.add_to_hand(c.p, c.game.base_id(t.card_id))


card("BG28_303", battlecry=_graverobber, battlecry_target="friendly:UNDEAD")                     # 伪装的盗墓者
card("BG28_582", activate=lambda c: (discard(c), gems(c, 3 * k(c))), activate_cost=0)             # 残躯强盗
card("BG30_123", choose=[lambda c: gem_upgrade(c.p, k(c), k(c)), lambda c: gems(c, 2 * k(c))])     # 无畏的美食家
card("BG31_326", end_of_turn=lambda c: [c.game.add_to_hand(c.p, "BG31_893") for _ in range(k(c))])  # 宝石鼠
card("BG31_327", start_of_turn=lambda c: c.p.counters.__setitem__("choose_both", k(c)))           # 荆棘开拓者
card("BG32_860", on_sell=lambda c: tavern_spell_bonus(c.p, k(c), k(c)))                          # 浅鳍秘法师
card("BG33_830", battlecry=lambda c: tavern_spell_bonus(c.p, k(c), 0))                           # 艾萨拉的刀客
card("BG33_924", **rally(lambda c: tavern_spell_bonus(c.p, 0, k(c))))                            # 蓝色幼龙
card("BG34_170", battlecry=lambda c: [c.game.add_to_hand(c.p, c.game.rng.choice(VOLS)) for _ in range(k(c))])
card("BG34_684", end_of_turn=lambda c: [c.game.add_to_hand(c.p, "BG28_698") for _ in range(k(c))])  # 堑壕战士
card("BG34_856", **deathrattle(lambda c: [refresh_buff(c.p, "random", 4, 4) for _ in range(k(c))]))
card("BG36_101", **deathrattle(lambda c: get_random_spell(c, n=k(c), cost=1)))
card("BG36_108", after_spell=lambda c: (buff(c.m, k(c), 2 * k(c)), deity(c, k(c), 2 * k(c)))
     if c.spell is not None and c.spell.card_id != "BG20_GEM" else None, status="approx")
card("BG36_112", battlecry=lambda c: [c.game.add_to_hand(c.p, "BG36_301t") for _ in range(k(c))])  # 腐蚀者
card("BG36_113", **deathrattle(lambda c: deity(c, 2 * k(c), k(c))))
card("BG36_240", activate=lambda c: [c.game.add_to_hand(c.p, c.game.rng.choice(
    ["BG34_634t", "BG34_635t", "BG34_636t", "BG34_637t"])) for _ in range(k(c))], activate_cost=2)  # 雇佣坐骑
card("BG36_311", activate=lambda c: (discard(c), get_random_spell(c, n=k(c))), activate_cost=0)
card("BG36_330", choose=[lambda c: free_refreshes(c.p, 2 * k(c)), lambda c: gems(c, 3 * k(c))])   # 狡猾的潜入者
card("BG36_346", activate=lambda c: [c.game.add_to_hand(c.p, "BG28_897") for _ in range(2 * k(c))], activate_cost=1)
card("BG36_369", on_sell=lambda c: discover(c, tier=7, exact_tier=True) if c.m.golden else None)  # 贪婪的诈骗犯
card("BG36_521", **deathrattle(lambda c: _lockbox(c, k(c))))                                     # 铐住的哗变者
card("BG36_730", **deathrattle(lambda c: c.p.data.__setitem__("fodder", c.p.data.get("fodder", 0) + 3)))
card("BG36_763", status="approx", note="造成 35 伤害后获得点金术：未实现")
card("BG36_854", **deathrattle(lambda c: [c.game.add_to_hand(c.p, "BG36_624") for _ in range(k(c))]))

# =============================== 4 本 ===============================
card("BG20_104", status="combat")
combat_only("BG25_016", "BG26_802", "BG26_817", "BG29_813", "BG36_245", "BG36_620", "BG34_604")
card("BG24_018", passive={"sell_bonus_if_lost": 4}, status="approx", note="卖价在 sell 时检查")
card("BG26_137", status="approx", note="在手牌里时随打出鱼人成长：未实现")
card("BG26_505", status="approx", note="法术迸发复制未实现")


def _percussionist(c):
    for _ in range(k(c)):
        discover(c, race="DEMON")


card("BG26_525", battlecry=_percussionist, status="approx", note="按发现到的星级扣血未实现")


def _courier(c):
    left = c.m.data.get("left", 5) - c.amount
    while left <= 0:
        for _ in range(k(c)):
            buff_all(friends(c.p, "PIRATE"), 2, 3)
        left += 5
    c.m.data["left"] = left


card("BG26_810", on_gold_spent=_courier, status="approx", note="数值按 +2/+3 估计")                 # 火药信使


def _balladist(c):
    t = c.target
    if t is not None:
        buff(t, 0, (2 + c.p.counters["gold_spent_turn"]) * k(c))


card("BG26_814", battlecry=_balladist, battlecry_target="friendly:PIRATE")                        # 苦情民谣歌手
card("BG28_583", on_gem=lambda c: None, status="approx", note="宝石连锁未实现")
card("BG30_121", passive={"gem_extra": 1})                                                       # 热气球勘测员
card("BG31_812", after_play=lambda c: give(c.other, "divineShield") if not is_self(c) and c.other.is_race("ELEMENTAL") else None,
     status="approx", note="“直到下回合”按永久处理")
card("BG32_336", battlecry=lambda c: [c.game.add_to_hand(c.p, "BG32_337") for _ in range(k(c))],
     **{k_: v for k_, v in deathrattle(lambda c: [c.game.add_to_hand(c.p, "BG32_337") for _ in range(k(c))]).items()})


def _maw_caster(c):
    t = c.target
    if t is not None and t in c.p.board:
        c.p.board.remove(t)
        for _ in range(k(c)):
            discover(c, race="UNDEAD")


card("BG32_340", battlecry=_maw_caster, battlecry_target="friendly:UNDEAD")                      # 巨口施法者
card("BG32_341", passive={}, after_play=lambda c: tavern_spell_bonus(c.p, k(c), 2 * k(c)) if is_self(c) else None,
     on_sell=lambda c: tavern_spell_bonus(c.p, -k(c), -2 * k(c)), status="approx", note="在场时加成")
card("BG32_873", status="approx", note="英雄受伤回溯未实现")
card("BG32_880", **deathrattle(lambda c: tavern_spell_bonus(c.p, k(c), 0)))
card("BG33_319")                                                                                # 霜鳞女祭司（法术迸发）
card("BG33_822", **rally(lambda c: [c.game.add_to_hand(c.p, c.game.rng.choice(
    ["BG33_811", "BG33_812", "BG33_813", "BG33_814", "BG33_815"])) for _ in range(k(c))]))       # 大亨强盗
card("BG34_171", **deathrattle(lambda c: [c.game.add_to_hand(c.p, c.game.rng.choice(VOLS)) for _ in range(k(c))]))
card("BG34_500", end_of_turn=lambda c: consume_tavern(c, c.m, "highest_hp", k(c)))               # 烈焰执行者
card("BG34_682", battlecry=lambda c: [c.game.add_to_hand(c.p, "BG34_689") for _ in range(k(c))])
card("BG34_690", **deathrattle(lambda c: undead_attack(c, 2 * k(c))))
card("BG34_865", battlecry=lambda c: [refresh_buff(c.p, "random", 10, 10) for _ in range(k(c))])  # 灯神
card("BG34_925", **rally(lambda c: None, note="主厨甄选在战斗中，未实现"))
card("BG35_341", after_play=lambda c: tavern_spell_bonus(c.p, k(c), 2 * k(c)) if is_self(c) else None,
     passive={"on_play_or_magnetize": lambda c: None}, status="approx", note="在场时加成")
card("BG35_881", battlecry=lambda c: [c.game.add_to_hand(c.p, "BG35_911") for _ in range(k(c))])
card("BG35_921", status="approx", note="按施法数成长：由引擎记数，未直接加到属性")
card("BG36_106", on_discard=lambda c: (buff(c.m, 4 * k(c), 4 * k(c)), deity(c, 4 * k(c), 4 * k(c))))
card("BG36_114", end_of_turn=lambda c: buff(leftmost(c.p), (2 + c.p.counters["discards"]) * k(c),
                                            (2 + c.p.counters["discards"]) * k(c)), status="approx",
     note="每弃一张的提升量按 +1 估计")
card("BG36_115", end_of_turn=lambda c: [c.game.add_to_hand(c.p, "BG36_301t") for _ in range(k(c))])  # 梦魇腐蚀者
def _prison_buy(c):
    mult = c.m.data.pop("absorb_next", 0)
    if mult and hasattr(c.other, "atk"):
        buff(c.m, c.other.atk * mult, c.other.hp * mult)


card("BG36_180", activate=lambda c: c.m.data.__setitem__("absorb_next", k(c)), activate_cost=1, after_buy=_prison_buy)
card("BG36_181", on_sell=_baller("both"), status="approx", note="同投球手")
card("BG36_204", **rally(lambda c: get_random_minion(c, n=k(c), race="BEAST")))
card("BG36_206", status="approx", note="诱饵未实现")
card("BG36_210", status="combat")
card("BG36_211", status="combat")
card("BG36_242", **rally(lambda c: [c.game.add_to_hand(c.p, c.game.rng.choice(
    ["BG34_634t", "BG34_635t", "BG34_636t", "BG34_637t"])) for _ in range(k(c))]))
card("BG36_243", status="approx", note="触发集结未实现")
card("BG36_308", on_sell=lambda c: get_random_minion(c, n=2 * k(c), race="ABERRATION"),
     status="approx", note="打出一张弃另一张未实现")
card("BG36_312", activate=lambda c: (discard(c), get_random_minion(c, n=k(c), race="ABERRATION")), activate_cost=0)
card("BG36_331", **rally(lambda c: get_random_minion(c, n=k(c), mechanic="chooseOne")))
card("BG36_332", choose=[lambda c: get_random_minion(c, n=k(c), race="QUILBOAR"),
                         lambda c: (c.p.counters.__setitem__("max_gold_bonus", c.p.counters.get("max_gold_bonus", 0) + k(c)),
                                    setattr(c.p, "max_gold", c.p.max_gold + k(c)))])
card("BG36_362", **deathrattle(lambda c: tavern_buff(c, 2 * k(c), 2 * k(c))),
     activate=lambda c: None, activate_cost=4)
card("BG36_372", status="combat")


def _jailer(c):
    for m in friends(c.p, "DEMON"):
        consume_tavern(c, m, mult=k(c))


card("BG36_503", activate=_jailer, activate_cost=2)                                              # 魂魄守卫
card("BG36_506", activate=lambda c: None, activate_cost=1, status="approx", note="磁力多吸附一次未实现")


def _cagey(c):
    for _ in range(2 * k(c)):
        sid = c.game.rng.choice(c.game.spell_pool)
        s = c.game.new_spell(sid)
        impl = c.game.impl(sid)
        tgt = c.m if impl and impl.spell_target else None
        c.game.cast_spell(c.p, s, tgt)


card("BG36_508", activate=_cagey, activate_cost=1, status="approx")
card("BG36_511", activate=lambda c: None, activate_cost=1, status="approx", note="未实现")
card("BG36_524", status="approx", note="按打出金色随从数成长，未实现")
card("BG36_701", activate=lambda c: _trigger_battlecry(c), activate_cost=1, activate_target="other_friendly")


def _trigger_battlecry(c):
    t = c.target
    if t is None:
        return
    impl = c.game.impl(t.card_id)
    if impl and impl.battlecry:
        for _ in range(k(c)):
            impl.battlecry(c.game.ctx(c.p, t, t.card_id))


card("BG36_703", spell_on_minion=lambda c: buff(next(iter(hand_minions(c.p)), None), 8 * k(c), 8 * k(c))
     if c.target is c.m else None)
card("BG36_731", **deathrattle(lambda c: [c.game.add_to_hand(c.p, "BG36_880") for _ in range(k(c))]))
card("BG36_764", end_of_turn=lambda c: get_random_spell(c, n=2 * k(c), cost=1))                 # 齿轮鱼人
card("BG36_853", status="approx", note="卫星吸附未实现")
card("BGS_116", battlecry=lambda c: free_refreshes(c.p, 2 * k(c)))                              # 刷新畸体
card("BGS_123", battlecry=lambda c: get_random_minion(c, n=k(c), race="ELEMENTAL"))             # 酒馆旋风

# =============================== 5 本 ===============================
card("BG21_004", after_play=lambda c: consume_tavern(c, c.m, mult=k(c))
     if not is_self(c) and c.other.is_race("DEMON") else None)                                   # 贪食的乌祖尔
card("BG23_008")                                                                                # 光鳞（法术迸发）
combat_only("BG23_318", "BG25_009", "BG26_523", "BG34_142", "BG33_318", "BG33_828", "BG34_403", "BG35_604",
            "BG36_102", "BG36_364", "BG36_514", "BGS_008", "BGS_018", "BG31_809", "BG31_925", "BG33_885",
            "BG33_883", "BG32_324", "BG36_848")
card("BG25_354", passive={"deathrattle_extra": 1}, status="combat")
card("BG26_ICC_901", passive={"end_of_turn_extra": 1})                                          # 达卡莱附魔师
card("BG28_550", battlecry=lambda c: discover_spell(c))                                          # 竞技表演者


def _felboar(c):
    if c.spell is None:
        return
    left = c.m.sd[0] - 1 if c.m.sd[0] else 2
    if left <= 0:
        consume_tavern(c, c.m, "highest_hp", k(c))
        left = c.game.card(c.m.card_id).get("scriptData", [3])[0] or 3
    c.m.sd[0] = left


card("BG28_633", after_spell=_felboar, status="approx", note="吞食哪个随从按最高生命近似")          # 邪能野猪人
card("BG28_707", after_spell=lambda c: [tavern_buff(c, 4, 3, "ELEMENTAL") for _ in range(k(c))]
     if c.spell is not None and c.spell.card_id != "BG20_GEM" else None)                        # 活化艾泽里特
card("BG28_741", after_spell=lambda c: buff_all([m for m in c.p.board if "divineShield" in m.kw], 4 * k(c), 0)
     if c.spell is not None and c.spell.card_id != "BG20_GEM" else None)                        # 冲锋的女沙皇
card("BG31_148", battlecry=lambda c: buff_all(friends(c.p, exclude=c.m), (3 + bonus_keywords(c.p)) * k(c),
                                              (2 + bonus_keywords(c.p)) * k(c)), status="approx",
     note="每种关键词的提升量按 +1/+1 估计")
card("BG31_920")                                                                                # 暗冠战略家（法术迸发）
card("BG32_111", battlecry=lambda c: [c.game.add_to_hand(c.p, "BG28_888") for _ in range(k(c))],
     **deathrattle(lambda c: [c.game.add_to_hand(c.p, "BG28_888") for _ in range(k(c))]))
card("BG32_231", battlecry=lambda c: [make_golden(c, m) for m in c.game.rng.sample(
    [x for x in friends(c.p, "PIRATE", c.m) if x.tier <= 4 and not x.golden],
    min(k(c), len([x for x in friends(c.p, "PIRATE", c.m) if x.tier <= 4 and not x.golden])))])  # 精英导航员
card("BG32_820", battlecry=lambda c: [c.game.add_to_hand(c.p, "BG28_168") for _ in range(k(c))],
     **deathrattle(lambda c: [c.game.add_to_hand(c.p, "BG28_168") for _ in range(k(c))]))
card("BG32_821", end_of_turn=lambda c: tavern_spell_bonus(c.p, k(c), k(c)))                      # 邪火咒术师
card("BG32_835")                                                                                # 宁静冥想者（法术迸发）
card("BG33_821", battlecry=lambda c: [c.game.add_to_hand(c.p, c.game.rng.choice(
    ["BG33_811", "BG33_812", "BG33_813", "BG33_814", "BG33_815"])) for _ in range(k(c))],
     **deathrattle(lambda c: [c.game.add_to_hand(c.p, c.game.rng.choice(
         ["BG33_811", "BG33_812", "BG33_813", "BG33_814", "BG33_815"])) for _ in range(k(c))]))  # 船难海贼
card("BG33_825", passive={"bounty_extra": 1})                                                   # 骄傲的私掠者
card("BG34_633", battlecry=lambda c: [c.game.add_to_hand(c.p, c.game.rng.choice(
    ["BG34_634t", "BG34_635t", "BG34_636t", "BG34_637t"])) for _ in range(k(c))])


def _air_revenant(c):
    left = c.m.data.get("left", 7) - c.amount
    while left <= 0:
        for _ in range(k(c)):
            refresh_buff(c.p, "random", 9, 9)
        left += 7
    c.m.data["left"] = left


card("BG34_858", on_gold_spent=_air_revenant, status="approx", note="门槛按 7 金估计")
card("BG35_123", end_of_turn=lambda c: [c.game.add_to_hand(c.p, c.p.data["last_spell"]) for _ in range(k(c))]
     if c.p.data.get("last_spell") else None)
card("BG35_602", after_summon=lambda c: buff(c.other, 3 * k(c), 0) if not is_self(c) and c.other.is_race("BEAST") else None,
     status="approx", note="提升未实现")
card("BG35_882", battlecry=lambda c: [c.game.add_to_hand(c.p, "BG35_910") for _ in range(k(c))])
card("BG36_097", on_discard=lambda c: buff_all(friends(c.p, exclude=c.m), 4 * k(c), 4 * k(c)))    # 心灵扭曲者
card("BG36_103", battlecry=lambda c: [c.game.add_to_hand(c.p, "BG36_371") for _ in range(k(c))],
     **deathrattle(lambda c: [c.game.add_to_hand(c.p, "BG36_371") for _ in range(k(c))]))
card("BG36_300", activate=lambda c: (discard(c), tavern_spell_bonus(c.p, k(c), k(c))), activate_cost=2)
card("BG36_318", **deathrattle(lambda c: deity(c, 2 * k(c), k(c))))


def _kthir(c):
    spells = [x for x in c.p.hand if not hasattr(x, "atk")][:2]
    for s in spells:
        discard(c, s)
        buff(c.m, 7, 7)


card("BG36_320", end_of_turn=_kthir)                                                             # 神秘的克熙尔


def _resourceful(c):
    for _ in range(k(c)):
        vid = c.game.rng.choice(VOLS)
        v = c.game.new_minion(vid)
        buff(c.m, v.atk, v.hp)
        c.m.kw |= v.kw
        c.m.attached.append(vid)
        c.game.add_to_hand(c.p, vid)


card("BG36_366", end_of_turn=_resourceful)                                                       # 足智多谋的机器人


def _escapee(c):
    left = c.m.data.get("left", 6) - c.amount
    while left <= 0:
        _lockbox(c, 1)
        left += 6
    c.m.data["left"] = left


card("BG36_523", on_gold_spent=_escapee, status="approx", note="门槛按 6 金估计")
card("BG36_621", activate=lambda c: [(buff(m, 8 * k(c), 8 * k(c)), give(m, c.game.rng.choice(["taunt", "divineShield", "windfury"])))
                                     for m in list(c.p.shop)], activate_cost=1, status="approx")
card("BG36_700", activate=lambda c: (buff(c.target, 7 * k(c), 7 * k(c)), random_keyword(c, c.target)),
     activate_cost=2, activate_target="friendly:MURLOC")                                         # 下水道逃犯
card("BG36_704", spell_on_minion=lambda c: buff_all(friends(c.p, "MURLOC") + hand_minions(c.p, "MURLOC"), 3 * k(c), 3 * k(c))
     if c.target is not None and c.target.is_race("MURLOC") else None)
card("BG36_762", spell_on_minion=lambda c: tavern_buff(c, k(c), 2 * k(c)) if c.target is c.m else None)  # 恶魔干扰者
card("BG36_851", after_play=lambda c: buff(c.other, 2 * k(c), 2) if not is_self(c) and c.other.is_race("MECHANICAL") else None,
     status="approx", note="卫星提升未实现")
card("BGS_020", battlecry=lambda c: [discover(c, race="MURLOC") for _ in range(k(c))]
     if friends(c.p, "MURLOC", c.m) else None)                                                    # 原始鳍 观察者
card("BGS_041", after_battlecry=lambda c: buff_all(friends(c.p, "DRAGON"), 2 * k(c), 2 * k(c)))   # 卡雷苟斯
card("BG_LOE_077", passive={"battlecry_extra": 1})                                               # 布莱恩·铜须

# =============================== 6 本 ===============================


def _murk_eye(c):
    for m in adjacent(c.p, c.m):
        impl = c.game.impl(m.card_id)
        if impl and impl.battlecry:
            for _ in range(k(c)):
                impl.battlecry(c.game.ctx(c.p, m, m.card_id))


card("BG22_403", end_of_turn=_murk_eye)                                                          # 小鱼人眼
card("BG23_017", battlecry=lambda c: gem_upgrade(c.p, 2 * k(c), k(c)), **deathrattle(lambda c: gem_upgrade(c.p, 2 * k(c), k(c))))
card("BG26_152", end_of_turn=lambda c: [buff(m, 4 * k(c) * len(m.attached), 5 * k(c) * len(m.attached))
                                        for m in c.p.board if m.attached])                       # 效用无人机
card("BG26_175", status="approx", note="可与任意元素三连：未实现")
combat_only("BG26_354", "BG31_810", "BG31_835", "BG36_208", "BG36_209", "BG36_241", "BG36_849", "BGS_040",
            "BG36_104")
card("BG31_035", after_play=lambda c: buff(c.m, (1 + c.p.counters["spells_cast"] // 3) * k(c),
                                           (1 + c.p.counters["spells_cast"] // 3) * k(c))
     if not is_self(c) and c.other.is_race("NAGA") else None)                                    # 破土者
card("BG31_323", after_choose_one=lambda c: [blood_gems(c, m, 2 * k(c)) for m in friends(c.p, "QUILBOAR")])
card("BG32_837", end_of_turn=lambda c: None, status="approx", note="对相邻随从施放自然祝福：未实现")


def _rogers(c):
    left = c.m.data.get("left", 9) - c.amount
    while left <= 0:
        for _ in range(k(c)):
            c.game.add_to_hand(c.p, c.game.rng.choice(["BG33_811", "BG33_812", "BG33_813", "BG33_814", "BG33_815"]))
        left += 9
    c.m.data["left"] = left


card("BG33_823", on_gold_spent=_rogers)                                                          # 空军上将罗杰斯
card("BG33_891", status="approx", note="教法术的鱼人未实现")
card("BG34_692", after_spell=lambda c: undead_attack(c, 3 * k(c))
     if c.spell is not None and c.spell.card_id != "BG20_GEM" else None)                        # 被遗忘者纺织工
card("BG35_155", after_sell=lambda c: c.p.data.__setitem__("fodder", c.p.data.get("fodder", 0) + k(c)),
     status="approx", note="饲料只记数")
card("BG35_342", status="combat")
card("BG35_883", passive={"friendly_spell_extra": 1})                                           # 巴琳达
card("BG36_109", on_card_added=lambda c: deity(c, 4 * k(c), 5 * k(c)), status="approx")
card("BG36_341", choose=[lambda c: [blood_gems(c, m, 3 * k(c)) for m in list(c.p.board)],
                         lambda c: [refresh_buff(c.p, "all", 2, 2) for _ in range(3 * k(c))]])   # 老兵大盗
card("BG36_343", battlecry=lambda c: get_random_minion(c, n=k(c), tier=4, exact=True, golden=True))  # 安静的投递员
card("BG36_344", after_discover=lambda c: buff_all(friends(c.p, "PIRATE", c.m), k(c), k(c)),
     status="approx", note="按金色随从提升未实现")


def _tempest(c):
    if not is_self(c) and c.other.is_race("ELEMENTAL"):
        left = c.m.data.get("left", 4) - 1
        if left <= 0 and c.p.shop:
            v = max(c.p.shop, key=lambda x: x.hp)
            buff(c.m, v.atk * k(c), v.hp * k(c))
            left = 4
        c.m.data["left"] = left


card("BG36_352", after_play=_tempest)                                                            # 不羁风暴
card("BG36_356", activate=lambda c: (setattr(c.target, "atk", 50 * k(c)), setattr(c.target, "hp", 50 * k(c)))
     if c.target else None, activate_cost=2, activate_target="other_friendly")                    # 泰瑞尔


def _reveille(c):
    left = c.m.data.get("left", 3) - 1
    if left <= 0:
        vid = c.game.rng.choice(VOLS)
        v = c.game.new_minion(vid)
        buff(c.m, v.atk, v.hp)
        c.m.attached.append(vid)
        for _ in range(k(c)):
            c.game.add_to_hand(c.p, vid)
        left = 3
    c.m.data["left"] = left


card("BG36_367", after_buy=_reveille, status="approx", note="门槛按 3 张估计")                    # 自动起床号


def _geomant(c):
    for _ in range(6 * k(c)):
        if len(c.p.hand) < c.game.rules.hand_size:
            c.game.add_to_hand(c.p, "BG20_GEM")
        elif c.p.board:
            buff(c.p.board[0], c.p.counters["blood_gem_atk"], c.p.counters["blood_gem_hp"])


card("BG36_370", activate=_geomant, activate_cost=2)                                             # 凯旋的地卜师
card("BG36_515", status="combat")
card("BG36_622", spell_on_minion=lambda c: buff_all(c.p.board, 2 * k(c), 3 * k(c))
     if c.target is not None and c.target.is_race("NAGA") else None)                            # 激流摧毁者
card("BG36_640", spell_on_minion=lambda c: [c.game.impl("BG28_888").cast(c.game.ctx(c.p, None, "BG28_888"))
                                            for _ in range(k(c))] if c.target is c.m else None)  # 守门融合怪
card("BG36_733", status="approx", note="英雄受伤触发：未实现")
card("BGFYM_005", on_discard=lambda c: deity(c, 2 * k(c), k(c)), status="approx")

# =============================== 7 本 ===============================
card("BG25_034", battlecry=lambda c: [make_golden(c, m) for m in c.game.rng.sample(
    [x for x in friends(c.p, exclude=c.m) if x.tier <= 6 and not x.golden],
    min(k(c), len([x for x in friends(c.p, exclude=c.m) if x.tier <= 6 and not x.golden])))])     # 杉德尔船长
card("BG26_149", on_magnetize=lambda c: [(buff(c.m, c.other.atk, c.other.hp), c.m.attached.append(c.other.card_id))
                                         for _ in range(k(c))] if c.target is not c.m else None)  # 极性 B-Box
card("BG27_016", battlecry=lambda c: tavern_buff(c, 8 * k(c), 8 * k(c)),
     **deathrattle(lambda c: tavern_buff(c, 8 * k(c), 8 * k(c))))                                # 萨格拉斯的勇士
combat_only("BG27_017", "BG31_999", "BG34_320", "BG34_322", "BG36_333")
card("BG27_514")                                                                                # 海巫扎尔吉拉（法术迸发）
card("BG34_145", end_of_turn=lambda c: buff(next(iter(hand_minions(c.p)), None), c.m.atk * k(c), c.m.hp * k(c)))  # 未来鱼人


def _ra(c):
    get_random_minion(c, n=k(c), tier=6, exact=True)


card("BG34_319", battlecry=_ra, **rally(_ra))                                                    # 莱，至高守护者


def _slab(c):
    if c.m.data.get("slab_turn") == c.game.turn or not hasattr(c.other, "atk"):
        return
    c.m.data["slab_turn"] = c.game.turn
    m = c.other
    buff(m, 20, 20)
    mult = 3 if c.g else 2
    m.atk *= mult
    m.hp *= mult


card("BG34_950", after_buy=_slab)                                                                # 石器时代石板
card("BG36_111", after_spell=lambda c: (buff_all(c.p.board, 3 * k(c), 3 * k(c)), deity(c, 3 * k(c), 3 * k(c)))
     if c.spell is not None and c.spell.card_id != "BG20_GEM" else None)                        # 恐惧之煞
