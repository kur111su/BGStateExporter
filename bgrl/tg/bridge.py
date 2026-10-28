"""A4：把插件导出的局面（schema 2）重建成 TavernGym 里的局面。

做法（编码只走 TavernGym 自己的 observe，避免论文里“两套编码器漂移”的问题）：
  1. 新开一局，种族固定为插件读到的本局种族；选英雄阶段把你的英雄放进候选并选上（你 = 0 号座位）。
  2. 其它 7 个座位按排行榜设置英雄、血量、护甲、等级；已淘汰的设为 0 血。
  3. 0 号座位：设置回合、金币、等级、血量、升本费用、免费刷新、下回合金币；饰品、英雄技能；
     清空后按插件局面重建场面（含站位、金色、攻防、关键词）、手牌、酒馆。
  4. 玩家级“本局”计数（酒馆法术加成等）写进存档 JSON 的 stats 后重新读档。
  5. 返回 TavernGym 的 Game；`compare(game, state)` 列出重建后和插件局面不一致的地方。

v2 起还会：
  - 手牌、酒馆按插件里的位置排（买 / 打出的动作编号按位置算）。
  - 随从计数器（counter）：按卡牌把插件的 scriptData 换算成 TavernGym 的 counter（COUNTERS 表）。
  - 黑暗之赐 / 磁力附着：从插件的附魔（enchantments）认出来，直接写进存档的 attached（不触发附着效果，
    攻防、关键词已经是插件里的实际值）。
  - 对手阵容：局面带 _seen（logs.load_game 生成）时，把每个对手最近一次交手看到的阵容放到他的场上，
    同时记成你“看到过”的阵容（scouted）。

还没做的：卡池余量（按默认卡池）、其它没映射的玩家标签。
"""
from __future__ import annotations

import json
import re
from typing import Any

A_CHOOSE = 30

# 插件（HDT 标签名）→ TavernGym PlayerStats 字段
STAT_TAGS = {
    "TAVERN_SPELL_ATTACK_INCREASE": "spell_atk",
    "TAVERN_SPELL_HEALTH_INCREASE": "spell_hp",
    "BACON_ELEMENTAL_BUFFATKVALUE": "tavern_elemental_atk",
    "BACON_ELEMENTAL_BUFFHEALTHVALUE": "tavern_elemental_hp",
    "NUM_SPELLS_PLAYED_THIS_GAME": "spells_cast",
    "BACON_ELEMENTAL_PLAY_COUNTER": "elementals_played",
}
# 宝石：HDT 和 TavernGym 记的都是基础 +1/+1 之外的额外部分（A5 回放核对过）
GEM_TAGS = {"BACON_BLOODGEMBUFFATKVALUE": "gem_atk", "BACON_BLOODGEMBUFFHEALTHVALUE": "gem_hp"}

KW = {"taunt": "TAUNT", "divineShield": "DIVINE_SHIELD", "windfury": "WINDFURY", "reborn": "REBORN",
      "venomous": "VENOMOUS", "poisonous": "POISONOUS", "stealth": "STEALTH"}


# 插件 scriptData（卡面上显示的数字，TAG_SCRIPT_DATA_NUM_1..4）→ TavernGym 随从 counter
# 参数：sd = scriptData 列表，g = 1（普通）/ 2（金色），c = 插件里的这张卡
COUNTERS = {
    "BG28_633": lambda sd, g, c: max(0, min(2, 3 - sd[0])),     # 邪能野猪人：再施放 sd[0] 个法术 → 已施放
    "BG31_327": lambda sd, g, c: sd[0],                          # 刺棘开拓者：本回合剩余次数
    "BG33_891": lambda sd, g, c: sd[0],                          # 魔鳍真菌学家：本回合剩余次数
    "BGFYM_005": lambda sd, g, c: max(0, sd[0] // 2 - g),        # 预言者阿菲拉斯：提升次数 × 基础值
    "BG36_362": lambda sd, g, c: max(0, sd[0] // 2 - g),         # 牺牲的愤怒卫士：同上
    "BG35_814": lambda sd, g, c: 1 if (c.get("attack") or 0) >= 6 else 0,  # 血色幸存飞龙：是否已触发
    "BG36_763": lambda sd, g, c: max(0, 35 - sd[1]),             # 财宝鹦鹉：已造成的伤害
    "BG24_715": lambda sd, g, c: max(0, sd[0] - 1),              # 耐心的侦查员：已提升的等级
}

_GIFT = re.compile(r"^(BG36_MidGameEffect_000t\d+t?)e\d*$")
_catalog: dict[str, dict] | None = None


def catalog() -> dict[str, dict]:
    """TavernGym 卡牌表：卡牌编号 → {cid, kind, text, ...}。"""
    global _catalog
    if _catalog is None:
        import taverngym
        _catalog = {c["id"]: c for c in json.loads(taverngym.catalog_json())}
    return _catalog


def attachments(c: dict) -> list[str]:
    """插件里一张随从的附魔 → TavernGym 的附着卡（黑暗之赐、磁力随从）。"""
    cat = catalog()
    out: list[str] = []
    for e in c.get("enchantments") or []:
        eid = e.get("cardId") or ""
        m = _GIFT.match(eid)
        cand = m.group(1) if m else None
        if cand is None and eid.endswith("e"):
            base = eid[:-1].rstrip("_")
            info = cat.get(base)
            if info and info.get("kind") == "minion" and "Magnetic" in (info.get("text") or ""):
                cand = base
        if cand and cand in cat and cand not in out:
            out.append(cand)
    return out


def counter_of(c: dict) -> int | None:
    f = COUNTERS.get(base_id(c.get("cardId") or ""))
    sd = c.get("scriptData")
    if f is None or not sd:
        return None
    sd = list(sd) + [0] * (4 - len(sd))
    try:
        return int(f(sd, 2 if c.get("golden") or (c.get("cardId") or "").endswith("_G") else 1, c))
    except (TypeError, ValueError, IndexError):
        return None


# 饰品 / 英雄技能等非随从卡的进度：插件 scriptData → TavernGym 的 turn_data / levels / counters
# 值：函数(sd) → [(字段, 数值)]
CARD_STATE = {
    "BG36_MagicItem_371": lambda sd: [("turn_data", max(0, sd[0] - 1))],   # 蜂巢戒指：本回合已提升次数
    "BG32_MagicItem_801t": lambda sd: [("levels", max(0, sd[0] - 1))],    # 森林之心：提升次数
}

# “每 N 次……（还剩 x 个！）”的饰品 / 奖励：插件 scriptData[0] = 还剩几次（0 = 刚触发，下一轮从 N 开始），
# TavernGym 用 counters 记已经累计了几次（Engine::tick_counter）。N 取自 TavernGym 源码。
TICK_EVERY = {
    "BG36_MagicItem_307": 3, "BG36_MagicItem_370": 2, "BG30_MagicItem_891": 2, "BG36_MagicItem_418": 3,
    "BG30_MagicItem_541": 3, "BG35_MagicItem_305": 2, "BG35_MagicItem_861": 11, "BG36_MagicItem_407": 13,
    "BG36_MagicItem_409": 10, "BG36_MagicItem_410": 25, "BG28_Reward_508": 5, "BG30_MagicItem_425": 2,
    "BG30_MagicItem_435": 3, "BG30_MagicItem_707": 3, "BG30_MagicItem_951": 6, "BG32_MagicItem_300": 2,
    "BG35_MagicItem_700": 6, "BG35_MagicItem_817": 2, "BG36_MagicItem_850": 5,
}
for _k, _n in TICK_EVERY.items():
    CARD_STATE.setdefault(_k, (lambda n: lambda sd: [("counters", (n - sd[0]) % n if 0 <= sd[0] <= n else 0)])(_n))


def _card_state(me: dict, cards: list[dict], warnings: list[str]) -> None:
    cat = catalog()
    for c in cards:
        f = CARD_STATE.get(c.get("cardId") or "")
        sd = c.get("scriptData")
        if f is None or not sd:
            continue
        cid = cat[c["cardId"]]["cid"]
        for field_, v in f(list(sd) + [0] * 4):
            lst = [x for x in me.get(field_) or [] if x[0] != cid]
            if v:
                lst.append([cid, v])
            me[field_] = lst


def spell_extra_dynamic(board: list[dict], trinkets: list[dict]) -> tuple[int, int]:
    """TavernGym 里不算进 stats 的酒馆法术加成（Engine::spell_extra 的场面光环和饰品部分）。"""
    a = h = 0
    for c in board:
        cid = c.get("cardId") or ""
        g = 2 if c.get("golden") or cid.endswith("_G") else 1
        b = base_id(cid)
        if b == "BG34_Giant_066":                 # 时光扭曲的拉伯尔
            a, h = a + g, h + g
        elif b == "BG32_341":                     # 胡蒙格斯
            a, h = a + g, h + 2 * g
        elif b == "BG35_341":                     # 附魔哨兵
            a, h = a + g, h + g
        for e in c.get("enchantments") or []:     # 磁力吸附的附魔哨兵
            if (e.get("cardId") or "").startswith("BG35_341"):
                a, h = a + 1, h + 1
    for t in trinkets:
        sd = list(t.get("scriptData") or []) + [0, 0]
        if t.get("cardId") in ("BG36_MagicItem_371", "BG32_MagicItem_801t", "BG36_MagicItem_373"):
            a, h = a + max(1, sd[0]), h + max(1, sd[1])
    return a, h


def _by_pos(cards):
    return sorted(cards or [], key=lambda x: x.get("position", 0))


def base_id(card_id: str) -> str:
    return card_id[:-2] if card_id and card_id.endswith("_G") else card_id


class BridgeError(Exception):
    pass


def _dbg(g, **cmd) -> Any:
    try:
        r = g.debug(json.dumps(cmd))
    except Exception as ex:                       # noqa: BLE001 —— TavernGym 用 ValueError 报错
        return {"error": str(ex), "cmd": cmd}
    try:
        return json.loads(r) if r else None
    except ValueError:
        return r


def _view(g, p: int = 0) -> dict:
    return json.loads(g.view(p))


def _pick_heroes(g, heroes: dict[int, str], warnings: list[str]) -> None:
    for p, h in heroes.items():
        if h:
            r = _dbg(g, cmd="offer_hero", p=p, hero=h)
            if isinstance(r, dict) and r.get("error"):
                warnings.append(f"座位 {p} 英雄 {h}：{r['error']}")
    for _ in range(64):
        progressed = False
        for p in range(g.num_players):
            if not g.needs_action(p):
                continue
            pend = ((_view(g, p).get("me") or {}).get("pending") or {})
            if pend.get("kind") == "hero" and heroes.get(p):
                ids = [o.get("id") for o in pend.get("options", [])]
                idx = ids.index(heroes[p]) if heroes[p] in ids else 0
                g.step(p, A_CHOOSE + idx)
            elif pend.get("kind") == "hero":
                g.step(p, g.bot_action(p))
            else:
                continue
            progressed = True
        if not progressed:
            break


def _clear(g, p: int, warnings: list[str]) -> None:
    for zone in ("board", "hand", "shop"):
        r = _dbg(g, cmd="clear_zone", p=p, zone=zone)
        if isinstance(r, dict) and r.get("error"):
            warnings.append(f"清空 {zone}：{r['error']}")


def _add_minion(g, p: int, zone: str, c: dict, warnings: list[str]) -> bool:
    golden = bool(c.get("golden")) or (c.get("cardId") or "").endswith("_G")
    r = _dbg(g, cmd="add_minion", p=p, zone=zone, card=base_id(c["cardId"]), golden=golden,
             atk=c.get("attack"), hp=c.get("health"))
    if isinstance(r, dict) and r.get("error"):
        warnings.append(f"{zone} 加随从 {c.get('name')}({c['cardId']})：{r['error']}")
        return False
    return True


def _put_board(g, p: int, cards: list[dict], warnings: list[str]) -> None:
    """按顺序放随从到 p 的场上，并把关键词改成插件里的实际值。"""
    n0 = len((json.loads(g.view(p)).get("me") or {}).get("board") or [])
    for c in cards:
        _add_minion(g, p, "board", c, warnings)
    for i, c in enumerate(cards):
        _dbg(g, cmd="edit_minion", p=p, zone="board", index=n0 + i, keywords=[v for k, v in KW.items() if c.get(k)])


def build_game(state: dict, seed: int = 0, config: dict | None = None) -> tuple[Any, list[str]]:
    """返回 (TavernGym Game, 警告列表)。state = 插件导出的一条招募阶段局面（schema 2）。"""
    import taverngym

    warnings: list[str] = []
    p0 = state.get("player") or {}
    if not p0:
        raise BridgeError("局面里没有玩家数据")
    cfg = dict(config or {})
    races = state.get("availableRaces") or []
    if races:
        cfg.setdefault("lobby_types", races)
    try:
        g = taverngym.Game(seed=seed, config_json=json.dumps(cfg))
    except Exception as ex:                       # noqa: BLE001
        warnings.append(f"配置 {cfg} 不被接受（{ex}），改用默认配置")
        g = taverngym.Game(seed=seed)

    # 座位：0 = 你，其余按排行榜顺序
    others = [e for e in state.get("leaderboard") or [] if not e.get("isPlayer")]
    heroes = {0: p0.get("_heroBase") or p0.get("heroCardId")}
    for i, e in enumerate(others[:7], start=1):
        heroes[i] = e.get("_heroBase") or e.get("heroCardId")
    _pick_heroes(g, heroes, warnings)
    for p, h in heroes.items():                        # 兜底：没选上就直接设置
        cur = ((_view(g, p).get("me") or {}).get("hero") or {}).get("id")
        if h and cur != h:
            r = _dbg(g, cmd="set_hero", p=p, hero=h)
            if isinstance(r, dict) and r.get("error"):
                warnings.append(f"座位 {p} 设置英雄 {h}：{r['error']}")
    if _view(g, 0).get("phase") != "recruit":
        warnings.append(f"重建后阶段是 {_view(g, 0).get('phase')}，不是招募阶段")

    turn = int(state.get("turn") or 1)
    _dbg(g, cmd="set_turn", turn=turn)
    for i, e in enumerate(others[:7], start=1):
        _dbg(g, cmd="set_player", p=i, health=0 if e.get("dead") else e.get("health"), armor=e.get("armor"),
             tier=max(1, e.get("tavernTier") or 1))

    # 0 号座位（饰品先给：有的饰品获得时会给牌、改酒馆，之后清空重建）
    for t in p0.get("trinkets") or []:
        r = _dbg(g, cmd="give_trinket", p=0, card=t["cardId"])
        if isinstance(r, dict) and r.get("error"):
            warnings.append(f"饰品 {t.get('name')}：{r['error']}")
    for slot, hp in enumerate(p0.get("heroPowers") or []):
        if hp.get("cardId"):
            r = _dbg(g, cmd="set_hero_power", p=0, slot=slot, card=hp["cardId"])
            if isinstance(r, dict) and r.get("error"):
                warnings.append(f"英雄技能 {hp.get('name')}：{r['error']}")
    for rw in p0.get("questRewards") or []:
        _dbg(g, cmd="give_reward", p=0, card=rw["cardId"])
    # 重建手牌、场面、酒馆时先把饰品拿掉（有的饰品会在每次改动后自动补满酒馆、给牌），最后放回去；
    # 饰品获得时的一次性效果（玩家计数等）已经生效
    st = json.loads(g.state_json())
    held_trinkets = st["players"][0].get("trinkets") or []
    if held_trinkets:
        st["players"][0]["trinkets"] = []
        g = taverngym.Game.from_json(json.dumps(st))
    _clear(g, 0, warnings)

    board, hand, tavern = _by_pos(p0.get("board")), _by_pos(p0.get("hand")), _by_pos(state.get("tavern"))
    _put_board(g, 0, board, warnings)
    for c in hand:
        if c.get("type") == "minion":
            _add_minion(g, 0, "hand", c, warnings)
        else:
            r = _dbg(g, cmd="add_spell", p=0, zone="hand", card=c["cardId"])
            if isinstance(r, dict) and r.get("error"):
                warnings.append(f"手牌法术 {c.get('name')}：{r['error']}")
    for c in tavern:
        if c.get("type") == "minion":
            _add_minion(g, 0, "shop", c, warnings)
        else:
            r = _dbg(g, cmd="add_spell", p=0, zone="shop", card=c["cardId"])
            if isinstance(r, dict) and r.get("error"):
                warnings.append(f"酒馆法术 {c.get('name')}：{r['error']}")

    # 对手：最近一次交手看到的阵容
    seen = state.get("_seen") or {}
    opp_seen: dict[int, int] = {}
    for i, e in enumerate(others[:7], start=1):
        sv = seen.get(e.get("_heroBase") or e.get("heroCardId") or "")
        if sv and not e.get("dead"):
            _dbg(g, cmd="clear_zone", p=i, zone="board")
            _put_board(g, i, _by_pos(sv.get("board"))[:7], warnings)
            opp_seen[i] = int(sv.get("turn") or 0)

    shop = p0.get("shop") or {}
    _dbg(g, cmd="set_player", p=0, health=p0.get("health"), armor=p0.get("armor"),
         tier=max(1, p0.get("tavernTier") or 1), gold=p0.get("gold"),
         gold_next_turn=p0.get("extraGoldNextTurn") or 0,
         free_refreshes=shop.get("freeRefreshes") or 0,
         upgrade_cost=shop.get("upgradeCost") if shop.get("canUpgrade") else None)

    # 改存档 JSON 再读档：玩家级计数、随从计数器、附着物、看到过的对手阵容
    tags = {k.upper(): v for k, v in (p0.get("playerTags") or {}).items()}
    patch = {f: tags[t] for t, f in STAT_TAGS.items() if t in tags}
    patch.update({f: tags[t] for t, f in GEM_TAGS.items() if t in tags})
    if "DAMAGE_DEALT_TO_HERO_LAST_TURN" in tags:             # 上一场战斗输了（蓝壳始祖龟等）
        patch["lost_last"] = bool(tags["DAMAGE_DEALT_TO_HERO_LAST_TURN"])
        if patch["lost_last"]:
            patch["won_last"] = False
    # 插件的酒馆法术加成是总数（含场上光环、饰品），TavernGym 的 spell_atk/hp 只是“本局”部分
    da, dh = spell_extra_dynamic(board, p0.get("trinkets") or [])
    for t, f, d in (("TAVERN_SPELL_ATTACK_INCREASE", "spell_atk", da), ("TAVERN_SPELL_HEALTH_INCREASE", "spell_hp", dh)):
        if t in tags:
            patch[f] = max(0, tags[t] - d)
    s = json.loads(g.state_json())
    me = s["players"][0]
    stats = me.get("stats")
    if isinstance(stats, dict):
        for k, v in patch.items():
            if k in stats:
                stats[k] = v
            else:
                warnings.append(f"TavernGym 没有计数字段 {k}")
    else:
        warnings.append("存档里找不到 players[0].stats")
    me["trinkets"] = held_trinkets
    _card_state(me, (p0.get("trinkets") or []) + (p0.get("questRewards") or []), warnings)
    if me.get("pending"):
        # 重建过程中（选英雄、给饰品等）产生的待选项不是插件局面里的；插件局面里的待选项（offered）暂不重建
        warnings.append(f"去掉重建时产生的待选项 {len(me['pending'])} 个")
        me["pending"] = []
    item_minion = lambda x: (x.get("item") or {}).get("Minion")      # noqa: E731
    zones = [("board", me.get("board") or [], board, lambda x: x),
             ("hand", me.get("hand") or [], hand, item_minion),
             ("shop", me.get("shop") or [], tavern, item_minion)]
    for i in opp_seen:
        zones.append((f"对手{i}场面", s["players"][i].get("board") or [],
                      _by_pos(seen[(others[i - 1].get("_heroBase") or others[i - 1].get("heroCardId"))]["board"])[:7],
                      lambda x: x))
    cat = catalog()
    for zone, items, cards, get in zones:
        if len(items) != len(cards):
            warnings.append(f"{zone} 数量 {len(items)} ≠ {len(cards)}，跳过计数器和附着物")
            continue
        for it, c in zip(items, cards):
            m = get(it)
            if not m:
                continue
            ctr = counter_of(c)
            if ctr is not None:
                m["counter"] = ctr
            for a in attachments(c):
                m.setdefault("attached", []).append(cat[a]["cid"])
                m.setdefault("attachment_data", []).append(0)
                if a == "BG36_MidGameEffect_000t15":
                    m["shield_max"] = 3
    if opp_seen:
        n = len(s["players"])
        sc = me.get("scouted") or []
        sc = sc + [{"board": [], "turn": 0} for _ in range(n - len(sc))]
        for i, t in opp_seen.items():
            sc[i] = {"board": s["players"][i].get("board") or [], "turn": t}
        me["scouted"] = sc
    g = taverngym.Game.from_json(json.dumps(s))
    for i, c in enumerate(tavern):                           # 酒馆随从的攻防、关键词改成插件里的实际值
        if c.get("type") == "minion":
            _dbg(g, cmd="edit_minion", p=0, zone="shop", index=i, atk=c.get("attack"), hp=c.get("health"),
                 keywords=[v for k, v in KW.items() if c.get(k)])
    return g, warnings


def compare(g, state: dict) -> list[str]:
    """重建后的 TavernGym 局面和插件局面的差异（只比 0 号座位能看到的东西）。"""
    v = _view(g, 0)
    me = v.get("me") or v.get("player") or v
    diffs: list[str] = []
    p0 = state["player"]

    def minions(lst):
        return [(base_id(m.get("id", "")), bool(m.get("golden")), m.get("atk"), m.get("hp"),
                 tuple(sorted(k for k in m.get("keywords", []) if k in KW.values())))
                for m in lst]

    want_board = [(base_id(c["cardId"]), bool(c.get("golden")) or c["cardId"].endswith("_G"), c["attack"], c["health"],
                   tuple(sorted(v for k, v in KW.items() if c.get(k))))
                  for c in sorted(p0.get("board") or [], key=lambda x: x.get("position", 0))]
    got_board = minions(me.get("board") or [])
    if got_board != want_board:
        diffs.append(f"场面 {got_board} ≠ {want_board}")
    want_hand = sorted(base_id(c["cardId"]) for c in p0.get("hand") or [])
    got_hand = sorted(base_id((h.get("minion") or h).get("id") or h.get("card", {}).get("id", "")) for h in me.get("hand") or [])
    if got_hand != want_hand:
        diffs.append(f"手牌 {got_hand} ≠ {want_hand}")
    want_shop = sorted(base_id(c["cardId"]) for c in state.get("tavern") or [])
    got_shop = sorted(base_id((s.get("minion") or s).get("id") or s.get("card", {}).get("id", "")) for s in me.get("shop") or [])
    if got_shop != want_shop:
        diffs.append(f"酒馆 {got_shop} ≠ {want_shop}")
    races = sorted(state.get("availableRaces") or [])
    if races and sorted(v.get("lobby_races") or []) != races:
        diffs.append(f"种族 {sorted(v.get('lobby_races') or [])} ≠ {races}")
    hero = (me.get("hero") or {}).get("id")
    want_hero = p0.get("_heroBase") or p0.get("heroCardId")
    if hero != want_hero:
        diffs.append(f"英雄 {hero} ≠ {want_hero}")
    got_hp = sorted((h or {}).get("id") for h in me.get("hero_powers") or [])
    want_hp = sorted(h.get("cardId") for h in p0.get("heroPowers") or [] if h.get("cardId"))
    if got_hp != want_hp:
        diffs.append(f"英雄技能 {got_hp} ≠ {want_hp}")
    got_tr = sorted(t.get("id") for t in me.get("trinkets") or [])
    want_tr = sorted(t.get("cardId") for t in p0.get("trinkets") or [])
    if want_tr and got_tr != want_tr:
        diffs.append(f"饰品 {got_tr} ≠ {want_tr}")
    opp = {o.get("id"): o for o in v.get("opponents") or []}
    others = [e for e in state.get("leaderboard") or [] if not e.get("isPlayer")]
    for i, e in enumerate(others[:7], start=1):
        o = opp.get(i)
        if o is None:
            continue
        oh = (o.get("hero") or {}).get("id")
        if oh != (e.get("_heroBase") or e.get("heroCardId")):
            diffs.append(f"对手{i}英雄 {oh} ≠ {e.get('_heroBase')}")
        if not e.get("dead") and (o.get("health"), o.get("tier")) != (e.get("health"), max(1, e.get("tavernTier") or 1)):
            diffs.append(f"对手{i} 血/等级 {(o.get('health'), o.get('tier'))} ≠ {(e.get('health'), e.get('tavernTier'))}")
    seen = state.get("_seen") or {}
    for i, e in enumerate(others[:7], start=1):
        sv = seen.get(e.get("_heroBase") or e.get("heroCardId") or "")
        o = opp.get(i)
        if not sv or o is None or e.get("dead"):
            continue
        want = [(base_id(c["cardId"]), c.get("attack"), c.get("health")) for c in _by_pos(sv.get("board"))[:7]]
        got = [(base_id(m.get("id", "")), m.get("atk"), m.get("hp")) for m in o.get("board") or []]
        if got != want or o.get("seen_turn") != sv.get("turn"):
            diffs.append(f"对手{i}阵容 {got}@{o.get('seen_turn')} ≠ {want}@{sv.get('turn')}")
    for m, c in zip(me.get("board") or [], _by_pos(p0.get("board"))):
        ctr = counter_of(c)
        if ctr is not None and m.get("counter") != ctr:
            diffs.append(f"计数器 {c['cardId']} {m.get('counter')} ≠ {ctr}")
        want_att = sorted(attachments(c))
        got_att = sorted(a.get("id") for a in m.get("attached") or [])
        if got_att != want_att:
            diffs.append(f"附着 {c['cardId']} {got_att} ≠ {want_att}")
    stats = json.loads(g.state_json())["players"][0].get("stats", {})
    tags = {k.upper(): val for k, val in (p0.get("playerTags") or {}).items()}
    da, dh = spell_extra_dynamic(_by_pos(p0.get("board")), p0.get("trinkets") or [])
    for t, f in STAT_TAGS.items():
        want = tags.get(t)
        if want is not None and f in ("spell_atk", "spell_hp"):
            want = max(0, want - (da if f == "spell_atk" else dh))
        if want is not None and stats.get(f) != want:
            diffs.append(f"计数 {f} {stats.get(f)} ≠ {want}")
    for key, want in (("gold", p0.get("gold")), ("tier", p0.get("tavernTier")), ("health", p0.get("health")),
                      ("armor", p0.get("armor"))):
        got = me.get(key)
        if got is not None and want is not None and got != want:
            diffs.append(f"{key} {got} ≠ {want}")
    return diffs


def observe(g, p: int = 0):
    """TavernGym 自己的观察编码：(features float32[OBS_FLOATS], cards int32[OBS_CARDS])。"""
    import numpy as np
    import taverngym
    f, c = g.observe(p)
    return np.frombuffer(f, dtype="<f4").copy(), np.frombuffer(c, dtype="<i4").copy()
