"""A4：把插件导出的局面（schema 2）重建成 TavernGym 里的局面。

做法（编码只走 TavernGym 自己的 observe，避免论文里“两套编码器漂移”的问题）：
  1. 新开一局，种族固定为插件读到的本局种族；选英雄阶段把你的英雄放进候选并选上（你 = 0 号座位）。
  2. 其它 7 个座位按排行榜设置英雄、血量、护甲、等级；已淘汰的设为 0 血。
  3. 0 号座位：设置回合、金币、等级、血量、升本费用、免费刷新、下回合金币；饰品、英雄技能；
     清空后按插件局面重建场面（含站位、金色、攻防、关键词）、手牌、酒馆。
  4. 玩家级“本局”计数（酒馆法术加成等）写进存档 JSON 的 stats 后重新读档。
  5. 返回 TavernGym 的 Game；`compare(game, state)` 列出重建后和插件局面不一致的地方。

还没做的（会在 compare 的报告里体现）：随从计数器（counter）、黑暗之赐等附着物、对手的当前阵容
（只能用上次交手时看到的）、卡池余量（按默认卡池）。
"""
from __future__ import annotations

import json
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
# 宝石：HDT 记的是一颗宝石的总攻防（基础 1），TavernGym 记的是额外部分
GEM_TAGS = {"BACON_BLOODGEMBUFFATKVALUE": "gem_atk", "BACON_BLOODGEMBUFFHEALTHVALUE": "gem_hp"}

KW = {"taunt": "TAUNT", "divineShield": "DIVINE_SHIELD", "windfury": "WINDFURY", "reborn": "REBORN",
      "venomous": "VENOMOUS", "poisonous": "POISONOUS", "stealth": "STEALTH"}


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
            pend = (_view(g, p).get("pending") or {})
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

    turn = int(state.get("turn") or 1)
    _dbg(g, cmd="set_turn", turn=turn)
    for i, e in enumerate(others[:7], start=1):
        _dbg(g, cmd="set_player", p=i, health=0 if e.get("dead") else e.get("health"), armor=e.get("armor"),
             tier=max(1, e.get("tavernTier") or 1))

    # 0 号座位
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
    _clear(g, 0, warnings)

    for c in sorted(p0.get("board") or [], key=lambda x: x.get("position", 0)):
        _add_minion(g, 0, "board", c, warnings)
    for c in p0.get("hand") or []:
        if c.get("type") == "minion":
            _add_minion(g, 0, "hand", c, warnings)
        else:
            r = _dbg(g, cmd="add_spell", p=0, zone="hand", card=c["cardId"])
            if isinstance(r, dict) and r.get("error"):
                warnings.append(f"手牌法术 {c.get('name')}：{r['error']}")
    for c in state.get("tavern") or []:
        if c.get("type") == "minion":
            _add_minion(g, 0, "shop", c, warnings)
        else:
            r = _dbg(g, cmd="add_spell", p=0, zone="shop", card=c["cardId"])
            if isinstance(r, dict) and r.get("error"):
                warnings.append(f"酒馆法术 {c.get('name')}：{r['error']}")
    # 关键词（加随从时用的是卡面关键词，这里改成实际的）
    for zone, cards in (("board", sorted(p0.get("board") or [], key=lambda x: x.get("position", 0))),
                        ("shop", [c for c in state.get("tavern") or [] if c.get("type") == "minion"])):
        for i, c in enumerate(cards):
            kws = [v for k, v in KW.items() if c.get(k)]
            _dbg(g, cmd="edit_minion", p=0, zone=zone, index=i, keywords=kws)

    shop = p0.get("shop") or {}
    _dbg(g, cmd="set_player", p=0, health=p0.get("health"), armor=p0.get("armor"),
         tier=max(1, p0.get("tavernTier") or 1), gold=p0.get("gold"),
         gold_next_turn=p0.get("extraGoldNextTurn") or 0,
         free_refreshes=shop.get("freeRefreshes") or 0,
         upgrade_cost=shop.get("upgradeCost") if shop.get("canUpgrade") else None)

    # 玩家级计数：改存档 JSON 再读档
    tags = {k.upper(): v for k, v in (p0.get("playerTags") or {}).items()}
    patch = {f: tags[t] for t, f in STAT_TAGS.items() if t in tags}
    patch.update({f: tags[t] - 1 for t, f in GEM_TAGS.items() if t in tags})
    if patch:
        s = json.loads(g.state_json())
        stats = s["players"][0].get("stats")
        if isinstance(stats, dict):
            for k, v in patch.items():
                if k in stats:
                    stats[k] = v
                else:
                    warnings.append(f"TavernGym 没有计数字段 {k}")
            g = taverngym.Game.from_json(json.dumps(s))
        else:
            warnings.append("存档里找不到 players[0].stats")
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
