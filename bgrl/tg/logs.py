"""读插件的对局记录（games/*.jsonl，schema 2），整理成 TavernGym 桥接和回放校验要用的形式。

  load_game(path)        → 本局所有招募阶段记录（去掉面板用的字段；标签名统一大写；补上 _heroBase、_seen）
  seen_boards(records)   → 每个回合开始战斗时看到的对手阵容 {回合: (对手英雄, 阵容)}
  recruit_snapshots(...) → 每回合取一条“局面稳定”的招募记录（A4 用）
  segment(records)       → 切成一个个动作，每个动作配上动作前、动作后的稳定局面（A5 用）

_seen：到这条记录为止，每个对手（按英雄）最近一次交手时的阵容，{"英雄": {"turn": 回合, "board": [...]}}。
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

DROP = ("simulation", "learning", "plugin", "turnLog")
_SKIN = re.compile(r"_SKIN_[A-Z0-9]+$")
_skins: dict[str, str] | None = None


def _skin_map() -> dict[str, str]:
    """皮肤 → 原英雄。优先用 bgrl/data/bg_cards.json 的 skins 表，没有就按卡牌编号的 _SKIN_ 后缀推断。"""
    global _skins
    if _skins is None:
        _skins = {}
        p = Path(__file__).resolve().parent.parent / "data" / "bg_cards.json"
        if p.exists():
            try:
                _skins = dict(json.loads(p.read_text(encoding="utf-8")).get("skins") or {})
            except (ValueError, OSError):
                _skins = {}
    return _skins


def hero_base(card_id: str | None) -> str | None:
    if not card_id:
        return card_id
    m = _skin_map()
    if card_id in m:
        return m[card_id]
    return _SKIN.sub("", card_id)


def _upper_keys(d):
    return {str(k).upper(): v for k, v in d.items()} if isinstance(d, dict) else d


def _normalize(s: dict) -> dict:
    """v0.9.0 的记录里字典键被转成了 bacoN_RALLY 这种形式，统一转回大写。"""
    p = s.get("player") or {}
    for k in ("playerTags", "heroTags"):
        if k in p:
            p[k] = _upper_keys(p[k])
    for c in (p.get("board") or []) + (p.get("hand") or []) + (s.get("tavern") or []):
        if "tags" in c:
            c["tags"] = _upper_keys(c["tags"])
    for h in p.get("heroPowers") or []:
        if "tags" in h:
            h["tags"] = _upper_keys(h["tags"])
    if "gameTags" in s:
        s["gameTags"] = _upper_keys(s["gameTags"])
    if p:
        p["_heroBase"] = hero_base(p.get("heroCardId"))
    for e in s.get("leaderboard") or []:
        e["_heroBase"] = hero_base(e.get("heroCardId"))
    return s


def read_records(path: str | os.PathLike) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                s = json.loads(line)
            except ValueError:
                continue
            if not s.get("isBattlegrounds") or s.get("mode") not in (None, "solo"):
                continue
            for k in DROP:
                s.pop(k, None)
            out.append(_normalize(s))
    return out


def seen_boards(records: list[dict]) -> dict[int, tuple[str, list[dict]]]:
    """每回合战斗开始时对手的阵容（这一回合第一条带对手阵容的战斗记录 = 战斗前的阵容）。"""
    out: dict[int, tuple[str, list[dict]]] = {}
    for s in records:
        if s.get("phase") != "combat":
            continue
        o = s.get("opponent") or {}
        hero = o.get("heroCardId") or ""
        t = int(s.get("turn") or 0)
        if t in out or not hero or "BaconShopBob" in hero:
            continue
        board = o.get("board") or []
        if board:
            out[t] = (hero_base(hero), sorted(board, key=lambda c: c.get("position", 0)))
    return out


def load_game(path: str | os.PathLike) -> list[dict]:
    """本局的招募阶段记录（按顺序），每条加上 _file 和 _seen（到这条为止对手最近一次的阵容）。"""
    recs = read_records(path)
    seen_by_turn = seen_boards(recs)
    name = os.path.basename(str(path))
    seen: dict[str, dict] = {}
    out = []
    for s in recs:
        t = int(s.get("turn") or 0)
        # 本回合招募阶段看到的是上一回合（及更早）战斗的阵容
        for tt in sorted(seen_by_turn):
            if tt < t:
                h, b = seen_by_turn[tt]
                if seen.get(h, {}).get("turn", -1) < tt:
                    seen[h] = {"turn": tt, "board": b}
        if s.get("phase") == "recruit" and s.get("player"):
            s["_file"] = name
            s["_seen"] = {h: v for h, v in seen.items()}
            out.append(s)
    return out


def _key(s: dict):
    p = s["player"]
    return (p.get("gold"), p.get("tavernTier"),
            tuple((c["entityId"], c.get("attack"), c.get("health")) for c in p.get("board") or []),
            tuple(c["entityId"] for c in p.get("hand") or []),
            tuple(c["entityId"] for c in s.get("tavern") or []))


def _settled(a: dict, b: dict) -> bool:
    """两条相邻记录的可见局面一样（动画结算完了）。"""
    return _key(a) == _key(b)


def recruit_snapshots(records: list[dict]) -> list[dict]:
    """每回合一条：本回合最后一条招募记录之前、局面已稳定的那条（A4 检查用）。"""
    by_turn: dict[int, list[dict]] = {}
    for s in records:
        by_turn.setdefault(int(s.get("turn") or 0), []).append(s)
    out = []
    for t, lst in sorted(by_turn.items()):
        if t < 1:
            continue
        mid = lst[len(lst) // 2:]
        pick = next((x for x, y in zip(mid, mid[1:]) if _settled(x, y)), mid[0])
        out.append(pick)
    return out


# ---------------- 动作切分（A5） ----------------

@dataclass
class Action:
    turn: int
    kind: str                     # buy / sell / play / cast / magnetize / upgrade / refresh / hero_power / move
    eid: int = 0                  # 相关实体（插件的 entityId）
    card_id: str = ""
    before: dict = field(default_factory=dict, repr=False)
    after: dict = field(default_factory=dict, repr=False)
    # 动作之后到下一个动作之前出现过的各个不同的稳定局面（动画、三连合成、发现之后的选择等会让局面分几步变化）
    afters: list = field(default_factory=list, repr=False)
    # 动作之后再往后的十几条记录（同一回合）：你操作比动画快时，攻防变化会晚到下一个动作之后才出现
    later: list = field(default_factory=list, repr=False)

    def to_json(self) -> dict:
        return {"turn": self.turn, "kind": self.kind, "eid": self.eid, "cardId": self.card_id}


def _base(card_id: str) -> str:
    return card_id[:-2] if card_id.endswith("_G") else card_id


def _golden(c: dict) -> bool:
    return bool(c.get("golden")) or (c.get("cardId") or "").endswith("_G")


def _ids(cards):
    return {c["entityId"] for c in cards}


def _sorted(cards):
    return sorted(cards, key=lambda c: c.get("position", 0))


def primary_actions(s0: dict, s1: dict) -> list[Action]:
    """两条相邻记录之间玩家做了哪些主要动作。"""
    p0, p1 = s0["player"], s1["player"]
    T0, H0, B0 = _ids(s0.get("tavern") or []), _ids(p0.get("hand") or []), _ids(p0.get("board") or [])
    T1, H1, B1 = _ids(s1.get("tavern") or []), _ids(p1.get("hand") or []), _ids(p1.get("board") or [])
    acts: list[Action] = []
    turn = int(s1.get("turn") or 0)
    if not p0.get("tavernTier"):          # 对局刚开始、还没初始化完（等级 0 → 1、金币 0 → 3 不是升级）
        return acts
    by0 = {c["entityId"]: c for c in (s0.get("tavern") or []) + (p0.get("hand") or []) + (p0.get("board") or [])}
    # 三连：同名的三张合成一张金色（不是玩家的动作；买第三张时可能和购买合在一条记录里）
    by1 = {c["entityId"]: c for c in (p1.get("hand") or []) + (p1.get("board") or [])}
    new_golden = {_base(by1[e]["cardId"]) for e in (H1 | B1) - H0 - B0 - T0 if _golden(by1[e])}
    # 合成可能分两条记录：先三张一起消失，下一条才出现金色的
    gone = Counter(_base(by0[e]["cardId"]) for e in ((H0 | B0) - H1 - B1) if by0[e].get("type") == "minion"
                   and not _golden(by0[e]))
    new_golden |= {b for b, n in gone.items() if n >= 2}
    for e in T0 & (H1 | B1):
        acts.append(Action(turn, "buy", e, by0[e]["cardId"]))
    for e in T0 - T1 - H1 - B1:
        if _base(by0[e]["cardId"]) in new_golden and (p1.get("gold") or 0) < (p0.get("gold") or 0):
            acts.append(Action(turn, "buy", e, by0[e]["cardId"]))
    for e in H0 & B1:
        acts.append(Action(turn, "play", e, by0[e]["cardId"]))
    for e in H0 - H1 - B1:
        c = by0[e]
        if c.get("type") == "minion" and _base(c["cardId"]) in new_golden:
            continue
        if c.get("type") == "spell":
            acts.append(Action(turn, "cast", e, c["cardId"]))
        elif c.get("type") == "minion":
            acts.append(Action(turn, "magnetize", e, c["cardId"]))
    for e in B0 - B1 - H1:
        if _base(by0[e]["cardId"]) in new_golden:
            continue
        if (p1.get("gold") or 0) > (p0.get("gold") or 0) or len(B0 - B1) == 1:
            acts.append(Action(turn, "sell", e, by0[e]["cardId"]))
    if (p1.get("tavernTier") or 0) > (p0.get("tavernTier") or 0):
        acts.append(Action(turn, "upgrade"))
    elif T1 and T0 and not (T0 & T1) and not any(a.kind == "buy" for a in acts) and \
            ((p1.get("gold") or 0) < (p0.get("gold") or 0) or ((p0.get("shop") or {}).get("freeRefreshes") or 0) > 0):
        acts.append(Action(turn, "refresh"))
    hp0 = {h.get("entityId"): h for h in p0.get("heroPowers") or []}
    for h in p1.get("heroPowers") or []:
        if h.get("used") and not hp0.get(h.get("entityId"), {}).get("used", True):
            acts.append(Action(turn, "hero_power", h.get("entityId") or 0, h.get("cardId") or ""))
    if not acts and B0 == B1 and len(B0) > 1:
        o0 = [c["entityId"] for c in _sorted(p0["board"])]
        o1 = [c["entityId"] for c in _sorted(p1["board"])]
        if o0 != o1:
            moved = next(e for e, f in zip(o0, o1) if e != f)
            acts.append(Action(turn, "move", moved, by0[moved]["cardId"]))
    return acts


def segment(records: list[dict]) -> list[Action]:
    """切成动作。动作前 = 动作发生前最后一条记录；动作后 = 动作之后第一条“稳定”的记录（和下一条可见局面一样），
    在下一个动作之前；找不到就用下一个动作前的最后一条。回合最后一个动作也一样处理，避免把回合结束时的效果
    （回合结束触发、酒馆清空）算进动作结果里。
    同一步里有多个主要动作（记录合并了）的，整组标成 multi，不单独检验。"""
    out: list[Action] = []
    starts: list[tuple[int, list[Action]]] = []        # (动作发生后第一条记录的下标, 动作)
    for i in range(len(records) - 1):
        s0, s1 = records[i], records[i + 1]
        if s1.get("turn") != s0.get("turn"):
            continue
        acts = primary_actions(s0, s1)
        if not acts:
            continue
        if len(acts) > 1:
            acts = [Action(acts[0].turn, "multi", 0, "+".join(a.kind for a in acts))]
        # 金币常常比卡牌移动早一两条记录变化（花钱先扣、出售先加）：往前找到金币变化之前的那条。
        # 只跨过“只有金币变了”的记录，方向要和动作一致（出售加钱，其它动作扣钱）
        sign = 1 if acts[0].kind == "sell" else -1
        k = j = i
        while j > 0 and i - j < 8 and records[j - 1].get("turn") == s0.get("turn") and \
                _key(records[j - 1])[1:] == _key(records[j])[1:] and \
                sign * ((records[j]["player"].get("gold") or 0) - (records[j - 1]["player"].get("gold") or 0)) >= 0:
            j -= 1
            if records[j]["player"].get("gold") != records[i]["player"].get("gold"):
                k = j          # 最早一条金币还没变的记录
                break
        for a in acts:
            a.before = records[k]
        starts.append((i + 1, acts))
    for k, (j, acts) in enumerate(starts):
        end = starts[k + 1][0] - 1 if k + 1 < len(starts) else len(records) - 1   # 下一个动作前最后一条
        turn = records[j].get("turn")
        while end > j and records[end].get("turn") != turn:
            end -= 1
        after = records[end]
        for x in range(j, end):
            if records[x + 1].get("turn") == turn and _settled(records[x], records[x + 1]):
                after = records[x]
                break
        cands, keys = [], set()
        for x in range(j, end + 1):
            if records[x].get("turn") != turn:
                continue
            if x < end and not _settled(records[x], records[x + 1]):
                continue
            k = _key(records[x])
            if k not in keys:
                keys.add(k)
                cands.append(records[x])
        later = [r for r in records[end + 1:end + 25] if r.get("turn") == turn]
        for a in acts:
            a.after = after
            a.afters = cands[:8] or [after]
            a.later = later
        out += acts
    return [a for a in out if a.before and a.after]
