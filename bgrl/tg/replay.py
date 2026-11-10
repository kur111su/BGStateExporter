"""对局记录里的动作 → TavernGym 动作（高分回放转决策样本用，见 tools/make_pro_demos.py）。

  visible_log / visible_tg  记录 / TavernGym 局面里看得见的部分（场面、手牌、酒馆、金币、等级），diff 比两边
  translate(action)         logs.segment 切出来的动作 → TavernGym 的动作编号
  best_choice(g, reals)     动作之后要选目标 / 发现 / 抉择的，挨个试，取最接近记录的一个（记录里没有选了哪个）
"""
from __future__ import annotations

import json
from collections import Counter

from .actions import (A_ACTIVATE, A_BUY, A_CHOOSE, A_HERO_POWER, A_MAGNETIZE, A_MOVE, A_PLAY, A_REFRESH, A_SELL,
                      A_UPGRADE, N_BOARD, N_POS)
from .bridge import KW, base_id, build_game, _by_pos
from .logs import Action

TRIPLE = "TB_BaconShop_Triples_01"


# ---------------- 可见局面 ----------------

def visible_log(s: dict) -> dict:
    p = s["player"]
    return {
        "board": [(base_id(c["cardId"]), bool(c.get("golden")) or c["cardId"].endswith("_G"), c.get("attack"),
                   c.get("health"), tuple(sorted(v for k, v in KW.items() if c.get(k)))) for c in _by_pos(p.get("board"))],
        "hand": Counter(base_id(c["cardId"]) for c in p.get("hand") or []),
        "shop": Counter(base_id(c["cardId"]) for c in s.get("tavern") or []),
        "gold": p.get("gold"), "tier": p.get("tavernTier"),
        "pending": bool(s.get("offered")),
    }


def _item_id(x: dict) -> str:
    return base_id((x.get("minion") or x).get("id") or (x.get("card") or {}).get("id", ""))


def visible_tg(g) -> dict:
    me = json.loads(g.view(0))["me"]
    pend = me.get("pending")
    return {
        "pending": isinstance(pend, dict) and bool(pend.get("options") or pend.get("targets")),
        "board": [(base_id(m.get("id", "")), bool(m.get("golden")), m.get("atk"), m.get("hp"),
                   tuple(sorted(k for k in m.get("keywords", []) if k in KW.values()))) for m in me.get("board") or []],
        "hand": Counter(_item_id(h) for h in me.get("hand") or []),
        "shop": Counter(_item_id(h) for h in me.get("shop") or []),
        "gold": me.get("gold"), "tier": me.get("tier"),
    }


def diff(pred: dict, real: dict, shop: bool = True) -> list[str]:
    out = []
    if pred.get("pending") != real.get("pending"):
        out.append(f"待选 {'有' if pred.get('pending') else '无'}≠{'有' if real.get('pending') else '无'}")
    if pred["gold"] != real["gold"]:
        out.append(f"金币 {pred['gold']}≠{real['gold']}")
    if pred["tier"] != real["tier"]:
        out.append(f"等级 {pred['tier']}≠{real['tier']}")
    pb, rb = pred["board"], real["board"]
    if len(pb) != len(rb) or [x[0] for x in pb] != [x[0] for x in rb]:
        out.append(f"场面 {[x[0] for x in pb]}≠{[x[0] for x in rb]}")
    else:
        for q, r in zip(pb, rb):
            if q[1:4] != r[1:4]:
                out.append(f"{r[0]} {q[2]}/{q[3]}{'金' if q[1] else ''}≠{r[2]}/{r[3]}{'金' if r[1] else ''}")
            elif set(q[4]) != set(r[4]):
                out.append(f"{r[0]} 关键词 {sorted(q[4])}≠{sorted(r[4])}")
    if pred["hand"] != real["hand"]:
        out.append(f"手牌 少{sorted((real['hand'] - pred['hand']).elements())} 多{sorted((pred['hand'] - real['hand']).elements())}")
    if shop and pred["shop"] != real["shop"]:
        out.append(f"酒馆 少{sorted((real['shop'] - pred['shop']).elements())} 多{sorted((pred['shop'] - real['shop']).elements())}")
    return out


# ---------------- 动作翻译 ----------------

def _index(cards: list[dict], eid: int) -> int | None:
    return next((i for i, c in enumerate(_by_pos(cards)) if c.get("entityId") == eid), None)


def translate(a: Action) -> list[int]:
    """返回候选动作编号（磁力目标未知时有多个）；空 = 翻译不了。"""
    b, af = a.before, a.after
    p0, p1 = b["player"], af["player"]
    if a.kind == "buy":
        i = _index(b.get("tavern") or [], a.eid)
        return [] if i is None else [A_BUY + i]
    if a.kind == "sell":
        i = _index(p0.get("board") or [], a.eid)
        return [] if i is None else [A_SELL + i]
    if a.kind in ("play", "cast"):
        h = _index(p0.get("hand") or [], a.eid)
        if h is None or h >= 10:
            return []
        pos = 0
        if a.kind == "play":
            pos = _index(p1.get("board") or [], a.eid)
            if pos is None:
                pos = len(p0.get("board") or [])
            pos = min(pos, len(p0.get("board") or []), N_POS - 1)
        return [A_PLAY + h * N_POS + pos]
    if a.kind == "magnetize":
        h = _index(p0.get("hand") or [], a.eid)
        if h is None:
            return []
        return [A_MAGNETIZE + h * N_BOARD + i for i in range(len(p0.get("board") or []))]
    if a.kind == "upgrade":
        return [A_UPGRADE]
    if a.kind == "refresh":
        return [A_REFRESH]
    if a.kind == "hero_power":
        hps = p0.get("heroPowers") or []
        slot = next((i for i, h in enumerate(hps) if h.get("entityId") == a.eid), 0)
        return [A_HERO_POWER + slot]
    if a.kind == "move":
        f = _index(p0.get("board") or [], a.eid)
        t = _index(p1.get("board") or [], a.eid)
        return [] if f is None or t is None else [A_MOVE + f * N_BOARD + t]
    return []


# ---------------- 执行 + 选择 ----------------

def _pending(g) -> dict | None:
    v = json.loads(g.view(0))
    pend = (v.get("me") or {}).get("pending")
    if not isinstance(pend, dict):
        return None
    if pend.get("options") is None and pend.get("targets") is not None:      # 选目标：选项是实体编号
        pend = dict(pend, options=[{"eid": t} for t in pend["targets"]])
    return pend if pend.get("options") is not None else None


def _score(g, reals: list[dict], shop: bool) -> tuple[int, list[str], dict, int]:
    """和各个候选的动作后局面比，取最接近的：(差异数, 差异, 预测局面, 第几个候选)。"""
    pred = visible_tg(g)
    best = None
    for i, real in enumerate(reals):
        d = diff(pred, real, shop)
        n = _weight(d)
        if best is None or n < best[0]:
            best = (n, d, pred, i)
        if not d:
            break
    return best


def _weight(d: list[str]) -> int:
    """差异打分：停在选择中途（待选项不一致）比选完之后攻防有点出入差得多。"""
    return sum(3 if x.startswith("待选") else 1 for x in d)


def best_choice(g, reals: list[dict], shop: bool = True) -> tuple[list[int], int]:
    """g 停在一个待选项上（选目标 / 发现 ……）：每个选项都试一遍（之后的待选项也按最接近的走），
    返回 (和记录一样接近的选项下标们, 差异数)。没有待选项返回 ([], -1)。用来从高手回放里读出“选了哪个”。"""
    pend = _pending(g)
    if not pend:
        return [], -1
    scores = []
    for i in range(min(len(pend.get("options") or []), 16)):
        g2 = g.copy()
        try:
            g2.step(0, A_CHOOSE + i)
        except ValueError:
            continue
        scores.append((_resolve(g2, reals, shop, 1)[0], i))
    if not scores:
        return [], -1
    best = min(s for s, _ in scores)
    return [i for s, i in scores if s == best], best


def _resolve(g, reals: list[dict], shop: bool, depth: int = 0) -> tuple:
    """动作之后的待选项：“先不选”（记录可能停在选择之前）和每个选项都试，返回最接近记录的
    (差异数, 差异, 预测局面, 候选下标, 局面)。"""
    here = _score(g, reals, shop) + (g,)
    pend = _pending(g) if depth < 4 else None
    if not pend and here[0] and depth < 4:
        # 三连奖励：你常常在插件记下它进手牌之前就用掉了（记录里直接是发现的结果）
        hand = json.loads(g.view(0))["me"].get("hand") or []
        tri = [i for i, h in enumerate(hand) if _item_id(h) == TRIPLE]
        if tri and all(r["hand"].get(TRIPLE, 0) < sum(1 for h in hand if _item_id(h) == TRIPLE) for r in reals):
            g2 = g.copy()
            try:
                g2.step(0, A_PLAY + tri[0] * N_POS)
            except ValueError:
                return here
            r = _resolve(g2, reals, shop, depth + 1)
            return r if r[0] < here[0] else here
    if not pend or here[0] == 0:
        return here
    opts = pend.get("options") or []
    want = Counter()
    for r in reals:
        want |= r["hand"] - visible_tg(g)["hand"]
    order = list(range(min(len(opts), 16)))
    # 发现类：先试和记录里新进手牌一致的选项
    order.sort(key=lambda i: 0 if want.get(base_id((opts[i] or {}).get("id") or "")) else 1)
    best = here
    for i in order:
        g2 = g.copy()
        try:
            g2.step(0, A_CHOOSE + i)
        except ValueError:
            continue
        r = _resolve(g2, reals, shop, depth + 1)
        if r[0] < best[0]:
            best = r
        if best[0] == 0:
            break
    return best
