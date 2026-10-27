"""A5 现实闭环回放：用你真实对局里的每一步检验 TavernGym。

对每个动作（logs.segment 切出来的）：
  1. 用动作前的稳定局面在 TavernGym 里重建（bridge.build_game）。
  2. 把你的动作翻译成 TavernGym 的动作编号执行；之后要选目标 / 发现 / 抉择的，挨个试，取最接近记录的一个
     （记录里没有你选了哪个）。结果不一致时换几个随机种子再试（随机效果）。
  3. 和记录里动作后的稳定局面比：场面（顺序、卡、金色、攻防、关键词）、手牌、酒馆、金币、等级。
  4. 结果分四类：exact（完全一致）、aggregate（总攻防、张数一致，只是分配不同，多半是随机目标）、
     mismatch（不一致）、illegal（TavernGym 不让做这个动作）；另有 skipped（记录合并了几个动作，或翻译不了）。
"""
from __future__ import annotations

import json
import random
from collections import Counter
from typing import Any

from .bridge import KW, base_id, build_game, _by_pos
from .logs import Action

A_END, A_REFRESH, A_FREEZE, A_UPGRADE, A_HERO_POWER = 0, 1, 2, 3, 4
A_BUY, A_SELL, A_ACTIVATE, A_CHOOSE, A_PLAY, A_MAGNETIZE, A_MOVE = 8, 16, 23, 30, 46, 126, 196
N_POS, N_BOARD = 8, 7


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


def aggregate_ok(pred: dict, real: dict) -> bool:
    """总量一致：金币、等级、场面张数和卡、总攻防、手牌张数（随机目标、发现 / 随机给的牌不同也算）。"""
    def tot(v):
        return (v["gold"], v["tier"], len(v["board"]), sum(b[2] or 0 for b in v["board"]),
                sum(b[3] or 0 for b in v["board"]), sorted(b[0] for b in v["board"]), sum(v["hand"].values()))
    return tot(pred) == tot(real)


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
        if best is None or len(d) < best[0]:
            best = (len(d), d, pred, i)
        if not d:
            break
    return best


def _resolve(g, reals: list[dict], shop: bool, depth: int = 0) -> tuple:
    """动作之后的待选项：“先不选”（记录可能停在选择之前）和每个选项都试，返回最接近记录的
    (差异数, 差异, 预测局面, 候选下标, 局面)。"""
    here = _score(g, reals, shop) + (g,)
    pend = _pending(g) if depth < 4 else None
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


def _reseed(g, seed: int):
    import taverngym
    s = json.loads(g.state_json())
    r = random.Random(seed)
    s["rng"]["s"] = [r.getrandbits(64) | 1 for _ in range(4)]
    return taverngym.Game.from_json(json.dumps(s))


def check_action(a: Action, seed: int = 0, trials: int = 4) -> dict:
    out = {"file": a.before.get("_file"), "turn": a.turn, "kind": a.kind, "card": base_id(a.card_id) or a.kind}
    if a.kind == "multi":
        out.update(result="skipped", why="记录合并了几个动作")
        return out
    acts = translate(a)
    if not acts:
        out.update(result="skipped", why="翻译不了")
        return out
    try:
        g0, warns = build_game(a.before, seed=seed)
    except Exception as ex:                       # noqa: BLE001
        out.update(result="skipped", why=f"重建失败 {type(ex).__name__}: {ex}")
        return out
    if a.kind in ("cast", "magnetize"):
        # 手牌消失也可能是“发动”（如：发动：弃一张牌……），记录里分不出来，一起试
        board = json.loads(g0.view(0))["me"].get("board") or []
        acts = acts + [A_ACTIVATE + i for i, m in enumerate(board) if m.get("activate_cost") is not None]
    reals = [visible_log(x) for x in (a.afters or [a.after])]
    shop = a.kind not in ("refresh", "upgrade")    # 刷新是随机的；升本后可能有“升级时刷新”的效果
    best = None
    legal = False
    for t in range(trials):
        gs = g0 if t == 0 else _reseed(g0, seed * 1000 + t)
        for act in acts:
            g = gs.copy()
            try:
                g.step(0, act)
            except ValueError as ex:
                out.setdefault("refused", str(ex))
                continue
            legal = True
            r = _resolve(g, reals, shop)
            if best is None or r[0] < best[0]:
                best = r
            if best[0] == 0:
                break
        if best is not None and best[0] == 0:
            break
    if not legal:
        out.update(result="illegal", why=out.pop("refused", ""))
        return out
    score, d, pred, ci, _ = best
    out.update(result="exact" if score == 0 else "aggregate" if aggregate_ok(pred, reals[ci]) else "mismatch",
               diff=d[:6], warnings=warns[:3], afterIndex=ci)
    return out
