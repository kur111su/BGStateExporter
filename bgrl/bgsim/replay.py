"""回放校验：用对局记录检验招募规则和卡牌实现。

做法：
  1. 把一局的招募阶段记录切成一个个“动作”（买 / 卖 / 打出 / 施法 / 升本 / 刷新 / 英雄技能 / 主动技能），
     动作前后各取“稳定后的局面”（下一个动作之前的最后一条记录，效果动画都结算完了）。
  2. 用动作前的局面在环境里重建玩家状态（随从 uid = 记录里的 entityId），执行同样的动作。
     记录里看不到的东西（计数器、宝石加成等）沿用上一步环境里的值。
  3. 对比环境算出的结果和记录里动作后的局面：场面（攻防、关键词）、手牌、金币。
     目标未知（记录里没有）和随机效果：枚举所有可能目标、换几个随机种子，取最接近的一次。
  4. 按“动作的卡”统计通过率。

用法：python -m bgsim.replay [对局记录目录或文件...] [--out 报告.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from . import cards as cardsmod
from .game import Game, IllegalAction
from .model import Minion, Player, Spell

KW_MAP = {"taunt": "taunt", "divineShield": "divineShield", "poisonous": "poisonous", "venomous": "venomous",
          "windfury": "windfury", "reborn": "reborn", "stealth": "stealth"}


@dataclass
class Action:
    turn: int
    kind: str                     # buy / sell / play / cast / magnetize / upgrade / refresh / hero_power / activate
    eid: int = 0                  # 相关实体
    card_id: str = ""
    before: dict = field(default_factory=dict)
    after: dict = field(default_factory=dict)
    pos: int = -1


def ids(cards: list[dict]) -> set[int]:
    return {c["entityId"] for c in cards}


def load_states(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                s = json.loads(line)
            except ValueError:
                continue
            if s.get("isBattlegrounds") and s.get("mode") == "solo" and s.get("phase") == "recruit" and s.get("player"):
                out.append(s)
    return out


def primary_actions(s0: dict, s1: dict) -> list[Action]:
    """两条相邻记录之间玩家做了哪些主要动作。"""
    p0, p1 = s0["player"], s1["player"]
    T0, H0, B0 = ids(s0["tavern"]), ids(p0["hand"]), ids(p0["board"])
    T1, H1, B1 = ids(s1["tavern"]), ids(p1["hand"]), ids(p1["board"])
    acts: list[Action] = []
    turn = s1["turn"]
    by0 = {c["entityId"]: c for c in s0["tavern"] + p0["hand"] + p0["board"]}
    for e in T0 & (H1 | B1):
        acts.append(Action(turn, "buy", e, by0[e]["cardId"]))
    for e in H0 & B1:
        acts.append(Action(turn, "play", e, by0[e]["cardId"]))
    gone_hand = H0 - H1 - B1
    for e in gone_hand:
        c = by0[e]
        if c["type"] == "spell":
            acts.append(Action(turn, "cast", e, c["cardId"]))
        elif c["type"] == "minion":
            acts.append(Action(turn, "magnetize", e, c["cardId"]))
    for e in B0 - B1 - H1:
        if p1["gold"] > p0["gold"] or len(B0 - B1) == 1:
            acts.append(Action(turn, "sell", e, by0[e]["cardId"]))
    if p1["tavernTier"] > p0["tavernTier"]:
        acts.append(Action(turn, "upgrade"))
    elif T1 and T0 and not (T0 & T1) and not any(a.kind == "buy" for a in acts) and \
            (p1["gold"] < p0["gold"] or (p0.get("shop") or {}).get("freeRefreshes", 0) > 0):
        acts.append(Action(turn, "refresh"))
    hp0 = {h["entityId"]: h for h in p0.get("heroPowers") or []}
    for h in p1.get("heroPowers") or []:
        if h.get("used") and not hp0.get(h["entityId"], {}).get("used", True):
            acts.append(Action(turn, "hero_power", h["entityId"], h.get("cardId") or ""))
    return acts


def segment(states: list[dict]) -> list[Action]:
    """切成动作，每个动作配上稳定后的前后局面。"""
    out: list[Action] = []
    last_stable = None
    pending: list[Action] = []
    for s0, s1 in zip(states, states[1:]):
        if s1["turn"] != s0["turn"]:
            for a in pending:
                a.after = s0
            out += pending
            pending = []
            last_stable = None
            continue
        acts = primary_actions(s0, s1)
        if acts:
            for a in pending:
                a.after = s0
            out += pending
            before = s0
            pending = []
            for a in acts:
                a.before = before
                pending.append(a)
            last_stable = s1
    if pending and states:
        for a in pending:
            a.after = states[-1]
        out += pending
    # 一步里同时有多个主要动作的（合并的记录），只保留可单独检验的
    return [a for a in out if a.before and a.after]


# ---------------- 从记录重建环境里的玩家 ----------------

def minion_from(game: Game, c: dict, carried: dict[int, Minion]) -> Minion:
    m = game.new_minion(c["cardId"], c["attack"], c["health"])
    m.uid = c["entityId"]
    m.golden = bool(c.get("golden"))
    m.kw = {v for k, v in KW_MAP.items() if c.get(k)}
    m.frozen = bool(c.get("frozen"))
    old = carried.get(m.uid)
    if old is not None and old.card_id == m.card_id:
        m.sd = list(old.sd)
        m.data = dict(old.data)
    return m


def spell_from(game: Game, c: dict) -> Spell:
    s = game.new_spell(c["cardId"])
    s.uid = c["entityId"]
    s.cost = c.get("cost", s.cost)
    return s


def sync_player(game: Game, p: Player, s: dict, carried: dict[int, Minion]) -> None:
    ps = s["player"]
    p.hero_id = ps.get("heroCardId") or p.hero_id
    p.health, p.armor = ps["health"], ps["armor"]
    p.tier, p.gold, p.max_gold = ps["tavernTier"], ps["gold"], ps["maxGold"]
    p.extra_gold_next = ps.get("extraGoldNextTurn", 0)
    sh = ps.get("shop") or {}
    if sh.get("buyCostKnown"):
        p.counters["buy_cost_mod"] = sh["buyCost"] - game.rules.buy_cost
    if sh.get("refreshCost") is not None:
        p.counters["refresh_cost_mod"] = sh["refreshCost"] - game.rules.refresh_cost if sh["refreshCost"] else 0
    p.free_refreshes = sh.get("freeRefreshes", 0) or (1 if sh.get("refreshCost") == 0 else 0)
    p.upgrade_cost = sh.get("upgradeCost", p.upgrade_cost) if sh.get("canUpgrade") else 99
    p.board = [minion_from(game, c, carried) for c in sorted(ps["board"], key=lambda x: x["position"])]
    p.hand = [minion_from(game, c, carried) if c["type"] == "minion" else spell_from(game, c) for c in ps["hand"]]
    p.shop = [minion_from(game, c, carried) for c in s["tavern"] if c["type"] == "minion"]
    p.shop_spells = [spell_from(game, c) for c in s["tavern"] if c["type"] == "spell"]
    p.pending = []
    p.trinkets = [t["cardId"] for t in ps.get("trinkets") or []]


def visible(p: Player) -> dict:
    return {"board": [(m.uid, m.card_id, m.atk, m.hp, tuple(sorted(m.kw))) for m in p.board],
            "hand": sorted((c.card_id, getattr(c, "atk", 0), getattr(c, "hp", 0)) for c in p.hand),
            "gold": p.gold, "tier": p.tier}


def visible_log(s: dict) -> dict:
    ps = s["player"]
    return {"board": [(c["entityId"], c["cardId"], c["attack"], c["health"],
                       tuple(sorted(v for k, v in KW_MAP.items() if c.get(k))))
                      for c in sorted(ps["board"], key=lambda x: x["position"])],
            "hand": sorted((c["cardId"], c["attack"] if c["type"] == "minion" else 0,
                            c["health"] if c["type"] == "minion" else 0) for c in ps["hand"]),
            "gold": ps["gold"], "tier": ps["tavernTier"]}


def diff(pred: dict, real: dict) -> list[str]:
    out = []
    if pred["gold"] != real["gold"]:
        out.append(f"金币 {pred['gold']}≠{real['gold']}")
    if pred["tier"] != real["tier"]:
        out.append(f"等级 {pred['tier']}≠{real['tier']}")
    pb = {b[0]: b for b in pred["board"]}
    rb = {b[0]: b for b in real["board"]}
    for uid, r in rb.items():
        q = pb.get(uid)
        if q is None:
            continue
        if (q[2], q[3]) != (r[2], r[3]):
            out.append(f"{r[1]} {q[2]}/{q[3]}≠{r[2]}/{r[3]}")
        elif set(q[4]) != set(r[4]):
            out.append(f"{r[1]} 关键词 {sorted(q[4])}≠{sorted(r[4])}")
    new_p = Counter((b[1], b[2], b[3]) for uid, b in pb.items() if uid not in rb)
    new_r = Counter((b[1], b[2], b[3]) for uid, b in rb.items() if uid not in pb)
    if new_p != new_r:
        out.append(f"新随从 {sorted(new_p.elements())}≠{sorted(new_r.elements())}")
    hp_, hr_ = Counter(pred["hand"]), Counter(real["hand"])
    if hp_ != hr_:
        miss, extra = hr_ - hp_, hp_ - hr_
        out.append(f"手牌 少{sorted(miss.elements())} 多{sorted(extra.elements())}")
    return out


def aggregate_ok(pred: dict, real: dict) -> bool:
    """总量一致（随机目标给错对象也算对）：金币、等级、场面总攻防、手牌卡名。"""
    def tot(v):
        return (v["gold"], v["tier"], sum(b[2] for b in v["board"]), sum(b[3] for b in v["board"]),
                len(v["board"]), Counter(h[0] for h in v["hand"]))
    return tot(pred) == tot(real)


def run_action(game: Game, p: Player, a: Action, seed: int, target: int | None) -> bool:
    game.rng.seed(seed)
    try:
        if a.kind == "buy":
            game.step(p, ("buy", a.eid))
        elif a.kind == "sell":
            game.step(p, ("sell", a.eid))
        elif a.kind == "play":
            pos = next((i for i, c in enumerate(sorted(a.after["player"]["board"], key=lambda x: x["position"]))
                        if c["entityId"] == a.eid), len(p.board))
            game.step(p, ("play", a.eid, pos, target))
        elif a.kind == "cast":
            game.step(p, ("play", a.eid, 0, target))
        elif a.kind == "magnetize":
            game.step(p, ("play", a.eid, -1, target))
        elif a.kind == "upgrade":
            game.step(p, ("upgrade",))
        elif a.kind == "refresh":
            game.step(p, ("refresh",))
        elif a.kind == "hero_power":
            game.step(p, ("hero_power", target))
        while p.pending:                       # 发现：选和记录里新进手牌一致的那张，没有就选第一张
            ch = p.pending[0]
            want = Counter(h[0] for h in visible_log(a.after)["hand"]) - Counter(h[0] for h in visible(p)["hand"])
            idx = next((i for i, o in enumerate(ch.options) if want.get(o.card_id)), 0)
            game.step(p, ("choose", idx))
        return True
    except IllegalAction:
        return False


def candidate_targets(game: Game, p: Player, a: Action) -> list[int | None]:
    c = p.find(a.eid)
    if a.kind == "magnetize":
        return [m.uid for m in p.board if m.is_race("MECHANICAL")] or [None]
    impl = game.impl(c.card_id) if c is not None else (game.hero_power_impl(p) if a.kind == "hero_power" else None)
    spec = None
    if impl:
        spec = impl.battlecry_target if a.kind == "play" else impl.spell_target if a.kind == "cast" else \
            impl.hp_target if a.kind == "hero_power" else None
    if not spec:
        return [None]
    return game.targets(p, spec, c if isinstance(c, Minion) else None) or [None]


def snapshot(p: Player) -> dict:
    import copy
    return copy.deepcopy({"board": p.board, "hand": p.hand, "shop": p.shop, "shop_spells": p.shop_spells,
                          "counters": p.counters, "data": p.data, "gold": p.gold, "tier": p.tier,
                          "upgrade_cost": p.upgrade_cost, "extra": p.extra_gold_next, "free": p.free_refreshes,
                          "pending": []})


def restore(p: Player, snap: dict) -> None:
    import copy
    s = copy.deepcopy(snap)
    p.board, p.hand, p.shop, p.shop_spells = s["board"], s["hand"], s["shop"], s["shop_spells"]
    p.counters, p.data, p.gold, p.tier = s["counters"], s["data"], s["gold"], s["tier"]
    p.upgrade_cost, p.extra_gold_next, p.free_refreshes, p.pending = s["upgrade_cost"], s["extra"], s["free"], []


def acting_card(game: Game, a: Action, p: Player) -> str:
    if a.kind == "hero_power":
        return a.card_id or "hero_power"
    if a.kind in ("upgrade", "refresh"):
        return a.kind
    return Game.base_id(a.card_id)


def replay_game(db: dict, path: str, trials: int = 6) -> list[dict]:
    states = load_states(path)
    if len(states) < 2:
        return []
    game = Game(db, seed=0, heroes=[states[0]["player"].get("heroCardId") or "TB_BaconShop_HERO_01"] * 8,
                races=states[0].get("availableRaces") or None)
    p = game.players[0]
    carried: dict[int, Minion] = {}
    results = []
    for a in segment(states):
        game.turn = a.turn
        sync_player(game, p, a.before, carried)
        if a.kind != "hero_power" and a.eid and p.find(a.eid) is None:
            continue
        base = snapshot(p)
        real = visible_log(a.after)
        best = None
        for target in candidate_targets(game, p, a):
            for seed in range(trials if a.kind not in ("upgrade",) else 1):
                restore(p, base)
                if not run_action(game, p, a, seed, target):
                    continue
                d = diff(visible(p), real)
                score = len(d)
                if best is None or score < best[0]:
                    best = (score, d, aggregate_ok(visible(p), real), {m.uid: m for m in p.board + p.hand
                                                                       if isinstance(m, Minion)}, snapshot(p))
                if score == 0:
                    break
            if best and best[0] == 0:
                break
        if best is None:
            results.append({"file": os.path.basename(path), "turn": a.turn, "kind": a.kind,
                            "card": acting_card(game, a, p), "result": "illegal", "diff": []})
            continue
        carried.update(best[3])
        p.counters = best[4]["counters"]
        p.data = best[4]["data"]
        triggers = sorted({Game.base_id(m.card_id) for m in p.board if game.impl(m.card_id)
                           and game.impl(m.card_id).status == "done"})
        results.append({"file": os.path.basename(path), "turn": a.turn, "kind": a.kind,
                        "card": acting_card(game, a, p),
                        "result": "exact" if best[0] == 0 else "aggregate" if best[2] else "mismatch",
                        "diff": best[1][:6], "board": triggers})
    return results


def summarize(results: list[dict], db: dict) -> dict:
    cards = db["cards"]
    per = defaultdict(Counter)
    examples = defaultdict(list)
    for r in results:
        per[r["card"]][r["result"]] += 1
        if r["result"] in ("mismatch", "illegal") and len(examples[r["card"]]) < 3:
            examples[r["card"]].append(f"{r['file']} 第{r['turn']}回合 {r['kind']}：{'；'.join(r['diff'])}")
    rows = []
    for cid, c in per.items():
        n = sum(c.values())
        rows.append({"card": cid, "name": cards.get(cid, {}).get("name", cid), "n": n, **dict(c),
                     "passRate": round((c["exact"] + c["aggregate"]) / n, 3), "examples": examples[cid]})
    rows.sort(key=lambda r: (r["passRate"], -r["n"]))
    tot = Counter(r["result"] for r in results)
    return {"actions": len(results), "byResult": dict(tot), "cards": rows}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", default=[os.path.join(os.environ.get("APPDATA", ""), "HearthstoneDeckTracker",
                                                               "BGStateExporter", "games")])
    ap.add_argument("--cards", default=str(cardsmod.DEFAULT_OUT))
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    db = cardsmod.load(args.cards)
    files: list[str] = []
    for pth in args.paths:
        files += sorted(glob.glob(os.path.join(pth, "*.jsonl"))) if os.path.isdir(pth) else [pth]
    results = []
    for f in files:
        r = replay_game(db, f)
        results += r
        c = Counter(x["result"] for x in r)
        print(f"{os.path.basename(f)}：{len(r)} 个动作 {dict(c)}", flush=True)
    s = summarize(results, db)
    print(f"合计 {s['actions']} 个动作：{s['byResult']}")
    for row in s["cards"][:25]:
        print(f"  {row['name']}({row['card']}) {row['n']} 次，通过 {row['passRate']:.0%}  {row['examples'][:1]}")
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": s, "results": results}, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
