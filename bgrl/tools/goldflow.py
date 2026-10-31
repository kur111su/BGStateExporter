"""前几回合的金币流向：模型 vs 6500+ 高手。

每回合：开局金币、花在 升本 / 买 / 刷新 / 发动（含选目标）/ 黑暗之赐 / 英雄技能 / 其它 上的钱、卖怪收入、
回合末剩下的钱、回合末等级、升本率、“回合开始时钱够升本”的回合里升了本的比例。

  bg.py goldflow                        用 runs/v3/latest.pt 在 TavernGym 里打 64 局（0 号座位是模型，其余脚本机器人），
                                        和 rl/pro_profile.json 里的高手数字并排打印，结果写 runs/<run>/goldflow.json
  bg.py goldflow --ckpt X --games 128 --sample
  bg.py goldflow --pro data\\firestone   从高分回放算高手那一列，写进 rl/pro_profile.json 的 gold_flow（只有汇总数字）

模型这边按竞技场的方式选动作（最大概率，--sample 改成按概率抽）。金币、等级、升本费直接读观察（第 7 / 10 / 11 个数）。
高手那边用回放里每一步前后的金币差（tg.logs.segment）；黑暗之赐在回放里看不到是哪一步花的，算在“其它”里。
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paths import ROOT, RUNS  # noqa: E402
from tg.actions import (A_ACTIVATE, A_BUY, A_CANCEL, A_CHOOSE, A_DARK_GIFT, A_END, A_FREEZE, A_HERO_POWER,  # noqa: E402
                        A_MAGNETIZE, A_MOVE, A_PLAY, A_REFRESH, A_SELL, A_UPGRADE)

F_GOLD, F_TIER, F_UPCOST = 7, 10, 11
PROFILE = ROOT / "rl" / "pro_profile.json"
COLS = ("upgrade", "buy", "refresh", "activate", "dark_gift", "hero_power", "other")
ZH = {"upgrade": "升本", "buy": "买", "refresh": "刷新", "activate": "发动", "dark_gift": "黑暗之赐",
      "hero_power": "技能", "other": "其它"}


def kind_of(a: int) -> str:
    if a == A_END:
        return "end"
    if a == A_REFRESH:
        return "refresh"
    if a == A_UPGRADE:
        return "upgrade"
    if a in (A_HERO_POWER, A_HERO_POWER + 1):
        return "hero_power"
    if a == A_DARK_GIFT:
        return "dark_gift"
    if a in (A_FREEZE, A_CANCEL) or a >= A_MOVE:
        return "noop"
    if A_BUY <= a < A_SELL:
        return "buy"
    if A_SELL <= a < A_ACTIVATE:
        return "sell"
    if A_ACTIVATE <= a < A_CHOOSE:
        return "activate"
    if A_CHOOSE <= a < A_PLAY:
        return "choose"
    if A_PLAY <= a < A_MAGNETIZE:
        return "play"
    return "magnetize"


class TurnLog:
    """一个槽位（一局）的逐回合记录。选项（发现 / 目标）里花的钱记在发起它的那一步上（例如“发动 → 选目标”记在发动上）。"""

    def __init__(self):
        self.turns: dict[int, dict] = {}
        self.prev = None                     # (回合, 金币, 等级, 记账类别)
        self.src = "other"

    def see(self, turn: int, gold: int, tier: int, up: int, a: int):
        d = self.turns.get(turn)
        if d is None:
            d = self.turns[turn] = {"gold": gold, "tier0": tier, "afford": int(up > 0 and gold >= up),
                                    "spend": Counter(), "income": 0, "left": gold, "tier": tier}
        if self.prev and self.prev[0] == turn:
            diff = self.prev[1] - gold
            if diff > 0:
                d["spend"][self.prev[3]] += diff
            elif diff < 0:
                d["income"] += -diff
        d["left"], d["tier"] = gold, tier
        k = kind_of(a)
        if k != "choose":
            self.src = k if k in COLS else "other"
        self.prev = (turn, gold, tier, self.src)


def _mean(rows, f):
    return round(st.mean([f(r) for r in rows]), 2) if rows else None


def summarize(games: list[dict], max_turn: int = 10) -> dict:
    out = {}
    for t in range(1, max_turn + 1):
        rows = [g[t] for g in games if t in g]
        if len(rows) < 3:
            continue
        aff = [r for r in rows if r["afford"]]
        out[str(t)] = {"n": len(rows), "gold": _mean(rows, lambda r: r["gold"]),
                       **{k: _mean(rows, lambda r, k=k: r["spend"].get(k, 0)) for k in COLS},
                       "income": _mean(rows, lambda r: r["income"]), "left": _mean(rows, lambda r: r["left"]),
                       "tier": _mean(rows, lambda r: r["tier"]),
                       "up": round(st.mean([r["tier"] > r["tier0"] for r in rows]), 2),
                       "afford": round(len(aff) / len(rows), 2),
                       "up_afford": round(st.mean([r["tier"] > r["tier0"] for r in aff]), 2) if aff else None}
        if any("gift" in r for r in rows):
            out[str(t)]["gift"] = round(st.mean([bool(r.get("gift")) for r in rows]), 2)
    return out


# ---------------- 模型 ----------------

def model_policy(ckpt: str, device: str, greedy: bool):
    """检查点 → (选动作的函数 act(features, cards, mask) -> 动作数组, 检查点信息)。"""
    import torch
    from rl.advisor import load_model
    model, info = load_model(ckpt, device)

    def act(f, c, m):
        with torch.no_grad():
            mm = torch.as_tensor(m, device=device)
            lg = model(torch.as_tensor(f, dtype=torch.float32, device=device),
                       torch.as_tensor(c, dtype=torch.long, device=device), mm)[0].float()
            lg = lg.masked_fill(~mm, float("-inf"))
            a = lg.argmax(-1) if greedy else torch.distributions.Categorical(logits=lg).sample()
        return a.cpu().numpy()
    info["name"] = Path(ckpt).name
    return act, info


def play_model(ckpt: str, games: int, seed: int, threads: int, greedy: bool, device: str, boosted: bool = False,
               policy=None):
    """boosted：7 个机器人从第 5 回合起每 2 回合换成高分对局里同一回合的真实阵容（runs/pro_boards.json +
    runs/h13_boards.json，和训练时的考试 eval_boost_place 一样，全强度）。policy：测试用，代替检查点。"""
    import time as _t
    from taverngym import BatchEnv
    from tools.cardvalue import PickStats
    print(f"读模型 {ckpt} …", flush=True)
    act, info = policy if policy is not None else model_policy(ckpt, device, greedy)
    env = BatchEnv(games, seed=seed, controlled=(0,), threads=threads)
    booster = None
    if boosted:
        from rl.calibrate import ProBoardBooster, load_pro_boards
        lib: dict = {}
        for pth in (RUNS / "pro_boards.json", RUNS / "h13_boards.json"):
            if pth.exists():
                for t, v in load_pro_boards(pth).items():
                    lib.setdefault(t, []).extend(v)
        if lib:
            booster = ProBoardBooster(lib, frac=1.0, start_turn=5, seed=seed, every=2)
            booster.strength = 1.0
            print(f"对手：高分阵容 {sum(len(v) for v in lib.values())} 个（第 5 回合起，全强度）", flush=True)
        else:
            print("没有 runs/pro_boards.json / runs/h13_boards.json，对手是普通脚本机器人", flush=True)
    picks = PickStats()
    sells: dict[str, int] = {}
    logs = [TurnLog() for _ in range(env.n_slots)]
    done = [False] * env.n_slots
    places = {}
    last_cards = [None] * env.n_slots                  # 每局最后一次招募时的场面（最终阵容）
    obs = env.observe()
    t0, shown = _t.time(), 0
    print(f"开始打 {games} 局（每打完 8 局报一次进度）…", flush=True)
    for _ in range(200000):
        if all(done):
            break
        ready = [i for i in np.flatnonzero(obs["ready"]) if not done[i]]
        # 已经打完的槽位会自动开新局：随便给个合法动作让它走（不算数）
        actions = obs["action_mask"].astype(bool).argmax(1).astype(np.int64)
        if ready:
            f = obs["features"][ready]
            m = obs["action_mask"][ready].astype(bool)
            a = np.asarray(act(f, obs["cards"][ready], m))
            for k, i in enumerate(ready):
                fi = f[k]
                last_cards[i] = obs["cards"][i][19:26].copy()
                ak = int(a[k])
                if 8 <= ak < 16:
                    picks.add(obs["cards"][i][36:44], ak - 8)
                elif 16 <= ak < 23 and obs["cards"][i][19 + ak - 16] > 0:
                    sid = picks._id(int(obs["cards"][i][19 + ak - 16]))
                    sells[sid] = sells.get(sid, 0) + 1
                logs[i].see(int(round(fi[0] * 20)), int(round(fi[F_GOLD] * 10)), int(round(fi[F_TIER] * 6)),
                            int(round(fi[F_UPCOST] * 10)), int(a[k]))
            actions[ready] = a
        _, dones, inf = env.step(actions)
        for i in np.flatnonzero(dones):
            if not done[i]:
                done[i] = True
                places[int(i)] = int(inf["placement"][i])
        if len(places) >= shown + 8 or (len(places) == games and shown < games):
            shown = len(places)
            ps = [p for p in places.values() if p > 0]
            el = _t.time() - t0
            print(f"  {shown}/{games} 局，平均名次 {np.mean(ps):.2f}，{el / 60:.1f} 分钟"
                  + (f"，预计还要 {el / shown * (games - shown) / 60:.0f} 分钟" if shown < games else ""), flush=True)
        obs = env.observe()
        if booster is not None and booster.boost(env, obs["features"]):
            obs = env.observe()
    print(f"模型 {info.get('iter')} 轮（{info.get('name', Path(ckpt).name)}）：{len(places)} 局，平均名次 "
          f"{np.mean([p for p in places.values() if p > 0]):.2f}（{'对高分阵容' if booster else '对脚本机器人'}）", flush=True)
    info["picks"], info["sells"] = picks, sells
    import taverngym
    finals = []
    for i, row in enumerate(last_cards):
        if row is None:
            continue
        ids = [taverngym.card_info(int(x))[0] for x in row if x > 0]
        finals.append({"board": [x[:-2] if x.endswith("_G") else x for x in ids], "golden": sum(x.endswith("_G") for x in ids),
                       "place": places.get(i, 0)})
    info["finals"] = finals
    return [x.turns for x in logs], info


# ---------------- 高手（回放） ----------------

def _pro_one(job):
    from tg.hsreplay import parse
    from tg.logs import load_records, segment
    from tools.make_pro_demos import _xml
    try:
        states = load_records(parse(_xml(job)), "pro")
        acts = segment(states)
    except Exception:                                   # noqa: BLE001
        return None
    T = {}
    for s in states:
        t = int(s["turn"])
        p = s["player"]
        sh = p.get("shop") or {}
        if t not in T:
            uc = sh.get("upgradeCost")
            T[t] = {"gold": p["gold"], "tier0": p["tavernTier"], "spend": Counter(), "income": 0,
                    "afford": int(bool(sh.get("canUpgrade")) and uc is not None and p["gold"] >= uc), "gift": False}
        T[t]["left"], T[t]["tier"] = p["gold"], p["tavernTier"]
        if (s.get("_dark_gifts") or {}).get("used_turn"):
            T[t]["gift"] = True
    for a in acts:
        d = T.get(int(a.turn))
        if d is None:
            continue
        g0 = a.before["player"]["gold"]
        g1 = (a.after or a.before)["player"]["gold"]
        k = a.kind if a.kind in COLS else "other"
        if g0 > g1:
            d["spend"][k] += g0 - g1
        elif g1 > g0:
            d["income"] += g1 - g0
    for d in T.values():                                # 回放里看不到是哪一步花的钱（主要是黑暗之赐）算“其它”
        miss = d["gold"] + d["income"] - d["left"] - sum(d["spend"].values())
        if miss > 0:
            d["spend"]["other"] += miss
    return T


def pro_table(src: Path) -> dict:
    from multiprocessing import Pool
    from tools.make_pro_demos import _jobs
    jobs = list(_jobs(src))
    if not jobs:
        raise SystemExit(f"{src} 里没有回放")
    with Pool() as pool:
        res = [r for r in pool.map(_pro_one, jobs, chunksize=4) if r]
    return {"games": len(res), "created": time.strftime("%Y-%m-%d"), "turns": summarize(res)}


# ---------------- 打印 ----------------

def show(model: dict, pro: dict | None, upto: int = 9):
    hdr = "回合 | 谁   | 金币 | " + " ".join(f"{ZH[k]:>4}" for k in COLS) + " | 卖入 | 剩 | 回合末等级 | 升本率 | 钱够时升本"
    print(hdr)
    print("-" * len(hdr.encode("gbk", "replace")))
    for t in range(1, upto + 1):
        for who, tab in (("模型", model), ("高手", (pro or {}).get("turns") or {})):
            r = tab.get(str(t))
            if not r:
                continue
            ua = f"{r['up_afford'] * 100:.0f}%" if r.get("up_afford") is not None else "-"
            print(f"{t:>4} | {who} | {r['gold']:>4} | " + " ".join(f"{(r.get(k) or 0):>6.2f}" for k in COLS) +
                  f" | {r['income']:>4} | {r['left']:>3} | {r['tier']:>10} | {r['up'] * 100:>5.0f}% | {ua:>8}")
    if pro:
        print("（高手一列：回放里认不出“发动”和黑暗之赐是哪一步花的钱，都算在“其它”里）")


def show_comps(finals: list[dict]) -> dict | None:
    """模型的最终阵容（场上 ≥ 5 个随从的局）和 rl/h13_profile.json（万三分段）对比。"""
    from tools.native_demos import ZH_RACE, comp_stats
    fb = [x for x in finals if len(x["board"]) >= 5]
    if not fb:
        return None
    m = comp_stats([x["board"] for x in fb], [x["golden"] for x in fb], [])
    hp = ROOT / "rl" / "h13_profile.json"
    h = json.loads(hp.read_text(encoding="utf-8"))["final"] if hp.exists() else None
    print(f"\n最终阵容（场上 ≥5 个随从的 {m['n']} 局）：金色 {m['golden']} 个" + (f"（万三 {h['golden']}）" if h else ""))
    fmt = lambda lst: "、".join(f"{ZH_RACE.get(k, k)} {v:.0%}" for k, v in lst[:6])   # noqa: E731
    print("  模型主种族：", fmt(m["main"]))
    if h:
        print("  万三主种族：", fmt(h["main"]))
    print("  模型常用卡：", "、".join(f"{n} {v:.0%}" for _, n, v in m["cards"][:12]))
    if h:
        print("  万三常用卡：", "、".join(f"{n} {v:.0%}" for _, n, v in h["cards"][:12]))
    # 配合程度（tools/native_demos.board_scores）
    import taverngym
    from tools.native_demos import board_scores, load_comp_refs
    pairs, key = load_comp_refs()
    det = {c["id"]: c for c in json.loads(taverngym.catalog_json())}
    sc = [board_scores(x["board"], det, pairs, key) for x in fb]
    m["scores"] = {k: round(float(np.mean([r[k] for r in sc])), 3) for k in ("coh", "syn", "key")}
    hs = (h or {}).get("scores") or {}
    print(f"  配合：主种族纯度 {m['scores']['coh']:.2f}（万三 {hs.get('coh', '-')}）、高手常见搭配 {m['scores']['syn']:.2f} 对"
          f"（{hs.get('syn', '-')}）、关键卡 {m['scores']['key']:.2f} 张（{hs.get('key', '-')}）")
    by_place = {}
    for x, r in zip(fb, sc):
        by_place.setdefault("前四" if 0 < x["place"] <= 4 else "后四", []).append(r)
    for k, rows in sorted(by_place.items()):
        print(f"  {k}名的局（{len(rows)}）：纯度 {np.mean([r['coh'] for r in rows]):.2f}、搭配 {np.mean([r['syn'] for r in rows]):.2f}、"
              f"关键卡 {np.mean([r['key'] for r in rows]):.2f}")
    return m


def show_picks(picks, sells: dict) -> dict | None:
    """模型买卡的挑中倍率和高手比（rl/card_value.json）：高手抢着买、模型不买的卡；模型偏爱、高手基本不买的卡；经济卡。"""
    cv = ROOT / "rl" / "card_value.json"
    if not cv.exists() or not picks.n:
        return None
    ref = json.loads(cv.read_text(encoding="utf-8"))
    hc = ref["cards"]
    mt = picks.table(15)
    both = [(k, mt[k]["lift"], hc[k]["lift"]) for k in mt if k in hc]
    if not both:
        return None
    name = lambda k: hc[k]["name"]                          # noqa: E731
    miss = sorted((x for x in both if x[2] >= 1.6), key=lambda x: x[1] - x[2])[:12]
    over = sorted((x for x in both if x[2] < 0.8), key=lambda x: x[2] - x[1])[:10]
    print(f"\n挑卡（模型买了 {picks.n} 次；倍率 = 实际买的次数 / 随便挑的期望，>1 是偏爱）")
    print("  高手抢着买、模型买得少：" + "、".join(f"{name(k)} {m:.1f}（高手 {p:.1f}）" for k, m, p in miss))
    print("  模型偏爱、高手很少买：" + "、".join(f"{name(k)} {m:.1f}（高手 {p:.1f}）" for k, m, p in over))
    eco = [k for k in ref.get("eco", []) if k in mt]
    if eco:
        print("  经济卡：" + "、".join(f"{name(k)} {mt[k]['lift']:.1f}（高手 {hc[k]['lift']:.1f}）" for k in eco))
    keyset = set(ref.get("key", []))
    kb = sum(mt[k]["picked"] for k in mt if k in keyset)
    print(f"  买的卡里关键卡占 {kb / max(1, picks.n):.0%}")
    corr = float(np.corrcoef([x[1] for x in both], [x[2] for x in both])[0, 1]) if len(both) > 2 else None
    if corr is not None:
        print(f"  和高手挑卡倍率的相关系数 {corr:.2f}（1 = 完全一样的偏好）")
    top_sold = sorted(sells.items(), key=lambda x: -x[1])[:10]
    if top_sold:
        print("  模型最常卖：" + "、".join(f"{(hc.get(k) or {}).get('name', k)} {v}" for k, v in top_sold))
    return {"key_share": round(kb / max(1, picks.n), 3), "corr": round(corr, 3) if corr is not None else None,
            "miss": [[name(k), m, p] for k, m, p in miss], "over": [[name(k), m, p] for k, m, p in over]}


def advice(model: dict, pro: dict | None) -> list[str]:
    """差距最大的几条，用一句话说。"""
    p = (pro or {}).get("turns") or {}
    out = []
    for t in ("2", "4", "5", "6", "7"):
        m, h = model.get(t), p.get(t)
        if not m or not h:
            continue
        if h["up"] - m["up"] >= 0.25:
            spent = sorted(((m.get(k) or 0, k) for k in COLS if k != "upgrade"), reverse=True)[:2]
            out.append(f"第 {t} 回合升本率 {m['up']:.0%}（高手 {h['up']:.0%}）；钱主要花在 "
                       + "、".join(f"{ZH[k]} {v:.1f}" for v, k in spent if v > 0))
        if (m.get("refresh") or 0) - (h.get("refresh") or 0) >= 1:
            out.append(f"第 {t} 回合刷新花 {m['refresh']:.1f}（高手 {h['refresh']:.1f}）")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(RUNS / "v3" / "latest.pt"))
    ap.add_argument("--games", type=int, default=64)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="")
    ap.add_argument("--sample", action="store_true", help="按概率抽动作（默认和竞技场一样取最大概率）")
    ap.add_argument("--pro", default="", help="高分回放目录：算高手那一列写进 rl/pro_profile.json（gold_flow）")
    ap.add_argument("--turns", type=int, default=9)
    ap.add_argument("--boosted", action="store_true", help="对手换成高分对局的真实阵容（和训练考试 eval_boost_place 一样）")
    a = ap.parse_args()
    prof = json.loads(PROFILE.read_text(encoding="utf-8")) if PROFILE.exists() else {}
    if a.pro:
        prof["gold_flow"] = pro_table(Path(a.pro))
        PROFILE.write_text(json.dumps(prof, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"高手金币流向：{prof['gold_flow']['games']} 局 → {PROFILE}")
        show({}, prof["gold_flow"], a.turns)
        return 0
    if not Path(a.ckpt).exists():
        print(f"没有检查点 {a.ckpt}")
        return 1
    dev = a.device
    if not dev:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    games, info = play_model(a.ckpt, a.games, a.seed, a.threads, not a.sample, dev, a.boosted)
    model = summarize(games)
    pro = prof.get("gold_flow")
    show(model, pro, a.turns)
    tips = advice(model, pro)
    if tips:
        print("\n差距最大的地方：")
        for x in tips:
            print("  ·", x)
    if not pro:
        print("\n（rl/pro_profile.json 里还没有高手的金币流向：bg.py goldflow --pro <高分回放目录>）")
    comps = show_comps(info.get("finals") or [])
    pk = show_picks(info.get("picks"), info.get("sells") or {})
    out = Path(a.ckpt).resolve().parent / ("comptest.json" if a.boosted else "goldflow.json")
    out.write_text(json.dumps({"ckpt": str(a.ckpt), "iter": info.get("iter"), "games": a.games,
                               "greedy": not a.sample, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                               "boosted": a.boosted, "turns": model, "final": comps, "picks": pk}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
