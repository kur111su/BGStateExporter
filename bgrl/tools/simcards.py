"""引擎实测的卡牌价值和卡牌配合：拿高分阵容库里的真实场面，在 TavernGym 里拿掉一张 / 两张卡再打，看胜率掉多少。

  bg.py simcards                         读 runs/pro_boards.json 和 runs/h13_boards.json，写 rl/sim_value.json（只有统计数字）
  bg.py simcards --per-turn 300          每回合多抽些场面（默认 150；越多越准、越慢，150 约 2 分钟）

对每个场面 B（第 5–13 回合、≥ 5 个随从），用 rl/boardsim.BoardJudge 量它打同期高分阵容的胜率 Φ：
  卡牌贡献  Φ(B) - Φ(B 去掉 x)                        这张卡在这个场面里值多少胜率
  配合      Φ(B) - Φ(B-x) - Φ(B-y) + Φ(B-x-y)         两张一起在场比各自单独多出来的部分（> 0 = 互相成就）
按卡牌（金色并到普通）汇总平均。和 rl/card_value.json（高手买得多不多）、rl/h13_profile.json（经常一起上场）不同，
这里量的是在这个模拟器里打起来到底强不强——训练（rl/ppo.py）和竞技场用的都是这个引擎。

训练里怎么用：--pro-weights 里的 simkey（买的是引擎实测贡献高的卡）、simsyn（买的卡和场上某张是引擎实测的好搭配），
阵容测试（bg.py goldflow --boosted）会列出模型买这些卡的倍率。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paths import ROOT, RUNS  # noqa: E402
from rl.comp import load_sim_refs  # noqa: E402,F401 —— 读 rl/sim_value.json（训练、goldflow 用）

OUT = ROOT / "rl" / "sim_value.json"


def _lab(seed: int):
    """实验局：脚本机器人打到第 2 回合（英雄都选好了），之后只用 0 号、1 号座位摆场面打。"""
    import taverngym
    g = taverngym.Game(seed=seed)
    while g.turn < 2 and not g.is_over():
        for p in range(g.num_players):
            if g.needs_action(p):
                g.step(p, g.bot_action(p))
    return g


def ablate(judge, lab, board: list, turn: int) -> dict | None:
    """一个场面的 Φ、每张卡拿掉后的 Φ、每两张拿掉后的 Φ。"""
    from rl.boardsim import put_board

    def phi(b):
        g = lab.copy()
        put_board(g, 0, b)
        return judge.judge(g, 0, turn)

    full = phi(board)
    if full is None:
        return None
    one = [phi(board[:i] + board[i + 1:]) for i in range(len(board))]
    two = {(i, j): phi([x for k, x in enumerate(board) if k not in (i, j)]) for i, j in combinations(range(len(board)), 2)}
    return {"full": full, "one": one, "two": two}


def build(lib: dict, per_turn: int = 150, turns=range(5, 14), seed: int = 0, min_pair: int = 8) -> dict:
    from rl.boardsim import BoardJudge
    from tg.catalog import base_id, catalog
    det = catalog()
    judge = BoardJudge(lib, k=16, n=4, seed=seed, lag=0)       # 只拿同一回合的打：高分场面之间胜率在一半左右，拿掉一张卡最看得出来
    rng = random.Random(seed)
    lab = _lab(seed)
    contrib, n_card, gold = defaultdict(float), defaultdict(int), defaultdict(int)
    by_turn = defaultdict(lambda: defaultdict(list))
    pair_sum, pair_n = defaultdict(float), defaultdict(int)
    t0, done, phis = time.time(), 0, defaultdict(list)
    for t in turns:
        pool = [e for e in lib.get(t, []) if len(e.get("board") or []) >= 5]
        for e in rng.sample(pool, min(per_turn, len(pool))):
            b = [list(x) for x in e["board"][:7]]
            r = ablate(judge, lab, b, t)
            if r is None:
                continue
            done += 1
            phis[t].append(r["full"])
            ids = [base_id(x[0]) for x in b]
            for i, cid in enumerate(ids):
                if r["one"][i] is None:
                    continue
                d = r["full"] - r["one"][i]
                contrib[cid] += d
                n_card[cid] += 1
                gold[cid] += int(str(b[i][0]).endswith("_G"))
                by_turn[cid][t].append(d)
            for (i, j), v in r["two"].items():
                if v is None or r["one"][i] is None or r["one"][j] is None or ids[i] == ids[j]:
                    continue
                key = tuple(sorted((ids[i], ids[j])))
                pair_sum[key] += r["full"] - r["one"][i] - r["one"][j] + v
                pair_n[key] += 1
        print(f"  第 {t} 回合：{len(phis[t])} 个场面，平均 Φ {sum(phis[t]) / max(1, len(phis[t])):.2f}，"
              f"已用 {time.time() - t0:.0f} 秒", flush=True)
    name = lambda x: (det.get(x) or {}).get("name_zh", x)  # noqa: E731
    cards = {cid: {"name": name(cid), "tier": (det.get(cid) or {}).get("tier"), "n": n_card[cid],
                   "contrib": round(contrib[cid] / n_card[cid], 4), "golden": round(gold[cid] / n_card[cid], 3),
                   "by_turn": {str(t): round(sum(v) / len(v), 4) for t, v in sorted(by_turn[cid].items()) if len(v) >= 5}}
             for cid in n_card if n_card[cid] >= 5}
    # 关键卡：至少出现 15 次，贡献在同一酒馆等级里排前 25%，而且绝对值 ≥ 0.05（拿掉它胜率掉 5 个百分点以上）
    tiers = defaultdict(list)
    for cid, r in cards.items():
        if r["n"] >= 15:
            tiers[r["tier"]].append(r["contrib"])
    cut = {t: sorted(v)[int(len(v) * 0.75)] for t, v in tiers.items() if v}
    key = sorted((cid for cid, r in cards.items() if r["n"] >= 15 and r["contrib"] >= max(0.05, cut.get(r["tier"], 1))),
                 key=lambda k: -cards[k]["contrib"])
    pairs = sorted(([round(pair_sum[k] / pair_n[k], 4), pair_n[k], name(k[0]), name(k[1]), k[0], k[1]]
                    for k in pair_sum if pair_n[k] >= min_pair), reverse=True)
    return {"source": "高分阵容库（runs/pro_boards.json 6500+、runs/h13_boards.json 万三）的场面在 TavernGym 里拿掉卡再打",
            "created": time.strftime("%Y-%m-%d"), "boards": done, "per_turn_phi": {str(t): round(sum(v) / len(v), 3)
                                                                                for t, v in phis.items() if v},
            "judge": {"boards": judge.k, "fights": judge.n, "lag": judge.lag},
            "key": key, "pairs": [p for p in pairs if p[0] > 0][:60], "anti_pairs": [p for p in pairs if p[0] < 0][-30:],
            "cards": cards}


def main() -> int:
    from rl.calibrate import merge_pro_boards
    ap = argparse.ArgumentParser()
    ap.add_argument("--boards", default=f"{RUNS / 'pro_boards.json'};{RUNS / 'h13_boards.json'}")
    ap.add_argument("--per-turn", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    paths = [Path(p) for p in a.boards.split(";") if p.strip()]
    lib = merge_pro_boards(p for p in paths if p.exists())
    if not lib:
        print(f"没有高分阵容库：{a.boards}（先运行 bg.py prolib / bg.py nativedemos）")
        return 1
    t0 = time.time()
    v = build(lib, a.per_turn, seed=a.seed)
    Path(a.out).write_text(json.dumps(v, ensure_ascii=False, indent=1), encoding="utf-8")
    cs = v["cards"]
    print(f"{v['boards']} 个场面，{time.time() - t0:.0f} 秒 → {a.out}")
    print(f"关键卡 {len(v['key'])} 张：" + "、".join(f"{cs[k]['name']}（{cs[k]['tier']} 本）+{cs[k]['contrib']:.2f}"
                                               for k in v["key"][:25]))
    print("最好的配合：" + "、".join(f"{p[2]}+{p[3]} +{p[0]:.2f}（{p[1]}）" for p in v["pairs"][:15]))
    weak = sorted((k for k, r in cs.items() if r["n"] >= 30), key=lambda k: cs[k]["contrib"])[:12]
    print("出现很多但几乎没贡献：" + "、".join(f"{cs[k]['name']} {cs[k]['contrib']:+.2f}" for k in weak))
    return 0


if __name__ == "__main__":
    sys.exit(main())
