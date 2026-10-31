"""别人的对局（已经是 TavernGym 观察格式的 .npz：features / cards / actions / game_idx / turn / synthetic）
→ 和高分回放一样的决策样本分片（runs/h13_demos/）+ 对手阵容库（runs/h13_boards.json）+ 汇总报告。

  bg.py nativedemos <文件.npz> [--mmr 13000] [--out runs\\h13_demos] [--boards runs\\h13_boards.json]

数据的情况（human_native_v0.npz，万三分段，503 局）：
  · 卡牌编号、观察布局和我们的 TavernGym 一致（酒馆随从的等级特征和卡牌等级 99.8% 对得上）。
  · 没有合法动作表：用 tg/supermask.py 从观察推一个超集。
  · 44% 的局面 turn = -1：回合数和金币都没写（观察里回合 = 1、金币 = 0），不能用 → 跳过。
  · “选择 / 黑暗之赐 / 英雄技能”大多和观察对不上（没有待选项、没有技能卡、黑暗之赐不可用），自己换到自己的
    “换位”也不合法 → 动作不在推出的集合里的都跳过。
留出每 10 局里的 1 局做验证（split = 1）。样本的 mmr 字段 = --mmr，训练日志里按它分开看命中率（pro_top1_hi）。
对手阵容：每局每个已知回合最后一个局面的场面（卡牌、攻 / 血、金色）和当时的酒馆等级，格式同 runs/pro_boards.json。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paths import RUNS  # noqa: E402
from tg.actions import (A_ACTIVATE, A_BUY, A_CANCEL, A_CHOOSE, A_DARK_GIFT, A_END, A_FREEZE, A_HERO_POWER,  # noqa: E402
                        A_MAGNETIZE, A_MOVE, A_PLAY, A_REFRESH, A_SELL, A_UPGRADE)
from tg.supermask import C_BOARD, C_HAND, NB, super_mask  # noqa: E402

MF, F_BOARD = 31, 68                 # 场上随从：每格 31 个浮点，从第 68 个开始；格内第 1/2 个 = 攻/10、血/10
SHARD = 4000


def kind_of(a: int, hand_card_kind: str = "") -> str | None:
    if a == A_END:
        return "end"
    if a == A_REFRESH:
        return "refresh"
    if a == A_FREEZE:
        return "freeze"
    if a == A_UPGRADE:
        return "upgrade"
    if a in (A_HERO_POWER, A_HERO_POWER + 1):
        return "hero_power"
    if a == A_DARK_GIFT:
        return "dark_gift"
    if a == A_CANCEL or A_CHOOSE <= a < A_PLAY:
        return None                                      # 示范类别里没有（KINDS）
    if A_BUY <= a < A_SELL:
        return "buy"
    if A_SELL <= a < A_ACTIVATE:
        return "sell"
    if A_ACTIVATE <= a < A_CHOOSE:
        return "activate"
    if A_PLAY <= a < A_MAGNETIZE:
        return "cast" if "spell" in hand_card_kind else "play"
    if A_MAGNETIZE <= a < A_MOVE:
        return "magnetize"
    return "move"


def convert(src: Path, mmr: int, val_every: int = 10):
    import taverngym
    from tools.make_demos import KINDS, MAX_CANDS
    from tools.make_pro_demos import soft_log_np        # 不需要 torch
    z = np.load(src, allow_pickle=False)
    f = z["features"].astype(np.float32)
    c = z["cards"].astype(np.int64)
    a = z["actions"].astype(np.int64)
    gi = z["game_idx"].astype(np.int64)
    T = z["turn"].astype(np.int64)
    n = len(a)
    if f.shape[1] != taverngym.OBS_FLOATS or c.shape[1] != taverngym.OBS_CARDS:
        raise SystemExit(f"观察长度 {f.shape[1]}/{c.shape[1]}，当前 TavernGym 是 {taverngym.OBS_FLOATS}/{taverngym.OBS_CARDS}")
    info: dict[int, tuple] = {}
    det = {x["id"]: x for x in json.loads(taverngym.catalog_json())}

    def card(i: int) -> tuple:
        if i not in info:
            info[i] = taverngym.card_info(int(i))
        return info[i]

    stats = Counter()
    m = super_mask(f, c)
    known = T > 0
    in_set = m[np.arange(n), np.clip(a, 0, m.shape[1] - 1)]
    keep, kinds = [], []
    for i in range(n):
        if not known[i]:
            stats["跳过：回合未知"] += 1
            continue
        if not in_set[i]:
            stats["跳过：动作和观察对不上"] += 1
            continue
        hk = ""
        if A_PLAY <= a[i] < A_MAGNETIZE:
            hc = int(c[i, C_HAND + (a[i] - A_PLAY) // 8])
            hk = (det.get(card(hc)[0]) or {}).get("kind", "") if hc > 0 else ""
        k = kind_of(int(a[i]), hk)
        if k is None:
            stats["跳过：选择 / 取消"] += 1
            continue
        keep.append(i)
        kinds.append(KINDS.index(k))
        stats[f"样本 {k}"] += 1
    keep = np.array(keep)
    cand = np.full((len(keep), MAX_CANDS), -1, dtype=np.int16)
    cand[:, 0] = a[keep]
    turn = ((T[keep] + 1) // 2).astype(np.int16)
    games = np.unique(gi)
    order = {g: k for k, g in enumerate(games)}
    split = np.array([1 if val_every and order[g] % val_every == val_every - 1 else 0 for g in gi[keep]], dtype=np.int8)
    data = {"f": soft_log_np(f[keep]).astype(np.float16), "c": c[keep].astype(np.int32),
            "m": np.packbits(m[keep], axis=1), "cand": cand, "kind": np.array(kinds, dtype=np.int8), "turn": turn,
            "split": split, "mmr": np.full(len(keep), mmr, np.int32)}
    # 对手阵容库：每局每个已知回合最后一个局面
    last = {}
    for i in np.flatnonzero(known):
        last[(int(gi[i]), int((T[i] + 1) // 2))] = i
    boards = defaultdict(list)
    for (g, t), i in sorted(last.items()):
        b = []
        for s in range(NB):
            x = int(c[i, C_BOARD + s])
            if x <= 0:
                continue
            cid = card(x)[0]
            if (det.get(cid[:-2] if cid.endswith("_G") else cid) or {}).get("kind") != "minion":
                continue
            atk = int(round(f[i, F_BOARD + MF * s + 1] * 10))
            hp = int(round(f[i, F_BOARD + MF * s + 2] * 10))
            b.append([cid, atk, max(1, hp)])
        if b:
            boards[t].append({"tier": int(round(f[i, 10] * 6)), "mmr": mmr, "board": b})
    # 汇总（只有统计数字）：每回合开始时的等级、本回合升本率；最终阵容；常一起上场的卡
    first, upg = {}, defaultdict(int)
    for i in np.flatnonzero(known):
        key = (int(gi[i]), int((T[i] + 1) // 2))
        first.setdefault(key, i)
        upg[key] |= int(a[i] == A_UPGRADE)
    per = defaultdict(lambda: ([], []))
    for key, i in first.items():
        per[key[1]][0].append(int(round(f[i, 10] * 6)))
        per[key[1]][1].append(upg[key])
    curve = {str(t): {"n": len(v[0]), "tier_start": round(float(np.mean(v[0])), 2), "up": round(float(np.mean(v[1])), 2)}
             for t, v in sorted(per.items()) if len(v[0]) >= 10}
    last_row = {}
    for i in range(n):
        last_row[int(gi[i])] = i
    fb, gl, tl = [], [], []
    for i in last_row.values():
        ids = []
        g_ = 0
        for s_ in range(NB):
            x = int(c[i, C_BOARD + s_])
            if x > 0:
                cid = card(x)[0]
                g_ += cid.endswith("_G")
                ids.append(cid[:-2] if cid.endswith("_G") else cid)
        if len(ids) >= 5:
            fb.append(ids)
            gl.append(g_)
            tl.append(int(round(f[i, 10] * 6)))
    profile = {"source": "他人对局（万三分段，TavernGym 观察格式）", "games": len(games), "mmr": mmr,
               "created": time.strftime("%Y-%m-%d"), "tempo": curve, "final": comp_stats(fb, gl, tl, det),
               "synergy": synergy(fb, det)}
    pairs = {tuple(sorted((r[4], r[5]))) for r in profile["synergy"] if r[0] >= 3}
    key = load_comp_refs()[1]
    sc = [board_scores(ids, det, pairs, key) for ids in fb]
    profile["final"]["scores"] = {k: round(float(np.mean([x[k] for x in sc])), 3) for k in ("coh", "syn", "key")} if sc else {}
    return data, boards, stats, len(games), profile


ZH_RACE = {"QUILBOAR": "野猪人", "UNDEAD": "亡灵", "BEAST": "野兽", "MURLOC": "鱼人", "DEMON": "恶魔", "MECHANICAL": "机械",
           "ELEMENTAL": "元素", "PIRATE": "海盗", "DRAGON": "龙", "NAGA": "纳迦", "ABERRATION": "畸变", "混合": "混合"}


def _catalog():
    import taverngym
    return {x["id"]: x for x in json.loads(taverngym.catalog_json())}


def comp_stats(boards: list[list[str]], goldens: list[int], tiers: list[int], det: dict | None = None) -> dict:
    """最终阵容汇总：boards = 每局最后的场面（卡牌 ID，金色已去掉 _G）。主种族：同一种族（含全种族）≥ 4 个。"""
    det = det or _catalog()
    main, cards = Counter(), Counter()
    for ids in boards:
        for x in set(ids):
            cards[x] += 1
        rc = Counter(r for x in ids for r in (det.get(x) or {}).get("races", []) if r != "ALL")
        allr = sum(1 for x in ids if "ALL" in (det.get(x) or {}).get("races", []))
        top = rc.most_common(1)
        main[top[0][0] if top and top[0][1] + allr >= 4 else "混合"] += 1
    n = max(1, len(boards))
    return {"n": len(boards), "tier": round(float(np.mean(tiers)), 2) if tiers else None,
            "golden": round(float(np.mean(goldens)), 2) if goldens else None,
            "main": [[k, round(v / n, 3)] for k, v in main.most_common()],
            "cards": [[x, (det.get(x) or {}).get("name_zh", x), round(v / n, 3)] for x, v in cards.most_common(40)]}


def synergy(boards: list[list[str]], det: dict, min_games: int = 12, top: int = 40) -> list:
    """经常一起上场的两张卡（提升度 = 一起出现的局数 / 各自独立时的期望）。"""
    from itertools import combinations
    one, two = Counter(), Counter()
    for ids in boards:
        u = sorted(set(ids))
        one.update(u)
        two.update(combinations(u, 2))
    n = len(boards)
    out = [[round(k * n / (one[x] * one[y]), 2), k, (det.get(x) or {}).get("name_zh", x), (det.get(y) or {}).get("name_zh", y), x, y]
           for (x, y), k in two.items() if k >= min_games]
    return sorted(out, reverse=True)[:top]


def board_scores(ids: list[str], det: dict, pairs: set, key: set) -> dict:
    """一个最终阵容的配合程度：
      coh   主种族纯度 = 最多的那个种族（全种族随从都算）的随从数 / 场上随从数
      syn   场上两两组合里，属于高手常见搭配（synergy 里提升度 ≥ 3）的有几对
      key   场上关键卡（rl/card_value.json 的 key）有几张"""
    from itertools import combinations
    races = Counter()
    for x in ids:
        rs = (det.get(x) or {}).get("races", [])
        for r in ({r for r in rs if r != "ALL"} if "ALL" not in rs else set(ZH_RACE) - {"混合"}):
            races[r] += 1
    u = sorted(set(ids))
    return {"coh": round(max(races.values()) / len(ids), 3) if races and ids else 0.0,
            "syn": sum(1 for x, y in combinations(u, 2) if (x, y) in pairs),
            "key": sum(1 for x in ids if x in key)}


def load_comp_refs() -> tuple[set, set]:
    """(高手常见搭配 {(id, id)}, 关键卡 {id})：rl/h13_profile.json 和 rl/card_value.json。"""
    root = HERE.parent / "rl"
    pairs, key = set(), set()
    hp, cv = root / "h13_profile.json", root / "card_value.json"
    if hp.exists():
        for row in json.loads(hp.read_text(encoding="utf-8")).get("synergy", []):
            if len(row) >= 6 and row[0] >= 3:
                pairs.add(tuple(sorted((row[4], row[5]))))
    if cv.exists():
        key = set(json.loads(cv.read_text(encoding="utf-8")).get("key", []))
    return pairs, key


def merge_boards(paths: list[str]) -> dict:
    """几个阵容库合在一起（训练时 --pro-boards "a;b" 用）。"""
    out: dict[int, list] = defaultdict(list)
    from rl.calibrate import load_pro_boards
    for p in paths:
        for t, v in load_pro_boards(p).items():
            out[t] += v
    return dict(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--mmr", type=int, default=13000)
    ap.add_argument("--out", default=str(RUNS / "h13_demos"))
    ap.add_argument("--boards", default=str(RUNS / "h13_boards.json"))
    ap.add_argument("--val-every", type=int, default=10)
    ap.add_argument("--profile", default=str(HERE.parent / "rl" / "h13_profile.json"), help="汇总（只有统计数字）")
    a = ap.parse_args()
    t0 = time.time()
    data, boards, stats, n_games, profile = convert(Path(a.src), a.mmr, a.val_every)
    Path(a.profile).write_text(json.dumps(profile, ensure_ascii=False, indent=1), encoding="utf-8")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("part_*.npz"):
        old.unlink()
    n = len(data["kind"])
    meta = {"source": f"{Path(a.src).name}（他人对局，TavernGym 观察格式）", "games": n_games, "mmr": a.mmr,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"), "samples": n, "stats": dict(stats)}
    k = 0
    for s in range(0, n, SHARD):
        part = {key: v[s:s + SHARD] for key, v in data.items()}
        np.savez_compressed(out / f"part_{k:03d}.npz", **part, n_actions=np.int32(245), compressed=np.int8(1),
                            meta=np.array(json.dumps(meta, ensure_ascii=False)))
        k += 1
    meta["shards"] = k
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    Path(a.boards).parent.mkdir(parents=True, exist_ok=True)
    Path(a.boards).write_text(json.dumps({"source": meta["source"], "games": n_games, "min_mmr": a.mmr,
                                          "created": meta["created"],
                                          "turns": {str(t): v for t, v in sorted(boards.items())}},
                                         ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"{n_games} 局 → {n} 条样本（验证 {int(data['split'].sum())}，{k} 片）→ {out}；"
          f"阵容 {sum(len(v) for v in boards.values())} 个 → {a.boards}；{time.time() - t0:.0f} 秒")
    print({kk: v for kk, v in sorted(stats.items())})
    fin = profile["final"]
    print(f"汇总 → {a.profile}：最终阵容 {fin['n']} 局，平均 {fin['tier']} 本、金色 {fin['golden']} 个；主种族 "
          + "、".join(f"{ZH_RACE.get(k, k)} {v:.0%}" for k, v in fin["main"][:6]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
