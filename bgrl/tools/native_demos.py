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
from tg.actions import A_MAGNETIZE, A_PLAY, A_UPGRADE, action_family  # noqa: E402
from tg.catalog import base_id, card_id, catalog, is_golden_id  # noqa: E402
from tg.obs import C_BOARD, C_HAND, N_BOARD as NB, board_stats, tier as obs_tier  # noqa: E402
from tg.supermask import super_mask  # noqa: E402
# 阵容统计在 rl/comp.py，示范格式在 rl/demos.py；这里的旧名字保留（goldflow / 测试 / 旧脚本在用）
from rl.comp import ZH_RACE, board_scores, comp_stats, load_comp_refs, synergy  # noqa: E402,F401
from rl.demos import KINDS, MAX_CANDS, reset_dir, soft_log_np, write_shard  # noqa: E402

SHARD = 4000


def kind_of(a: int, hand_card_kind: str = "") -> str | None:
    """动作 → 示范类别（tools/make_demos.KINDS）；选择 / 取消不在示范类别里，返回 None。
    打出手牌：手里那张是法术记成 cast，否则 play。"""
    fam = action_family(a)
    if fam in ("cancel", "choose"):
        return None
    if fam == "play":
        return "cast" if "spell" in hand_card_kind else "play"
    return fam


def convert(src: Path, mmr: int, val_every: int = 10):
    import taverngym
    z = np.load(src, allow_pickle=False)
    f = z["features"].astype(np.float32)
    c = z["cards"].astype(np.int64)
    a = z["actions"].astype(np.int64)
    gi = z["game_idx"].astype(np.int64)
    T = z["turn"].astype(np.int64)
    n = len(a)
    if f.shape[1] != taverngym.OBS_FLOATS or c.shape[1] != taverngym.OBS_CARDS:
        raise SystemExit(f"观察长度 {f.shape[1]}/{c.shape[1]}，当前 TavernGym 是 {taverngym.OBS_FLOATS}/{taverngym.OBS_CARDS}")
    det = catalog()

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
            hk = (det.get(card_id(hc)) or {}).get("kind", "") if hc > 0 else ""
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
            cid = card_id(x)
            if (det.get(base_id(cid)) or {}).get("kind") != "minion":
                continue
            atk, hp = board_stats(f[i], s)
            b.append([cid, atk, max(1, hp)])
        if b:
            boards[t].append({"tier": int(obs_tier(f[i])), "mmr": mmr, "board": b})
    # 汇总（只有统计数字）：每回合开始时的等级、本回合升本率；最终阵容；常一起上场的卡
    first, upg = {}, defaultdict(int)
    for i in np.flatnonzero(known):
        key = (int(gi[i]), int((T[i] + 1) // 2))
        first.setdefault(key, i)
        upg[key] |= int(a[i] == A_UPGRADE)
    per = defaultdict(lambda: ([], []))
    for key, i in first.items():
        per[key[1]][0].append(int(obs_tier(f[i])))
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
                cid = card_id(x)
                g_ += is_golden_id(cid)
                ids.append(base_id(cid))
        if len(ids) >= 5:
            fb.append(ids)
            gl.append(g_)
            tl.append(int(obs_tier(f[i])))
    profile = {"source": "他人对局（万三分段，TavernGym 观察格式）", "games": len(games), "mmr": mmr,
               "created": time.strftime("%Y-%m-%d"), "tempo": curve, "final": comp_stats(fb, gl, tl, det),
               "synergy": synergy(fb, det)}
    pairs = {tuple(sorted((r[4], r[5]))) for r in profile["synergy"] if r[0] >= 3}
    key = load_comp_refs()[1]
    sc = [board_scores(ids, det, pairs, key) for ids in fb]
    profile["final"]["scores"] = {k: round(float(np.mean([x[k] for x in sc])), 3) for k in ("coh", "syn", "key")} if sc else {}
    return data, boards, stats, len(games), profile


def merge_boards(paths: list[str]) -> dict:
    """几个阵容库合在一起（训练时 --pro-boards "a;b" 用）。"""
    from rl.calibrate import merge_pro_boards
    return merge_pro_boards(paths)


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
    out = reset_dir(Path(a.out))
    n = len(data["kind"])
    meta = {"source": f"{Path(a.src).name}（他人对局，TavernGym 观察格式）", "games": n_games, "mmr": a.mmr,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"), "samples": n, "stats": dict(stats)}
    k = 0
    for s in range(0, n, SHARD):
        write_shard(out, k, {key: v[s:s + SHARD] for key, v in data.items()}, 245, meta)
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
