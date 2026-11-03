"""目标阵容的“配套卡”：高手已经在做某一套（场上 + 手里核心卡过半、种族阵容场上本族随从 ≥ 2）时，比平时明显更爱买的卡。
这些就是成型以后用来提升实力（养场面的卡）、倒转经济（发宝石 / 发牌 / 免费刷新）、卡牌联动（扩音机配机械、宰割配亡灵）的那些。
另外找出高手做这套时留着不卖的核心 / 配套卡（keep：热气球测绘员、布莱恩·铜须、提图斯这类；尖利的鲨鱼这种买来就是要卖的不算）。

  bg.py playbook                 读配方里的 pro_data（runs/pro_demos + runs/h13_demos），写进 rl/target_comps.json 的 support
  bg.py playbook --min-lift 1.5  比平时多买多少倍才算（默认 1.5）

算法：每次高手买卡的局面，先认出他在做哪一套（rl/comp.CompTables.building(best=True)），
对这套阵容统计酒馆里每张卡“出现时被买的比例”，除以这张卡在所有局面里被买的比例 = 倍率。
出现至少 15 次、买了至少 5 次、倍率 ≥ min-lift、不是核心卡 → 配套卡（每套最多 8 张）。
只把统计结果（卡牌 ID、倍率、次数）写进 rl/target_comps.json，不带任何对局内容。

训练里怎么用：酒馆里有正在做的 S / A 阵容的配套卡时，老师那一步 ×teacher_core_mult；模仿高手时买配套卡 ×tgtsup；
阵容测试列出模型买配套卡的比例（高手的比例也存在这里）。
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
from paths import ROOT  # noqa: E402

OUT = ROOT / "rl" / "target_comps.json"


def _path(p: str) -> Path:
    q = Path(str(p).replace("\\", "/"))
    return q if q.exists() or q.is_absolute() else ROOT / q


def load(data: str):
    from rl.demos import SHARD_KEYS, load_dir
    parts = [x for x in (load_dir(_path(p)) for p in data.split(";") if p.strip()) if x is not None]
    if not parts:
        return None
    return {k: np.concatenate([p[k] for p in parts]) for k in SHARD_KEYS}


def build(d, min_lift: float = 1.5, min_off: int = 15, min_got: int = 5, top: int = 8) -> dict:
    from rl.comp import CompTables, load_target_comps
    from rl.demos import KINDS
    from tg.actions import A_BUY
    from tg.catalog import base_of_index, n_cards
    from tg.obs import C_SHOP, N_SHOP, lobby_bits
    ids = [base_of_index(i) if i else "" for i in range(n_cards())]
    comps = load_target_comps()
    tab = CompTables(comps, ids)
    buy = np.flatnonzero(d["kind"] == KINDS.index("buy"))
    slot = d["cand"][buy, 0].astype(np.int64) - A_BUY
    ok = (slot >= 0) & (slot < N_SHOP)
    buy, slot = buy[ok], slot[ok]
    c = d["c"][buy]
    got_id = c[np.arange(len(buy)), C_SHOP + slot]
    build_m = tab.building(c, lobby_bits(d["f"][buy]), best=True)
    off_all, got_all = Counter(), Counter()
    off, got, n_state = defaultdict(Counter), defaultdict(Counter), Counter()
    for r in range(len(buy)):
        shop = {ids[int(x)] for x in c[r, C_SHOP:C_SHOP + N_SHOP] if x > 0}
        b = ids[int(got_id[r])]
        for x in shop:
            off_all[x] += 1
        got_all[b] += 1
        for k in np.flatnonzero(build_m[r]):
            n_state[k] += 1
            for x in shop:
                off[k][x] += 1
            got[k][b] += 1
    out = {}
    for k, cp in enumerate(tab.comps):
        rows = []
        for x, o in off[k].items():
            if o < min_off or x in cp["core"] or not x:
                continue
            r0 = got_all[x] / max(1, off_all[x])
            lift = (got[k][x] / o) / r0 if r0 > 0 else 0.0
            if lift >= min_lift and got[k][x] >= min_got:
                rows.append((round(lift, 2), got[k][x], o, x))
        rows.sort(reverse=True)
        out[cp["name"]] = {"states": int(n_state[k]), "support": [r[3] for r in rows[:top]],
                           "lift": [r[0] for r in rows[:top]], "n": [[r[1], r[2]] for r in rows[:top]]}
    # 高手的买卡里，有多少是“正在做的 S / A 阵容的配套卡”（阵容测试拿模型的比例和它比）
    sa = np.array([cp["tier"] in "SA" for cp in tab.comps])
    supp = {cp["name"]: set(out[cp["name"]]["support"]) for cp in tab.comps}
    hit = sum(ids[int(got_id[r])] in set().union(*(supp[tab.comps[k]["name"]] for k in np.flatnonzero(build_m[r] & sa)))
              for r in range(len(buy)) if (build_m[r] & sa).any())
    keep = keepers(d, comps, ids, out)
    return {"comps": out, "buys": int(len(buy)), "sa_states": int((build_m & sa).any(1).sum()),
            "pro_support_share": round(hit / max(1, len(buy)), 4), **keep}


def keepers(d, comps, ids, sup: dict, min_expo: int = 60, ratio: float = 0.5) -> dict:
    """高手留着不卖的核心 / 配套卡：正在做那套（核心卡开了头 / 配套卡时在做这套）时，这张卡在场上的每个局面里被卖掉的比例
    不到所有随从平均的 ratio 倍。尖利的鲨鱼、新锐植物学家这类本来就是买了卖的（卖掉触发效果），不算。
    也算高手卖掉“该留的牌”占他所有卖牌的比例（阵容测试拿模型的比例比）。"""
    from rl.comp import CompTables
    from rl.demos import KINDS
    from tg.actions import A_SELL
    from tg.obs import C_BOARD, C_HAND, N_BOARD, N_HAND, lobby_bits
    comps = [dict(c, support=set(sup.get(c["name"], {}).get("support", []))) for c in comps]
    tab = CompTables(comps, ids, tiers="SA")
    c = d["c"]
    bits = lobby_bits(d["f"])
    frac, tribe = tab.progress(c)
    started = ((frac > 0) | (tab.tribal & (tribe >= 2))) & tab.reach(bits)
    bm = tab.building(c, bits)
    board = np.clip(c[:, C_BOARD:C_BOARD + N_BOARD], 0, len(ids) - 1)
    rel = (tab.core[board] & started[:, None, :]).any(2) | (tab.supp[board] & bm[:, None, :]).any(2)   # [行, 场上格]
    sell = np.where(d["kind"] == KINDS.index("sell"), d["cand"][:, 0].astype(np.int64) - A_SELL, -1)
    sold = (sell[:, None] == np.arange(N_BOARD)[None, :]) & (board > 0)
    expo, got = Counter(), Counter()
    for x, r_, s_ in zip(board[rel], np.ones(int(rel.sum())), sold[rel]):
        expo[ids[int(x)]] += 1
        got[ids[int(x)]] += int(s_)
    occ = board > 0
    base = float(sold[occ].sum() / max(1, occ.sum()))
    keep = sorted(b for b in expo if expo[b] >= min_expo and got[b] / expo[b] < base * ratio)
    kset = set(keep)
    # 高手的卖牌里，卖掉“正在做的阵容里该留的牌”（而且手里 / 场上没有第二张）占多少
    own = np.concatenate([c[:, C_BOARD:C_BOARD + N_BOARD], c[:, C_HAND:C_HAND + N_HAND]], 1)
    rows = np.flatnonzero((sell >= 0) & (sell < N_BOARD))
    bad = 0
    for r in rows:
        k = int(sell[r])
        x = int(board[r, k])
        if x and rel[r, k] and ids[x] in kset and sum(ids[int(y)] == ids[x] for y in own[r] if y > 0) < 2:
            bad += 1
    return {"keep": keep, "sell_base": round(base, 4), "pro_keep_sold_share": round(bad / max(1, len(rows)), 4),
            "keep_rate": {b: round(got[b] / expo[b], 4) for b in keep}}


def main(argv=None) -> int:
    from tg.catalog import catalog
    from tools.train import load_recipe
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", default=str(ROOT / "configs" / "v3.json"))
    ap.add_argument("--pro-data", default="")
    ap.add_argument("--min-lift", type=float, default=1.5)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    rc = load_recipe(a.recipe).get("config") or {}
    data = a.pro_data or rc.get("pro_data") or "runs/pro_demos"
    d = load(data)
    if d is None:
        print(f"没有高分决策样本 {data}（先运行 prodemos.bat）", flush=True)
        return 1
    r = build(d, a.min_lift)
    p = Path(a.out)
    v = json.loads(p.read_text(encoding="utf-8")) if p.exists() else json.loads(OUT.read_text(encoding="utf-8"))
    det = catalog()
    name = lambda x: (det.get(x) or {}).get("name_zh", x)  # noqa: E731
    for c in v["comps"]:
        s = r["comps"].get(c["name"]) or {}
        c["support"] = s.get("support", [])
        c["support_names"] = [name(x) for x in c["support"]]
        c["support_lift"] = s.get("lift", [])
        c["support_states"] = s.get("states", 0)
    v["playbook"] = {"source": "高分示范（Firestone 6500+、万三）里高手正在做这套时比平时多买的卡、留着不卖的卡（bg.py playbook，只有统计数字）",
                     "created": time.strftime("%Y-%m-%d"), "min_lift": a.min_lift, "buys": r["buys"],
                     "sa_states": r["sa_states"], "pro_support_share": r["pro_support_share"],
                     "keep": r["keep"], "keep_names": [name(x) for x in r["keep"]], "sell_base": r["sell_base"],
                     "pro_keep_sold_share": r["pro_keep_sold_share"]}
    p.write_text(json.dumps(v, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{r['buys']} 次买卡，其中 {r['sa_states']} 次高手正在做 S / A 阵容；买配套卡占全部买卡 {r['pro_support_share']:.1%} → {p}")
    print(f"留着不卖的核心 / 配套卡 {len(r['keep'])} 张（场上每个局面被卖的比例不到平均 {r['sell_base']:.3f} 的一半）："
          + "、".join(name(x) for x in r["keep"][:30]) + f"；高手的卖牌里卖掉这些占 {r['pro_keep_sold_share']:.1%}")
    for c in v["comps"]:
        print(f"  {c['tier']} {c['name']}（{c['support_states']} 次）：" +
              ("、".join(f"{n} ×{l}" for n, l in zip(c["support_names"], c["support_lift"])) or "样本不够"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
