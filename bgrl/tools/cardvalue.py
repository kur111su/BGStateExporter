"""高手怎么挑卡：每次“买”的时候酒馆里摆着哪些卡、买了哪张 → 每张卡的挑中倍率、关键卡、经济卡。

  bg.py cardvalue                      读 runs/pro_demos 和 runs/h13_demos，写 rl/card_value.json（只有统计数字）
  bg.py cardvalue runs\\pro_demos ...   指定样本目录

挑中倍率 = 实际被买的次数 / 按“在场的选项里随便挑一张”的期望次数（每次买时酒馆有 n 张卡，每张期望 1/n）。
> 1 说明高手看到它就倾向于买。关键卡：倍率 ≥ 1.6、至少出现 40 次，或者在万三最终阵容里出现 ≥ 5% 的随从。
经济卡：说明里有“获得 N 枚铸币 / 铸币上限 / 下回合获得”的卡（酒馆币、慎重投资、南海卖艺者……）。
训练里 --pro-weights "key:2" 让买关键卡的示范权重 ×2；bg.py goldflow 用它比较模型和高手挑卡。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paths import ROOT, RUNS  # noqa: E402

from tg.obs import C_SHOP, N_SHOP as NS  # noqa: E402
OUT = ROOT / "rl" / "card_value.json"
ECO = re.compile(r"获得\d+枚铸币|铸币上限|下回合获得|获得（\d+）枚铸币")


class PickStats:
    """累计每次买的时候的选项和选中的卡（卡牌 ID，金色去掉 _G）。"""

    def __init__(self):
        self.offered = defaultdict(int)
        self.expected = defaultdict(float)
        self.picked = defaultdict(int)
        self.n = 0

    def _id(self, x: int) -> str:
        from tg.catalog import base_of_index
        return base_of_index(int(x))

    def add(self, shop_row, slot: int) -> None:
        """shop_row：cards[36:44]（卡牌编号）；slot：买的是第几格。"""
        opts = [int(x) for x in shop_row if x > 0]
        if not opts or not (0 <= slot < len(shop_row)) or shop_row[slot] <= 0:
            return
        self.n += 1
        for x in opts:
            cid = self._id(x)
            self.offered[cid] += 1
            self.expected[cid] += 1.0 / len(opts)
        self.picked[self._id(int(shop_row[slot]))] += 1

    def table(self, min_offered: int = 20) -> dict:
        return {cid: {"offered": o, "picked": self.picked.get(cid, 0),
                      "lift": round(self.picked.get(cid, 0) / max(1e-9, self.expected[cid]), 2)}
                for cid, o in self.offered.items() if o >= min_offered}


STYLE = ("tribe", "syn", "key", "eco", "simkey", "simsyn", "engine", "engcard", "triple")


def from_demos(dirs: list[Path], style: dict | None = None) -> tuple[PickStats, PickStats]:
    """(6500+ 回放, 万分以上) 两份。style：顺便统计高手的买法（rl/demos.bought_in 各类占买卡的比例，{类: [命中, 总数]}）。"""
    from rl.demos import KINDS, bought_in, load_dir
    lo, hi = PickStats(), PickStats()
    kb = KINDS.index("buy")
    for d in dirs:
        x = load_dir(d)
        if x is None:
            continue
        sel = np.flatnonzero(x["kind"] == kb)
        for i in sel:
            slot = int(x["cand"][i, 0]) - 8
            (hi if int(x["mmr"][i]) >= 10000 else lo).add(x["c"][i, C_SHOP:C_SHOP + NS], slot)
        if style is not None:
            for w in STYLE:
                s = bought_in({"kind": x["kind"], "c": x["c"], "cand": x["cand"]}, w)
                if s is not None:
                    acc = style.setdefault(w, [0, 0])
                    acc[0] += int(s[sel].sum())
                    acc[1] += len(sel)
    return lo, hi


def build(dirs: list[Path]) -> dict:
    from tg.catalog import catalog
    det = catalog()
    style: dict = {}
    lo, hi = from_demos(dirs, style)
    both = PickStats()
    for src in (lo, hi):
        for k, v in src.offered.items():
            both.offered[k] += v
        for k, v in src.expected.items():
            both.expected[k] += v
        for k, v in src.picked.items():
            both.picked[k] += v
        both.n += src.n
    t_all, t_lo, t_hi = both.table(), lo.table(10), hi.table(10)
    hp = ROOT / "rl" / "h13_profile.json"
    final = {}
    if hp.exists():
        for row in json.loads(hp.read_text(encoding="utf-8"))["final"]["cards"]:
            final[row[0]] = row[2]
    cards = {}
    for cid, r in t_all.items():
        c = det.get(cid) or {}
        cards[cid] = {"name": c.get("name_zh", cid), "tier": c.get("tier"), "kind": c.get("kind"), **r,
                      "lift_6500": (t_lo.get(cid) or {}).get("lift"), "lift_13k": (t_hi.get(cid) or {}).get("lift"),
                      "final_13k": final.get(cid, 0.0), "eco": bool(ECO.search(c.get("text_zh") or ""))}
    key = sorted((cid for cid, r in cards.items()
                  if (r["lift"] >= 1.6 and r["offered"] >= 40) or (r["kind"] == "minion" and r["final_13k"] >= 0.05)),
                 key=lambda k: -cards[k]["lift"])
    eco = sorted((cid for cid, r in cards.items() if r["eco"]), key=lambda k: -cards[k]["lift"])
    return {"source": "高分示范（runs/pro_demos 6500+ 回放 + runs/h13_demos 万三）里每次买卡时的选项", "created": time.strftime("%Y-%m-%d"),
            "buys": both.n, "buys_6500": lo.n, "buys_13k": hi.n, "key": key, "eco": eco,
            "style": {w: round(a / max(1, n), 3) for w, (a, n) in style.items()}, "cards": cards}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    dirs = [Path(x) for x in a.dirs] or [RUNS / "pro_demos", RUNS / "h13_demos"]
    v = build([d for d in dirs if d.exists()])
    Path(a.out).write_text(json.dumps(v, ensure_ascii=False, indent=1), encoding="utf-8")
    cs = v["cards"]
    print(f"{v['buys']} 次买卡（6500+ {v['buys_6500']}，万三 {v['buys_13k']}）→ {a.out}")
    print(f"关键卡 {len(v['key'])} 张：" + "、".join(f"{cs[k]['name']}×{cs[k]['lift']}" for k in v["key"][:30]))
    print(f"经济卡 {len(v['eco'])} 张：" + "、".join(f"{cs[k]['name']}×{cs[k]['lift']}" for k in v["eco"][:15]))
    low = sorted((k for k, r in cs.items() if r["offered"] >= 60), key=lambda k: cs[k]["lift"])[:15]
    print("高手基本不买的：" + "、".join(f"{cs[k]['name']}×{cs[k]['lift']}" for k in low))
    return 0


if __name__ == "__main__":
    sys.exit(main())
