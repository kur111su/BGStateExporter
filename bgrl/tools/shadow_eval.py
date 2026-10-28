"""影子模式评估：把影子模式记录（reports/shadow/日期.jsonl）和插件的对局记录对齐，看模型建议和你实际操作有多接近。

用法：python shadow_eval.py [影子模式记录.jsonl ...] [--games 对局记录目录] [--out 报告.json]
  不给记录文件：读 bgrl/reports/shadow 里全部；对局记录默认读插件的 games 目录。

对你的每一个动作（买、卖、打出、施法、磁力、升本、刷新、英雄技能；调站位和合并的记录不算），
找同一回合、动作之前最后一次的模型方案（而且是在你上一个动作之后算的，局面一致），比较：
  同类   模型方案的第一步和你的动作是同一类（买 / 卖 / 打出 / 法术 / 升本 / 刷新 / 英雄技能）
  同一步 同类而且是同一张卡（按中文名）
  在方案里 你的动作出现在模型整个本回合方案的任何一步
按模型版本（训练轮数）分开统计。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paths import REPORTS, games_dir  # noqa: E402
from tg.actions import (A_ACTIVATE, A_BUY, A_CHOOSE, A_DARK_GIFT, A_END, A_HERO_POWER, A_MAGNETIZE,  # noqa: E402
                        A_MOVE, A_PLAY, A_REFRESH, A_SELL, A_UPGRADE)
from tg.logs import load_game, segment  # noqa: E402

KIND = {"buy": "买", "sell": "卖", "play": "打出", "magnetize": "打出", "cast": "法术", "upgrade": "升本",
        "refresh": "刷新", "hero_power": "英雄技能"}


def step_type(a: int, text: str) -> str:
    if a == A_END:
        return "结束"
    if a == A_REFRESH:
        return "刷新"
    if a == A_UPGRADE:
        return "升本"
    if A_HERO_POWER <= a < A_DARK_GIFT:
        return "英雄技能"
    if a == A_DARK_GIFT:
        return "黑暗之赐"
    if A_BUY <= a < A_SELL:
        return "买"
    if A_SELL <= a < A_ACTIVATE:
        return "卖"
    if A_ACTIVATE <= a < A_CHOOSE:
        return "发动"
    if A_CHOOSE <= a < A_PLAY:
        return "选择"
    if A_PLAY <= a < A_MAGNETIZE:
        return "法术" if text.startswith("使用") else "打出"
    if A_MAGNETIZE <= a < A_MOVE:
        return "打出"
    return "调站位"


def _names() -> dict[str, str]:
    try:
        from tg.bridge import catalog
        return {k: v.get("name_zh") or v.get("name") or k for k, v in catalog().items()}
    except Exception:                              # noqa: BLE001 —— 没装 TavernGym 时只按类别比
        return {}


def _game_start(path: str) -> datetime | None:
    try:
        return datetime.strptime(os.path.basename(path)[:15], "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def evaluate(shadow_rows: list[dict], game_files: list[str]) -> dict:
    names = _names()
    games = sorted((g, _game_start(g)) for g in game_files if _game_start(g))
    starts = [s for _, s in games]
    by_game: dict[str, list[dict]] = defaultdict(list)
    for r in shadow_rows:
        t = datetime.fromisoformat(r["time"])
        idx = max((i for i, s in enumerate(starts) if s <= t), default=None)
        if idx is not None:
            by_game[games[idx][0]].append(r)
    per_model: dict = defaultdict(lambda: {"n": 0, "type": 0, "exact": 0, "in_plan": 0, "confusion": Counter(),
                                           "by_kind": defaultdict(Counter), "examples": []})
    for path, rows in by_game.items():
        recs = load_game(path)
        hero = (recs[0]["player"] or {}).get("heroCardId") if recs else None
        rows = [r for r in rows if not hero or r.get("hero") == hero]
        acts = [a for a in segment(recs) if a.kind in KIND]
        prev_seq = -1
        for a in acts:
            seq = a.before.get("seq") or 0
            cand = [r for r in rows if r["turn"] == a.turn and prev_seq < (r.get("seq") or 0) <= seq]
            prev_seq = seq
            if not cand:
                continue
            r = max(cand, key=lambda x: x["seq"])
            steps = [(x, t) for x, t in zip(r.get("actions") or [], r.get("steps") or [])
                     if step_type(x, t) not in ("选择",)]
            if not steps:
                continue
            m = per_model[(r.get("model") or {}).get("iter")]
            mine = KIND[a.kind]
            name = names.get(a.card_id.removesuffix("_G"), "") if a.card_id else ""
            first_t = step_type(*steps[0])

            def same(x, t):
                return step_type(x, t) == mine and (not name or mine in ("升本", "刷新") or name in t)
            m["n"] += 1
            m["type"] += first_t == mine
            m["exact"] += same(*steps[0])
            hit = any(same(x, t) for x, t in steps)
            m["in_plan"] += hit
            m["confusion"][(mine, first_t)] += 1
            m["by_kind"][mine]["n"] += 1
            m["by_kind"][mine]["type"] += first_t == mine
            if not hit and len(m["examples"]) < 40:
                m["examples"].append(f"{os.path.basename(path)[:15]} 第{a.turn}回合 你：{mine} {name}；"
                                     f"模型：{' / '.join(t for _, t in steps[:4])}")
    out = {}
    for it, m in sorted(per_model.items(), key=lambda kv: (kv[0] is None, kv[0])):
        n = max(1, m["n"])
        out[str(it)] = {"动作数": m["n"], "第一步同类": round(m["type"] / n, 3), "第一步同一张卡": round(m["exact"] / n, 3),
                        "在模型方案里": round(m["in_plan"] / n, 3),
                        "按你的动作": {k: {"次数": v["n"], "第一步同类": round(v["type"] / max(1, v["n"]), 3)}
                                  for k, v in sorted(m["by_kind"].items(), key=lambda kv: -kv[1]["n"])},
                        "你做的 → 模型第一步（最多的）": [f"{a} → {b}：{c}" for (a, b), c in m["confusion"].most_common(12)],
                        "没对上的例子": m["examples"]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--games", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    files = a.files or sorted(glob.glob(str(REPORTS / "shadow" / "*.jsonl")))
    rows = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            rows += [json.loads(x) for x in fh if x.strip()]
    gdir = a.games or str(games_dir())
    res = evaluate(rows, sorted(glob.glob(os.path.join(gdir, "*.jsonl"))))
    for it, r in res.items():
        print(f"模型第 {it} 轮：{r['动作数']} 个动作，第一步同类 {r['第一步同类']:.0%}，同一张卡 {r['第一步同一张卡']:.0%}，"
              f"在模型方案里 {r['在模型方案里']:.0%}")
        for line in r["你做的 → 模型第一步（最多的）"][:8]:
            print("   ", line)
    out = a.out or str(REPORTS / "shadow_eval.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("详细结果：", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
