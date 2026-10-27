"""A5 现实闭环回放：用对局记录里你的每一步检验 TavernGym（做法见 bgrl/tg/replay.py）。

用法：
  python tg_replay.py 对局记录.jsonl ...          直接读插件的对局记录（games 目录里的文件）
  python tg_replay.py --fixture 回放.jsonl.gz      读 make_replay_fixture 生成的精简文件（CI 用）
  --annotate  把摘要和问题最多的卡打印成 GitHub Actions 注释
  --out 报告.json
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from tg.logs import Action, load_game, segment  # noqa: E402
from tg.replay import check_action  # noqa: E402


def actions_from_fixture(path: str) -> list[Action]:
    out = []
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            a = Action(d["turn"], d["kind"], d.get("eid", 0), d.get("cardId", ""))
            a.before = d["before"]
            a.afters = d["afters"]
            a.after = a.afters[0]
            out.append(a)
    return out


def summarize(results: list[dict]) -> dict:
    per = defaultdict(Counter)
    examples = defaultdict(list)
    kinds = Counter()
    for r in results:
        per[r["card"]][r["result"]] += 1
        if r["result"] in ("mismatch", "illegal") and len(examples[r["card"]]) < 3:
            examples[r["card"]].append(f"{r['file']} 第{r['turn']}回合 {r['kind']}：{'；'.join(r.get('diff') or [r.get('why', '')])}"[:300])
        for d in r.get("diff") or []:
            kinds[d.split(" ")[0] if d.split(" ")[0] in ("金币", "等级", "场面", "手牌", "酒馆") else "攻防/关键词"] += 1
    rows = []
    for cid, c in per.items():
        n = sum(c.values())
        bad = c["mismatch"] + c["illegal"]
        rows.append({"card": cid, "n": n, **dict(c), "bad": bad, "examples": examples[cid]})
    rows.sort(key=lambda r: (-r["bad"], -r["n"]))
    return {"actions": len(results), "byResult": dict(Counter(r["result"] for r in results)), "diffKinds": dict(kinds),
            "cards": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--fixture", default="")
    ap.add_argument("--annotate", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--trials", type=int, default=4)
    a = ap.parse_args()
    from tg.bridge import catalog
    names = {k: v.get("name_zh") or v.get("name") for k, v in catalog().items()}

    acts: list[Action] = []
    if a.fixture:
        acts = actions_from_fixture(a.fixture)
    files: list[str] = []
    for p in a.paths:
        files += sorted(glob.glob(os.path.join(p, "*.jsonl"))) if os.path.isdir(p) else [p]
    for f in files:
        acts += segment(load_game(f))
    t0 = time.time()
    results = [check_action(x, seed=i, trials=a.trials) for i, x in enumerate(acts)]
    s = summarize(results)
    checked = sum(v for k, v in s["byResult"].items() if k != "skipped")
    ok = s["byResult"].get("exact", 0) + s["byResult"].get("aggregate", 0)
    summary = (f"A5 回放：{len(results)} 个动作，检验 {checked} 个，完全一致 {s['byResult'].get('exact', 0)}，"
               f"总量一致 {s['byResult'].get('aggregate', 0)}（合计 {ok / max(1, checked):.1%}）；"
               f"{s['byResult']}；差异类别 {s['diffKinds']}；用时 {time.time() - t0:.0f} 秒")
    print(summary)
    for row in s["cards"][:25]:
        if row["bad"]:
            print(f"  {names.get(row['card'], row['card'])}({row['card']}) {row['n']} 次，不一致 {row['bad']}  {row['examples'][:1]}")
    if a.annotate:
        def ann(level, msg):
            print(f"::{level}::" + msg.replace("\n", " ").replace("%", "%25")[:3000])
        ann("notice", summary)
        for row in [r for r in s["cards"] if r["bad"]][:12]:
            ann("warning", f"{names.get(row['card'], row['card'])}({row['card']}) {row['n']} 次，不一致 {row['bad']}：" +
                " | ".join(row["examples"][:2]))
    if a.out:
        Path(a.out).write_text(json.dumps({"summary": summary, **s, "results": results}, ensure_ascii=False, indent=1),
                               encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
