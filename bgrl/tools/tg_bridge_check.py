"""A4 检查：把插件记录的招募局面逐条重建进 TavernGym，统计重建后和插件局面一致的比例。

用法：python tg_bridge_check.py [局面文件.jsonl ...] [--annotate]
  默认读 bgrl/tests/fixtures/recruit_states_v2.jsonl；
  --annotate：把摘要和前几个差异打印成 GitHub Actions 注释（CI 用）。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from tg.bridge import build_game, compare, observe  # noqa: E402

DEFAULT = HERE.parent / "tests" / "fixtures" / "recruit_states_v2.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", default=[str(DEFAULT)])
    ap.add_argument("--annotate", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    import taverngym

    states = []
    for f in a.files:
        with open(f, encoding="utf-8") as fh:
            states += [json.loads(line) for line in fh if line.strip()]
    ok = 0
    kinds: Counter = Counter()
    warn_kinds: Counter = Counter()
    examples: list[str] = []
    rows = []
    obs_shape = None
    stats_keys = None
    for i, s in enumerate(states):
        try:
            g, warns = build_game(s, seed=i)
            diffs = compare(g, s)
            f, c = observe(g, 0)
            obs_shape = (len(f), len(c))
            if stats_keys is None:
                st = json.loads(g.state_json())["players"][0].get("stats", {})
                stats_keys = sorted(st.keys()) if isinstance(st, dict) else str(type(st))
        except Exception as ex:                   # noqa: BLE001
            diffs, warns = [f"异常 {type(ex).__name__}: {ex}"], []
        for d in diffs:
            kinds[d.split(" ")[0]] += 1
        for w in warns:
            warn_kinds[w.split("：")[0][:40]] += 1
        if not diffs:
            ok += 1
        elif len(examples) < 8:
            examples.append(f"{s.get('_file')} 第{s.get('turn')}回合：{'；'.join(diffs)[:400]}")
        rows.append({"file": s.get("_file"), "turn": s.get("turn"), "diffs": diffs, "warnings": warns})
    summary = (f"TavernGym {getattr(taverngym, 'version', lambda: '?')()}：重建 {len(states)} 个局面，完全一致 {ok}；"
               f"差异类别 {dict(kinds)}；观察向量 {obs_shape}")
    print(summary)
    print("警告类别：", dict(warn_kinds.most_common(15)))
    print("stats 字段：", stats_keys)
    for e in examples:
        print("  ", e)
    if a.annotate:
        def ann(level, msg):
            print(f"::{level}::" + msg.replace("\n", " ").replace("%", "%25")[:3000])
        ann("notice", summary)
        ann("notice", "警告类别：" + json.dumps(dict(warn_kinds.most_common(15)), ensure_ascii=False))
        ann("notice", "stats 字段：" + json.dumps(stats_keys, ensure_ascii=False))
        for e in examples[:8]:
            ann("warning", e)
    if a.out:
        Path(a.out).write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1),
                               encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
