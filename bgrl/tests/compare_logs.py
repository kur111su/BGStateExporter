"""比较两份训练日志（log.jsonl）是不是逐轮一样：重构“只搬代码不改逻辑”的验收（.github/workflows/bgrl-repro.yml）。

  python tests/compare_logs.py A/log.jsonl B/log.jsonl

计时、时间戳、显存这类每次都不一样的字段不比；其余数字要几乎相等（相对误差 1e-6），其它值要完全相等。
有差别就打印出来、退出码 1。
"""
from __future__ import annotations

import json
import math
import sys

SKIP_PREFIX = ("t_",)
SKIP = {"time", "dps", "gpu_mem_gb", "snapshot", "restored_from"}


def load(p):
    return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]


def same(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if math.isnan(a) and math.isnan(b):
            return True
        return abs(a - b) <= 1e-6 * max(1.0, abs(a), abs(b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def diff(A, B) -> list[str]:
    out = []
    if len(A) != len(B):
        out.append(f"行数不同：{len(A)} / {len(B)}")
    for i, (a, b) in enumerate(zip(A, B)):
        keys = {k for k in a.keys() | b.keys() if k not in SKIP and not k.startswith(SKIP_PREFIX)}
        for k in sorted(keys):
            if k not in a or k not in b:
                out.append(f"第 {i} 行（iter {a.get('iter')}）：{k} 只在一边有")
            elif not same(a[k], b[k]):
                out.append(f"第 {i} 行（iter {a.get('iter')}）：{k} = {json.dumps(a[k], ensure_ascii=False)[:120]} / "
                           f"{json.dumps(b[k], ensure_ascii=False)[:120]}")
    return out


def main(argv=None) -> int:
    a, b = (argv or sys.argv[1:])[:2]
    d = diff(load(a), load(b))
    for line in d[:60]:
        print(line)
    print(f"{a} vs {b}: " + ("一样" if not d else f"{len(d)} 处不同"))
    return 1 if d else 0


if __name__ == "__main__":
    sys.exit(main())
