"""把某个快照设成 best.pt（python bg.py set-best runs/v3/snapshots/it27800.pt）。

用途：best.pt 被一个更差的版本盖掉了（第 28850 轮那次：退回以后考试守护清零，第一次凑满的平均 4.57 覆盖了第 27800 轮的 4.375），
想退回更早的某一版、而且让考试守护拿它的成绩当标准。做的事：
  1. 原来的 best.pt 改名为 best_it<轮>.pt（不删）；
  2. 快照的参数 + 它那一轮的考试平均（从 log.jsonl 找 guard_avg，或者 --score 指定）写成新的 best.pt。
之后在面板点“退回 best.pt 再练”（或 bg.py train --resume --restore runs/v3/best.pt），守护的最好成绩就是这一版的。
训练停着时用。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def score_from_log(log: Path, it: int):
    """log.jsonl 里第 it 轮的考试平均（guard_avg）。"""
    if not log.exists():
        return None
    for line in log.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("iter") == it and r.get("guard_avg") is not None:
            return float(r["guard_avg"])
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("snapshot", help="例如 runs/v3/snapshots/it27800.pt")
    ap.add_argument("--run", default="runs/v3")
    ap.add_argument("--score", type=float, default=None, help="这一版的考试平均名次（默认从 log.jsonl 找）")
    a = ap.parse_args(argv)
    import torch
    run, snap = ROOT / a.run, Path(a.snapshot)
    snap = snap if snap.is_absolute() else ROOT / snap
    m = re.search(r"it(\d+)", snap.name)
    if not snap.exists() or not m:
        sys.exit(f"找不到快照 {snap}（文件名要像 it27800.pt）")
    it = int(m.group(1))
    score = a.score if a.score is not None else score_from_log(run / "log.jsonl", it)
    if score is None:
        sys.exit(f"log.jsonl 里第 {it} 轮没有考试平均（guard_avg），用 --score 指定")
    ck = torch.load(snap, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    best = run / "best.pt"
    mkw = None
    for p in (best, run / "latest.pt"):
        if p.exists():
            mkw = torch.load(p, map_location="cpu", weights_only=False).get("model_kwargs")
            if mkw:
                break
    if best.exists():
        old = torch.load(best, map_location="cpu", weights_only=False)
        q = run / f"best_it{old.get('iter') or 0}.pt"
        best.replace(q)
        print(f"原来的 best.pt（第 {old.get('iter')} 轮，考试平均 {(old.get('guard') or {}).get('eval_avg')}）改名为 {q.name}")
    torch.save({"model": sd, "model_kwargs": mkw, "iter": it,
                "guard": {"eval_avg": round(score, 3), "time": time.strftime("%Y-%m-%d %H:%M:%S"), "from": snap.name}}, best)
    print(f"best.pt = 第 {it} 轮（考试平均 {score:.3f}）。现在可以点“退回 best.pt 再练”。")


if __name__ == "__main__":
    main()
