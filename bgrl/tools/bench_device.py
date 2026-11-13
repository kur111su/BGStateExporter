"""测哪种设备组合训练最快（python bg.py bench-device），结果写进 reports/device.json，训练 device = auto 时照它选。

每种组合（更新用的设备 + 采样用的设备）跑几轮 train --bench，比每秒决策数（第一轮是预热，不算）。
Mac（Apple 芯片）：mps+cpu、mps+mps、cpu+cpu；有 NVIDIA 显卡：cuda+cuda、cuda+cpu；都没有只测 cpu。约 5–15 分钟。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from rl.device import BENCH, available, default_threads  # noqa: E402


def run_one(dev: str, rdev: str, iters: int, threads: int, size: str, recipe: str) -> dict:
    cmd = [sys.executable, "-X", "utf8", str(HERE / "tools" / "train.py"), "--bench", "--iters", str(iters), "--device", dev,
           "--rollout-device", rdev, "--threads", str(threads), "--verbose"]
    if recipe:
        cmd += ["--recipe", recipe]
    if size:
        cmd += ["--size", size]
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(HERE), capture_output=True, text=True, encoding="utf-8", errors="replace")
    recs = []
    for line in p.stdout.splitlines():
        line = line.strip().removeprefix("::notice::")
        if line.startswith("{") and '"dps"' in line:
            try:
                recs.append(json.loads(line))
            except ValueError:
                pass
    out = {"device": dev, "rollout": rdev, "ok": p.returncode == 0 and bool(recs), "sec": round(time.time() - t0, 1)}
    if not out["ok"]:
        out["error"] = (p.stdout + p.stderr).strip().splitlines()[-3:]
        return out
    use = recs[1:] or recs
    for k in ("dps", "t_collect", "t_env", "t_net", "t_update"):
        v = [r[k] for r in use if isinstance(r.get(k), (int, float))]
        if v:
            out[k] = round(sum(v) / len(v), 2)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--size", default="")
    ap.add_argument("--recipe", default="configs/v3.json")
    a = ap.parse_args(argv)
    av = available()
    combos = [("cpu", "cpu")]
    if "mps" in av:
        combos = [("mps", "cpu"), ("mps", "mps")] + combos
    if "cuda" in av:
        combos = [("cuda", "cuda"), ("cuda", "cpu")] + combos
    threads = a.threads or default_threads()
    recipe = a.recipe if (HERE / a.recipe).exists() else ""
    print(f"可用设备 {'/'.join(av)}；模拟器线程 {threads}；每种组合 {a.iters} 轮（第一轮预热不算）", flush=True)
    res = []
    for dev, rdev in combos:
        print(f"  测 更新 {dev} + 采样 {rdev} …", flush=True)
        r = run_one(dev, rdev, a.iters, threads, a.size, recipe)
        res.append(r)
        if r["ok"]:
            print(f"    {r.get('dps', 0):.0f} 决策/秒（采样 {r.get('t_collect', 0):.1f}s，其中模拟器 {r.get('t_env', 0):.1f}s、"
                  f"模型 {r.get('t_net', 0):.1f}s；更新 {r.get('t_update', 0):.1f}s）", flush=True)
        else:
            print("    失败：" + " | ".join(r.get("error", [])), flush=True)
    ok = [r for r in res if r["ok"]]
    if not ok:
        print("所有组合都失败了，训练会用默认设备", flush=True)
        return 1
    best = max(ok, key=lambda r: r.get("dps", 0))
    BENCH.parent.mkdir(parents=True, exist_ok=True)
    BENCH.write_text(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "threads": threads, "results": res,
                                 "best": {"device": best["device"], "rollout": best["rollout"], "dps": best.get("dps", 0)}},
                                ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"最快：更新 {best['device']} + 采样 {best['rollout']}，{best.get('dps', 0):.0f} 决策/秒 → {BENCH}"
          "（训练设备选 auto 时照这个）", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
