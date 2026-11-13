"""算力选择：Windows / Linux 上有 NVIDIA 显卡用 CUDA，Mac（Apple 芯片）用 MPS（Metal），都没有用 CPU。

训练分两段：采样（模拟器 + 模型一次几百个局面的前向）和更新（大批量反向传播）。Apple 芯片上小批量前向用 CPU
往往比 MPS 快（每次调 GPU 有固定开销），大批量更新用 MPS 快——所以采样和更新可以用不同的设备
（Config.rollout_device；内存是统一的，两边同步参数很便宜）。哪种组合最快由 `bg.py bench-device` 实测，
结果写进 reports/device.json，device = "auto" 时照它选；没测过就用下面的默认。
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")      # MPS 不支持的算子退回 CPU，不报错

BENCH = Path(__file__).resolve().parents[1] / "reports" / "device.json"


def is_mac() -> bool:
    return platform.system() == "Darwin"


def available() -> list[str]:
    import torch
    out = ["cpu"]
    if torch.cuda.is_available():
        out.append("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        out.append("mps")
    return out


def bench_choice() -> dict | None:
    try:
        d = json.loads(BENCH.read_text(encoding="utf-8"))
        return d if isinstance(d.get("best"), dict) else None
    except (OSError, ValueError):
        return None


def resolve(device: str = "auto", rollout: str = "auto") -> tuple[str, str]:
    """→ (更新用的设备, 采样用的设备)。"""
    av = available()
    best = (bench_choice() or {}).get("best") or {}
    if device in ("", "auto"):
        device = best.get("device") if best.get("device") in av else ("cuda" if "cuda" in av else "mps" if "mps" in av else "cpu")
    elif device not in av:
        print(f"设备 {device} 用不了（能用的：{', '.join(av)}），改用 CPU", flush=True)
        device = "cpu"
    if rollout in ("", "auto"):
        rollout = best.get("rollout") if best.get("device") == device and best.get("rollout") in av else device
    elif rollout == "same":
        rollout = device
    elif rollout not in av:
        rollout = device
    return device, rollout


def perf_cores() -> int:
    """性能核数量（Apple 芯片的效率核跑模拟器很慢，线程数按性能核算）；别的平台 = 逻辑核数。"""
    if is_mac():
        try:
            n = int(subprocess.run(["sysctl", "-n", "hw.perflevel0.physicalcpu"], capture_output=True, text=True,
                                   timeout=5).stdout.strip())
            if n > 0:
                return n
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return os.cpu_count() or 4


def default_threads() -> int:
    """模拟器的线程数默认值：bench-device 实测最快的；没测过 = 性能核数（至少 2）。"""
    t = ((bench_choice() or {}).get("best") or {}).get("threads")
    return int(t) if isinstance(t, int) and t > 0 else max(2, perf_cores())


def describe() -> str:
    import torch
    av = available()
    parts = [f"torch {torch.__version__}", "可用设备 " + "/".join(av)]
    if "cuda" in av:
        parts.append(torch.cuda.get_device_name(0))
    if is_mac():
        parts.append(f"{platform.machine()} macOS {platform.mac_ver()[0]}，性能核 {perf_cores()}")
    b = bench_choice()
    if b:
        parts.append(f"实测最快：更新 {b['best']['device']} + 采样 {b['best']['rollout']}、线程 {b['best'].get('threads', '?')}"
                     f"（{b['best'].get('dps', 0):.0f} 决策/秒，{b.get('time', '')}）")
    return "，".join(parts)
