"""bgrl 用到的路径，集中在这里（其它模块和脚本都从这里取默认值）。"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent          # bgrl/
RUNS = ROOT / "runs"                            # 训练目录（每次训练一个子目录，含 log.jsonl、latest.pt、snapshots/）
REPORTS = ROOT / "reports"                      # 各种报告（TavernBench、安装日志）
FIXTURES = ROOT / "tests" / "fixtures"          # CI 用的局面样本
DEFAULT_RUN = RUNS / "v3"                       # 当前的正式训练（第二阶段：机器人按真人强度加强；v2 是第一阶段）


