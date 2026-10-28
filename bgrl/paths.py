"""bgrl 用到的路径和端口，集中在这里（其它模块和脚本都从这里取默认值）。"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent          # bgrl/
RUNS = ROOT / "runs"                            # 训练目录（每次训练一个子目录，含 log.jsonl、latest.pt、snapshots/）
REPORTS = ROOT / "reports"                      # 各种报告（回放、测速、影子模式记录）
FIXTURES = ROOT / "tests" / "fixtures"          # CI 用的局面 / 回放样本
DEFAULT_RUN = RUNS / "v2"                       # 当前的正式训练

PLUGIN_PORT = 8765                              # 插件的网页面板和 /state.json
SHADOW_PORT = 8766                              # 影子模式的 /advice.json


def plugin_data() -> Path:
    """插件的数据目录：%AppData%\\HearthstoneDeckTracker\\BGStateExporter（非 Windows 上用 BGSE_DATA 指定）。"""
    env = os.environ.get("BGSE_DATA")
    if env:
        return Path(env)
    return Path(os.environ.get("APPDATA", "")) / "HearthstoneDeckTracker" / "BGStateExporter"


def games_dir() -> Path:
    """插件写对局记录的目录（每局一个 yyyyMMdd_HHmmss.jsonl）。"""
    return plugin_data() / "games"
