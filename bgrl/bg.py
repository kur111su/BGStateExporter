"""bgrl 的统一入口：python bg.py <命令> [参数]（Windows 上双击对应的 .bat 即可）。

  train     训练（train.bat 用它接着 runs/v2 练；参数同 tools/train.py：--smoke、--bench、--run、--iters …）
  shadow    影子模式：插件面板显示模型建议（参数同 tools/shadow.py：--ckpt、--device …）
  replay    A5 回放检查：用你的对局记录逐步检查模拟器（参数同 tools/tg_replay.py：--last N …）
  shadow-eval  影子模式评估：模型建议和你实际操作对得上多少（tools/shadow_eval.py）
  tavernbench  用 TavernBench（论文的 368 题评测集）测模型，和 TavernAgent、大语言模型比（tools/tavernbench.py）
  demos     把你的对局记录变成示范数据 runs/demos.npz（训练时模仿你的操作；tools/make_demos.py）
  prolib    高分对局（Firestone 6500+ 分享包）→ 高分阵容库 runs/pro_boards.json + 汇总曲线 rl/pro_profile.json
  prodemos  高分对局回放（Firestone 6500+ 分享包的 rawxml-sanitized）→ 高分玩家决策样本 runs/pro_demos/
  bridge    A4 局面重建检查（tools/tg_bridge_check.py）
  profile   B1 对手强度画像：真人 vs 脚本机器人（tools/opp_profile.py）
  bench     A1 TavernGym 测速（tools/tg_bench.py）
  test      跑单元测试（tests/ 里的 test_calibrate、test_rl、test_shadow、test_arena）
  info      看环境：Python、torch / CUDA、TavernGym 版本、训练进度

每个命令后面加 --help 看它自己的参数。
"""
from __future__ import annotations

import importlib
import json
import runpy
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

TOOLS = {"train": "train", "shadow": "shadow", "replay": "tg_replay", "shadow-eval": "shadow_eval",
         "tavernbench": "tavernbench", "demos": "make_demos", "prolib": "pro_library", "prodemos": "make_pro_demos",
         "bridge": "tg_bridge_check",
         "profile": "opp_profile", "bench": "tg_bench"}


def _run_tool(name: str, args: list[str]) -> int:
    sys.argv = [f"bg.py {name}"] + args
    mod = importlib.import_module(f"tools.{TOOLS[name]}")
    return int(mod.main() or 0)


def _test() -> int:
    rc = 0
    for t in ("test_calibrate.py", "test_rl.py", "test_shadow.py", "test_arena.py", "test_shaping.py", "test_hsreplay.py"):
        print(f"=== {t} ===", flush=True)
        try:
            runpy.run_path(str(ROOT / "tests" / t), run_name="__main__")
        except SystemExit as ex:
            if ex.code not in (None, 0):
                rc = 1
        except Exception:                          # noqa: BLE001 —— 一个测试文件出错不影响跑下一个
            traceback.print_exc()
            rc = 1
    return rc


def _info() -> int:
    from paths import DEFAULT_RUN, games_dir
    print("Python", sys.version.split()[0])
    try:
        import torch
        print("torch", torch.__version__, "CUDA", torch.cuda.is_available(),
              torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    except ImportError:
        print("torch 没装（运行 setup.bat）")
    try:
        import taverngym
        print("TavernGym", taverngym.version())
    except ImportError:
        print("TavernGym 没装（运行 setup.bat）")
    g = games_dir()
    print("对局记录", g, f"（{len(list(g.glob('*.jsonl')))} 局）" if g.exists() else "（不存在）")
    log = DEFAULT_RUN / "log.jsonl"
    if log.exists():
        rows = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines() if x.strip()]
        last = next((r for r in reversed(rows) if "loss" in r), {})
        ev = next((r for r in reversed(rows) if "eval_place" in r), {})
        print(f"训练 {DEFAULT_RUN.name}：第 {last.get('iter')} 轮（{last.get('time')}）；"
              f"最近一次考试第 {ev.get('iter')} 轮平均名次 {ev.get('eval_place')}")
    else:
        print("训练", DEFAULT_RUN, "还没有日志")
    return 0


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd in TOOLS:
        return _run_tool(cmd, args)
    if cmd == "test":
        return _test()
    if cmd == "info":
        return _info()
    print(f"不认识的命令：{cmd}\n{__doc__}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
