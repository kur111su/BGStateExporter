"""清理旧文件（面板“维护”里的按钮；原来的 cleanup.bat）。

  bg.py cleanup                     只列出会删什么、各有多大（不删）
  bg.py cleanup --yes               真的删
  bg.py cleanup --keep-from 20000   快照留到哪一轮为止（默认 19000；更早的每 1000 轮留一个）

会删：
  runs/_bench、_smoke、main、v2、pro_demos_old    旧的训练目录和测试目录
  runs/v3/snapshots 里 --keep-from 以前的快照     （每 1000 轮留一个；下次训练启动时自动从对手库里去掉）
  已经去掉的功能留下的文件                        影子模式、回放检查、自己对局的示范、插件源码、旧的 .bat
不动：runs/v3 的 latest.pt、best.pt、日志、看板，teacher.pt，高分样本（pro_demos、h13_demos）和阵容库。
训练在跑的时候不删（日志 3 分钟内更新过算在跑；--force 跳过这个检查）。
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paths import ROOT, RUNS  # noqa: E402

OLD_RUNS = ("_bench", "_smoke", "main", "v2", "pro_demos_old")
OLD_RUN_FILES = ("comptest.json", "demos.npz", "v3/best_it15450.pt", "v3/comptest_look.json", "v3/comptest_seeded_look.json")
OLD_BATS = ("arena.bat", "arena_loop.bat", "arena_match.bat", "arena_practice.bat", "arena_view.bat", "board.bat",
            "cleanup.bat", "comptest.bat", "comptest_seeded.bat", "comptest_lookahead.bat", "goldflow.bat", "prodemos.bat",
            "replay.bat", "shadow.bat", "tavernbench.bat", "teacher.bat", "train.bat")
OLD_FILES = ("rl/advisor.py", "tests/test_shadow.py", "tests/fixtures/replay_v3.jsonl.gz", "tools/shadow.py",
             "tools/shadow_eval.py", "tools/tg_replay.py", "tools/make_demos.py", "tools/make_fixtures.py",
             "tools/make_replay_fixture.py", "reports/a5_replay.json", "reports/a5_replay.txt", "reports/shadow")
OLD_TOP = ("plugin", "docs", "examples")          # 仓库根目录下（bgrl 的上一层）


def _size(p: Path) -> int:
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return p.stat().st_size if p.exists() else 0


def plan(keep_from: int, root: Path = ROOT, runs: Path = RUNS) -> tuple[list[tuple[str, Path, int]], int]:
    """([(哪一类, 路径, 字节数)], 留下的快照数)。"""
    out = []
    for d in OLD_RUNS:
        if (runs / d).exists():
            out.append(("旧的训练目录", runs / d, _size(runs / d)))
    for f in OLD_RUN_FILES:
        if (runs / f).exists():
            out.append(("旧文件", runs / f, _size(runs / f)))
    kept = 0
    snaps = runs / "v3" / "snapshots"
    for p in sorted(snaps.glob("it*.pt")) if snaps.exists() else []:
        m = re.fullmatch(r"it(\d+)", p.stem)
        if not m:
            continue
        n = int(m.group(1))
        if n >= keep_from or n % 1000 == 0:
            kept += 1
        else:
            out.append(("旧快照", p, _size(p)))
    for f in OLD_BATS + OLD_FILES:
        if (root / f).exists():
            out.append(("去掉的功能", root / f, _size(root / f)))
    for d in OLD_TOP:
        p = root.parent / d
        if p.exists() and (root.parent / "bgrl").resolve() == root.resolve():
            out.append(("去掉的功能", p, _size(p)))
    return out, kept


def training_running(runs: Path = RUNS, within: float = 180.0) -> bool:
    p = runs / "v3" / "log.jsonl"
    return p.exists() and time.time() - p.stat().st_mtime < within


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep-from", type=int, default=19000)
    ap.add_argument("--yes", action="store_true", help="真的删（不加只列出来）")
    ap.add_argument("--force", action="store_true", help="训练在跑也删")
    a = ap.parse_args()
    items, kept = plan(a.keep_from)
    by = {}
    for kind, _, n in items:
        c = by.setdefault(kind, [0, 0])
        c[0] += 1
        c[1] += n
    total = sum(n for _, _, n in items)
    for kind, p, n in items:
        if kind != "旧快照":
            print(f"  {kind}  {p}  {n / 1e6:.1f} MB")
    for kind, (c, n) in by.items():
        print(f"{kind}：{c} 项，{n / 1e9:.2f} GB")
    print(f"快照留 {kept} 个（第 {a.keep_from} 轮起的全部 + 更早的每 1000 轮一个）")
    print(f"一共 {len(items)} 项，{total / 1e9:.2f} GB")
    if not a.yes:
        print("（只是列出来，没有删。）")
        return 0
    if training_running() and not a.force:
        print("训练好像还在跑（日志 3 分钟内更新过）。先停止训练再清理。")
        return 1
    failed = 0
    for _, p, _n in items:
        try:
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
        except OSError as ex:
            failed += 1
            print(f"删不掉 {p}：{ex}")
    print(f"删完了：{len(items) - failed} 项" + (f"，{failed} 项没删掉" if failed else "") + "。下次训练启动时会自动把删掉的快照从对手库里去掉。")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
