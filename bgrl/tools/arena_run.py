"""竞技场（TavernArena）的几步流程，面板里的竞技场按钮跑的就是这个（原来的 arena*.bat）。

  bg.py arena check            打包模型（arena/export.py）→ 核对契约和卡牌表（arena/prep.py）→ 官方连接器 --check（不排队）
  bg.py arena practice         打一局练习（不计分）
  bg.py arena match            打一局计分对局
  bg.py arena loop [--update]  一局接一局打计分对局；--update：训练出的模型比打包的新 2 小时以上就重新打包
  bg.py arena view             对局查看器 http://127.0.0.1:8767/

连接器在单独的环境 arena/.venv 里跑（Python 3.12+、numpy、PyTorch CPU 版，不装 TavernGym）。
服务器地址默认 https://rizling.top，arena/url.txt 可以改。
参赛密钥：环境变量 ARENA_API_KEY，或者单独放在 arena/key.txt 里——只传给连接器进程的环境变量，
不写进命令行、网址、日志和结果文件。打包出来的模型、密钥、对局结果都只在本机（不进 Git）。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ARENA = ROOT / "arena"
BUNDLE, PREV = ARENA / "model_bundle", ARENA / "model_bundle.prev"
RESULTS = ARENA / "results"
VENV = ARENA / ".venv"
APY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def url() -> str:
    p = ARENA / "url.txt"
    u = p.read_text(encoding="utf-8").strip() if p.exists() else ""
    return u or "https://rizling.top"


def _run(argv, **kw) -> int:
    return subprocess.run([str(a) for a in argv], cwd=str(ROOT), **kw).returncode


def _base_python() -> list[str] | None:
    """建 arena/.venv 用的 Python（3.12 以上）。"""
    cands = [["py", f"-{v}"] for v in ("3.12", "3.13", "3.14")] if os.name == "nt" else []
    cands += [["python"], ["python3"], [sys.executable]]
    for c in cands:
        try:
            ok = subprocess.run(c + ["-c", "import sys; sys.exit(0 if sys.version_info[:2] >= (3, 12) else 1)"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        except OSError:
            ok = False
        if ok:
            return c
    return None


def ensure_venv() -> bool:
    if not APY.exists():
        base = _base_python()
        if base is None:
            print("竞技场连接器要 Python 3.12 以上，这台电脑上没找到。")
            return False
        print("建 arena\\.venv ……", flush=True)
        if _run(base + ["-m", "venv", VENV]):
            return False
    if _run([APY, "-c", "import numpy, torch"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL):
        print("往 arena\\.venv 里装 numpy 和 PyTorch CPU 版（约 200 MB）……", flush=True)
        _run([APY, "-m", "pip", "install", "-q", "numpy"])
        if _run([APY, "-m", "pip", "install", "-q", "torch", "--index-url", "https://download.pytorch.org/whl/cpu"]):
            print("装不上 PyTorch。")
            return False
    if _run([APY, "-I", "-c", "import importlib.util,sys; sys.exit(1 if importlib.util.find_spec('taverngym') else 0)"]):
        print("arena\\.venv 里不能有 TavernGym。删掉 arena\\.venv 再来一次。")
        return False
    return True


def package(extra=()) -> bool:
    """打包 + 核对；不成就换回上一次打包的模型（竞技场对局接着用它）。"""
    shutil.rmtree(PREV, ignore_errors=True)
    if BUNDLE.exists():
        BUNDLE.rename(PREV)
    ok = _run([sys.executable, "-X", "utf8", "arena/export.py", *extra]) == 0
    if ok:
        print("核对竞技场契约和卡牌表 ……", flush=True)
        ok = _run([APY, "-X", "utf8", "arena/prep.py", "--url", url()]) == 0
    if ok:
        shutil.rmtree(PREV, ignore_errors=True)
        return True
    shutil.rmtree(BUNDLE, ignore_errors=True)
    if PREV.exists():
        PREV.rename(BUNDLE)
    print("\n没通过——保留上一次打包的模型（竞技场对局接着用它）。提示竞技场连不上的话过一会儿再试；别的情况把这段输出发给我。")
    return False


def compat() -> dict | None:
    p = ARENA / "compat.json"
    if not p.exists():
        print("先跑一次“打包并检查”。")
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def key_env() -> dict | None:
    """带着参赛密钥的环境变量（只给连接器进程）。密钥不打印。"""
    k = os.environ.get("ARENA_API_KEY", "").strip()
    p = ARENA / "key.txt"
    if not k and p.exists():
        k = (p.read_text(encoding="utf-8").strip().splitlines() or [""])[0].strip()
    if not k:
        print("没有参赛密钥。把它单独放进 bgrl\\arena\\key.txt（只放密钥这一行）。")
        return None
    return dict(os.environ, ARENA_API_KEY=k)


def _client(args, out: Path, env=None) -> int:
    """跑官方连接器，结果（JSON）存到 out 并打印出来。"""
    RESULTS.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        rc = _run([APY, "-E", "-s", "-X", "utf8", "arena/arena_client.py", "--url", url(), *args], stdout=f, env=env)
    text = out.read_text(encoding="utf-8", errors="replace").strip()
    print(text, flush=True)
    return rc


def check(extra=()) -> int:
    if not ensure_venv():
        return 1
    print("[1/3] 打包模型 ……", flush=True)
    if not package(extra):
        return 1
    c = compat()
    if c is None:
        return 1
    print("[3/3] 官方连接器 --check（不排队）……", flush=True)
    return _client(["--check", "--policy", "agent:act", "--catalog-sha256", c["catalog_sha256"]], RESULTS / "check.json")


def play(rated: bool) -> int:
    """打一局。返回连接器的退出码。"""
    if not APY.exists():
        print("先跑一次“打包并检查”。")
        return 1
    c, env = compat(), key_env()
    if c is None or env is None:
        return 1
    out = RESULTS / ("arena_last.json" if rated else "practice_last.json")
    print(("计分对局（一局）" if rated else "练习局（不计分）") + f"，模型 {c['model_version']} ……", flush=True)
    rc = _client(["--mode", "arena" if rated else "practice", "--games", "1", "--policy", "agent:act",
                  "--catalog-sha256", c["catalog_sha256"], "--model-version", c["model_version"]], out, env)
    text = out.read_text(encoding="utf-8", errors="replace").strip()
    if text:
        with (RESULTS / "history.jsonl").open("a", encoding="utf-8") as f:
            f.write(text + "\n")
    return rc


def _newer_model() -> bool:
    a, b = ROOT / "runs" / "v3" / "latest.pt", BUNDLE / "model.pt"
    return a.exists() and (not b.exists() or a.stat().st_mtime > b.stat().st_mtime + 7200)


def loop(update: bool, max_games: int = 0, wait: float = 60.0) -> int:
    n, fails = 0, 0
    while not max_games or n < max_games:
        if update and _newer_model():
            print("训练出了更新的模型，重新打包 ……", flush=True)
            package()
        n += 1
        print(f"\n===== 第 {n} 局计分对局（{time.strftime('%Y-%m-%d %H:%M:%S')}）=====", flush=True)
        rc = play(True)
        _run([APY, "-X", "utf8", "arena/viewer.py", "--print-last", RESULTS / "arena_last.json"])
        if rc == 0:
            fails = 0
            continue
        if rc == 130:
            return 130
        fails += 1
        if fails >= 3:
            print("连续 3 次失败，停下。把这段输出发给我。")
            return 1
        print(f"连接器出错停了，{wait:.0f} 秒后再试 ……", flush=True)
        time.sleep(wait)
    return 0


def view() -> int:
    py = APY if APY.exists() else Path(sys.executable)
    print("对局查看器：http://127.0.0.1:8767/ （看完在面板里点停止）", flush=True)
    return _run([py, "-X", "utf8", "arena/viewer.py"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["check", "practice", "match", "loop", "view"])
    ap.add_argument("--update", action="store_true", help="loop：训练出的模型比打包的新 2 小时以上就重新打包")
    ap.add_argument("--ckpt", default="", help="check：打包这个检查点（默认 runs/v3/latest.pt）")
    a = ap.parse_args()
    if a.mode == "check":
        return check(["--ckpt", a.ckpt] if a.ckpt else [])
    if a.mode == "view":
        return view()
    if a.mode == "loop":
        return loop(a.update)
    return play(a.mode == "match")


if __name__ == "__main__":
    raise SystemExit(main())
