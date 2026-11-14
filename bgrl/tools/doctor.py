"""检查项目完整（python bg.py doctor）：环境、代码、训练进度、数据、竞技场，一项一项打勾；--fix 顺手清掉明显多余的东西。

  bg.py doctor            检查（约 10–30 秒：会读一遍 latest.pt / best.pt 并装进模型）
  bg.py doctor --deep     另外把每个快照文件完整校验一遍（CRC，几 GB 要一两分钟）
  bg.py doctor --fix      清掉：名字里带反斜杠的错目录（Windows 路径在 Mac 上建出来的，只在里面没有模型文件时）、
                          退回用剩的 restore_src.pt、访达留下的 .DS_Store

只读不改，除非加 --fix。不打印密钥内容。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paths import RUNS  # noqa: E402

EXT_MD5 = "6ea4d35fdf436513d315d04fe9cbc3c8"           # 外部模型 model.pt（作者 README 里写的）


class Report:
    def __init__(self):
        self.bad, self.warn = 0, 0

    def ok(self, msg, hint=""):
        print(f"  ✔ {msg}", flush=True)

    def no(self, msg, hint=""):
        self.bad += 1
        print(f"  ✖ {msg}" + (f"\n      → {hint}" if hint else ""), flush=True)

    def maybe(self, msg, hint=""):
        self.warn += 1
        print(f"  ! {msg}" + (f"\n      → {hint}" if hint else ""), flush=True)


def gb(n: float) -> str:
    return f"{n / 1e9:.2f} GB" if n >= 1e8 else f"{n / 1e6:.1f} MB"


def du(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def zip_ok(p: Path, deep: bool) -> str:
    """torch 的 .pt 是 zip：打得开（deep 时每个成员 CRC 都对）→ ""，否则错误说明。"""
    try:
        with zipfile.ZipFile(p) as z:
            if deep:
                bad = z.testzip()
                if bad:
                    return f"成员 {bad} 校验不对"
            elif not any(n.endswith("data.pkl") for n in z.namelist()):
                return "里面没有 data.pkl"
        return ""
    except (zipfile.BadZipFile, OSError) as ex:
        return f"{type(ex).__name__}: {ex}"


def check_env(R: Report) -> None:
    print("环境", flush=True)
    v = sys.version_info
    (R.ok if (3, 10) <= v[:2] <= (3, 13) else R.no)(f"Python {v.major}.{v.minor}.{v.micro}（{sys.executable}）")
    try:
        import torch
        from rl.device import available, describe
        R.ok(describe())
        if sys.platform == "darwin" and "mps" not in available():
            R.maybe("这台 Mac 上 MPS 用不了，训练只能用 CPU（很慢）", "macOS 要 12.3 以上；重新运行 setup.sh 装 PyTorch")
        del torch
    except ImportError:
        R.no("PyTorch 没装", "运行 setup（Mac：./setup.sh；Windows：setup.bat）")
    try:
        import taverngym
        pin = (ROOT / "taverngym" / "PIN.txt").read_text(encoding="utf-8")
        commit = next((ln.split("=", 1)[1].strip() for ln in pin.splitlines() if ln.startswith("commit=")), "")
        stamp = (ROOT / "wheels" / ".installed")
        got = stamp.read_text(encoding="utf-8").strip() if stamp.exists() else ""
        want = f"taverngym-{commit[:7]}"
        if got and got != want:
            R.no(f"TavernGym 装的是 {got}，锁定的版本是 {want}", "重新运行 setup")
        else:
            from importlib.metadata import version
            R.ok(f"TavernGym {version('taverngym')}（{want}）")
    except ImportError:
        R.no("TavernGym 没装", "运行 setup")
    try:
        import PySide6  # noqa: F401
        import pyqtgraph  # noqa: F401
        R.ok("面板需要的 PySide6 / pyqtgraph")
    except ImportError:
        R.maybe("面板需要的 PySide6 / pyqtgraph 没装", "pip install -r requirements.txt")
    st = shutil.disk_usage(ROOT)
    (R.ok if st.free > 20e9 else R.maybe)(f"磁盘剩 {gb(st.free)}", "训练每 50 轮存一个快照（约 33 MB），定期在面板“维护”里清理")


def check_code(R: Report) -> None:
    print("代码", flush=True)
    git = shutil.which("git")
    if not git or not (ROOT.parent / ".git").exists():
        R.maybe("没有 git 记录，查不了代码文件是不是完整", "用 git clone 拿代码，以后 git pull 更新")
        return
    run = lambda *a: subprocess.run([git, *a], cwd=ROOT.parent, capture_output=True, text=True, encoding="utf-8")  # noqa: E731
    head = run("log", "-1", "--format=%h %cd", "--date=short").stdout.strip()
    br = run("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    st = [ln for ln in run("status", "--porcelain", "--untracked-files=no").stdout.splitlines() if ln.strip()]
    gone = [ln[3:] for ln in st if ln[:2].strip() == "D"]
    mod = [ln[3:] for ln in st if ln[:2].strip() != "D"]
    if gone:
        R.no(f"{len(gone)} 个代码文件不见了：{', '.join(gone[:5])}", "git checkout -- <文件> 恢复，或 git pull")
    if mod:
        R.maybe(f"{len(mod)} 个代码文件和 git 里的不一样（本地改过）：{', '.join(mod[:5])}", "不是你改的话：git checkout -- <文件>")
    if not gone and not mod:
        R.ok(f"代码完整（{br} {head}）")


def check_run(R: Report, run_dir: Path, deep: bool, fix: bool) -> None:
    print(f"训练进度（{run_dir.relative_to(ROOT)}）", flush=True)
    if not run_dir.exists():
        R.no("训练目录不在", "从旧电脑拷 runs/v3 整个文件夹过来；或者在面板“训练”页从头练")
        return
    latest, best = run_dir / "latest.pt", run_dir / "best.pt"
    info = {}
    for p in (latest, best):
        if not p.exists():
            (R.no if p is latest else R.maybe)(f"{p.name} 不在")
            continue
        err = zip_ok(p, deep)
        if err:
            R.no(f"{p.name} 坏了：{err}", "从旧电脑重新拷，或者用 snapshots 里最近的快照 --restore")
            continue
        try:
            import torch
            from rl.model import build_model
            ck = torch.load(p, map_location="cpu", weights_only=False)
            sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
            kw = ck.get("model_kwargs") if isinstance(ck, dict) else None
            if kw is None:
                cfgp = run_dir / "config.json"
                kw = (json.loads(cfgp.read_text(encoding="utf-8")).get("model_kwargs") if cfgp.exists() else None) or {}
            m = build_model(**kw)
            missing, extra = m.load_state_dict(sd, strict=False)
            missing = [k for k in missing if k != "attr_proj.weight"]
            if missing or extra:
                R.no(f"{p.name}：参数对不上（少 {len(missing)} 个、多 {len(extra)} 个）", "代码和模型版本不一致：git pull 到最新")
            else:
                it = ck.get("iter") if isinstance(ck, dict) else None
                info[p.name] = it
                R.ok(f"{p.name}：第 {it} 轮，模型 {kw}，{gb(p.stat().st_size)}")
        except ImportError:
            R.maybe(f"{p.name} 打得开（没装 torch，没法装进模型试）")
        except Exception as ex:                          # noqa: BLE001
            R.no(f"{p.name} 读不了：{type(ex).__name__}: {ex}")
    log = run_dir / "log.jsonl"
    if log.exists():
        last = None
        with open(log, "rb") as fh:
            fh.seek(max(0, log.stat().st_size - 200_000))
            for ln in fh.read().decode("utf-8", "replace").splitlines():
                if '"decisions"' in ln:
                    try:
                        last = json.loads(ln)
                    except ValueError:
                        pass
        if last:
            li, ci = last.get("iter"), info.get("latest.pt")
            msg = f"log.jsonl 最后一轮 {li}（{last.get('time', '')}，{last.get('dps', '?')} 决策/秒，设备 {last.get('device', '?')}）"
            (R.ok if ci is None or li is None or abs(int(li) - int(ci)) <= 60 else R.maybe)(
                msg, f"latest.pt 是第 {ci} 轮，和日志差得多：可能拷的是不同时间的文件")
    else:
        R.maybe("没有 log.jsonl（看板会是空的）")
    snaps = run_dir / "snapshots"
    files = {p.name: p for p in snaps.glob("*.pt")} if snaps.exists() else {}
    lg = run_dir / "league.json"
    if lg.exists():
        try:
            mem = json.loads(lg.read_text(encoding="utf-8")).get("members", [])
            names = [str(m.get("path") or "").replace("\\", "/").rsplit("/", 1)[-1] for m in mem if m.get("path")]
            miss = [n for n in names if n not in files]
            orphan = sorted(set(files) - set(names))
            (R.no if miss else R.ok)(f"对手库 {len(names)} 个快照" + (f"，{len(miss)} 个文件不在：{', '.join(miss[:4])}" if miss else "，文件都在"),
                                    "从旧电脑拷 runs/v3/snapshots；不拷的话训练会把它们从对手库去掉")
            if orphan:
                R.maybe(f"{len(orphan)} 个快照不在对手库里（{gb(sum(files[n].stat().st_size for n in orphan))}）",
                        "面板“维护”里的清理会删掉用不到的快照")
        except ValueError as ex:
            R.no(f"league.json 读不了：{ex}")
    bad = [n for n, p in sorted(files.items()) if zip_ok(p, deep)]
    (R.no if bad else R.ok)(f"{len(files)} 个快照文件{'（完整校验）' if deep else ''}" + (f"，坏的：{', '.join(bad[:5])}" if bad else "都打得开"),
                            "坏的删掉（训练会自动从对手库去掉）")
    rs = run_dir / "restore_src.pt"
    if rs.exists():
        if fix:
            size = rs.stat().st_size
            rs.unlink()
            R.ok(f"删了退回用剩的 restore_src.pt（{gb(size)}）")
        else:
            R.maybe("restore_src.pt 是“退回 best.pt 再练”用剩的临时副本", "bg.py doctor --fix 删掉")
    print(f"  · 占用：{gb(du(run_dir))}（快照 {gb(du(snaps)) if snaps.exists() else '0'}）", flush=True)


def check_data(R: Report) -> None:
    print("数据", flush=True)
    items = [("runs/pro_demos", "高分示范（Firestone 6500+）", "模仿学习用不了", True),
             ("runs/h13_demos", "高分示范（万三）", "模仿学习少一部分样本", False),
             ("runs/pro_boards.json", "高分阵容库", "加强局、阵容测试、成型阵容课程用不了", True),
             ("runs/h13_boards.json", "高分阵容库（万三）", "阵容库少一部分", False)]
    for rel, name, lose, must in items:
        p = ROOT / rel
        if not p.exists():
            (R.no if must else R.maybe)(f"{name}（{rel}）不在", f"从旧电脑拷；没有的话{lose}")
            continue
        if p.is_dir():
            n = len(list(p.glob("*.npz"))) + len(list(p.glob("*.jsonl")))
            (R.ok if n else R.no)(f"{name}：{n} 个文件，{gb(du(p))}")
        else:
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(d, dict) and isinstance(d.get("turns"), dict):     # {turns: {回合: [整板, …]}, games: 局数}
                    n = sum(len(v) for v in d["turns"].values() if isinstance(v, list))
                    R.ok(f"{name}：{d.get('games', '?')} 局、{n} 个整板（{len(d['turns'])} 个回合）")
                else:
                    R.ok(f"{name}：{len(d)} 条")
            except ValueError as ex:
                R.no(f"{name} 读不了：{ex}")
    ext = ROOT / "data" / "external" / "model.pt"
    if ext.exists():
        h = hashlib.md5(ext.read_bytes()).hexdigest()
        (R.ok if h == EXT_MD5 else R.no)(f"外部模型 data/external/model.pt（md5 {'对' if h == EXT_MD5 else '不对：' + h}）",
                                         "重新从 huggingface.co/jimmypretend/hearthstone-bg-ppo 下载")
    else:
        R.maybe("外部模型 data/external/model.pt 不在：311 维卡牌属性和“学卖牌”的老师这次都没有")
    t = ROOT / "runs" / "teacher.pt"
    if t.exists():
        R.ok(f"旧老师模型 runs/teacher.pt（{gb(t.stat().st_size)}；现在的配方没用它）")


def check_arena(R: Report) -> None:
    print("竞技场", flush=True)
    a = ROOT / "arena"
    key = a / "key.txt"
    (R.ok if key.exists() and key.read_text(encoding="utf-8").strip() else R.maybe)(
        "参赛密钥 arena/key.txt" + ("（有）" if key.exists() else "不在"), "没有密钥只能打练习局")
    cl = a / "arena_client.py"
    (R.ok if cl.exists() else R.no)("官方连接器 arena/arena_client.py" + ("" if cl.exists() else " 不在"),
                                    "它不在 Git 里：从旧电脑拷 bgrl/arena/arena_client.py，或从竞技场网站重新下载；没有它竞技场的检查、练习局、计分局都跑不了")
    res = list((a / "results").glob("*.json")) if (a / "results").exists() else []
    R.ok(f"对局结果 {len(res)} 局") if res else R.maybe("arena/results 里没有对局（对局查看器是空的）")


def check_strays(R: Report, fix: bool) -> None:
    print("多余的东西", flush=True)
    found = False
    for p in RUNS.iterdir() if RUNS.exists() else []:
        if "\\" in p.name and p.is_dir():               # Mac / Linux 上按 Windows 路径建出来的目录（例如 runs\v3）
            found = True
            pts = list(p.rglob("*.pt"))
            if fix and not pts:
                shutil.rmtree(p)
                R.ok(f"删了错目录 runs/{p.name}（里面没有模型）")
            else:
                R.maybe(f"错目录 runs/{p.name}（Windows 路径在这台电脑上建出来的，{len(pts)} 个模型文件）",
                        "bg.py doctor --fix 删掉（有模型文件的要自己看一眼再删）")
    ds = [p for p in ROOT.parent.rglob(".DS_Store") if ".venv" not in p.parts]
    if ds:
        found = True
        if fix:
            for p in ds:
                p.unlink(missing_ok=True)
            R.ok(f"删了 {len(ds)} 个 .DS_Store")
        else:
            R.maybe(f"{len(ds)} 个 .DS_Store（访达留下的，无害）", "bg.py doctor --fix 删掉")
    if not found:
        R.ok("没有")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=str(RUNS / "v3"))
    ap.add_argument("--deep", action="store_true", help="快照文件完整校验（CRC）")
    ap.add_argument("--fix", action="store_true", help="清掉明显多余的东西（见上）")
    a = ap.parse_args(argv)
    R = Report()
    check_env(R)
    check_code(R)
    check_run(R, Path(a.run).resolve(), a.deep, a.fix)
    check_data(R)
    check_arena(R)
    check_strays(R, a.fix)
    print(f"\n结果：{R.bad} 个问题，{R.warn} 个提醒" + ("" if R.bad else "；能训练"), flush=True)
    return 1 if R.bad else 0


if __name__ == "__main__":
    sys.exit(main())
