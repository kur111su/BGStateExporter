"""面板的任务层：启动、排队、停止后台任务，看状态和输出。只用标准库，不依赖界面（界面在 panel/app.py）。

每个任务一个文件 reports/panel/jobs/<id>.json（状态）+ <id>.log（输出）+ <id>.beat（心跳）+ <id>.stop（停止请求）。
任务由独立的进程 panel/runner.py 跑：关掉面板、关掉浏览器都不影响它；重新打开面板能接着看。
同一组（group）的任务一次只跑一个，后来的排队；不同组可以同时跑（训练时可以跑一个阵容测试）。

能跑什么全写在 SPECS 里：每个任务的命令由这里的代码拼出来，参数只接受 SPECS 里列的那几项、按类型和范围检查——
界面上没有能输入任意命令的地方。竞技场密钥不经过这里（tools/arena_run.py 自己从 arena/key.txt 读，传给连接器的环境变量）。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "reports" / "panel" / "jobs"
RUN = ROOT / "runs" / "v3"
STOP_FILE = RUN / "STOP"                 # 训练每轮结束看一眼：有这个文件就存盘退出（rl/ppo.Trainer.train）
BEAT_DEAD = 30.0                         # 心跳超过这么多秒没更新：跑任务的进程已经不在了
ACTIVE = ("queued", "running")
STATUS_ZH = {"queued": "排队中", "running": "运行中", "done": "完成", "failed": "失败", "stopped": "已停止",
             "lost": "中断（进程不在了）"}


def _bg(*args) -> list[str]:
    return [sys.executable, "-X", "utf8", "bg.py", *[str(a) for a in args]]


def _train(p):
    init = [] if (RUN / "latest.pt").exists() or not (ROOT / "runs" / "v2" / "latest.pt").exists() \
        else ["--init", "runs/v2/latest.pt"]
    return [_bg("train", "--recipe", "configs/v3.json", "--resume", *init, "--threads", p["threads"])]


def _ckpt(p) -> list[str]:
    return [] if p.get("ckpt", "latest") == "latest" else ["--ckpt", "runs/v3/best.pt"]


def _arena(mode):
    return lambda p: [_bg("arena", mode, *(["--update"] if p.get("update") else []))]


_GAMES = ("int", 64, 16, 1024)
_CKPT = ("choice", "latest", ("latest", "best"))
# 任务 → 标题、分组、说明、参数 {名字: (类型, 默认, …)}、怎么拼命令、怎么停（file = 让训练自己存盘退出）、要不要二次确认
SPECS: dict[str, dict] = {
    "train": dict(title="训练", group="train", stop="file", params={"threads": ("int", os.cpu_count() or 4, 1, 256)},
                  desc="接着 runs\\v3 练（configs\\v3.json）。停止 = 这一轮跑完、存好进度再退出。", build=_train),
    "teacher": dict(title="老师模型", group="train", params={},
                    desc="只模仿高分示范，挑留出局上最好的一版 → runs\\teacher.pt。约 15–25 分钟，和训练不能同时跑。",
                    build=lambda p: [_bg("teacher")]),
    "comptest": dict(title="阵容测试", group="test", params={"games": _GAMES, "ckpt": _CKPT},
                     desc="模型打换成高分阵容的对手：金币流向、最终阵容、成型比例，和高手并排比。64 局约 2 分钟。",
                     build=lambda p: [_bg("goldflow", "--boosted", "--games", p["games"], *_ckpt(p))]),
    "comptest_seeded": dict(title="拿着成型阵容测试", group="test", params={"games": ("int", 256, 16, 1024), "ckpt": _CKPT},
                            desc="第 8–10 回合把模型的场面换成高手的成型阵容，再让它自己打：看拿到阵容以后能打第几。",
                            build=lambda p: [_bg("goldflow", "--boosted", "--games", p["games"], "--seed-comps", "8,9,10",
                                                 *_ckpt(p))]),
    "goldflow": dict(title="金币流向", group="test", params={"games": _GAMES, "ckpt": _CKPT},
                     desc="前几回合的钱花在哪（升本、买、刷新），和 6500+ 高手并排比。对手是普通机器人。",
                     build=lambda p: [_bg("goldflow", "--games", p["games"], *_ckpt(p))]),
    "tavernbench": dict(title="TavernBench 评测", group="test", params={},
                        desc="论文的 368 题评测集。只用 CPU，训练时也能跑。", build=lambda p: [_bg("tavernbench")]),
    "selftest": dict(title="自检", group="test", params={}, desc="跑一遍单元测试（几分钟）。", build=lambda p: [_bg("test")]),
    "info": dict(title="环境信息", group="misc", params={}, desc="Python、PyTorch / 显卡、TavernGym 版本、训练进度。",
                 build=lambda p: [_bg("info")]),
    "board": dict(title="重画看板", group="misc", params={}, desc="按最新日志重画 runs\\v3\\dashboard.md 和曲线图。",
                  build=lambda p: [_bg("board")]),
    "prodemos": dict(title="高分样本", group="data", params={},
                     desc="bgrl\\data\\firestone 里的 6500+ 分享包 → 决策样本 runs\\pro_demos。约 10–20 分钟。",
                     build=lambda p: [_bg("prodemos", "data/firestone")]),
    "cardvalue": dict(title="卡牌价值", group="data", params={}, desc="高手怎么挑卡：挑中倍率、关键卡、经济卡 → rl\\card_value.json。",
                      build=lambda p: [_bg("cardvalue")]),
    "playbook": dict(title="阵容配套卡", group="data", params={},
                     desc="高手做某一套阵容时比平时多买的卡 → rl\\target_comps.json 的配套卡。改过目标阵容以后跑一次。",
                     build=lambda p: [_bg("playbook")]),
    "arena_check": dict(title="竞技场：打包并检查", group="arena", params={},
                        desc="把 runs\\v3\\latest.pt 打包成竞技场用的模型，核对契约和卡牌表（不排队、不计分）。",
                        build=_arena("check")),
    "arena_practice": dict(title="竞技场：练习局", group="arena", params={}, desc="打一局练习（不计分）。", build=_arena("practice")),
    "arena_match": dict(title="竞技场：计分局（1 局）", group="arena", params={}, confirm="这一局计入竞技场排名。",
                        desc="打一局计分对局，打完就停。", build=_arena("match")),
    "arena_loop": dict(title="竞技场：连续计分局", group="arena", params={"update": ("bool", False)},
                       confirm="会一直打计分对局，直到你点停止或者连续失败 3 次；每一局都计入竞技场排名。",
                       desc="一局接一局打计分对局。勾上“自动换新模型”：训练出的模型比打包的新 2 小时以上就重新打包。",
                       build=_arena("loop")),
    "cleanup": dict(title="清理旧文件", group="train", params={"keep_from": ("int", 19000, 0, 10_000_000)},
                    confirm="删除以后找不回来：旧的训练目录、这一轮以前的快照（每 1000 轮留一个）、已去掉的功能留下的文件。",
                    desc="先点“看看会删什么”，确认以后再删。训练在跑的时候会排在训练后面。",
                    build=lambda p: [_bg("cleanup", "--keep-from", p["keep_from"], "--yes")]),
    "cleanup_preview": dict(title="清理：看看会删什么", group="misc", params={"keep_from": ("int", 19000, 0, 10_000_000)},
                            desc="只列出来，不删。", build=lambda p: [_bg("cleanup", "--keep-from", p["keep_from"])]),
}


def check_params(key: str, params: dict | None) -> dict:
    """按 SPECS 检查参数：只留认识的名字，类型、范围、可选值不对就报 ValueError。"""
    spec = SPECS[key]
    out = {}
    for name, (kind, default, *rest) in spec["params"].items():
        v = (params or {}).get(name, default)
        if kind == "int":
            if isinstance(v, bool) or not isinstance(v, int) or not rest[0] <= v <= rest[1]:
                raise ValueError(f"{name} 要是 {rest[0]}–{rest[1]} 的整数：{v!r}")
        elif kind == "bool":
            if not isinstance(v, bool):
                raise ValueError(f"{name} 要是 是 / 否：{v!r}")
        elif kind == "choice":
            if v not in rest[0]:
                raise ValueError(f"{name} 只能是 {rest[0]}：{v!r}")
        out[name] = v
    extra = set(params or {}) - set(spec["params"])
    if extra:
        raise ValueError(f"不认识的参数：{sorted(extra)}")
    return out


# ---------------- 任务文件 ----------------

def _path(jid: str, ext: str = "json", root: Path | None = None) -> Path:
    return (root or DIR) / f"{jid}.{ext}"


def _write(p: Path, obj: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    for k in range(20):                                  # Windows：面板正好在读这个文件时换不了，等一下再换
        try:
            os.replace(tmp, p)
            return
        except PermissionError:
            if k == 19:
                raise
            time.sleep(0.05)


def load(jid: str, root: Path | None = None) -> dict | None:
    for _ in range(3):                                   # 正好撞上另一边在换文件：再读一次
        try:
            return json.loads(_path(jid, root=root).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (ValueError, OSError):
            time.sleep(0.05)
    return None


def _status(j: dict, root: Path | None = None) -> str:
    """文件里写着排队 / 运行，可心跳停了 → lost（电脑重启过、进程被杀了）。"""
    if j["status"] in ACTIVE:
        try:
            age = time.time() - _path(j["id"], "beat", root).stat().st_mtime
        except OSError:
            age = time.time() - j["created"]
        if age > BEAT_DEAD:
            return "lost"
    return j["status"]


def jobs(root: Path | None = None, limit: int = 60) -> list[dict]:
    """最近的任务（新的在前），每个带上算过心跳的 state。"""
    d = root or DIR
    out = []
    for p in sorted(d.glob("*.json"), reverse=True)[:limit] if d.exists() else []:
        j = load(p.stem, root)
        if j:
            j["state"] = _status(j, root)
            out.append(j)
    return out


def active(root: Path | None = None, group: str | None = None) -> list[dict]:
    return [j for j in jobs(root) if j["state"] in ACTIVE and (group is None or j["group"] == group)]


def start(key: str, params: dict | None = None, root: Path | None = None, steps: list | None = None,
          spawn: bool = True) -> dict:
    """排一个任务，起一个独立进程去跑它，返回任务记录。同一个任务已经在排队 / 在跑就不再排（报 RuntimeError）。
    steps、spawn 只给测试用（不走 SPECS 的命令、不起进程）。"""
    spec = SPECS[key] if steps is None else dict(SPECS.get(key, {}), title=key, group=key.split(":")[0])
    p = check_params(key, params) if steps is None else dict(params or {})
    if any(j["key"] == key for j in active(root)):
        raise RuntimeError(f"“{spec['title']}”已经在排队或运行了")
    now = time.time()
    jid = time.strftime("%Y%m%d-%H%M%S", time.localtime(now)) + f"-{int(now * 1000) % 1000:03d}-{key.replace(':', '_')}"
    j = {"id": jid, "key": key, "title": spec["title"], "group": spec["group"], "params": p,
         "steps": steps if steps is not None else spec["build"](p), "stop": spec.get("stop", "kill"),
         "status": "queued", "created": now, "started": None, "ended": None, "rc": None, "pid": None}
    _write(_path(jid, root=root), j)
    _path(jid, "beat", root).write_text("", encoding="utf-8")
    if spawn:
        kw = {"creationflags": 0x08000000 | 0x00000200} if os.name == "nt" else {"start_new_session": True}
        subprocess.Popen([sys.executable, "-X", "utf8", "-m", "panel.runner", jid, *(["--root", str(root)] if root else [])],
                         cwd=str(ROOT), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kw)
    return j


def stop(jid: str, hard: bool = False, root: Path | None = None) -> None:
    """请求停止。训练（stop = file）先让它自己存盘退出；hard = 直接结束进程。排队中的直接取消。"""
    _path(jid, "stop", root).write_text("hard" if hard else "soft", encoding="utf-8")


def stop_requested(jid: str, root: Path | None = None) -> str:
    try:
        return _path(jid, "stop", root).read_text(encoding="utf-8").strip() or "soft"
    except OSError:
        return ""


def tail(jid: str, max_bytes: int = 24000, root: Path | None = None) -> str:
    """任务输出的最后一段。"""
    p = _path(jid, "log", root)
    try:
        with p.open("rb") as f:
            f.seek(0, 2)
            n = f.tell()
            f.seek(max(0, n - max_bytes))
            data = f.read()
    except OSError:
        return ""
    text = data.decode("utf-8", errors="replace")
    return text if n <= max_bytes else text.split("\n", 1)[-1]


def prune(root: Path | None = None, keep: int = 200) -> int:
    """只留最近 keep 个已经结束的任务的记录和输出。"""
    d = root or DIR
    old = [p for p in sorted(d.glob("*.json"), reverse=True)][keep:] if d.exists() else []
    n = 0
    for p in old:
        j = load(p.stem, root)
        if j and _status(j, root) in ACTIVE:
            continue
        for ext in ("json", "log", "beat", "stop"):
            _path(p.stem, ext, root).unlink(missing_ok=True)
        n += 1
    return n


def train_state(run: Path | None = None) -> dict:
    """训练进度（读日志最后一段，不解析整个文件）：{iter, time, age（秒：最后一轮到现在）, eval（最近一次考试等效名次）}。"""
    p = (run or RUN) / "log.jsonl"
    out = {"iter": None, "time": "", "age": None, "eval": None, "eval_iter": None}
    try:
        with p.open("rb") as f:
            f.seek(0, 2)
            n = f.tell()
            f.seek(max(0, n - 400_000))
            lines = f.read().decode("utf-8", errors="replace").splitlines()
        out["age"] = time.time() - p.stat().st_mtime
    except OSError:
        return out
    for line in reversed(lines):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if "decisions" not in r:
            continue
        if out["iter"] is None:
            out["iter"], out["time"] = r.get("iter"), r.get("time", "")
        if out["eval"] is None and r.get("eval_boost_score", r.get("eval_boost_place", r.get("eval_place"))) is not None:
            out["eval"] = r.get("eval_boost_score", r.get("eval_boost_place", r.get("eval_place")))
            out["eval_iter"] = r.get("iter")
        if out["eval"] is not None:
            break
    return out
