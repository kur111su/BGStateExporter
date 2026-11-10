"""面板的任务层（panel/jobs.py、panel/runner.py）和它跑的两个新命令（tools/cleanup.py、tools/arena_run.py）。
不需要 Streamlit、torch、TavernGym；界面（panel/app.py）在 CI 里另外用 Streamlit 的 AppTest 打开一遍。"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from panel import jobs as J  # noqa: E402
from panel import runner as R  # noqa: E402

PY = sys.executable


def _wait(root, jid, states, timeout=30.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = J.load(jid, root)
        if j and j["status"] in states:
            return j
        time.sleep(0.1)
    raise AssertionError((jid, J.load(jid, root), J.tail(jid, root=root)))


def check_specs():
    """每个任务的命令都是 bg.py 的固定命令；参数按类型、范围检查，不认识的不收；密钥不出现在命令里。"""
    for key, spec in J.SPECS.items():
        p = J.check_params(key, None)
        steps = spec["build"](p)
        assert steps and all(s[0] == PY and s[1:4] == ["-X", "utf8", "bg.py"] for s in steps), (key, steps)
        assert all(isinstance(a, str) for s in steps for a in s), (key, steps)
        assert not any("key" in a.lower() and "arena" in a.lower() for s in steps for a in s), (key, steps)
    assert J.SPECS["comptest"]["build"](J.check_params("comptest", {"games": 256, "ckpt": "best"}))[0][4:] == \
        ["goldflow", "--boosted", "--games", "256", "--ckpt", "runs/v3/best.pt"]
    assert J.SPECS["arena_loop"]["build"]({"update": True})[0][4:] == ["arena", "loop", "--update"]
    assert J.SPECS["arena_match"].get("confirm") and J.SPECS["arena_loop"].get("confirm") and J.SPECS["cleanup"].get("confirm")
    for bad in ({"games": 5}, {"games": "64; rm -rf"}, {"games": True}, {"ckpt": "../x.pt"}, {"cmd": "x"}):
        try:
            J.check_params("comptest", bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def check_run_and_queue(root: Path):
    """独立进程跑任务：输出进日志；同组的排队，按先后一个个跑；不同组的同时跑。"""
    a = J.start("g:a", steps=[[PY, "-c", "import time; print('A1', flush=True); time.sleep(2.5); print('A2')"]], root=root)
    time.sleep(0.05)
    b = J.start("g:b", steps=[[PY, "-c", "print('B')"], [PY, "-c", "print('B2')"]], root=root)
    c = J.start("h:c", steps=[[PY, "-c", "print('C')"]], root=root)
    try:
        J.start("g:a", steps=[[PY, "-c", "pass"]], root=root)
        raise AssertionError("同一个任务排了两次")
    except RuntimeError:
        pass
    jc = _wait(root, c["id"], ("done",))
    assert J.load(b["id"], root)["status"] == "queued", J.load(b["id"], root)       # a 还没完，b 等着；c 不同组，已经跑完
    ja = _wait(root, a["id"], ("done",))
    jb = _wait(root, b["id"], ("done",))
    assert jb["started"] >= ja["ended"] - 0.05 and jc["ended"] < ja["ended"], (ja, jb, jc)
    assert "A1" in J.tail(a["id"], root=root) and "A2" in J.tail(a["id"], root=root)
    assert J.tail(b["id"], root=root).count("$ ") == 2 and "B2" in J.tail(b["id"], root=root)
    assert [j["id"] for j in J.jobs(root)][:1] == [c["id"]] and not J.active(root)
    f = J.start("g:f", steps=[[PY, "-c", "import sys; sys.exit(3)"], [PY, "-c", "print('never')"]], root=root)
    jf = _wait(root, f["id"], ("failed",))
    assert jf["rc"] == 3 and "never" not in J.tail(f["id"], root=root)


def check_stop(root: Path):
    """停止：普通任务直接结束（连它起的子进程）；排队中的取消；训练那种先放 STOP 文件让它自己退出。"""
    a = J.start("s:a", steps=[[PY, "-c", "import time; print('go', flush=True); time.sleep(120)"]], root=root)
    b = J.start("s:b", steps=[[PY, "-c", "print('B')"]], root=root)
    _wait(root, a["id"], ("running",))
    J.stop(b["id"], root=root)
    assert _wait(root, b["id"], ("stopped",))["started"] is None            # 排队中的：没跑就取消了
    t0 = time.time()
    J.stop(a["id"], root=root)
    ja = _wait(root, a["id"], ("stopped",))
    assert time.time() - t0 < 20 and ja["ended"] and "go" in J.tail(a["id"], root=root)

    old = J.STOP_FILE
    J.STOP_FILE = root / "run" / "STOP"
    try:
        code = ("import time, pathlib, sys; p = pathlib.Path(sys.argv[1]); print('loop', flush=True)\n"
                "while not p.exists(): time.sleep(0.1)\nprint('saved', flush=True)")
        j = J.start("train", steps=[[PY, "-c", code, str(J.STOP_FILE)]], root=root, spawn=False)
        j["stop"] = "file"
        J._write(J._path(j["id"], root=root), j)
        th = threading.Thread(target=R.run, args=(j["id"], root, 0.1))
        th.start()
        _wait(root, j["id"], ("running",))
        J.stop(j["id"], root=root)
        th.join(30)
        jj = J.load(j["id"], root)
        assert jj["status"] == "stopped" and jj["rc"] == 0 and "saved" in J.tail(j["id"], root=root), jj
        assert not J.STOP_FILE.exists()                                     # 用完收走，下次启动训练不会被它停掉
    finally:
        J.STOP_FILE = old


def check_lost_and_prune(root: Path):
    """心跳停了的“运行中”算中断，不挡后面的任务；prune 只留最近的记录。"""
    j = J.start("g:lost", steps=[[PY, "-c", "pass"]], root=root, spawn=False)
    j["status"] = "running"
    J._write(J._path(j["id"], root=root), j)
    old = time.time() - J.BEAT_DEAD - 5
    os.utime(J._path(j["id"], "beat", root), (old, old))
    assert J.jobs(root)[0]["state"] == "lost" and not J.active(root, "g")
    n = len(list(root.glob("*.json")))
    assert J.prune(root, keep=2) == n - 2 and len(list(root.glob("*.json"))) == 2


def check_train_state(tmp: Path):
    run = tmp / "v3"
    run.mkdir()
    (run / "log.jsonl").write_text(
        '{"iter": 10, "decisions": 5, "time": "2026-11-10 10:00:00", "eval_boost_score": 4.5}\n'
        '{"iter": 11, "rollback": "x"}\n{"iter": 12, "decisions": 5, "time": "2026-11-10 10:01:00"}\n', encoding="utf-8")
    s = J.train_state(run)
    assert s["iter"] == 12 and s["eval"] == 4.5 and s["eval_iter"] == 10 and s["age"] < 60, s
    assert J.train_state(tmp / "none")["iter"] is None


def check_cleanup(tmp: Path):
    """清理：旧目录、界线以前的快照（每 1000 轮留一个）、旧 bat；不碰 latest.pt 和界线以后的快照。"""
    from tools import cleanup as C
    root, runs = tmp / "repo" / "bgrl", tmp / "repo" / "bgrl" / "runs"
    (runs / "v3" / "snapshots").mkdir(parents=True)
    (runs / "v2").mkdir()
    (runs / "v2" / "latest.pt").write_bytes(b"x" * 10)
    (runs / "v3" / "latest.pt").write_bytes(b"x")
    for n in (50, 1000, 18950, 19000, 19050):
        (runs / "v3" / "snapshots" / f"it{n:05d}.pt").write_bytes(b"x" * 4)
    (root / "train.bat").write_text("x")
    (root / "panel.bat").write_text("x")
    (root.parent / "plugin").mkdir()
    items, kept = C.plan(19000, root, runs)
    names = {p.name for _, p, _ in items}
    assert names == {"v2", "it00050.pt", "it18950.pt", "train.bat", "plugin"} and kept == 3, (names, kept)
    assert sum(n for _, _, n in items) == 10 + 8 + 1


def check_arena_key(tmp: Path):
    """密钥只进连接器的环境变量；没有密钥就不打。"""
    from tools import arena_run as A
    old, old_env = A.ARENA, os.environ.pop("ARENA_API_KEY", None)
    A.ARENA = tmp / "arena"
    A.ARENA.mkdir()
    try:
        assert A.key_env() is None and A.url() == "https://rizling.top"
        (A.ARENA / "key.txt").write_text("  secret-123 \n", encoding="utf-8")
        (A.ARENA / "url.txt").write_text("https://example.test\n", encoding="utf-8")
        assert A.key_env()["ARENA_API_KEY"] == "secret-123" and A.url() == "https://example.test"
    finally:
        A.ARENA = old
        if old_env is not None:
            os.environ["ARENA_API_KEY"] = old_env


def main() -> int:
    check_specs()
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        check_run_and_queue(d / "q")
        check_stop(d / "s")
        check_lost_and_prune(d / "q")
        check_train_state(d)
        check_cleanup(d)
        check_arena_key(d)
    print("panel OK：任务排队 / 停止 / 中断、清理、竞技场密钥")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
