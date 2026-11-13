"""面板的任务层（panel/jobs.py、panel/runner.py）和它跑的两个新命令（tools/cleanup.py、tools/arena_run.py）。
不需要 torch、TavernGym；界面（panel/gui.py）在 CI 里另外用 tests/gui_smoke.py 打开一遍。"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

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
    tr, rs = J.SPECS["train"]["build"]({"threads": 8})[0], J.SPECS["train_restore"]["build"]({"threads": 8})[0]
    assert rs == tr + ["--restore", "runs/v3/best.pt"] and "--restore" not in tr and J.SPECS["train_restore"].get("confirm")
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
        j = J.start("train", steps=[[PY, "-c", code, str(root / "arena" / "STOP")]], root=root, spawn=False)
        j.update(stop="file", stop_file=str(root / "arena" / "STOP"))        # 连续计分局：自己的停止文件
        J._write(J._path(j["id"], root=root), j)
        th = threading.Thread(target=R.run, args=(j["id"], root, 0.1))
        th.start()
        _wait(root, j["id"], ("running",))
        J.stop(j["id"], root=root)
        th.join(30)
        assert J.load(j["id"], root)["status"] == "stopped" and "saved" in J.tail(j["id"], root=root)
        assert not (root / "arena" / "STOP").exists() and not J.STOP_FILE.exists()
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


def check_arena_loop_stop(tmp: Path):
    """连续计分局：点停止 = 这一局打完就不打下一局（不中途退出对局）。"""
    from tools import arena_run as A
    old = (A.STOP, A.play)
    A.STOP = tmp / "loop" / "STOP"
    A.STOP.parent.mkdir()
    played = []

    def play(rated):
        played.append(rated)
        if len(played) == 2:
            A.STOP.write_text("panel", encoding="utf-8")                    # 第 2 局打着的时候点了停止
        return 0
    A.play = play
    try:
        A.STOP.write_text("stale", encoding="utf-8")                        # 上次留下的不算
        assert A.loop(False, max_games=5) == 0 and played == [True, True] and not A.STOP.exists()
        assert J.SPECS["arena_loop"]["stop"] == "file" and J.SPECS["arena_loop"]["stop_file"].endswith("STOP")
        assert J.SPECS["arena_match"].get("stop", "kill") == "kill"
    finally:
        A.STOP, A.play = old


def check_arena_summary(tmp: Path):
    """一局打完只打印名次、分数、最后的场面；结果读不出来时原样打印。"""
    import contextlib
    import importlib.util
    import io
    import json
    spec = importlib.util.spec_from_file_location("arena_viewer", Path(__file__).resolve().parent.parent / "arena" / "viewer.py")
    V = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(V)
    V.RESULTS, V.SITE = tmp / "none.jsonl", tmp / "none.json"
    m = {"id": "m1", "category": "arena", "human_count": 3, "bot_count": 5, "turns": 13,
         "participants": [{"username": "me", "kind": "agent", "placement": 2, "score_before": 1500.0, "score_after": 1512.4},
                          {"username": "b1", "kind": "bot", "placement": 1}],
         "final_lineups": [{"username": "me", "hero": {"name_zh": "托里姆"}, "tavern_tier": 5,
                            "board": [{"name_zh": "布莱恩·铜须", "attack": 12, "health": 14, "golden": True}]}]}
    p = tmp / "last.json"
    p.write_text(json.dumps({"type": "match_finished", "match": m}, ensure_ascii=False), encoding="utf-8")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert V.print_last(str(p)) == 0
    out = buf.getvalue()
    assert "第 2 名" in out and "1512.4" in out and "+12.4" in out and "★布莱恩·铜须 12/14" in out and "托里姆" in out, out
    assert len(out.strip().splitlines()) == 3 and "participants" not in out, out
    m2 = dict(m, id="m0", participants=[{"username": "me", "kind": "agent", "placement": 5}, {"username": "other", "kind": "agent", "placement": 1}])
    m["participants"].append({"username": "rival", "kind": "agent", "placement": 3})
    V.RESULTS = tmp / "hist.jsonl"                          # 这一局有两个 AI：按历史对局里每局都在的账号认出自己
    V.RESULTS.write_text(json.dumps({"type": "match_finished", "match": m2}, ensure_ascii=False) + "\n", encoding="utf-8")
    p.write_text(json.dumps({"type": "match_finished", "match": m}, ensure_ascii=False), encoding="utf-8")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert V.print_last(str(p)) == 0
    assert "第 2 名" in buf.getvalue(), buf.getvalue()
    p.write_text('{"error": "queue timeout"}', encoding="utf-8")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert V.print_last(str(p)) == 1
    assert "queue timeout" in buf.getvalue()


def check_data(tmp: Path):
    """面板要显示的数据（panel/data.py）：看板的指标 / 曲线 / 表，竞技场的战绩、对局列表、单局每一步。"""
    import gui_smoke as S
    from panel import data as D
    from rl import board as B
    S.fake_run(tmp / "run")
    S.fake_arena(tmp / "arena2")
    rows, _ev = B.load(tmp / "run")
    ks = D.kpis(rows)
    assert len(ks) >= 12 and ks[0][0] == "考试等效名次" and ks[0][3] is True and ks[0][2] is not None, ks[:2]
    ch = D.charts(rows, 300)
    assert len(ch) >= 9 and ch[0][2] and all(len(ln[1]) == len(ln[2]) == len(ln[3]) for c in ch for ln in c[3])
    assert min(ch[2][3][0][1]) >= rows[-1]["iter"] - 300 and min(ch[0][3][0][1]) < rows[-1]["iter"] - 300      # 考试那张看全程
    tabs = D.md_tables(tmp / "run")
    cg = next(t for t in tabs if t["title"].startswith("送整套阵容"))
    assert cg["head"][0] == "阵容" and any(r[0] == "圣盾流龙族" for r in cg["rows"]) and "第 5–11 回合" in cg["text"], cg
    assert "第 22200 轮" in D.header(rows) and D.header([]) == "还没有训练日志"
    V = D.load_viewer(tmp / "arena2")
    s = V.summary()
    assert s["model"] == "capri-it22170" and s["matches"][0]["live"] and len(s["matches"]) == 177
    st = D.arena_stats(s["matches"])
    assert st["n"] == 176 and sum(st["dist"]) == 176 and st["top1"] == st["dist"][0] and len(st["recent"]) == 100
    assert [c[0] for c in st["cats"]][0] == "all" and sum(c[2] for c in st["cats"][1:]) == 176
    one = D.arena_stats(s["matches"], "standard")
    assert one["n"] == one["elo_n"] > 0 and one["elo"] is not None and D.arena_stats(s["matches"], "practice")["elo"] is None
    assert D.match_line(s["matches"][0])[0] == "●" and "预计" in D.match_line(s["matches"][0])[1]
    live = D.match_view(V.match("live001"), follow=True)
    assert live["live"] and live["turn"] == 6 and live["n"] == live["steps"][-1][0] and live["cands"][0] == ("升本", 0.62, True)
    assert dict(live["zones"])["手牌"] == ["★希望信使 2/3"] and ("gold", 6) in live["scalars"] and not live["final"]
    done = D.match_view(V.match("m0175"), turn=2)
    assert not done["live"] and done["title"].startswith("第 ") and done["turn"] == 2 and done["n"] == done["steps"][0][0]
    assert len(done["final"]) == 2 and done["final"][0][5] != done["final"][1][5]
    assert D.match_view(V.match("m0100"))["note"] and not D.match_view(V.match("m0100"))["turns"]


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
        check_arena_summary(d)
        check_arena_loop_stop(d)
        check_data(d)
    print("panel OK：任务排队 / 停止 / 中断、清理、竞技场密钥")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
