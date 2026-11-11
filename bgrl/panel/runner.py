"""跑一个面板任务的独立进程（panel/jobs.start 起的）：python -m panel.runner <任务 id>

等同组里排在前面的任务跑完 → 一步步跑命令，输出写进 <id>.log → 记下结果。每秒更新心跳、看有没有停止请求。
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from panel import jobs as J  # noqa: E402


def _kill(p: subprocess.Popen) -> None:
    """结束命令和它起的所有子进程。"""
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            os.killpg(p.pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        pass


def run(jid: str, root: Path | None = None, poll: float = 1.0) -> int:
    j = J.load(jid, root)
    if j is None:
        return 2
    path, beat, log = J._path(jid, root=root), J._path(jid, "beat", root), J._path(jid, "log", root)

    def save(**kw):
        j.update(kw)
        J._write(path, j)

    def touch():
        try:
            os.utime(beat, None)
        except OSError:
            beat.write_text("", encoding="utf-8")

    save(pid=os.getpid())
    while True:                                            # 排队：同组里比我早、还没结束的任务
        touch()
        if J.stop_requested(jid, root):
            save(status="stopped", ended=time.time())
            return 0
        ahead = [o for o in J.active(root, j["group"]) if o["id"] != jid and (o["created"], o["id"]) < (j["created"], jid)]
        if not ahead:
            break
        time.sleep(poll)
    save(status="running", started=time.time())
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    kw = {"creationflags": 0x00000200} if os.name == "nt" else {"start_new_session": True}
    rc, stopped, asked = 0, False, False
    stop_file = Path(j["stop_file"]) if j.get("stop_file") else J.STOP_FILE
    with log.open("ab") as out:
        for step in j["steps"]:
            out.write((f"$ {' '.join(Path(step[0]).name if k == 0 else str(a) for k, a in enumerate(step))}\n").encode("utf-8"))
            out.flush()
            try:
                p = subprocess.Popen(step, cwd=str(J.ROOT), stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                                     env=env, **kw)
            except OSError as ex:
                out.write(f"启动不了：{ex}\n".encode("utf-8"))
                rc = 127
                break
            while p.poll() is None:
                touch()
                how = J.stop_requested(jid, root)
                if how:
                    stopped = True
                    if j["stop"] == "file" and how != "hard":
                        if not asked:                      # 放一个 STOP 文件：训练这一轮跑完存盘退出 / 连续计分局这一局打完就停
                            stop_file.parent.mkdir(parents=True, exist_ok=True)
                            stop_file.write_text("panel", encoding="utf-8")
                            out.write("已请求停止：手上这一轮 / 这一局做完就退出……\n".encode("utf-8"))
                            out.flush()
                            asked = True
                    else:
                        _kill(p)
                        try:
                            p.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            p.kill()
                time.sleep(poll)
            rc = p.returncode
            if rc != 0 or stopped:
                break
    if asked:
        stop_file.unlink(missing_ok=True)
    save(status="stopped" if stopped else ("done" if rc == 0 else "failed"), rc=rc, ended=time.time())
    touch()
    return 0


def main() -> int:
    a = sys.argv[1:]
    root = Path(a[a.index("--root") + 1]) if "--root" in a else None
    return run(a[0], root)


if __name__ == "__main__":
    raise SystemExit(main())
