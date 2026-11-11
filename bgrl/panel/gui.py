"""程序看板（桌面程序，PySide6 + pyqtgraph）：训练看板、所有操作、竞技场对局，都在一个窗口里。双击 panel.bat。

只是界面：任务的启动 / 排队 / 停止在 panel/jobs.py（独立进程跑，关掉这个窗口不影响训练，再打开能接着看）；
要显示的数据在 panel/data.py 算。不开网页、不监听端口。
"""
from __future__ import annotations

import sys
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pyqtgraph as pg  # noqa: E402
from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402

from panel import data as D  # noqa: E402
from panel import jobs as J  # noqa: E402

PALETTE = ["#2f6fdf", "#e0702a", "#2e9e5b", "#c23b5b", "#7a5bd6", "#8a8a8a"]
ICON = {"queued": "🕒", "running": "▶", "done": "✔", "failed": "✘", "stopped": "■", "lost": "⚠"}
PARAM_ZH = {"games": "局数", "ckpt": "模型", "threads": "线程数", "update": "自动换新模型", "keep_from": "快照留到第几轮"}
CKPT_ZH = {"latest": "最新（latest.pt）", "best": "考试最好的（best.pt）"}
SHORT = {
    "train": "接着 runs\\v3 练。停止 = 这一轮跑完存盘再退出。",
    "train_restore": "模型退回考试最好的那一版，从那里接着练。",
    "teacher": "只模仿高分示范的老师模型，约 15–25 分钟。",
    "board": "按最新日志重画 dashboard.md 和曲线图。",
    "comptest": "打高分阵容的对手：成型比例、最终阵容、金币流向。",
    "comptest_seeded": "第 8–10 回合换成高手的成型场面，再自己打。",
    "goldflow": "前几回合的钱花在哪，和 6500+ 高手比。",
    "tavernbench": "论文的 368 题评测集，只用 CPU。",
    "prodemos": "data\\firestone 的高分对局 → 决策样本。",
    "cardvalue": "高手怎么挑卡：挑中倍率、关键卡、经济卡。",
    "playbook": "各套目标阵容的配套卡。改过阵容表后跑一次。",
    "cleanup_preview": "只列出会删什么、各有多大，不删。",
    "cleanup": "删旧训练目录、旧快照、去掉的功能留下的文件。",
    "selftest": "跑一遍单元测试，几分钟。",
    "info": "Python、PyTorch / 显卡、TavernGym 版本。",
}
PAGES = [("看板", None, ""),
         ("训练", ["train", "train_restore", "teacher", "board"], "训练和老师模型不能同时跑（后点的排队）。训练时可以同时跑一个测试。"),
         ("测试", ["comptest", "comptest_seeded", "goldflow", "tavernbench"], "测试一次跑一个，后点的排队。阵容测试的汇总也会出现在看板上。"),
         ("数据", ["prodemos", "cardvalue", "playbook"], "从本机的高分对局数据重新生成训练用的样本和统计。原始数据和样本只在本机。"),
         ("竞技场", None, ""),
         ("维护", ["cleanup_preview", "cleanup", "selftest", "info"], "清理前先点“看看会删什么”。训练在跑的时候，清理会排在训练后面。"),
         ("任务输出", None, "")]
ARENA_KEYS = ["arena_check", "arena_practice", "arena_match", "arena_loop"]
CARD_W, CARD_H = 300, 230

STYLE = """
QWidget { font-size: 13px; color: #1d1d1f; }
QMainWindow, #page { background: #f6f5f2; }
#card, #tile, #panel { background: #ffffff; border: 1px solid #e3e1dc; border-radius: 8px; }
#nav { background: #ecebe7; border: none; outline: none; font-size: 14px; }
#nav::item { padding: 10px 14px; }
#nav::item:selected { background: #2f6fdf; color: #ffffff; }
#mute { color: #6b6b70; }
#big { font-size: 20px; font-weight: 600; }
#h1 { font-size: 16px; font-weight: 600; }
#h2 { font-size: 14px; font-weight: 600; }
QPushButton { padding: 5px 14px; border: 1px solid #cfcdc7; border-radius: 6px; background: #ffffff; }
QPushButton:hover { border-color: #2f6fdf; }
QPushButton:disabled { color: #a0a0a6; background: #f0efec; }
QPushButton#primary { background: #2f6fdf; color: #ffffff; border-color: #2f6fdf; }
QPushButton#primary:disabled { background: #a9c1ee; border-color: #a9c1ee; color: #ffffff; }
QPushButton#danger { color: #c23b5b; }
QPushButton:checked { background: #2f6fdf; color: #ffffff; border-color: #2f6fdf; }
QPlainTextEdit { background: #1e1e22; color: #e6e6ea; border-radius: 6px; font-family: Consolas, "Cascadia Mono", "Noto Sans Mono CJK SC", monospace; font-size: 12px; }
QTableWidget { background: #ffffff; border: 1px solid #e3e1dc; gridline-color: #ecebe7; }
QHeaderView::section { background: #f0efec; border: none; border-bottom: 1px solid #e3e1dc; padding: 4px 6px; font-weight: 600; }
QListWidget#matches { background: #ffffff; border: 1px solid #e3e1dc; border-radius: 8px; }
QListWidget#matches::item { padding: 6px; border-bottom: 1px solid #f0efec; }
QListWidget#matches::item:selected { background: #e2ebfb; color: #1d1d1f; }
"""


def fmt_time(t) -> str:
    return time.strftime("%m-%d %H:%M:%S", time.localtime(t)) if t else "-"


def fmt_dur(sec) -> str:
    if sec is None:
        return "-"
    sec = int(sec)
    return f"{sec // 3600} 小时 {sec % 3600 // 60} 分" if sec >= 3600 else (f"{sec // 60} 分 {sec % 60} 秒" if sec >= 60 else f"{sec} 秒")


def label(text="", name="", wrap=False) -> QtWidgets.QLabel:
    w = QtWidgets.QLabel(text)
    if name:
        w.setObjectName(name)
    w.setWordWrap(wrap)
    w.setTextInteractionFlags(Qt.TextSelectableByMouse)
    return w


def clear_layout(lay) -> None:
    while lay.count():
        it = lay.takeAt(0)
        if it.widget():
            it.widget().deleteLater()
        elif it.layout():
            clear_layout(it.layout())


class LogView(QtWidgets.QPlainTextEdit):
    """任务输出。只把新增的内容接在后面（不整段重写）：选中的文字、滚动位置都不动，能正常选中、Ctrl+C 复制；
    本来停在最底下才跟着往下滚。右键菜单里有复制 / 全选。"""

    def __init__(self, tail_bytes: int = 200_000):
        super().__init__()
        self.setReadOnly(True)
        self.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.setMaximumBlockCount(20000)
        self.tail_bytes = tail_bytes
        self.path, self.pos, self.note = None, 0, None

    def show_note(self, text: str) -> None:
        """没有任务可看时显示一句说明。"""
        if self.note != text:
            self.path, self.pos, self.note = None, 0, text
            self.setPlainText(text)

    def follow(self, path) -> None:
        """跟着这个输出文件：换了文件就从头（最后 tail_bytes）读，否则只读新增的部分。"""
        path = Path(path)
        if path != self.path:
            self.path, self.pos, self.note = path, 0, None
            self.clear()
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            self.pos = max(0, size - self.tail_bytes)
            self._read(skip_partial=self.pos > 0, to_end=True)
            return
        self._read()

    def _read(self, skip_partial: bool = False, to_end: bool = False) -> None:
        try:
            with self.path.open("rb") as f:
                f.seek(0, 2)
                size = f.tell()
                if size < self.pos:                        # 文件变短了（重新开始写）：从头读
                    self.pos = 0
                    self.clear()
                if size == self.pos:
                    return
                f.seek(self.pos)
                data = f.read(size - self.pos)
        except OSError:
            return
        cut = len(data)
        while cut > 0 and cut > len(data) - 4 and (data[cut - 1] & 0xC0) == 0x80:      # 别把一个汉字从中间截开
            cut -= 1
        if 0 < cut <= len(data) and (data[cut - 1] & 0xC0) == 0xC0:
            cut -= 1
        data = data[:cut] if cut else data
        self.pos += len(data)
        text = data.decode("utf-8", errors="replace").replace("\r\n", "\n")
        if skip_partial:
            text = text.split("\n", 1)[-1]
        if not text:
            return
        bar = self.verticalScrollBar()
        at_bottom = to_end or bar.value() >= bar.maximum() - 4
        keep = bar.value()
        cur = QtGui.QTextCursor(self.document())           # 另开一个光标往末尾接：不动用户的选区
        cur.movePosition(QtGui.QTextCursor.End)
        cur.insertText(text)
        bar.setValue(bar.maximum() if at_bottom else keep)


def fill_table(tb: QtWidgets.QTableWidget, head: list[str], rows: list[list], bold_rows=()) -> None:
    tb.setColumnCount(len(head))
    tb.setHorizontalHeaderLabels(head)
    tb.setRowCount(len(rows))
    for i, r in enumerate(rows):
        for k in range(len(head)):
            it = QtWidgets.QTableWidgetItem(str(r[k]) if k < len(r) else "")
            if i in bold_rows:
                f = it.font()
                f.setBold(True)
                it.setFont(f)
            tb.setItem(i, k, it)
    tb.resizeColumnsToContents()
    tb.horizontalHeader().setStretchLastSection(True)


def make_table(max_rows: int = 14) -> QtWidgets.QTableWidget:
    tb = QtWidgets.QTableWidget()
    tb.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
    tb.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
    tb.verticalHeader().setVisible(False)
    tb.setAlternatingRowColors(True)
    tb.setProperty("max_rows", max_rows)
    return tb


def fit_table(tb: QtWidgets.QTableWidget) -> None:
    """表格高度 = 行数（最多 max_rows 行，再多自己滚）。"""
    n = min(tb.rowCount(), int(tb.property("max_rows") or 14))
    tb.setFixedHeight(tb.horizontalHeader().height() + n * tb.verticalHeader().defaultSectionSize() + 6)


class Plot(pg.PlotWidget):
    """曲线图：左键拖动平移、右键拖动缩放、左下角 A 复原；滚轮留给页面滚动。"""

    def __init__(self, height: int = 230):
        super().__init__()
        self.setBackground("w")
        self.setFixedHeight(height)
        self.showGrid(x=True, y=True, alpha=0.25)
        self.setMenuEnabled(False)
        for ax in ("left", "bottom"):
            self.getAxis(ax).setTextPen("#6b6b70")
            self.getAxis(ax).setPen("#d0cec8")

    def wheelEvent(self, ev):  # noqa: N802
        ev.ignore()


class Async(QtCore.QObject):
    """在后台线程里算，算完把结果送回界面线程。"""
    done = QtCore.Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.busy = False

    def run(self, fn) -> None:
        if self.busy:
            return
        self.busy = True

        def work():
            try:
                res = fn()
            except Exception:                              # noqa: BLE001
                res = RuntimeError(traceback.format_exc())
            self.done.emit(res)
        threading.Thread(target=work, daemon=True).start()


# ---------------- 任务卡片、任务页 ----------------

class JobCard(QtWidgets.QFrame):
    """一个任务：标题 + 状态、一句说明、参数、开始 / 停止。所有卡片一样大。要确认的任务点开始时弹确认框。"""

    def __init__(self, key: str, win):
        super().__init__()
        self.key, self.win, self.spec = key, win, J.SPECS[key]
        self.setObjectName("card")
        self.setFixedSize(CARD_W, CARD_H)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(6)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(label(self.spec["title"], "h2"))
        top.addStretch(1)
        self.state = label("", "mute")
        top.addWidget(self.state)
        lay.addLayout(top)
        d = label(SHORT.get(key, self.spec["desc"]), "mute", wrap=True)
        d.setToolTip(self.spec["desc"])
        lay.addWidget(d)
        self.inputs = {}
        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 2, 0, 0)
        for name, (kind, default, *rest) in self.spec["params"].items():
            if kind == "int":
                w = QtWidgets.QSpinBox()
                w.setRange(rest[0], rest[1])
                w.setValue(default)
            elif kind == "bool":
                w = QtWidgets.QCheckBox()
                w.setChecked(default)
            else:
                w = QtWidgets.QComboBox()
                for v in rest[0]:
                    w.addItem(CKPT_ZH.get(v, str(v)), v)
                w.setCurrentIndex(list(rest[0]).index(default))
            self.inputs[name] = w
            form.addRow(PARAM_ZH.get(name, name), w)
        lay.addLayout(form)
        lay.addStretch(1)
        self.note = label("", "mute", wrap=True)
        lay.addWidget(self.note)
        row = QtWidgets.QHBoxLayout()
        self.go = QtWidgets.QPushButton("开始")
        self.go.setObjectName("primary")
        self.go.clicked.connect(self.start)
        self.stop = QtWidgets.QPushButton("停止")
        self.stop.clicked.connect(lambda: self.ask_stop(False))
        self.kill = QtWidgets.QPushButton("直接结束")
        self.kill.setObjectName("danger")
        self.kill.clicked.connect(lambda: self.ask_stop(True))
        for b in (self.go, self.stop, self.kill):
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        self.job = None
        self.refresh(None)

    def params(self) -> dict:
        out = {}
        for name, w in self.inputs.items():
            out[name] = w.value() if isinstance(w, QtWidgets.QSpinBox) else (
                w.isChecked() if isinstance(w, QtWidgets.QCheckBox) else w.currentData())
        return out

    def start(self) -> None:
        self.win.start_job(self.key, self.params())

    def ask_stop(self, hard: bool) -> None:
        if self.job:
            J.stop(self.job["id"], hard=hard, root=self.win.jobs_root)
            self.win.tick()

    def refresh(self, job: dict | None) -> None:
        self.job = job
        on = job is not None
        self.go.setVisible(not on)
        self.stop.setVisible(on)
        asked = J.stop_requested(job["id"], self.win.jobs_root) if on else ""
        self.stop.setEnabled(on and not asked)
        self.kill.setVisible(bool(on and asked and job["stop"] == "file"))
        for w in self.inputs.values():
            w.setEnabled(not on)
        if on:
            self.state.setText(f"{ICON[job['state']]} {J.STATUS_ZH[job['state']]}")
            self.note.setText("已请求停止" + ("：这一轮跑完、存好进度就退出。" if job["stop"] == "file" else "。") if asked
                              else f"{fmt_time(job['started'] or job['created'])} 起")
        else:
            self.state.setText("")
            self.note.setText("")


class JobsPage(QtWidgets.QWidget):
    """一页任务：一排一样大的卡片，下面是这一页最近那个任务的输出。"""

    def __init__(self, keys: list[str], note: str, win):
        super().__init__()
        self.setObjectName("page")
        self.keys, self.win = keys, win
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 12)
        lay.addWidget(label(note, "mute", wrap=True))
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(12)
        self.cards = {k: JobCard(k, win) for k in keys}
        for c in self.cards.values():
            row.addWidget(c)
        row.addStretch(1)
        lay.addLayout(row)
        self.head = label("", "h2")
        lay.addWidget(self.head)
        self.out = LogView()
        lay.addWidget(self.out, 1)

    def refresh(self, jobs: list[dict]) -> None:
        act = {j["key"]: j for j in jobs if j["state"] in J.ACTIVE}
        for k, c in self.cards.items():
            c.refresh(act.get(k))
        mine = [j for j in jobs if j["key"] in self.keys]
        if not mine:
            self.head.setText("输出")
            self.out.show_note("这一页还没有跑过任务。点上面的“开始”，输出会显示在这里。")
            return
        j = next((x for x in mine if x["state"] == "running"), mine[0])
        dur = ((j["ended"] or time.time()) - j["started"]) if j["started"] else None
        self.head.setText(f"输出：{j['title']}　{ICON[j['state']]} {J.STATUS_ZH[j['state']]} · 开始 {fmt_time(j['started'])} · 用时 {fmt_dur(dur)}")
        self.out.follow(J._path(j["id"], "log", self.win.jobs_root))


class OutputPage(QtWidgets.QWidget):
    """任务输出：选一个任务看它的输出；下面是任务记录。"""

    def __init__(self, win):
        super().__init__()
        self.setObjectName("page")
        self.win = win
        self.ids: list[str] = []
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 12)
        self.pick = QtWidgets.QComboBox()
        lay.addWidget(self.pick)
        self.info = label("", "mute")
        lay.addWidget(self.info)
        self.out = LogView(2_000_000)
        lay.addWidget(self.out, 2)
        lay.addWidget(label("任务记录", "h2"))
        self.table = make_table(8)
        lay.addWidget(self.table, 1)

    def refresh(self, jobs: list[dict]) -> None:
        ids = [j["id"] for j in jobs]
        cur = self.pick.currentData()
        want = self.win.watch if self.win.watch in ids else (cur if cur in ids else (ids[0] if ids else None))
        self.win.watch = None
        self.pick.blockSignals(True)
        self.pick.clear()
        for j in jobs:
            self.pick.addItem(f"{ICON[j['state']]} {j['title']} · {fmt_time(j['created'])} · {J.STATUS_ZH[j['state']]}", j["id"])
        if want in ids:
            self.pick.setCurrentIndex(ids.index(want))
        self.pick.blockSignals(False)
        j = next((x for x in jobs if x["id"] == want), None)
        if j is None:
            self.info.setText("还没有跑过任务。")
            self.out.show_note("")
        else:
            dur = ((j["ended"] or time.time()) - j["started"]) if j["started"] else None
            self.info.setText(f"状态：{J.STATUS_ZH[j['state']]} · 开始 {fmt_time(j['started'])} · 用时 {fmt_dur(dur)}"
                              + (f" · 退出码 {j['rc']}" if j["rc"] not in (None, 0) else ""))
            self.out.follow(J._path(j["id"], "log", self.win.jobs_root))
        fill_table(self.table, ["任务", "状态", "排队", "开始", "结束", "用时", "参数"],
                   [[j["title"], f"{ICON[j['state']]} {J.STATUS_ZH[j['state']]}", fmt_time(j["created"]), fmt_time(j["started"]),
                     fmt_time(j["ended"]), fmt_dur((j["ended"] - j["started"]) if j["ended"] and j["started"] else None),
                     "、".join(f"{PARAM_ZH.get(k, k)}={v}" for k, v in j["params"].items())] for j in jobs])


# ---------------- 看板 ----------------

class Tile(QtWidgets.QFrame):
    def __init__(self, name: str, value: str, sub: str = "", color: str = "", width: int = 0):
        super().__init__()
        self.setObjectName("tile")
        if width:
            self.setFixedWidth(width)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(0)
        lay.addWidget(label(name, "mute"))
        lay.addWidget(label(value, "big"))
        s = label(sub, "mute")
        if color:
            s.setStyleSheet(f"color: {color};")
        lay.addWidget(s)


class DashboardPage(QtWidgets.QScrollArea):
    """训练看板：关键指标、曲线、阵容表、事件。日志在后台线程里读，读完再画。"""
    WINDOWS = [(500, "最近 500 轮"), (1000, "最近 1000 轮"), (3000, "最近 3000 轮"), (10000, "最近 10000 轮"), (10 ** 9, "全部")]

    def __init__(self, win):
        super().__init__()
        self.win = win
        self.setWidgetResizable(True)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        body = QtWidgets.QWidget()
        body.setObjectName("page")
        self.setWidget(body)
        self.lay = QtWidgets.QVBoxLayout(body)
        self.lay.setContentsMargins(16, 12, 16, 16)
        top = QtWidgets.QHBoxLayout()
        self.head = label("读训练日志……", "h1")
        top.addWidget(self.head, 1)
        self.win_pick = QtWidgets.QComboBox()
        for v, name in self.WINDOWS:
            self.win_pick.addItem(name, v)
        self.win_pick.setCurrentIndex(2)
        self.win_pick.currentIndexChanged.connect(self.draw)
        top.addWidget(label("曲线范围", "mute"))
        top.addWidget(self.win_pick)
        self.lay.addLayout(top)
        self.lay.addWidget(label("大数字是最近 50 轮的平均（考试指标是最近 3 次考试）；小字是和 500 轮前比，绿 = 变好，红 = 变差。", "mute"))
        self.kpi = QtWidgets.QGridLayout()
        self.kpi.setSpacing(8)
        self.lay.addLayout(self.kpi)
        self.lay.addWidget(label("曲线", "h1"))
        self.lay.addWidget(label("细线是每一轮的值，粗线是滑动平均。左键拖动平移，右键拖动缩放，左下角 A 复原。", "mute"))
        self.grid = QtWidgets.QGridLayout()
        self.grid.setSpacing(10)
        self.lay.addLayout(self.grid)
        self.plots: dict[str, tuple] = {}
        self.tables = QtWidgets.QVBoxLayout()
        self.lay.addLayout(self.tables)
        self.lay.addStretch(1)
        self.rows, self.events, self.stamp, self.last_load = [], [], None, 0.0
        self.loader = Async(self)
        self.loader.done.connect(self.loaded)

    def _stamp(self):
        p = self.win.run / "log.jsonl"
        try:
            s = p.stat()
            return (s.st_mtime, s.st_size)
        except OSError:
            return None

    def refresh(self, jobs=None, sync: bool = False) -> None:
        """日志变了（而且离上次读过了 30 秒）就重读。"""
        st = self._stamp()
        if st is None:
            self.head.setText("还没有训练日志（runs\\v3\\log.jsonl）。到“训练”页点开始。")
            return
        if st == self.stamp or (not sync and time.time() - self.last_load < 30 and self.stamp is not None):
            return
        from rl import board as B
        run = self.win.run
        if sync:
            self.loaded((B.load(run), st))
        else:
            self.loader.run(lambda: (B.load(run), st))

    def loaded(self, res) -> None:
        self.loader.busy = False
        self.last_load = time.time()
        if isinstance(res, Exception):
            self.head.setText(f"读日志出错：{res}")
            return
        (self.rows, self.events), self.stamp = res
        self.draw()

    def draw(self, *_a) -> None:
        rows = self.rows
        self.head.setText(D.header(rows))
        clear_layout(self.kpi)
        for i, (name, a, b, lower, d) in enumerate(D.kpis(rows)):
            sub, color = "", ""
            if b is not None and abs(a - b) > 1e-9:
                sub = f"{a - b:+.{d}f}"
                if lower is not None:
                    color = "#1f8a4c" if (a < b) == bool(lower) else "#c23b5b"
            self.kpi.addWidget(Tile(name, f"{a:.{d}f}", sub or "—", color), i // 6, i % 6)
        for i, (title, note, full, lines) in enumerate(D.charts(rows, int(self.win_pick.currentData()))):
            if title not in self.plots:
                box = QtWidgets.QFrame()
                box.setObjectName("panel")
                bl = QtWidgets.QVBoxLayout(box)
                bl.setContentsMargins(10, 8, 10, 8)
                bl.addWidget(label(title + ("（全程）" if full else ""), "h2"))
                leg = label("", wrap=True)                 # 图例放在图上面（放在图里会挡住线）
                leg.setTextFormat(Qt.RichText)
                bl.addWidget(leg)
                p = Plot()
                bl.addWidget(p)
                bl.addWidget(label(note, "mute", wrap=True))
                self.grid.addWidget(box, len(self.plots) // 2, len(self.plots) % 2)
                self.plots[title] = (p, leg)
            p, leg = self.plots[title]
            p.clear()
            leg.setText("　".join(f"<span style='color:{PALETTE[k % len(PALETTE)]}'>━</span> {ln[0]}" for k, ln in enumerate(lines)))
            for k, (name, xs, raw, sm, sparse) in enumerate(lines):
                c = PALETTE[k % len(PALETTE)]
                if sparse:
                    p.plot(xs, sm, pen=pg.mkPen(c, width=2), symbol="o", symbolSize=4, symbolBrush=c, symbolPen=None)
                else:
                    qc = QtGui.QColor(c)
                    qc.setAlpha(60)
                    p.plot(xs, raw, pen=pg.mkPen(qc, width=1))
                    p.plot(xs, sm, pen=pg.mkPen(c, width=2))
            ys = [v for ln in lines for v in ln[2]]
            if max(ys) - min(ys) < 1e-6:                   # 全是同一个数：自动范围会把刻度写成一长串小数
                p.setYRange(min(ys) - 0.05, max(ys) + 0.05, padding=0)
                p.enableAutoRange(axis="x")
            else:
                p.enableAutoRange()
        clear_layout(self.tables)
        for t in D.md_tables(self.win.run):
            self.tables.addWidget(label(t["title"], "h1"))
            if t["text"]:
                self.tables.addWidget(label(t["text"].replace("**", ""), "mute", wrap=True))
            if t["rows"]:
                tb = make_table()
                fill_table(tb, t["head"], t["rows"])
                fit_table(tb)
                self.tables.addWidget(tb)
        if self.events:
            self.tables.addWidget(label("事件（最近 15 条）", "h1"))
            tb = make_table(15)
            fill_table(tb, ["轮", "时间", "事件"], [[e.get("iter"), e.get("time", ""), e.get("text", "")] for e in self.events[-15:][::-1]])
            fit_table(tb)
            self.tables.addWidget(tb)


# ---------------- 竞技场 ----------------

PLACE_COLOR = {1: "#c99a12", 2: "#7d8ea8", 3: "#a86b3c", 4: "#2f6fdb"}


class CardChip(QtWidgets.QFrame):
    """一张随从卡片：名字、右上角等级 / 价格、底下攻 / 血；金色的描金边。small = 最终阵容里用的小号。"""

    def __init__(self, c: dict, small: bool = False):
        super().__init__()
        w, h = (78, 50) if small else (104, 66)
        self.setFixedSize(w, h)
        gold = c.get("golden")
        self.setStyleSheet(f"QFrame{{background:{'#fff7dc' if gold else '#ffffff'};border:{2 if gold else 1}px solid "
                           f"{'#d4a017' if gold else '#d6d3cc'};border-radius:6px}} QLabel{{border:none;background:transparent}}")
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(5, 3, 5, 3)
        lay.setSpacing(0)
        top = QtWidgets.QHBoxLayout()
        top.setSpacing(2)
        nm = QtWidgets.QLabel(("★" if gold else "") + c["name"])
        nm.setWordWrap(True)
        nm.setStyleSheet(f"font-size:{10 if small else 11}px;color:#1d1d1f")
        nm.setToolTip(("金色 " if gold else "") + c["name"])
        top.addWidget(nm, 1, Qt.AlignTop)
        tag = " ".join(x for x in (f"{c['tier']}本" if c.get("tier") else "", f"{c['cost']}金" if c.get("cost") is not None else "") if x)
        if tag and not small:
            t = QtWidgets.QLabel(tag)
            t.setStyleSheet("font-size:10px;color:#8a8a8a")
            top.addWidget(t, 0, Qt.AlignTop)
        lay.addLayout(top, 1)
        if c.get("atk") is not None:
            st = QtWidgets.QLabel(f"<span style='color:#c26a00'>{c['atk']}</span> / <span style='color:#c23b5b'>{c['hp']}</span>")
            st.setStyleSheet(f"font-size:{11 if small else 13}px;font-weight:600")
            lay.addWidget(st, 0, Qt.AlignRight)


class CardRow(QtWidgets.QWidget):
    """一个区（场面 / 手牌 / 酒馆 / 待选）：左边区名，右边一排卡片（多了自动换行）。"""

    def __init__(self, name: str, cards: list[dict], per_row: int = 8):
        super().__init__()
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lab = label(name, "mute")
        lab.setFixedWidth(40)
        lay.addWidget(lab, 0, Qt.AlignTop)
        g = QtWidgets.QGridLayout()
        g.setHorizontalSpacing(6)
        g.setVerticalSpacing(6)
        if not cards:
            g.addWidget(label("（空）", "mute"), 0, 0)
        for i, c in enumerate(cards):
            g.addWidget(CardChip(c), i // per_row, i % per_row)
        lay.addLayout(g)
        lay.addStretch(1)


class MatchRow(QtWidgets.QWidget):
    """对局列表一行：左边名次圆点，右边三行字（赛道 · 英雄 / 时间 · 回合 · 模型 / Elo 或说明）。"""

    def __init__(self, p: dict):
        super().__init__()
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(8)
        col = "#1f8a4c" if p["live"] else PLACE_COLOR.get(p["place"], "#8a8a8a")
        dot = QtWidgets.QLabel(p["badge"])
        dot.setFixedSize(24, 24)
        dot.setAlignment(Qt.AlignCenter)
        dot.setStyleSheet(f"background:{col};color:#ffffff;border-radius:12px;font-weight:600;font-size:12px")
        lay.addWidget(dot, 0, Qt.AlignTop)
        txt = QtWidgets.QVBoxLayout()
        txt.setSpacing(0)
        a = QtWidgets.QLabel(p["top"])
        a.setStyleSheet("font-weight:600;font-size:12px")
        txt.addWidget(a)
        b = QtWidgets.QLabel(p["mid"])
        b.setStyleSheet("color:#6b6b70;font-size:11px")
        txt.addWidget(b)
        if p["low"]:
            c = QtWidgets.QLabel(p["low"])
            c.setStyleSheet(f"color:{'#1f8a4c' if '+' in p['low'] else '#c23b5b' if 'Elo' in p['low'] else '#6b6b70'};font-size:11px")
            txt.addWidget(c)
        lay.addLayout(txt, 1)


class StandRow(QtWidgets.QFrame):
    """对局结束：一席——名次、选手、英雄 · 等级、最后阵容（小卡片）。自己那一席底色不一样。"""

    def __init__(self, r: tuple, cards: list[dict]):
        super().__init__()
        place, who, hero, tier, _txt, mine = r
        self.setStyleSheet("QFrame{background:%s;border-radius:6px} QLabel{background:transparent}" % ("#e2ebfb" if mine else "transparent"))
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(10)
        try:
            pn = int(str(place).split("（")[0])
        except ValueError:
            pn = None
        dot = QtWidgets.QLabel(str(place))
        dot.setFixedSize(26, 26)
        dot.setAlignment(Qt.AlignCenter)
        dot.setStyleSheet(f"background:{PLACE_COLOR.get(pn, '#8a8a8a')};color:#fff;border-radius:13px;font-weight:600")
        lay.addWidget(dot, 0, Qt.AlignVCenter)
        info = QtWidgets.QVBoxLayout()
        info.setSpacing(0)
        a = QtWidgets.QLabel(who or "-")
        a.setStyleSheet("font-weight:600" if mine else "")
        info.addWidget(a)
        info.addWidget(label(" · ".join(x for x in (hero, f"{tier} 本" if tier else "") if x), "mute"))
        w = QtWidgets.QWidget()
        w.setFixedWidth(150)
        w.setLayout(info)
        lay.addWidget(w)
        for c in cards:
            lay.addWidget(CardChip(c, small=True))
        lay.addStretch(1)

class ArenaPage(QtWidgets.QWidget):
    """竞技场：上面一排操作和战绩，左边对局列表，右边单局（默认跟着正在打的那一局）。"""

    def __init__(self, win):
        super().__init__()
        self.setObjectName("page")
        self.win = win
        self.V = D.load_viewer(win.arena_dir)
        self.sum, self.sel, self.turn, self.step, self.follow, self.data = None, None, None, None, True, None
        self.view, self.sig_stats, self.sig_list, self.shown = None, None, None, None
        self.matches: list[dict] = []
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(16, 10, 16, 10)
        lay.setSpacing(8)

        bar = QtWidgets.QHBoxLayout()
        self.btn = {}
        for key, text in zip(ARENA_KEYS, ["打包并检查", "练习局", "计分局（1 局）", "连续计分局"]):
            b = QtWidgets.QPushButton(text)
            b.setToolTip(J.SPECS[key]["desc"])
            b.clicked.connect(lambda _=False, k=key: self.start(k))
            self.btn[key] = b
            bar.addWidget(b)
        self.update_box = QtWidgets.QCheckBox("自动换新模型")
        self.update_box.setToolTip("连续计分局：训练出的模型比打包的新 2 小时以上就重新打包")
        bar.addWidget(self.update_box)
        self.stop_btn = QtWidgets.QPushButton("停止")
        self.stop_btn.setObjectName("danger")
        self.stop_btn.clicked.connect(self.stop)
        bar.addWidget(self.stop_btn)
        self.job_state = label("", "mute")
        bar.addWidget(self.job_state, 1)
        self.cat = QtWidgets.QComboBox()
        self.cat.currentIndexChanged.connect(lambda _i: self.draw_stats())
        bar.addWidget(self.cat)
        self.sync_btn = QtWidgets.QPushButton("同步竞技场战绩")
        self.sync_btn.clicked.connect(self.do_sync)
        bar.addWidget(self.sync_btn)
        lay.addLayout(bar)

        stats = QtWidgets.QHBoxLayout()
        stats.setSpacing(8)
        self.tiles = QtWidgets.QHBoxLayout()
        self.tiles.setSpacing(6)
        stats.addLayout(self.tiles)
        dbox, self.dist = self._mini("名次分布（第 1–8 名的局数）", 260)
        stats.addWidget(dbox)
        rbox, self.recent = self._mini("近期表现", 0)
        self.recent_title = rbox.findChild(QtWidgets.QLabel)
        stats.addWidget(rbox, 1)
        lay.addLayout(stats)

        split = QtWidgets.QSplitter(Qt.Horizontal)
        left = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        lh = QtWidgets.QHBoxLayout()
        lh.addWidget(label("对局（新的在上）", "mute"), 1)
        self.follow_btn = QtWidgets.QPushButton("跟随正在打的")
        self.follow_btn.setCheckable(True)
        self.follow_btn.setChecked(True)
        self.follow_btn.clicked.connect(self.set_follow)
        lh.addWidget(self.follow_btn)
        ll.addLayout(lh)
        self.list = QtWidgets.QListWidget()
        self.list.setObjectName("matches")
        self.list.setMinimumWidth(300)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list.setSpacing(0)
        self.list.currentRowChanged.connect(self.pick)
        ll.addWidget(self.list)
        split.addWidget(left)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        det = QtWidgets.QFrame()
        det.setObjectName("panel")
        scroll.setWidget(det)
        dl = QtWidgets.QVBoxLayout(det)
        dl.setContentsMargins(14, 10, 14, 12)
        dl.setSpacing(6)
        th = QtWidgets.QHBoxLayout()
        self.d_title = label("选左边一局查看。", "big")
        th.addWidget(self.d_title)
        self.d_sub = label("", "mute")
        th.addWidget(self.d_sub, 1)
        dl.addLayout(th)
        self.d_elo = label("")
        dl.addWidget(self.d_elo)
        self.curve_lbl = label("模型预计名次（每一步；越低越好）", "mute")
        dl.addWidget(self.curve_lbl)
        self.curve = Plot(120)
        self.curve.getPlotItem().invertY(True)
        dl.addWidget(self.curve)
        self.turns = QtWidgets.QTabBar()
        self.turns.setExpanding(False)
        self.turns.setDrawBase(False)
        self.turns.currentChanged.connect(self.pick_turn)
        dl.addWidget(self.turns)
        mid = QtWidgets.QHBoxLayout()
        mid.setSpacing(12)
        c1 = QtWidgets.QVBoxLayout()
        self.steps_lbl = label("这一回合的操作", "mute")
        c1.addWidget(self.steps_lbl)
        self.steps = QtWidgets.QListWidget()
        self.steps.setFixedHeight(150)
        self.steps.setMinimumWidth(260)
        self.steps.currentRowChanged.connect(self.pick_step)
        c1.addWidget(self.steps)
        mid.addLayout(c1, 2)
        self.stat_box = QtWidgets.QGridLayout()                # 本数 / 血量 / 金币 / 剩余人数
        self.stat_box.setSpacing(6)
        sw = QtWidgets.QWidget()
        sw.setLayout(self.stat_box)
        mid.addWidget(sw, 1, Qt.AlignTop)
        dl.addLayout(mid)
        self.zones = QtWidgets.QVBoxLayout()                   # 场面 / 手牌 / 酒馆 / 待选：一排排小卡片
        self.zones.setSpacing(2)
        dl.addLayout(self.zones)
        self.final_lbl = label("最终名次", "h2")
        dl.addWidget(self.final_lbl)
        self.final = QtWidgets.QVBoxLayout()
        self.final.setSpacing(2)
        dl.addLayout(self.final)
        self.d_note = label("", "mute", wrap=True)
        dl.addWidget(self.d_note)
        dl.addStretch(1)
        split.addWidget(scroll)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([330, 1000])
        bottom = QtWidgets.QWidget()                       # 对局区和输出区之间可以上下拖动分隔条
        bl2 = QtWidgets.QVBoxLayout(bottom)
        bl2.setContentsMargins(0, 0, 0, 0)
        bl2.setSpacing(4)
        self.out_head = label("竞技场任务的输出", "mute")
        bl2.addWidget(self.out_head)
        self.out = LogView()
        bl2.addWidget(self.out, 1)
        vsplit = QtWidgets.QSplitter(Qt.Vertical)
        vsplit.addWidget(split)
        vsplit.addWidget(bottom)
        vsplit.setStretchFactor(0, 1)
        vsplit.setStretchFactor(1, 0)
        vsplit.setSizes([640, 150])
        vsplit.setChildrenCollapsible(False)
        lay.addWidget(vsplit, 1)
        self.syncer = Async(self)
        self.syncer.done.connect(self.synced)

    def _mini(self, title: str, width: int):
        box = QtWidgets.QFrame()
        box.setObjectName("panel")
        if width:
            box.setFixedWidth(width)
        bl = QtWidgets.QVBoxLayout(box)
        bl.setContentsMargins(8, 4, 8, 4)
        bl.setSpacing(0)
        bl.addWidget(label(title, "mute"))
        p = Plot(92)
        p.setMouseEnabled(False, False)
        p.hideButtons()
        bl.addWidget(p)
        return box, p

    # ---- 操作
    def start(self, key: str) -> None:
        self.win.start_job(key, {"update": self.update_box.isChecked()} if key == "arena_loop" else {})

    def stop(self) -> None:
        """第一次点：连续计分局这一局打完就停（别的竞技场任务直接结束）；已经请求过再点：直接结束。"""
        for j in J.active(self.win.jobs_root, "arena"):
            asked = J.stop_requested(j["id"], self.win.jobs_root)
            J.stop(j["id"], hard=bool(asked) or j["stop"] != "file", root=self.win.jobs_root)
        self.win.tick()

    def do_sync(self) -> None:
        self.sync_btn.setEnabled(False)
        self.sync_btn.setText("同步中…")
        self.syncer.run(self.V.sync)

    def synced(self, _res) -> None:
        self.syncer.busy = False
        self.sync_btn.setEnabled(True)
        self.sync_btn.setText("同步竞技场战绩")
        self.refresh(None)

    def set_follow(self, on: bool) -> None:
        self.follow = bool(on)
        if self.follow:
            self.sel = None
        self.refresh(None)

    def _unfollow(self) -> None:
        self.follow = False
        self.follow_btn.setChecked(False)

    # ---- 刷新
    def refresh(self, jobs) -> None:
        jobs = J.jobs(self.win.jobs_root, 60) if jobs is None else jobs
        mine = [j for j in jobs if j["key"] in ARENA_KEYS]
        act = [j for j in mine if j["state"] in J.ACTIVE]
        for b in self.btn.values():
            b.setEnabled(not act)
        self.update_box.setEnabled(not act)
        self.stop_btn.setEnabled(bool(act))
        waiting = [j for j in act if j["stop"] == "file" and J.stop_requested(j["id"], self.win.jobs_root) == "soft"]
        self.stop_btn.setText("不等了，直接结束" if waiting else "停止")
        self.stop_btn.setToolTip("连续计分局：这一局打完就停，不会中途退出对局。再点一次才是直接结束（这一局算退赛）。")
        self.job_state.setText("　".join(f"{ICON[j['state']]} {j['title']}：" + ("这一局打完就停" if j in waiting else J.STATUS_ZH[j["state"]])
                                        + f"（{fmt_time(j['started'] or j['created'])} 起）" for j in act))
        if mine:
            j = act[0] if act else mine[0]
            self.out_head.setText(f"输出：{j['title']}　{ICON[j['state']]} {J.STATUS_ZH[j['state']]}")
            self.out.follow(J._path(j["id"], "log", self.win.jobs_root))
        try:
            self.sum = self.V.summary()
        except Exception as ex:                            # noqa: BLE001 —— 结果文件读不了不影响别的
            self.d_note.setText(f"读对局记录出错：{ex}")
            return
        self.matches = self.sum["matches"]
        live = next((m for m in self.matches if m.get("live")), None)
        if self.follow:
            want = live["id"] if live else (self.matches[0]["id"] if self.matches else None)
            if want != self.sel:
                self.sel, self.turn, self.step = want, None, None
        self.follow_btn.setText("跟随中：正在打的对局" if self.follow and live else ("跟随中：最近一局" if self.follow else "跟随正在打的"))
        self.draw_stats()
        self.draw_list()
        self.load_match()

    def draw_stats(self) -> None:
        if not self.sum:
            return
        cur = self.cat.currentData() or "all"
        sig = (cur, self.sum.get("model"), [(m["id"], m.get("placement"), m.get("score_after")) for m in self.matches])
        if sig == self.sig_stats:                          # 没有新结果：不重画（省得每次刷新闪一下）
            return
        self.sig_stats = sig
        st = D.arena_stats(self.matches, cur)
        self.cat.blockSignals(True)
        self.cat.clear()
        for k, name, n in st["cats"]:
            self.cat.addItem(f"{name} {n}", k)
        self.cat.setCurrentIndex(max(0, [c[0] for c in st["cats"]].index(cur) if cur in [c[0] for c in st["cats"]] else 0))
        self.cat.blockSignals(False)
        n = st["n"]
        clear_layout(self.tiles)
        items = [("参赛结果", str(n), ""), ("平均名次", f"{st['avg']:.2f}" if n else "-", ""),
                 ("第一", f"{st['top1'] / n:.0%}" if n else "-", f"{st['top1']} 局"),
                 ("前四", f"{st['top4'] / n:.0%}" if n else "-", f"{st['top4']} 局")]
        if st["elo"] is not None:
            items.append(("纯 AI Elo", f"{st['elo']:.1f}", f"{st['elo_n']} 局"))
        items.append(("当前模型", str(self.sum.get("model") or "-"), ""))
        for name, v, sub in items:
            self.tiles.addWidget(Tile(name, v, sub))
        self.dist.clear()
        self.dist.addItem(pg.BarGraphItem(x=list(range(1, 9)), height=st["dist"], width=0.6, brush="#2f6fdf", pen=None))
        self.dist.getAxis("bottom").setTicks([[(p, f"#{p}") for p in range(1, 9)]])
        self.dist.setXRange(0.4, 8.6, padding=0)
        self.dist.setYRange(0, max(1, max(st["dist"])) * 1.15, padding=0)
        self.recent.clear()
        rc = st["recent"]
        if rc:
            self.recent.plot(list(range(len(rc))), rc, pen=pg.mkPen("#2f6fdf", width=2))
        self.recent.getPlotItem().invertY(True)
        self.recent.setYRange(0.5, 8.5, padding=0)
        self.recent.getAxis("left").setTicks([[(1, "1"), (4, "4"), (8, "8")]])
        self.recent.getAxis("bottom").setTicks([[]])

        def f(v):
            return f"{v:.2f}" if v is not None else "-"
        self.recent_title.setText(f"近期表现：最近 {len(rc)} 局 · 均名 20 局 {f(st['avg20'])} / 50 局 {f(st['avg50'])} / 100 局 {f(st['avg100'])}"
                                  + (f" · 最近一局第 {rc[-1]} 名" if rc else ""))

    def draw_list(self) -> None:
        sig = [D.match_line(m) for m in self.matches]
        if sig == self.sig_list:                           # 列表没变：只把选中的那一行对上（不动滚动位置）
            row = next((i for i, m in enumerate(self.matches) if m["id"] == self.sel), -1)
            if row != self.list.currentRow():
                self.list.blockSignals(True)
                self.list.setCurrentRow(row)
                self.list.blockSignals(False)
            return
        self.sig_list = sig
        self.list.blockSignals(True)
        self.list.clear()
        row = -1
        for i, m in enumerate(self.matches):
            it = QtWidgets.QListWidgetItem()
            it.setData(Qt.UserRole, m["id"])
            w = MatchRow(D.match_parts(m))
            it.setSizeHint(QtCore.QSize(10, w.sizeHint().height() + 2))
            self.list.addItem(it)
            self.list.setItemWidget(it, w)
            if m["id"] == self.sel:
                row = i
        if row >= 0:
            self.list.setCurrentRow(row)
        self.list.blockSignals(False)

    def pick(self, row: int) -> None:
        if 0 <= row < len(self.matches):
            self._unfollow()
            self.sel, self.turn, self.step = self.matches[row]["id"], None, None
            self.load_match(force=True)

    def pick_turn(self, i: int) -> None:
        if self.data is not None and 0 <= i < len(self.view["turns"]):
            self._unfollow()
            self.turn, self.step = self.view["turns"][i], None
            self.draw_match()

    def pick_step(self, i: int) -> None:
        if self.data is not None and 0 <= i < len(self.view["steps"]):
            self._unfollow()
            self.step = self.view["steps"][i][0]
            self.draw_match()

    def load_match(self, force: bool = False) -> None:
        if not self.sel:
            return
        if not force and self.shown == self.sel and self.data is not None and self.data.get("result") is not None:
            return                                         # 打完的对局不会变：不用每次刷新都重读重画
        try:
            self.data = self.V.match(self.sel)
        except Exception as ex:                            # noqa: BLE001
            self.d_note.setText(f"读这一局出错：{ex}")
            return
        if self.follow and self.data.get("result") is None:
            self.turn, self.step = None, None
        self.shown = self.sel
        self.draw_match()

    def draw_match(self) -> None:
        v = self.view = D.match_view(self.data, self.turn, self.step, self.follow)
        self.turn, self.step = v["turn"], v["n"]
        self.d_title.setText(v["title"])
        self.d_sub.setText(v["sub"])
        self.d_elo.setText(v["elo"])
        self.d_elo.setVisible(bool(v["elo"]))
        has = bool(v["turns"])
        for w in (self.curve_lbl, self.curve, self.turns, self.steps, self.steps_lbl):
            w.setVisible(has)
        self.curve.clear()
        if len(v["curve"]) >= 2:
            self.curve.plot([c[0] for c in v["curve"]], [c[1] for c in v["curve"]], pen=pg.mkPen("#2f6fdf", width=2))
            self.curve.setYRange(0.5, 8.5, padding=0)
            if v["n"] is not None:
                self.curve.addItem(pg.InfiniteLine(pos=v["n"], angle=90, pen=pg.mkPen("#c23b5b", width=1)))
        self.turns.blockSignals(True)
        while self.turns.count():
            self.turns.removeTab(0)
        for t in v["turns"]:
            self.turns.addTab("选将" if t == 0 else f"第 {t} 回合")
        if v["turn"] in v["turns"]:
            self.turns.setCurrentIndex(v["turns"].index(v["turn"]))
        self.turns.blockSignals(False)
        self.steps.blockSignals(True)
        self.steps.clear()
        for k, (n, text, guard, place, used) in enumerate(v["steps"]):
            self.steps.addItem(f"{text}" + (f"（{guard}）" if guard else "") + f"　预计 {place if place is not None else '-'} 名")
            if n == v["n"]:
                self.steps.setCurrentRow(k)
        self.steps.blockSignals(False)
        clear_layout(self.stat_box)
        for k, (name, val) in enumerate(v["stats"]):
            self.stat_box.addWidget(Tile(name, val, width=110), k // 2, k % 2)
        clear_layout(self.zones)
        for name, cards in v["cards"]:
            self.zones.addWidget(CardRow(name, cards))
        clear_layout(self.final)
        self.final_lbl.setVisible(bool(v["final"]))
        for r, cards in zip(v["final"], v["final_cards"]):
            self.final.addWidget(StandRow(r, cards))
        self.d_note.setText(v["note"])


# ---------------- 主窗口 ----------------

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, run: Path | None = None, jobs_root: Path | None = None, arena_dir: Path | None = None, timers: bool = True):
        super().__init__()
        self.run, self.jobs_root, self.arena_dir = Path(run or J.RUN), jobs_root, arena_dir
        self.watch = None
        self.setWindowTitle("酒馆战旗 AI 面板")
        self.resize(1500, 940)
        central = QtWidgets.QWidget()
        central.setObjectName("page")
        self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        bar = QtWidgets.QFrame()
        bar.setObjectName("panel")
        bl = QtWidgets.QHBoxLayout(bar)
        bl.setContentsMargins(16, 6, 16, 6)
        self.stat = {}
        for k, name in (("train", "训练"), ("iter", "轮数"), ("age", "最后一轮"), ("eval", "最近考试（等效名次）"), ("jobs", "正在跑 / 排队")):
            box = QtWidgets.QVBoxLayout()
            box.setSpacing(0)
            box.addWidget(label(name, "mute"))
            self.stat[k] = label("-", "big")
            box.addWidget(self.stat[k])
            bl.addLayout(box, 1)
        self.hint = label("", "mute", wrap=True)
        bl.addWidget(self.hint, 2)
        outer.addWidget(bar)

        body = QtWidgets.QHBoxLayout()
        body.setSpacing(0)
        self.nav = QtWidgets.QListWidget()
        self.nav.setObjectName("nav")
        self.nav.setFixedWidth(120)
        self.stack = QtWidgets.QStackedWidget()
        self.pages = []
        for name, keys, note in PAGES:
            self.nav.addItem(name)
            page = DashboardPage(self) if name == "看板" else ArenaPage(self) if name == "竞技场" else \
                OutputPage(self) if name == "任务输出" else JobsPage(keys, note, self)
            self.pages.append(page)
            self.stack.addWidget(page)
        self.nav.currentRowChanged.connect(self.show_page)
        body.addWidget(self.nav)
        body.addWidget(self.stack, 1)
        outer.addLayout(body, 1)
        self.nav.setCurrentRow(0)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        if timers:
            self.timer.start(2500)
        self.tick()

    def closeEvent(self, ev) -> None:  # noqa: N802
        """关窗口：还有任务在跑就问一下——让它们接着跑，还是都停掉再退出。"""
        act = J.active(self.jobs_root)
        if act and not getattr(self, "_quiet_close", False):
            box = QtWidgets.QMessageBox(self)
            box.setWindowTitle("退出面板")
            box.setIcon(QtWidgets.QMessageBox.Question)
            box.setText(f"还有 {len(act)} 个任务在跑：" + "、".join(j["title"] for j in act))
            box.setInformativeText("让它们继续：面板关掉，任务在后台接着跑，下次打开面板能接着看。\n"
                                   "全部停止：训练这一轮跑完存盘后退出，连续计分局这一局打完后退出，别的任务直接结束。")
            keep = box.addButton("让它们继续，只关面板", QtWidgets.QMessageBox.AcceptRole)
            stop = box.addButton("全部停止并退出", QtWidgets.QMessageBox.DestructiveRole)
            box.addButton("取消", QtWidgets.QMessageBox.RejectRole)
            box.exec()
            if box.clickedButton() is stop:
                for j in act:
                    J.stop(j["id"], root=self.jobs_root)
            elif box.clickedButton() is not keep:
                ev.ignore()
                return
        self.timer.stop()
        ev.accept()

    def show_page(self, i: int) -> None:
        self.stack.setCurrentIndex(i)
        self.tick()

    def start_job(self, key: str, params: dict) -> None:
        spec = J.SPECS[key]
        if spec.get("confirm"):
            ok = QtWidgets.QMessageBox.question(self, spec["title"], spec["confirm"] + "\n\n确定开始吗？",
                                                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
            if ok != QtWidgets.QMessageBox.Yes:
                return
        try:
            j = J.start(key, params, root=self.jobs_root)
        except (RuntimeError, ValueError) as ex:
            QtWidgets.QMessageBox.warning(self, spec["title"], str(ex))
            return
        self.watch = j["id"]
        self.tick()

    def tick(self, sync: bool = False) -> None:
        """定时刷新：顶上的状态 + 当前这一页。"""
        jobs = J.jobs(self.jobs_root, 60)
        ts = J.train_state(self.run)
        act = [j for j in jobs if j["state"] in J.ACTIVE]
        tr = any(j["key"] == "train" and j["state"] == "running" for j in act)
        outside = not tr and ts["age"] is not None and ts["age"] < 180
        self.stat["train"].setText("运行中" if tr else ("运行中（面板外）" if outside else "没在跑"))
        self.stat["iter"].setText(str(ts["iter"]) if ts["iter"] is not None else "-")
        self.stat["age"].setText(fmt_dur(ts["age"]) + "前" if ts["age"] is not None else "-")
        self.stat["eval"].setText(f"{ts['eval']:.2f}" if isinstance(ts["eval"], (int, float)) else "-")
        self.stat["jobs"].setText(f"{sum(j['state'] == 'running' for j in act)} / {sum(j['state'] == 'queued' for j in act)}")
        self.hint.setText("训练是在面板外启动的（比如旧的 train.bat 窗口）：面板停不了它，要停请关那个窗口。" if outside else "")
        page = self.stack.currentWidget()
        if isinstance(page, DashboardPage):
            page.refresh(jobs, sync=sync)
        else:
            page.refresh(jobs)


def make_app() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    pg.setConfigOptions(antialias=True, foreground="#3f3f46")
    for fam in ("Microsoft YaHei UI", "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC"):
        if fam in QtGui.QFontDatabase.families():
            app.setFont(QtGui.QFont(fam, 10))
            break
    app.setStyleSheet(STYLE)
    return app


def main() -> int:
    log = ROOT / "reports" / "panel" / "gui.log"

    def hook(tp, val, tb):                                 # 没有黑窗口：出错写进日志，再弹出来
        text = "".join(traceback.format_exception(tp, val, tb))
        try:
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a", encoding="utf-8") as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S\n") + text + "\n")
        except OSError:
            pass
        if QtWidgets.QApplication.instance():
            QtWidgets.QMessageBox.critical(None, "面板出错了", text[-1500:] + f"\n\n（已记到 {log}，把它发给 Claude）")
    sys.excepthook = hook
    app = make_app()
    win = MainWindow()
    win.show()
    code = app.exec()
    import os
    os._exit(int(code))                                    # 窗口关了就让面板进程干净地退掉（不等后台线程、不跑清理钩子）；任务是别的进程，不受影响


if __name__ == "__main__":
    raise SystemExit(main())
