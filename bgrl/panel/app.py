"""程序看板（界面层，Streamlit）：训练看板 + 所有操作的按钮 + 任务输出。双击 panel.bat 或 python bg.py panel。

只负责显示和点按钮：任务的启动 / 排队 / 停止都在 panel/jobs.py（任务层，独立进程跑，关掉这个页面不影响训练）。
看板的数字和曲线直接读 runs/v3/log.jsonl（rl/board.py 的 load / KPIS / PANELS），表格取 runs/v3/dashboard.md 里现成的。
只监听本机 127.0.0.1:8770。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from panel import jobs as J  # noqa: E402
from rl import board as B  # noqa: E402

RUN = J.RUN
ICON = {"queued": "🕒", "running": "▶️", "done": "✅", "failed": "❌", "stopped": "⏹️", "lost": "⚠️"}
PARAM_ZH = {"games": "局数", "ckpt": "用哪个模型", "threads": "线程数", "update": "自动换新模型", "keep_from": "快照留到第几轮为止"}
CKPT_ZH = {"latest": "最新（latest.pt）", "best": "考试最好的（best.pt）"}

st.set_page_config(page_title="酒馆战旗 AI 面板", page_icon="🍺", layout="wide")


def fmt_time(t) -> str:
    return time.strftime("%m-%d %H:%M:%S", time.localtime(t)) if t else "-"


def fmt_dur(sec) -> str:
    if sec is None:
        return "-"
    sec = int(sec)
    return f"{sec // 3600} 小时 {sec % 3600 // 60} 分" if sec >= 3600 else (f"{sec // 60} 分 {sec % 60} 秒" if sec >= 60 else f"{sec} 秒")


# ---------------- 任务：按钮、状态、输出 ----------------

@st.cache_data(ttl=1.5, show_spinner=False)
def all_jobs() -> list[dict]:
    """最近的任务。页面上十来块东西各自几秒刷新一次，都从这里拿：1.5 秒内只读一次磁盘。"""
    return J.jobs(limit=60)


def active_jobs() -> list[dict]:
    return [j for j in all_jobs() if j["state"] in J.ACTIVE]


def param_inputs(key: str) -> dict:
    """SPECS 里这个任务的参数 → 输入框，返回填的值。"""
    out = {}
    for name, (kind, default, *rest) in J.SPECS[key]["params"].items():
        label = PARAM_ZH.get(name, name)
        wid = f"p_{key}_{name}"
        if kind == "int":
            out[name] = int(st.number_input(label, min_value=rest[0], max_value=rest[1], value=default, step=1, key=wid))
        elif kind == "bool":
            out[name] = bool(st.checkbox(label, value=default, key=wid))
        elif kind == "choice":
            out[name] = st.selectbox(label, list(rest[0]), index=list(rest[0]).index(default), key=wid,
                                     format_func=lambda v: CKPT_ZH.get(v, v))
    return out


# 卡片上的一句话说明（详细的在标题旁边的问号里）；每行 4 张卡片，同一行一样高
SHORT = {
    "train": "接着 runs\\v3 练。停止 = 这一轮跑完存盘再退出。",
    "teacher": "只模仿高分示范的老师模型，约 15–25 分钟。",
    "board": "按最新日志重画 dashboard.md 和曲线图。",
    "comptest": "打高分阵容的对手：成型比例、最终阵容、金币流向。",
    "comptest_seeded": "第 8–10 回合换成高手的成型场面，再自己打。",
    "goldflow": "前几回合的钱花在哪，和 6500+ 高手比。",
    "tavernbench": "论文的 368 题评测集，只用 CPU。",
    "prodemos": "data\\firestone 的高分对局 → 决策样本。",
    "cardvalue": "高手怎么挑卡：挑中倍率、关键卡、经济卡。",
    "playbook": "各套目标阵容的配套卡。改过阵容表后跑一次。",
    "arena_check": "打包 latest.pt，核对竞技场契约（不计分）。",
    "arena_practice": "打一局练习，不计分。",
    "arena_match": "打一局计分对局，打完就停。",
    "arena_loop": "一局接一局打计分对局，直到点停止。",
    "cleanup_preview": "只列出会删什么、各有多大，不删。",
    "cleanup": "删旧训练目录、旧快照、去掉的功能留下的文件。",
    "selftest": "跑一遍单元测试，几分钟。",
    "info": "Python、PyTorch / 显卡、TavernGym 版本。",
}
PER_ROW = 4


def card_height(key: str) -> int:
    """卡片要多高：标题 + 一句说明 + 每个参数一行 + 确认框 + 按钮。同一行取最高的那张。"""
    spec = J.SPECS[key]
    h = 200
    for kind, *_ in spec["params"].values():
        h += 52 if kind == "bool" else 90
    return h + (96 if spec.get("confirm") else 0)


def job_card(key: str, height: int) -> None:
    """一个任务：标题、一句说明、参数、开始按钮（要确认的先勾确认）；正在跑的话显示状态和停止按钮。"""
    spec = J.SPECS[key]
    with st.container(border=True, height=height):
        mine = [j for j in active_jobs() if j["key"] == key]
        state = f"　{ICON[mine[0]['state']]} {J.STATUS_ZH[mine[0]['state']]}" if mine else ""
        st.markdown(f"**{spec['title']}**{state}", help=spec["desc"])
        st.caption(SHORT.get(key, spec["desc"]))
        if mine:
            j = mine[0]
            st.caption(f"{fmt_time(j['started'] or j['created'])} 起 · 输出在下面")
            asked = J.stop_requested(j["id"])
            if st.button("停止", key=f"stop_{key}", disabled=bool(asked)):
                J.stop(j["id"])
                all_jobs.clear()
                st.rerun()
            if asked:
                st.caption("已请求停止" + ("：这一轮跑完、存好进度就退出（一般 1 分钟内）。" if j["stop"] == "file" else "。"))
                if j["stop"] == "file" and st.button("不等了，直接结束", key=f"kill_{key}"):
                    J.stop(j["id"], hard=True)
                    all_jobs.clear()
                    st.rerun()
            return
        params = param_inputs(key)
        ok = True
        if spec.get("confirm"):
            ok = st.checkbox(f"我确认：{spec['confirm']}", key=f"ok_{key}")
        if st.button("开始", key=f"go_{key}", type="primary", disabled=not ok):
            try:
                j = J.start(key, params)
                st.session_state["watch"] = j["id"]
                st.toast(f"已开始：{spec['title']}")
                all_jobs.clear()
                st.rerun()
            except (RuntimeError, ValueError) as ex:
                st.error(str(ex))


@st.fragment(run_every=6)
def cards(keys: list[str]) -> None:
    """一页的任务卡片：每行 PER_ROW 张、一样宽，同一行一样高。"""
    for i in range(0, len(keys), PER_ROW):
        row = keys[i:i + PER_ROW]
        h = max(card_height(k) for k in row)
        for col, key in zip(st.columns(PER_ROW), row):
            with col:
                job_card(key, h)


@st.fragment(run_every=4)
def tab_output(keys: list[str]) -> None:
    """这一页最近跑的那个任务的输出（最后一段）；全部任务的输出在“任务输出”页。"""
    js = [j for j in all_jobs()[:40] if j["key"] in keys]
    if not js:
        st.caption("这一页还没有跑过任务。点上面的“开始”，输出会显示在这里。")
        return
    j = next((x for x in js if x["state"] == "running"), js[0])
    dur = ((j["ended"] or time.time()) - j["started"]) if j["started"] else None
    st.markdown(f"**输出：{j['title']}**　{ICON[j['state']]} {J.STATUS_ZH[j['state']]} · 开始 {fmt_time(j['started'])} · 用时 {fmt_dur(dur)}")
    with st.container(border=True, height=380):
        st.code("\n".join((J.tail(j["id"], 12000) or "（还没有输出）").splitlines()[-60:]), language=None)


def page(keys: list[str], note: str) -> None:
    st.caption(note)
    cards(keys)
    tab_output(keys)


@st.fragment(run_every=3)
def status_bar() -> None:
    ts = J.train_state()
    act = active_jobs()
    tr = next((j for j in act if j["key"] == "train" and j["state"] == "running"), None)
    outside = tr is None and ts["age"] is not None and ts["age"] < 180
    c = st.columns(5)
    c[0].metric("训练", "运行中" if tr else ("运行中（面板外）" if outside else "没在跑"))
    c[1].metric("轮数", ts["iter"] if ts["iter"] is not None else "-")
    c[2].metric("最后一轮", fmt_dur(ts["age"]) + "前" if ts["age"] is not None else "-")
    c[3].metric("最近考试（等效名次）", f"{ts['eval']:.2f}" if isinstance(ts["eval"], (int, float)) else "-",
                help=f"第 {ts['eval_iter']} 轮；越低越好" if ts["eval_iter"] else None)
    c[4].metric("正在跑 / 排队", f"{sum(j['state'] == 'running' for j in act)} / {sum(j['state'] == 'queued' for j in act)}")
    if outside:
        st.caption("训练是在面板外启动的（比如旧的 train.bat 窗口）：面板停不了它，要停请关那个窗口。")


@st.fragment(run_every=2)
def output_view() -> None:
    """任务输出：默认看最近开始的那个，可以换。"""
    js = all_jobs()[:40]
    if not js:
        st.caption("还没有跑过任务。")
        return
    ids = [j["id"] for j in js]
    watch = st.session_state.get("watch")
    idx = ids.index(watch) if watch in ids else 0
    by = {j["id"]: j for j in js}
    pick = st.selectbox("看哪个任务的输出", ids, index=idx, key="pick_job",
                        format_func=lambda i: f"{ICON[by[i]['state']]} {by[i]['title']} · {fmt_time(by[i]['created'])} · "
                                              f"{J.STATUS_ZH[by[i]['state']]}")
    st.session_state["watch"] = pick
    j = by[pick]
    dur = ((j["ended"] or time.time()) - j["started"]) if j["started"] else None
    st.caption(f"状态：{J.STATUS_ZH[j['state']]} · 开始 {fmt_time(j['started'])} · 用时 {fmt_dur(dur)}"
               + (f" · 退出码 {j['rc']}" if j["rc"] not in (None, 0) else ""))
    st.code(J.tail(pick) or "（还没有输出）", language=None)


def history() -> None:
    js = all_jobs()[:60]
    if not js:
        st.caption("还没有跑过任务。")
        return
    st.dataframe([{"任务": j["title"], "状态": f"{ICON[j['state']]} {J.STATUS_ZH[j['state']]}", "排队时间": fmt_time(j["created"]),
                   "开始": fmt_time(j["started"]), "结束": fmt_time(j["ended"]),
                   "用时": fmt_dur((j["ended"] - j["started"]) if j["ended"] and j["started"] else None),
                   "参数": "、".join(f"{PARAM_ZH.get(k, k)}={v}" for k, v in j["params"].items())} for j in js],
                 hide_index=True)


# ---------------- 看板 ----------------

@st.cache_resource(show_spinner="读训练日志……", max_entries=1)
def load_log(path: str, mtime: float, size: int):
    """日志解析一次，结果原样留着用（不用 cache_data：它每次取都要把几万行复制一遍）。只读，别改返回的东西。"""
    return B.load(Path(path))


MAX_POINTS = 240                         # 每条线最多画这么多个点：浏览器里十几张图、每张几万个点会卡


def thin(pts: list) -> list:
    """等间隔抽点，留最后一个。"""
    if len(pts) <= MAX_POINTS:
        return pts
    k = -(-len(pts) // MAX_POINTS)
    out = pts[::k]
    return out if out[-1] == pts[-1] else out + [pts[-1]]


@st.cache_data(show_spinner=False, max_entries=4)
def chart_frames(path: str, mtime: float, size: int, window: int):
    """[(标题, 说明, 是不是全程, 画图用的表)]：滑动平均以后抽点。日志没变、范围没变就不重算。"""
    import pandas as pd
    rows, _ = load_log(path, mtime, size)
    if not rows:
        return []
    it1 = rows[-1]["iter"]
    lo = max(rows[0]["iter"], it1 - window)
    out = []
    for _f, title, keys, note, _lower, full in B.PANELS:
        x0 = rows[0]["iter"] if full else lo
        data = {}
        for k, name in keys:
            raw = [p for p in B.series(rows, k) if p[0] >= x0]
            if raw:
                sparse = k.startswith("eval_") or k.startswith("guard")
                data[name] = dict(thin(B.smooth(raw, 6 if sparse else 25)))
        if data:
            # 稀疏的线（考试每 50 轮一个点）和密的线画在一张图里：中间按轮数连起来，不然断成一个个点
            df = pd.DataFrame(data).sort_index().interpolate(method="index", limit_area="inside")
            out.append((title, note, full, df))
    return out


def _log():
    """(每轮的记录, 事件, 日志的 (修改时间, 大小))。"""
    p = RUN / "log.jsonl"
    if not p.exists():
        return [], [], (0.0, 0)
    s = p.stat()
    return (*load_log(str(RUN), s.st_mtime, s.st_size), (s.st_mtime, s.st_size))


def md_sections() -> dict[str, str]:
    """runs/v3/dashboard.md 按二级标题切开（训练时每 10 轮自动重画；表格直接用它的）。"""
    p = RUN / "dashboard.md"
    if not p.exists():
        return {}
    out, name, buf = {}, None, []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            if name:
                out[name] = "\n".join(buf)
            name, buf = line[3:].strip(), []
        elif name:
            buf.append(line)
    if name:
        out[name] = "\n".join(buf)
    return out


@st.fragment(run_every=120)
def dashboard() -> None:
    rows, events, stamp = _log()
    if not rows:
        st.info("还没有训练日志（runs\\v3\\log.jsonl）。到“训练”那一页点开始。")
        return
    it1 = rows[-1]["iter"]
    st.caption(f"第 {it1} 轮 · 最后一轮 {rows[-1].get('time', '')} · 每 2 分钟自动刷新")
    st.subheader("关键指标")
    st.caption("大数字是最近 50 轮的平均（考试指标是最近 3 次考试）；下面的小字是和 500 轮前比的变化，绿色 = 变好。")
    shown = []
    for k, name, lower, d in B.KPIS:
        ev = k.startswith("eval")
        a = B._avg(rows, k, it1 - (150 if ev else 50), it1 + 1)
        b = B._avg(rows, k, it1 - (650 if ev else 550), it1 - 500 + 1)
        if a is not None:
            shown.append((name, a, b, lower, d))
    for i in range(0, len(shown), 6):
        for col, (name, a, b, lower, d) in zip(st.columns(6), shown[i:i + 6]):
            delta = None if b is None else round(a - b, d)
            col.metric(name, f"{a:.{d}f}", delta=delta,
                       delta_color="off" if lower is None or not delta else ("inverse" if lower else "normal"))
    st.subheader("曲线")
    st.caption("画的是滑动平均，每条线最多 240 个点。")
    window = st.select_slider("看最近多少轮", options=[500, 1000, 3000, 10000, 100000], value=3000,
                              format_func=lambda v: "全部" if v >= 100000 else f"{v} 轮", key="win")
    frames = chart_frames(str(RUN), stamp[0], stamp[1], int(window))
    for i in range(0, len(frames), 2):
        for col, (title, note, full, df) in zip(st.columns(2), frames[i:i + 2]):
            with col:
                st.markdown(f"**{title}**" + ("（全程）" if full else ""))
                st.line_chart(df, height=260)
                if note:
                    st.caption(note)
    sec = md_sections()
    for name, body in sec.items():
        if name.startswith(("关键指标", "曲线", "事件")) or not body.strip():
            continue
        st.subheader(name)
        st.markdown(body)
    if events:
        st.subheader("事件（最近 15 条）")
        st.dataframe([{"轮": e.get("iter"), "时间": e.get("time", ""), "事件": e.get("text", "")} for e in events[-15:][::-1]],
                     hide_index=True)


# ---------------- 竞技场对局查看器（嵌在竞技场那一页里） ----------------

VIEWER_PORT = 8767


@st.cache_resource(show_spinner=False)
def viewer_server() -> str:
    """在面板进程里起 arena/viewer.py 的本机服务（只起一次，跟着面板开关）。端口已经有人在用（比如另外开着的查看器）
    就直接用那个。返回 "" = 好了，否则是出错的说明。"""
    import importlib.util
    import socket
    import threading
    from http.server import ThreadingHTTPServer
    with socket.socket() as sk:
        sk.settimeout(0.5)
        if sk.connect_ex(("127.0.0.1", VIEWER_PORT)) == 0:
            return ""
    try:
        spec = importlib.util.spec_from_file_location("arena_viewer", ROOT / "arena" / "viewer.py")
        V = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(V)
        srv = ThreadingHTTPServer(("127.0.0.1", VIEWER_PORT), V.H)
    except Exception as ex:                                # noqa: BLE001 —— 查看器起不来不影响面板其它部分
        return f"{type(ex).__name__}: {ex}"
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    threading.Thread(target=V.sync, daemon=True).start()   # 启动时同步一次公开战绩（只读公开接口，不用密钥）
    return ""


def arena_viewer() -> None:
    st.subheader("对局查看器")
    err = viewer_server()
    if err:
        st.warning(f"查看器没起来：{err}")
        return
    url = f"http://127.0.0.1:{VIEWER_PORT}/"
    st.caption(f"战绩、名次分布、每一局每一步的选择；正在打的对局每几秒刷新。也可以单独打开：{url}")
    import streamlit.components.v1 as components
    components.iframe(url, height=900, scrolling=True)


# ---------------- 页面 ----------------

st.title("酒馆战旗 AI 面板")
status_bar()
tabs = st.tabs(["📈 看板", "🏋️ 训练", "🧪 测试", "📦 数据", "🏟️ 竞技场", "🧹 维护", "📜 任务输出"])
with tabs[0]:
    dashboard()
with tabs[1]:
    page(["train", "teacher", "board"], "训练和老师模型不能同时跑（后点的排队）。训练时可以同时跑一个测试。")
with tabs[2]:
    page(["comptest", "comptest_seeded", "goldflow", "tavernbench"], "测试一次跑一个，后点的排队。阵容测试的汇总也会出现在看板上。")
with tabs[3]:
    page(["prodemos", "cardvalue", "playbook"], "从本机的高分对局数据重新生成训练用的样本和统计。原始数据和样本只在本机。")
with tabs[4]:
    st.caption("参赛密钥单独放在 bgrl\\arena\\key.txt，面板不显示也不记录它。模型只在竞技场的模拟对局里出动作。")
    cards(["arena_check", "arena_practice", "arena_match", "arena_loop"])
    arena_viewer()
    with st.expander("竞技场任务的输出"):
        tab_output(["arena_check", "arena_practice", "arena_match", "arena_loop"])
with tabs[5]:
    page(["cleanup_preview", "cleanup", "selftest", "info"], "清理前先点“看看会删什么”。训练在跑的时候，清理会排在训练后面。")
with tabs[6]:
    output_view()
    with st.expander("任务记录"):
        history()
