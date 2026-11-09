"""训练看板的画图部分（命令行在 tools/board.py；训练循环在 rl/ppo.Trainer._redraw_board 里调 render）。

训练看板（像 wandb 那样看曲线）：读 runs/<run>/log.jsonl，写 runs/<run>/dashboard.md + runs/<run>/board/ 里每张图一个 SVG。
VS Code（Ctrl+Shift+V 预览）、Typora 等打开 dashboard.md 就能看；训练时每 board_every 轮自动重画（文件一变预览就跟着变）。

  bg.py board                     画一次（默认 runs/v3，最近 3000 轮）
  bg.py board --window 1000       只看最近 1000 轮
  bg.py board --run runs/v3 --open  画完用默认程序打开

图里：浅色细线是每一轮的原始值，深色粗线是滑动平均；竖线是事件（红 = 变乱回退，橙 = 考试守护换回 best.pt，
灰 = 重新启动 / --restore）。考试那张图看全程，其余看最近 --window 轮。
只读日志，不碰模型；画图出错不影响训练。
"""
from __future__ import annotations

import html
import json
import time
from pathlib import Path

PALETTE = ["#2f6fdf", "#e0702a", "#2e9e5b", "#c23b5b", "#7a5bd6", "#8a8a8a"]

# (文件名, 标题, [(字段, 图例名)], 说明, 越小越好?, 全程?)
PANELS = [
    ("exam", "考试：全强度高分阵容（每 50 轮 128 局）", [("eval_boost_score", "等效名次（按名次奖励表）"),
                                                    ("eval_boost_place", "平均名次"), ("guard_avg", "守护：最近 6 次平均")],
     "越低越好。等效名次 = 4.5 − 3.5 × 名次奖励平均分；守护拿它和历史最好比", True, True),
    ("win", "吃鸡率 / 前四率（考试）", [("eval_boost_top1", "吃鸡率"), ("eval_boost_top4", "前四率")],
     "8 个人随便打：吃鸡 12.5%、前四 50%", False, True),
    ("place", "训练对局名次", [("place_bot", "机器人局"), ("place_boost", "加强局"), ("place_seeded", "成型阵容课程局"),
                          ("place_practice", "阵容练习局"), ("place_gift", "送牌局"), ("place_league", "对历史版本")],
     "越低越好", True, False),
    ("boost", "加强局强度（自适应）", [("boost_strength", "前期换成高分阵容的概率"), ("boost_floor", "下限")],
     "模型越强它越高；第 10 回合起固定全强度；下限从第 19650 轮起慢慢抬到 0.7", False, False),
    ("ent", "熵 / 熵系数", [("ent", "熵"), ("ent_coef_x100", "熵系数 × 100")],
     "熵目标 1.2；超过 1.8 或比最近中位数高 35% 连续 10 轮会回退", None, False),
    ("lr", "学习率 / KL", [("lr_x1e4", "学习率 × 1e4"), ("kl_x100", "KL × 100")], "回退时学习率减半，之后每 50 轮翻倍", None, False),
    ("strength", "场面强度（结束回合的场面打同期高分阵容的胜率）", [("cs_wr", "全部回合"), ("cs_late", "第 10 回合以后"),
                                                    ("grow", "后期每回合自己养了多少（结束 − 开始）")],
     "越高越好；高手第 12–14 回合约 0.6–0.7", False, False),
    ("econ", "经济（每回合）", [("unspent", "剩下没花的钱"), ("up_turn", "升本"), ("buy_turn", "买牌"), ("ref_turn", "刷新")],
     "剩钱越少越好（高手 0.01–0.26）", None, False),
    ("comp", "阵容", [("comp_q", "主种族纯度"), ("tgt_q", "离 S/A 目标阵容多近"), ("eng_q", "引擎配燃料"), ("comp_acc", "阵容预测准确率")],
     "", False, False),
    ("imit", "模仿", [("pro_top1", "和高手同一动作（留出局）"), ("teacher_agree", "和老师同一动作"), ("pro_bc", "模仿高手的损失")],
     "", None, False),
    ("speed", "速度", [("dps", "每秒决策数")], "", None, False),
]

KPIS = [("eval_boost_score", "考试等效名次", True, 3), ("eval_boost_place", "考试平均名次", True, 3),
        ("eval_boost_top1", "吃鸡率", False, 3), ("eval_boost_top4", "前四率", False, 3),
        ("place_bot", "机器人局名次", True, 3), ("place_boost", "加强局名次", True, 3), ("boost_strength", "加强局强度", False, 3),
        ("cs_late", "第 10 回合后场面强度", False, 3), ("grow", "后期每回合养场面", False, 3),
        ("place_practice", "阵容练习局名次", True, 3), ("practice_tgt", "练习局局末阵容做到几成", False, 3),
        ("expert_follow", "练习局自己就照着凑的比例", False, 3), ("place_gift", "送牌局名次", True, 3), ("unspent", "每回合剩钱", True, 3), ("ent", "熵", None, 3),
        ("tgt_q", "目标阵容接近度", False, 3), ("eng_q", "引擎配燃料", False, 3), ("pro_top1", "和高手同一动作", False, 3)]


def load(run: Path) -> tuple[list[dict], list[dict]]:
    """(每轮的记录, 事件)。事件：变乱回退、守护换回、--restore、重新启动（轮数往回跳）。"""
    rows, events, prev = [], [], None
    p = run / "log.jsonl"
    if not p.exists():
        return rows, events
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            it = r.get("iter")
            if "restored_from" in r:
                events.append({"iter": it, "time": r.get("time", ""), "kind": "restore",
                               "text": f"--restore {Path(str(r['restored_from'])).name}"})
                continue
            if "rollback" in r and "decisions" not in r:
                events.append({"iter": it, "time": r.get("time", ""), "kind": "rollback", "text": str(r["rollback"])[:80]})
                continue
            if "decisions" not in r or it is None:
                continue
            if prev is not None and it <= prev["iter"]:      # 轮数往回跳：变乱回退（刚记过）或者重新启动
                just_rb = events and events[-1]["kind"] in ("rollback", "restore") and \
                    events[-1].get("iter") is not None and events[-1]["iter"] >= prev["iter"] - 1
                if not just_rb:
                    events.append({"iter": it, "time": r.get("time", ""), "kind": "restart",
                                   "text": f"重新启动（第 {prev['iter']} 轮 → {it}）"})
            if r.get("guard_fix"):
                events.append({"iter": it, "time": r.get("time", ""), "kind": "guard", "text": str(r["guard_fix"])[:120]})
            if isinstance(r.get("ent_coef"), (int, float)):
                r["ent_coef_x100"] = r["ent_coef"] * 100
            if isinstance(r.get("lr"), (int, float)):
                r["lr_x1e4"] = r["lr"] * 1e4
            if isinstance(r.get("kl"), (int, float)):
                r["kl_x100"] = r["kl"] * 100
            rows.append(r)
            prev = r
    # 回退 / 重启以后轮数会往回跳：同一轮只留最后一次
    last = {}
    for r in rows:
        last[r["iter"]] = r
    return [last[k] for k in sorted(last)], events


def series(rows, key):
    return [(r["iter"], float(r[key])) for r in rows if isinstance(r.get(key), (int, float))]


def smooth(pts, k):
    if k <= 1 or len(pts) < 2:
        return pts
    out, acc, q = [], 0.0, []
    for x, y in pts:
        q.append(y)
        acc += y
        if len(q) > k:
            acc -= q.pop(0)
        out.append((x, acc / len(q)))
    return out


def _nice(lo, hi):
    if hi - lo < 1e-9:
        lo, hi = lo - 0.5, hi + 0.5
    pad = (hi - lo) * 0.06
    return lo - pad, hi + pad


def svg_chart(title, lines, events, x0, x1, note="", w=760, h=260):
    """lines：[(名字, 原始点, 平滑点)]。白底，暗色主题的预览里也看得清。"""
    L, R, T, B = 52, 16, 34, 46
    pw, ph = w - L - R, h - T - B
    ys = [y for _, raw, sm in lines for _, y in (sm or raw)]
    ys += [y for _, raw, _ in lines for _, y in raw] if len(ys) < 3 else []
    if not ys or x1 <= x0:
        return None
    lo, hi = _nice(min(ys), max(ys))
    X = lambda x: L + (x - x0) / (x1 - x0) * pw          # noqa: E731
    Y = lambda y: T + (1 - (y - lo) / (hi - lo)) * ph    # noqa: E731
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
         f'font-family="Segoe UI, Microsoft YaHei, sans-serif" font-size="11">',
         f'<rect width="{w}" height="{h}" fill="#ffffff" rx="6"/>',
         f'<text x="{L}" y="20" font-size="13" font-weight="600" fill="#222">{html.escape(title)}</text>']
    for i in range(5):                                    # 横网格 + 纵轴刻度
        v = lo + (hi - lo) * i / 4
        y = Y(v)
        o.append(f'<line x1="{L}" y1="{y:.1f}" x2="{L + pw}" y2="{y:.1f}" stroke="#e6e6e6"/>')
        dec = 1 if hi - lo >= 5 else 2 if hi - lo >= 0.5 else 3
        o.append(f'<text x="{L - 6}" y="{y + 4:.1f}" text-anchor="end" fill="#666">{v:.{dec}f}</text>')
    for i in range(6):                                    # 横轴刻度（轮）
        v = x0 + (x1 - x0) * i / 5
        x = X(v)
        o.append(f'<text x="{x:.1f}" y="{T + ph + 16}" text-anchor="middle" fill="#666">{int(v)}</text>')
    o.append(f'<line x1="{L}" y1="{T + ph}" x2="{L + pw}" y2="{T + ph}" stroke="#999"/>')
    colors = {"rollback": "#d33", "guard": "#e8901a", "restore": "#888", "restart": "#bbb"}
    for e in events:
        if e.get("iter") is not None and x0 <= e["iter"] <= x1:
            x = X(e["iter"])
            o.append(f'<line x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{T + ph}" stroke="{colors.get(e["kind"], "#bbb")}" '
                     f'stroke-dasharray="3,3"><title>{html.escape(e["text"])}</title></line>')
    for i, (name, raw, sm) in enumerate(lines):
        c = PALETTE[i % len(PALETTE)]
        if raw and sm is not raw:
            pts = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in raw if x0 <= x <= x1)
            o.append(f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-opacity="0.22" stroke-width="1"/>')
        main = [p for p in (sm or raw) if x0 <= p[0] <= x1]
        if len(main) == 1:
            o.append(f'<circle cx="{X(main[0][0]):.1f}" cy="{Y(main[0][1]):.1f}" r="3" fill="{c}"/>')
        elif main:
            pts = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in main)
            o.append(f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="2"/>')
        lx = L + i * (pw / max(1, len(lines)))
        last = f"{main[-1][1]:.3g}" if main else "-"
        o.append(f'<rect x="{lx:.1f}" y="{h - 18}" width="10" height="3" fill="{c}"/>')
        o.append(f'<text x="{lx + 14:.1f}" y="{h - 13}" fill="#333">{html.escape(name)}：{last}</text>')
    if note:
        o.append(f'<text x="{L + pw}" y="20" text-anchor="end" fill="#888">{html.escape(note)}</text>')
    o.append("</svg>")
    return "\n".join(o)


def _avg(rows, key, a, b):
    v = [r[key] for r in rows if a <= r["iter"] < b and isinstance(r.get(key), (int, float))]
    return sum(v) / len(v) if v else None


def _fmt(v, d=3):
    return "-" if v is None else f"{v:.{d}f}"


def render(run: Path, window: int = 3000) -> Path | None:
    rows, events = load(run)
    if not rows:
        return None
    out = run / "board"
    out.mkdir(parents=True, exist_ok=True)
    it1 = rows[-1]["iter"]
    lo = max(rows[0]["iter"], it1 - window)
    imgs = []
    for fname, title, keys, note, _lower, full in PANELS:
        x0 = rows[0]["iter"] if full else lo
        lines = []
        for k, name in keys:
            raw = [p for p in series(rows, k) if p[0] >= x0]
            if not raw:
                continue
            sparse = k.startswith("eval_") or k.startswith("guard")
            sm = smooth(raw, 6 if sparse else 25)
            lines.append((name, [] if sparse else raw, sm))
        evs = [e for e in events if e["kind"] in ("guard", "restore")] if full else events   # 全程图只标大事
        svg = svg_chart(title, lines, evs, x0, it1, note) if lines else None
        if svg:                                            # 文件名带轮数：预览不会拿缓存的旧图
            for old in out.glob(f"{fname}_*.svg"):
                old.unlink(missing_ok=True)
            (out / f"{fname}_{it1}.svg").write_text(svg, encoding="utf-8")
            imgs.append((f"{fname}_{it1}", title))
    last = rows[-1]
    g_best = next((r.get("guard_best") for r in reversed(rows) if r.get("guard_best") is not None), None)
    g_ref = next((r.get("guard_ref") for r in reversed(rows) if r.get("guard_ref") is not None), None)
    g_avg = next((r.get("guard_avg") for r in reversed(rows) if r.get("guard_avg") is not None), None)
    g_scale = next((r.get("guard_scale") for r in reversed(rows) if r.get("guard_scale") is not None), None)
    fixes = [e for e in events if e["kind"] == "guard"]
    rbs = [e for e in events if e["kind"] == "rollback" and e["iter"] is not None and e["iter"] >= lo]
    t_last = last.get("time", "")
    speed = None
    recent = [r for r in rows if r["iter"] > it1 - 200 and r.get("time")]
    if len(recent) > 10:
        try:
            ta = time.mktime(time.strptime(recent[0]["time"], "%Y-%m-%d %H:%M:%S"))
            tb = time.mktime(time.strptime(recent[-1]["time"], "%Y-%m-%d %H:%M:%S"))
            speed = (recent[-1]["iter"] - recent[0]["iter"]) / max(1.0, tb - ta) * 3600
        except (ValueError, OverflowError):
            speed = None
    md = [f"# 训练看板 · {run.name}", "",
          f"第 **{it1}** 轮 · 最后更新 {t_last} · 画图 {time.strftime('%Y-%m-%d %H:%M:%S')}"
          + (f" · 约 {speed:.0f} 轮/小时" if speed else ""), "",
          f"学习率 {_fmt(last.get('lr'), 6)} · 熵 {_fmt(last.get('ent'))} · 熵系数 {_fmt(last.get('ent_coef'), 4)} · "
          f"守护：最近 6 次 {_fmt(g_avg)}，最好 {_fmt(g_best if g_best is not None else g_ref)}，"
          f"模仿系数缩放 {_fmt(g_scale, 2)}，出手 {len(fixes)} 次 · 最近 {window} 轮回退 {len(rbs)} 次", "",
          "## 关键指标", "",
          f"| 指标 | 最近 50 轮 | 500 轮前 | 变化 |", "| --- | --- | --- | --- |"]
    for k, name, lower, d in KPIS:
        a = _avg(rows, k, it1 - 50 if not k.startswith("eval") else it1 - 150, it1 + 1)
        b = _avg(rows, k, it1 - 550 if not k.startswith("eval") else it1 - 650, it1 - 500 + 1)
        if a is None and b is None:
            continue
        if a is not None and b is not None and lower is not None and abs(a - b) > 1e-9:
            better = (a < b) if lower else (a > b)
            arrow = ("↓" if a < b else "↑") + ("（好）" if better else "（差）")
        else:
            arrow = ""
        md.append(f"| {name} | {_fmt(a, d)} | {_fmt(b, d)} | {arrow} |")
    md += ["", "考试指标取最近 3 次（150 轮）的平均；其余取最近 50 轮。", ""]
    ct = run / "comptest.json"
    if ct.exists():
        try:
            c = json.loads(ct.read_text(encoding="utf-8"))
            pl = c.get("places") or []
            line = f"最近一次阵容测试：{c.get('created', '')}，第 {c.get('iter')} 轮，{c.get('games')} 局"
            if pl:
                line += (f"，平均名次 {sum(pl) / len(pl):.2f}，吃鸡 {sum(p == 1 for p in pl) / len(pl):.1%}，"
                         f"前四 {sum(p <= 4 for p in pl) / len(pl):.0%}")
            md += ["## 阵容测试", "", line, ""]
        except (ValueError, OSError):
            pass
    cs = run / "comptest_seeded.json"                      # comptest_seeded.bat：拿着高手的成型阵容对全强度高分阵容
    if cs.exists():
        try:
            c = json.loads(cs.read_text(encoding="utf-8"))
            sd = c.get("seeded") or {}
            got = sd.get("seeded") or []
            line = (f"拿着成型阵容的阵容测试：{c.get('created', '')}，第 {c.get('iter')} 轮，第 {sd.get('turns', '')} 回合换成高手的成型场面")
            if got:
                line += (f"，{len(got)} 局平均名次 {sum(got) / len(got):.2f}，吃鸡 {sum(p == 1 for p in got) / len(got):.1%}，"
                         f"前四 {sum(p <= 4 for p in got) / len(got):.0%}")
            md += [line, ""]
        except (ValueError, OSError):
            pass
    pagg = {}                                              # 阵容练习局：各套 [局数, 名次和, 吃鸡和, 做成这套的和]
    for r in rows:
        if r["iter"] >= lo and isinstance(r.get("practice_by_comp"), dict):
            for k, v in r["practice_by_comp"].items():
                n, avg, top1, fm = v[:4]
                a = pagg.setdefault(k, [0, 0.0, 0.0, 0.0])
                a[0] += n
                a[1] += avg * n
                a[2] += top1 * n
                a[3] += fm * n
    if pagg:
        tot = sum(a[0] for a in pagg.values())
        md += ["## 阵容练习局（最近 %d 轮）" % window, "",
               "开局给模型定一套本局凑得成的 S / A，第 7–13 回合每回合酒馆里换进 2 张这套的核心 / 配套卡（场面、手牌、金币不动），"
               "第 7 回合起按“照着凑”的规则给建议（升本、刷新找核心卡、买核心卡、卖杂牌换位置）。“做成了”= 局末场面按严格标准"
               "成了练的这一套。做成的比例往上走，说明它学会了沿着一套凑；平均名次比机器人局好，说明自己凑出来的阵容也值钱。", "",
               f"一共 {tot} 局，做成了 {sum(a[3] for a in pagg.values()) / max(1, tot):.0%}，"
               f"平均名次 {sum(a[1] for a in pagg.values()) / max(1, tot):.2f}，"
               f"吃鸡 {sum(a[2] for a in pagg.values()) / max(1, tot):.0%}", "",
               "| 阵容 | 局数 | 做成了 | 平均名次 | 吃鸡率 |", "| --- | --- | --- | --- | --- |"]
        for k, (n, s_avg, s_top, s_fm) in sorted(pagg.items(), key=lambda kv: -kv[1][3] / max(1, kv[1][0])):
            if n >= 5:
                md.append(f"| {k} | {n} | {s_fm / n:.0%} | {s_avg / n:.2f} | {s_top / n:.0%} |")
        md.append("")
    agg = {}                                               # 成型阵容课程：各套阵容拿到手以后的名次（最近 window 轮）
    for r in rows:
        if r["iter"] >= lo and isinstance(r.get("seeded_by_comp"), dict):
            for k, v in r["seeded_by_comp"].items():
                n, avg, top1 = v[:3]
                nb, avgb, topb = (v[3:6] if len(v) >= 6 else (0, 0.0, 0.0))
                a = agg.setdefault(k, [0, 0.0, 0.0, 0, 0.0, 0.0])
                a[0] += n
                a[1] += avg * n
                a[2] += top1 * n
                a[3] += nb
                a[4] += avgb * nb
                a[5] += topb * nb
    if agg:
        split = any(a[3] for a in agg.values())
        md += ["## 拿着成型阵容打（成型阵容课程，最近 %d 轮）" % window, "",
               "模型在第 6–11 回合被换成这套高手的成型场面以后，自己接着打的结果。平均名次低、吃鸡率高的阵容，说明在模拟器里确实能赢。"
               + ("“加强局”那几列是对手换过高分阵容的局（第 10 回合起全强度），更接近考试。" if split else ""), ""]
        if split:
            md += ["| 阵容 | 局数 | 平均名次 | 吃鸡率 | 加强局 | 加强局平均名次 | 加强局吃鸡率 |",
                   "| --- | --- | --- | --- | --- | --- | --- |"]
        else:
            md += ["| 阵容 | 局数 | 平均名次 | 吃鸡率 |", "| --- | --- | --- | --- |"]
        for k, (n, s_avg, s_top, nb, sb_avg, sb_top) in sorted(agg.items(), key=lambda kv: kv[1][1] / max(1, kv[1][0])):
            if n >= 5:
                row = f"| {k} | {n} | {s_avg / n:.2f} | {s_top / n:.0%} |"
                if split:
                    row += (f" {nb} | {sb_avg / nb:.2f} | {sb_top / nb:.0%} |" if nb else " 0 | - | - |")
                md.append(row)
        md.append("")
    md += ["## 曲线", ""]
    for fname, title in imgs:
        md += [f"![{title}](board/{fname}.svg)", ""]
    ev = [e for e in events if e["kind"] in ("guard", "rollback", "restore") or e["kind"] == "restart"][-15:]
    if ev:
        md += ["## 事件（最近 15 条）", "", "| 轮 | 时间 | 事件 |", "| --- | --- | --- |"]
        name = {"rollback": "变乱回退", "guard": "守护换回 best.pt", "restore": "--restore", "restart": "重新启动"}
        for e in reversed(ev):
            md.append(f"| {e['iter']} | {e['time']} | {name[e['kind']]}：{e['text'].replace('|', '/')} |")
        md.append("")
    p = run / "dashboard.md"
    tmp = run / "dashboard.md.tmp"
    tmp.write_text("\n".join(md) + "\n", encoding="utf-8")
    tmp.replace(p)                                         # 一次换掉，预览不会读到半个文件
    return p
