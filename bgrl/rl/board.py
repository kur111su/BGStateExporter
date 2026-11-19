"""训练看板（命令行在 tools/board.py；训练循环在 rl/ppo.Trainer._redraw_board 里调 render）。

读 runs/<run>/log.jsonl，写 runs/<run>/dashboard.md：关键指标（最近 50 轮和 500 轮前比）、阵容测试、送整套阵容 /
成型阵容课程的表、事件（变乱回退、考试守护换回 best.pt、--restore、重新启动）。面板的“看板”页显示同样的内容。

  bg.py board                     写一次（默认 runs/v3；表里的“最近”是 3000 轮）
  bg.py board --window 1000       表只算最近 1000 轮
  bg.py board --run runs/v3 --open  写完用默认程序打开

只读日志，不碰模型；出错不影响训练。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

KPIS = [("eval_boost_score", "考试等效名次", True, 3), ("eval_boost_place", "考试平均名次", True, 3),
        ("eval_boost_top1", "吃鸡率", False, 3), ("eval_boost_top4", "前四率", False, 3),
        ("place_bot", "机器人局名次", True, 3), ("place_boost", "加强局名次", True, 3), ("boost_strength", "加强局强度", False, 3),
        ("cs_late", "第 10 回合后场面强度", False, 3), ("grow", "后期每回合养场面", False, 3),
        ("place_gift", "送牌局名次", True, 3), ("place_tribe", "种族课程局名次", True, 3), ("tribe_newcomp_free", "种族课程局成新阵容（没给起步牌）", False, 3), ("tribe_newcomp_start", "种族课程局成新阵容（给了起步牌）", False, 3), ("place_cgift", "送整套阵容的局名次", True, 3),
        ("cgift_formed", "送整套阵容：最后做成这套的比例", False, 3), ("unspent", "每回合剩钱", True, 3), ("ent", "熵", None, 3), ("ev", "价值头解释方差", False, 3),
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


def _avg(rows, key, a, b):
    v = [r[key] for r in rows if a <= r["iter"] < b and isinstance(r.get(key), (int, float))]
    return sum(v) / len(v) if v else None


def _fmt(v, d=3):
    return "-" if v is None else f"{v:.{d}f}"


def render(run: Path, window: int = 3000) -> Path | None:
    rows, events = load(run)
    if not rows:
        return None
    it1 = rows[-1]["iter"]
    lo = max(rows[0]["iter"], it1 - window)
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
          "| 指标 | 最近 50 轮 | 500 轮前 | 变化 |", "| --- | --- | --- | --- |"]
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
    cs = run / "comptest_seeded.json"                      # 面板的“拿着成型阵容测试”：拿着高手的成型阵容对全强度高分阵容
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
    cagg = {}                                              # 送整套阵容：各套 [局数, 名次和, 吃鸡和, 做成这套的和]
    for r in rows:
        if r["iter"] >= lo and isinstance(r.get("cgift_by_comp"), dict):
            for k, v in r["cgift_by_comp"].items():
                n, avg, top1, fm = v[:4]
                a = cagg.setdefault(k, [0, 0.0, 0.0, 0.0])
                a[0] += n
                a[1] += avg * n
                a[2] += top1 * n
                a[3] += fm * n
    if cagg:
        tot = sum(a[0] for a in cagg.values())
        md += ["## 送整套阵容（最近 %d 轮）" % window, "",
               "开局给模型定一套本局凑得成的目标阵容，第 5–11 回合把这套的核心卡一张张送到手里，其它全看它自己。"
               "“做成了”= 局末场面按严格标准成了送的这一套（只有 S / A 级的会算）。做成的比例高、名次比机器人局好的，是它已经会用的阵容。", "",
               f"一共 {tot} 局，做成了 {sum(a[3] for a in cagg.values()) / max(1, tot):.0%}，"
               f"平均名次 {sum(a[1] for a in cagg.values()) / max(1, tot):.2f}，"
               f"吃鸡 {sum(a[2] for a in cagg.values()) / max(1, tot):.0%}", "",
               "| 阵容 | 局数 | 做成了 | 平均名次 | 吃鸡率 |", "| --- | --- | --- | --- | --- |"]
        for k, (n, s_avg, s_top, s_fm) in sorted(cagg.items(), key=lambda kv: -kv[1][3] / max(1, kv[1][0])):
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
