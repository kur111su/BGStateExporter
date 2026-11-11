"""面板要显示的数据（不依赖界面库，单元测试直接测）：看板的指标 / 曲线 / 表格，竞技场的战绩和单局每一步。

看板读 runs/v3/log.jsonl（rl/board.py 的 load / KPIS / PANELS）和 runs/v3/dashboard.md 里现成的表；
竞技场读 arena/viewer.py 的 summary() / match()（本机的对局结果和逐步记录），统计在这里算。
"""
from __future__ import annotations

import importlib.util
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# ---------------- 看板 ----------------

def kpis(rows: list[dict]) -> list[tuple]:
    """[(名字, 最近的值, 500 轮前的值或 None, 越小越好? (None = 不分好坏), 小数位)]。考试指标取最近 3 次，其余最近 50 轮。"""
    from rl import board as B
    if not rows:
        return []
    it1 = rows[-1]["iter"]
    out = []
    for k, name, lower, d in B.KPIS:
        ev = k.startswith("eval")
        a = B._avg(rows, k, it1 - (150 if ev else 50), it1 + 1)
        b = B._avg(rows, k, it1 - (650 if ev else 550), it1 - 500 + 1)
        if a is not None:
            out.append((name, a, b, lower, d))
    return out


def charts(rows: list[dict], window: int) -> list[tuple]:
    """[(标题, 说明, 是不是全程, [(图例, [轮], [原始值], [滑动平均])])]。稀疏的线（考试）原始值就是点。"""
    from rl import board as B
    if not rows:
        return []
    it1 = rows[-1]["iter"]
    lo = max(rows[0]["iter"], it1 - window)
    out = []
    for _f, title, keys, note, _lower, full in B.PANELS:
        x0 = rows[0]["iter"] if full else lo
        lines = []
        for k, name in keys:
            raw = [p for p in B.series(rows, k) if p[0] >= x0]
            if not raw:
                continue
            sparse = k.startswith("eval_") or k.startswith("guard")
            sm = B.smooth(raw, 6 if sparse else 25)
            lines.append((name, [p[0] for p in raw], [p[1] for p in raw], [p[1] for p in sm], sparse))
        if lines:
            out.append((title, note, full, lines))
    return out


def header(rows: list[dict]) -> str:
    if not rows:
        return "还没有训练日志"
    r = rows[-1]

    def f(k, d=3):
        v = r.get(k)
        return f"{v:.{d}f}" if isinstance(v, (int, float)) else "-"
    return f"第 {r['iter']} 轮 · {r.get('time', '')} · 学习率 {f('lr', 6)} · 熵 {f('ent')} · 熵系数 {f('ent_coef', 4)}"


def md_tables(run: Path) -> list[dict]:
    """runs/v3/dashboard.md 里的表（训练时每 10 轮重画）：[{title, text（表前后的说明）, head, rows}]，不含关键指标 / 曲线 / 事件。"""
    p = Path(run) / "dashboard.md"
    if not p.exists():
        return []
    out, cur = [], None
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            cur = {"title": line[3:].strip(), "text": [], "head": [], "rows": []}
            if not cur["title"].startswith(("关键指标", "曲线", "事件")):
                out.append(cur)
            continue
        if cur is None or not line.strip():
            continue
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):
                continue
            if not cur["head"]:
                cur["head"] = cells
            else:
                cur["rows"].append(cells)
        elif not line.startswith("!["):
            cur["text"].append(line.strip())
    return [dict(t, text=" ".join(t["text"])) for t in out if t["text"] or t["rows"]]


# ---------------- 竞技场 ----------------

CAT = {"standard": "纯 AI", "bots": "补位", "arena": "补位", "practice": "练习", "custom": "约战"}


def cat_key(c) -> str:
    return "bots" if c == "arena" else str(c)


def load_viewer(arena: Path | None = None):
    """按路径加载 arena/viewer.py（它是独立脚本）；arena 给了就把它读写的目录换成那里（测试用）。"""
    spec = importlib.util.spec_from_file_location("arena_viewer", ROOT / "arena" / "viewer.py")
    V = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(V)
    if arena is not None:
        arena = Path(arena)
        V.HERE, V.RESULTS, V.RECORDS, V.SITE = arena, arena / "results" / "history.jsonl", arena / "records", \
            arena / "results" / "site_matches.json"
    return V


def fmt_time(t) -> str:
    return time.strftime("%m-%d %H:%M", time.localtime(t)) if t else ""


def arena_stats(matches: list[dict], cat: str = "all") -> dict:
    """战绩（打完的对局，旧的在前）：局数、平均名次、第一 / 前四、Elo、名次分布、近期名次、最近 20 / 50 / 100 局均名、各赛道局数。"""
    done = sorted((m for m in matches if m.get("placement")), key=lambda m: m.get("created") or 0)
    cats = [("all", "全部", len(done))]
    for k in dict.fromkeys(cat_key(m.get("category")) for m in done):
        cats.append((k, CAT.get(k, k), sum(cat_key(m.get("category")) == k for m in done)))
    g = done if cat == "all" else [m for m in done if cat_key(m.get("category")) == cat]
    n = len(g)
    elo = [m for m in g if m.get("score_after") is not None]

    def avg(k):
        q = g[-k:]
        return sum(m["placement"] for m in q) / len(q) if q else None
    return {"cats": cats, "n": n, "avg": avg(n) if n else None,
            "top1": sum(m["placement"] == 1 for m in g), "top4": sum(m["placement"] <= 4 for m in g),
            "elo": elo[-1]["score_after"] if elo else None, "elo_n": len(elo),
            "dist": [sum(m["placement"] == p for m in g) for p in range(1, 9)],
            "recent": [m["placement"] for m in g[-100:]], "avg20": avg(20), "avg50": avg(50), "avg100": avg(100)}


def match_line(m: dict) -> tuple[str, str]:
    """对局列表里的一行：(名次那一格, 说明)。"""
    badge = str(m["placement"]) if m.get("placement") else ("●" if m.get("live") else "?")
    top = CAT.get(m.get("category"), str(m.get("category") or ""))
    if m.get("hero"):
        top += f" · {m['hero']}" + (f" {m['tier']} 本" if m.get("tier") else "")
    sub = f"{fmt_time(m.get('created'))} · {m.get('turns') if m.get('turns') is not None else '-'} 回合 · {m.get('model') or ''}"
    if m.get("score_after") is not None and m.get("score_before") is not None:
        sub += f" · Elo {m['score_after'] - m['score_before']:+.1f}"
    if m.get("live") and m.get("place_est"):
        sub += f" · 预计 {m['place_est']} 名"
    if not m.get("records"):
        sub += " · 无逐步记录"
    return badge, f"{top}\n{sub}"


def match_parts(m: dict) -> dict:
    """对局列表一行：名次圆点；赛道 · 英雄 本数 / 时间 · 回合 / 模型 / Elo、预计名次、无逐步记录（各占一行）。"""
    badge = str(m["placement"]) if m.get("placement") else ("●" if m.get("live") else "?")
    top = "进行中" if m.get("live") else CAT.get(m.get("category"), str(m.get("category") or ""))
    if m.get("hero"):
        top += f" · {m['hero']}" + (f" {m['tier']} 本" if m.get("tier") else "")
    mid = " · ".join(x for x in (fmt_time(m.get("created")), f"{m['turns']} 回合" if m.get("turns") is not None else "") if x)
    low = []
    if m.get("score_after") is not None and m.get("score_before") is not None:
        low.append(f"Elo {m['score_before']:.0f} → {m['score_after']:.0f}（{m['score_after'] - m['score_before']:+.1f}）")
    if m.get("live") and m.get("place_est"):
        low.append(f"预计 {m['place_est']} 名")
    if not m.get("records"):
        low.append("无逐步记录")
    return {"badge": badge, "place": m.get("placement"), "live": bool(m.get("live")), "top": top, "mid": mid,
            "model": m.get("model") or "", "low": " · ".join(low)}


def card_info(c: dict) -> dict:
    """一张随从 / 法术卡片要显示的：名字、攻 / 血、金色、等级、价格。"""
    a, h = c.get("attack", c.get("atk")), c.get("health", c.get("hp"))
    return {"name": str(c.get("name_zh") or c.get("name") or c.get("id") or "?"),
            "atk": a, "hp": h, "golden": bool(c.get("golden") or str(c.get("id") or "").endswith("_G")),
            "tier": c.get("tier"), "cost": c.get("cost")}


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _zone(me: dict, keys) -> list | None:
    for k in keys:
        v = me.get(k)
        if isinstance(v, list):
            return v
        if isinstance(v, dict):
            for kk in ("cards", "options", "minions", "items"):
                if isinstance(v.get(kk), list):
                    return v[kk]
    return None


def card_text(c) -> str:
    if not isinstance(c, dict):
        return ""
    a, h = c.get("attack", c.get("atk")), c.get("health", c.get("hp"))
    g = c.get("golden") or str(c.get("id") or "").endswith("_G")
    s = ("★" if g else "") + str(c.get("name_zh") or c.get("name") or c.get("id") or "?")
    if a is not None:
        s += f" {a}/{h}"
    if c.get("cost") is not None:
        s += f"（{c['cost']} 金）"
    return s


ZONES = (("场面", ("board", "warband")), ("手牌", ("hand",)), ("酒馆", ("shop", "tavern", "store")),
         ("待选", ("choices", "options", "pending", "discover", "choice")))


def match_view(d: dict, turn=None, n=None, follow: bool = False) -> dict:
    """单局要显示的东西（d = viewer.match()）。turn / n = 看哪一回合、哪一步（None = 自动：最后一回合；
    跟随正在打的对局时取最后一步，否则这一回合的第一步）。"""
    res, recs, me_name = d.get("result"), d.get("records") or [], d.get("me")
    out = {"live": res is None, "title": "进行中", "sub": "", "elo": "", "curve": [], "turns": [], "turn": None, "n": None,
           "steps": [], "cands": [], "scalars": [], "zones": [], "cards": [], "stats": [], "final": [], "final_cards": [],
           "note": ""}
    if res:
        me = next((p for p in res.get("participants") or [] if p.get("username") == me_name), {})
        out["title"] = f"第 {me.get('placement', '?')} 名"
        out["sub"] = (f"{CAT.get(res.get('category'), res.get('category'))} · {res.get('turns')} 回合 · "
                      f"{round(res.get('duration_seconds') or 0)} 秒 · {res.get('human_count')} 个 AI + {res.get('bot_count')} 个机器人")
        if me.get("score_after") is not None and me.get("score_before") is not None:
            out["elo"] = f"Elo {me['score_before']:.1f} → {me['score_after']:.1f}（{me['score_after'] - me['score_before']:+.1f}）"
        for lu in sorted(res.get("final_lineups") or [], key=lambda x: x.get("placement") or 9):
            out["final"].append((f"{lu.get('placement')}" + ("（退赛）" if lu.get("forfeited") else ""),
                                 f"{lu.get('agent') or lu.get('kind') or ''} {lu.get('username') or ''}".strip(),
                                 (lu.get("hero") or {}).get("name_zh") or "", str(lu.get("tavern_tier") or ""),
                                 "，".join(card_text(c) for c in lu.get("board") or []), lu.get("username") == me_name))
            out["final_cards"].append([card_info(c) for c in lu.get("board") or [] if isinstance(c, dict)])
    else:
        out["sub"] = "每 3 秒刷新" + (" · 跟着最新一步" if follow else "")
    out["curve"] = [(r["n"], r["place"], r.get("turn")) for r in recs if r.get("place") is not None]
    turns = list(dict.fromkeys(r.get("turn") for r in recs))
    out["turns"] = turns
    if not turns:
        out["note"] = "这一局没有逐步记录（记录功能加上之前打的，或从竞技场同步来的）。"
        return out
    if turn not in turns:
        turn = turns[-1]
    tr = [r for r in recs if r.get("turn") == turn]
    if n is None or not any(r["n"] == n for r in tr):
        n = tr[-1]["n"] if (follow and res is None) else tr[0]["n"]
    out["turn"], out["n"] = turn, n
    out["steps"] = [(r["n"], str(r.get("text") or ""), r.get("guard") or "", r.get("place"), r.get("used")) for r in tr]
    cur = next(r for r in tr if r["n"] == n)
    out["cands"] = [(str(t), float(p), a == cur.get("action")) for a, p, t in (cur.get("top") or [])]
    view = None
    for r in recs:                                         # 对手部分每回合只存第一步：往前找最近的一份，再盖上这一步的
        if r["n"] > n:
            break
        if r.get("view") and r["view"].get("opponents"):
            view = dict(r["view"])
        if r["n"] == n and r.get("view"):
            view = dict(view or {}, **r["view"])
    me = (view or {}).get("me") or {}
    out["scalars"] = [(k, v) for k, v in me.items() if v is not None and not isinstance(v, (dict, list))][:24]
    if _num(me.get("tier")):                               # 小卡片：本数、血量（+护甲）、金币、剩余人数
        out["stats"].append(("酒馆", f"{int(me['tier'])} 本"))
    if _num(me.get("health")):
        arm = me.get("armor") if _num(me.get("armor")) else 0
        out["stats"].append(("血量", f"{int(me['health'])}" + (f" +{int(arm)}" if arm else "")))
    if _num(me.get("gold")):
        out["stats"].append(("金币", f"{int(me['gold'])}" + (f" / {int(me['max_gold'])}" if _num(me.get("max_gold")) else "")))
    opp = (view or {}).get("opponents")
    if isinstance(opp, list) and opp and all(isinstance(o, dict) and "alive" in o for o in opp):
        out["stats"].append(("剩余人数", f"{1 + sum(bool(o['alive']) for o in opp)} 人"))
    for label, keys in ZONES:
        z = _zone(me, keys)
        if z is not None:
            out["zones"].append((label, [card_text(c) for c in z if isinstance(c, dict)]))
            out["cards"].append((label, [card_info(c) for c in z if isinstance(c, dict)]))
    return out
