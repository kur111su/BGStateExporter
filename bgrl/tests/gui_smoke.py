"""桌面面板（panel/gui.py）的冒烟测试：用假数据把每一页都打开、点一遍、截图。CI 里无屏幕运行（QT_QPA_PLATFORM=offscreen）。

  python tests/gui_smoke.py <放截图的目录>
不需要 torch、TavernGym；要 PySide6、pyqtgraph。槽函数里出的错也算失败（Qt 默认只打印不退出）。
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import tempfile
import time
import traceback
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

NAMES = ["布莱恩·铜须", "守门融合怪", "巴琳达·斯通赫尔斯", "热气球测绘员", "执念诗心龙", "希望信使", "英勇的巢母"]


def fake_run(run: Path) -> None:
    from rl import board as B
    run.mkdir(parents=True)
    rnd = random.Random(0)
    t0 = time.time() - 700 * 30
    with (run / "log.jsonl").open("w", encoding="utf-8") as f:
        for i in range(1, 701):
            it = 21500 + i
            r = {"iter": it, "decisions": 9000, "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0 + i * 30)),
                 "lr": 1.5e-4, "ent": 1.2 + 0.1 * math.sin(i / 40) + rnd.uniform(-.05, .05), "ent_coef": 0.0044, "kl": 0.003,
                 "place_bot": 3.5 - i / 3000 + rnd.uniform(-.3, .3), "place_boost": 4.1 + rnd.uniform(-.3, .3), "boost_strength": 0.7,
                 "boost_floor": 0.7, "cs_wr": 0.5 + rnd.uniform(-.05, .05), "cs_late": 0.58 + rnd.uniform(-.05, .05),
                 "grow": 0.08 + rnd.uniform(-.02, .02), "unspent": 0.3 + rnd.uniform(-.05, .05), "up_turn": 0.4, "buy_turn": 1.6,
                 "ref_turn": 0.8, "comp_q": 0.5, "tgt_q": 0.07 + i / 30000, "eng_q": 0.05, "comp_acc": 0.6, "teacher_agree": 0.5,
                 "pro_bc": 1.1, "dps": 4000 + rnd.uniform(-200, 200), "place_gift": 3.1 + rnd.uniform(-.3, .3),
                 "place_cgift": 3.2 + rnd.uniform(-.3, .3), "cgift_formed": 0.15 + rnd.uniform(-.05, .05), "place_league": 3.9}
            if it % 50 == 0:
                r.update(eval_boost_place=4.4 + rnd.uniform(-.2, .2), eval_boost_score=4.6 + rnd.uniform(-.2, .2),
                         eval_boost_top1=0.03, eval_boost_top4=0.55, guard_avg=4.65, guard_best=4.55, pro_top1=0.45,
                         cgift_by_comp={"圣盾流龙族": [40, 2.1, 0.41, 0.86], "喝茶流": [80, 3.3, 0.23, 0.03]},
                         seeded_by_comp={"喝茶流": [90, 2.5, 0.3, 50, 3.1, 0.09]})
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (run / "comptest.json").write_text(json.dumps({"created": "2026-11-10 11:46:09", "iter": 21850, "games": 256,
                                                   "places": [1, 2, 3, 4, 5, 6, 7, 8] * 32}), encoding="utf-8")
    B.render(run)


def fake_arena(d: Path) -> None:
    (d / "records").mkdir(parents=True)
    (d / "results").mkdir()
    rnd = random.Random(1)
    now, elo, lines = time.time(), 1500.0, []
    for i in range(176):
        pl = rnd.choice([1, 1, 1, 1, 2, 2, 3, 4, 5, 7])
        cat = rnd.choice(["standard", "arena", "practice"])
        e0, elo = elo, elo + (4.5 - pl) * 2
        board = [{"name_zh": rnd.choice(NAMES), "attack": rnd.randint(3, 60), "health": rnd.randint(3, 60),
                  "golden": rnd.random() < .2} for _ in range(7)]
        me = {"username": "capri", "kind": "agent", "placement": pl, "model_version": "capri-it22170"}
        if cat == "standard":
            me.update(score_before=e0, score_after=elo)
        m = {"id": f"m{i:04d}", "category": cat, "created": now - (176 - i) * 1500, "finished": now - (176 - i) * 1500 + 900,
             "turns": rnd.randint(9, 15), "duration_seconds": 800, "human_count": 3, "bot_count": 5,
             "participants": [me, {"username": f"o{i}", "kind": "agent", "placement": 8 if pl != 8 else 1}],
             "final_lineups": [{"username": "capri", "placement": pl, "agent": "capri", "hero": {"name_zh": "托里姆"},
                                "tavern_tier": 5, "board": board},
                               {"username": f"o{i}", "placement": 8 if pl != 8 else 1, "agent": "other", "hero": {"name_zh": "伊瑟拉"},
                                "tavern_tier": 4, "board": board[:4]}]}
        lines.append(json.dumps({"type": "match_finished", "match": m}, ensure_ascii=False))
    (d / "results" / "history.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (d / "compat.json").write_text(json.dumps({"model_version": "capri-it22170"}), encoding="utf-8")

    def recs():
        out, n = [], 0
        for t in range(0, 7):
            for k in range(4):
                n += 1
                out.append({"n": n, "turn": t, "text": rnd.choice(["买 布莱恩·铜须", "升本", "刷新", "结束回合", "打出 守门融合怪"]),
                            "place": round(rnd.uniform(2, 5), 1), "used": k, "action": 3, "model": "capri-it22170",
                            "top": [[3, .62, "升本"], [8, .2, "买 热气球测绘员"], [0, .1, "结束回合"]],
                            "view": {"me": {"gold": 6, "tier": 3, "health": 31,
                                            "board": [{"name_zh": x, "attack": 5, "health": 6} for x in NAMES[:5]],
                                            "hand": [{"name_zh": NAMES[5], "attack": 2, "health": 3, "golden": True}],
                                            "shop": [{"name_zh": x, "attack": 3, "health": 3, "tier": 3, "cost": 3} for x in NAMES[2:6]]},
                                     "opponents": [{"alive": k < 4} for k in range(7)]}})
        return "\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n"
    (d / "records" / "live001.jsonl").write_text(recs(), encoding="utf-8")      # 正在打的
    (d / "records" / "m0175.jsonl").write_text(recs(), encoding="utf-8")        # 最近打完的一局有逐步记录


def fake_jobs(root: Path) -> None:
    from panel import jobs as J
    py = sys.executable
    a = J.start("train", {"threads": 8}, root=root, steps=[[py, "-c", "pass"]], spawn=False)
    a.update(status="running", started=time.time() - 3700, stop="file")
    J._write(J._path(a["id"], root=root), a)
    J._path(a["id"], "log", root).write_text("$ python bg.py train\n" + "\n".join(
        f"第 {22000 + i} 轮  31s | 熵 1.18（系数 0.0044） 学习率 1.5e-04 | 名次 机器人 3.42 加强 4.10（强度 0.70） | 后期场面 0.59 | 剩钱 0.28"
        for i in range(40)) + "\n", encoding="utf-8")
    time.sleep(0.01)
    b = J.start("comptest", {"games": 256, "ckpt": "best"}, root=root, steps=[[py, "-c", "pass"]], spawn=False)
    b.update(status="done", started=time.time() - 900, ended=time.time() - 700, rc=0)
    J._write(J._path(b["id"], root=root), b)
    J._path(b["id"], "log", root).write_text("$ python bg.py goldflow --boosted --games 256\n256 局：平均名次 4.07，吃鸡 3.5%，前四 63%\n", encoding="utf-8")
    time.sleep(0.01)
    c = J.start("arena_practice", {}, root=root, steps=[[py, "-c", "pass"]], spawn=False)
    c.update(status="failed", started=time.time() - 300, ended=time.time() - 290, rc=1)
    J._write(J._path(c["id"], root=root), c)
    J._path(c["id"], "log", root).write_text("===== 这一局的结果 =====\n最终排名：第 2 名（练习局，1 个 AI + 7 个机器人，13 回合）\n分数：这一局不计分\n"
                                             "场面（托里姆，5 本）：★布莱恩·铜须 12/14，守门融合怪 30/28\n", encoding="utf-8")


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "gui_shots")
    out.mkdir(parents=True, exist_ok=True)
    errors = []
    sys.excepthook = lambda tp, val, tb: errors.append("".join(traceback.format_exception(tp, val, tb)))
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        fake_run(d / "v3")
        fake_arena(d / "arena")
        fake_jobs(d / "jobs")
        from PySide6 import QtWidgets
        from panel import gui as G
        from panel import jobs as J
        app = G.make_app()
        started = []
        J.start = lambda key, params=None, root=None, **kw: (started.append((key, params)) or {"id": "x"})
        QtWidgets.QMessageBox.question = staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Yes)
        win = G.MainWindow(d / "v3", d / "jobs", d / "arena", timers=False)
        win.show()

        def shot(name):
            for _ in range(5):
                app.processEvents()
            win.grab().save(str(out / f"{name}.png"))

        for i, (name, _k, _n) in enumerate(G.PAGES):
            win.nav.setCurrentRow(i)
            win.tick(sync=True)
            shot(f"{i}_{name}")
        dash = win.pages[0]
        assert dash.rows and len(dash.plots) >= 8 and dash.kpi.count() >= 10, (len(dash.rows), len(dash.plots), dash.kpi.count())
        dash.win_pick.setCurrentIndex(0)
        tests = win.pages[2]
        assert set(tests.cards) == {"comptest", "comptest_seeded", "goldflow", "tavernbench"}
        tests.cards["comptest"].inputs["games"].setValue(128)
        tests.cards["comptest"].go.click()
        assert started[-1] == ("comptest", {"games": 128, "ckpt": "latest"}), started
        train = win.pages[1].cards["train"]
        assert not train.go.isVisible() and train.stop.isEnabled()          # 训练在跑：只有停止
        ar = win.pages[4]
        win.nav.setCurrentRow(4)
        win.tick()
        assert ar.follow and ar.sel == "live001" and ar.view["live"] and ar.view["turn"] == 6, (ar.sel, ar.view["turn"])
        assert ar.list.count() == 177 and ar.tiles.count() == 6
        ar.btn["arena_match"].click()
        assert started[-1] == ("arena_match", {}), started
        ar.list.setCurrentRow(1)                                             # 点一局打完的（有逐步记录）：停止跟随
        assert not ar.follow and ar.sel == "m0175" and ar.view["final"] and ar.view["turns"], ar.sel
        assert ar.final.count() == 2 and len(ar.view["final_cards"][0]) == 7
        ar.turns.setCurrentIndex(2)
        ar.steps.setCurrentRow(1)
        assert ar.view["turn"] == 2 and ar.view["n"] == ar.view["steps"][1][0]
        shot("4b_竞技场_看一局")
        ar.detail_scroll.verticalScrollBar().setValue(ar.detail_scroll.verticalScrollBar().maximum())
        shot("4d_竞技场_最终名次")
        ar.list.setCurrentRow(5)                                             # 没有逐步记录的一局
        assert ar.view["note"] and not ar.curve.isVisible()
        ar.list.setCurrentRow(0)
        assert dict(ar.view["stats"]) == {"酒馆": "3 本", "血量": "31", "金币": "6", "剩余人数": "5 人"}, ar.view["stats"]
        assert "场面" in ar.zones.text() and "酒馆" in ar.zones.text() and ar.stat_box.count() == 4 and not ar.final_box.isVisibleTo(ar)
        shot("4c_竞技场_进行中")
        ar.follow_btn.click()
        assert ar.follow and ar.sel == "live001"
        out_view = win.pages[1].out                                           # 输出：只接新增的，选区和滚动位置不动
        log = J._path(next(j["id"] for j in J.jobs(d / "jobs") if j["key"] == "train"), "log", d / "jobs")
        win.nav.setCurrentRow(1)
        win.tick()
        n0 = out_view.document().blockCount()
        cur = out_view.textCursor()
        cur.setPosition(0)
        cur.setPosition(30, G.QtGui.QTextCursor.KeepAnchor)
        out_view.setTextCursor(cur)
        picked = out_view.textCursor().selectedText()
        out_view.verticalScrollBar().setValue(0)
        with log.open("a", encoding="utf-8") as f:
            f.write("新的一行：第 22041 轮\n")
        win.tick()
        assert out_view.document().blockCount() == n0 + 1 and "第 22041 轮" in out_view.toPlainText()
        assert out_view.textCursor().selectedText() == picked and len(picked) == 30 and out_view.verticalScrollBar().value() == 0
        win._quiet_close = True
        win.close()
    if errors:
        print("\n".join(errors))
        return 1
    print("gui OK：", ", ".join(sorted(p.name for p in out.glob("*.png"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
