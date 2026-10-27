"""
第 1 步：检查 Bob's Buddy 能否在 HDT 之外运行，并测速。

第二版：通过 bgrl/simhost 编译出的 BGSimHost.dll 批量调用（第一版在 Python 里逐个设置随从属性，
每秒只能构造 76 场战斗，成了瓶颈）。

做三件事：
  1. 加载 BGSimHost.dll + HDT 目录里的 BobsBuddy.dll，跑一场冒烟测试
  2. 用你对局记录里的真实战斗重新模拟，和当时 HDT 里 Bob's Buddy 的胜率对比（两种调用方式都试）
  3. 测速：大量不同战斗各模拟一次（训练环境的用法）

用法：双击 bgrl\\step1_bb_check.bat（会先编译 BGSimHost）。
也可以：python bb_check.py [--hdt D:\\claude\\HDT] [--games 对局记录目录] [--limit 0]
结果写到 bgrl\\reports\\bb_check_时间.txt 和 .json。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import statistics
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from bgsim.bobsbuddy import BobsBuddy, combat_from_states, find_hdt  # noqa: E402

REPORT_DIR = HERE.parent / "reports"


class Report:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.data: dict = {"time": datetime.now().isoformat(timespec="seconds"), "version": 2}

    def log(self, msg: str = "") -> None:
        print(msg, flush=True)
        self.lines.append(msg)

    def save(self) -> Path:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        txt = REPORT_DIR / f"bb_check_{stamp}.txt"
        txt.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        (REPORT_DIR / f"bb_check_{stamp}.json").write_text(
            json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")
        return txt


# ---------------------------------------------------------------- 从对局记录里取真实战斗

@dataclass
class Combat:
    game: str
    turn: int
    races: list[str]
    anomaly_dbf: int | None
    me: dict                       # 招募阶段最后一刻的 player
    opp: dict                      # 战斗开始时的 opponent
    recorded: dict                 # 当时 HDT 里 Bob's Buddy 的结果
    ours: dict = field(default_factory=dict)

    def to_input(self) -> dict:
        return combat_from_states(self.turn, self.races, self.anomaly_dbf, self.me, self.opp)


def extract_combats(games_dir: str, log=print) -> list[Combat]:
    """每局每回合取：招募阶段最后的我方阵容、战斗阶段第一次看到的对手阵容、HDT 的模拟结果。"""
    combats: list[Combat] = []
    files = sorted(glob.glob(os.path.join(games_dir, "*.jsonl")))
    for i, f in enumerate(files, 1):
        last_recruit: dict[int, dict] = {}
        first_combat: dict[int, dict] = {}
        result: dict[int, dict] = {}
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    s = json.loads(line)
                except ValueError:
                    continue
                if not s.get("isBattlegrounds") or s.get("mode") != "solo":
                    continue
                turn = s.get("turn", 0)
                phase = s.get("phase")
                if phase == "recruit" and s.get("player"):
                    last_recruit[turn] = s
                elif phase == "combat" and s.get("opponent") and s["opponent"].get("board") and turn not in first_combat:
                    first_combat[turn] = s
                live = ((s.get("simulation") or {}).get("live")) or {}
                if live.get("hasResult") and live.get("errorState") == "None" and live.get("simulationCount", 0) > 0:
                    result[live["turn"]] = live           # 同一回合取最后一次（最终结果）
        n = 0
        for turn, c in sorted(first_combat.items()):
            r = last_recruit.get(turn)
            live = result.get(turn)
            if not r or not live or not r["player"].get("board"):
                continue
            combats.append(Combat(
                game=os.path.basename(f), turn=turn,
                races=c.get("availableRaces") or [], anomaly_dbf=c.get("anomalyDbfId"),
                me=dict(r["player"]), opp=c["opponent"],
                recorded={k: live.get(k) for k in ("winRate", "tieRate", "lossRate", "winLethal", "lossLethal",
                                                   "simulationCount")},
            ))
            n += 1
        log(f"  [{i}/{len(files)}] {os.path.basename(f)}：{n} 场战斗")
    return combats


def board_text(board: list[dict]) -> str:
    return " ".join(f"{c.get('name') or c.get('cardId')}{'(金)' if c.get('golden') else ''}{c['attack']}/{c['health']}"
                    for c in sorted(board, key=lambda x: x.get("position", 0)))


def compare(combats: list[Combat], results: list[dict]) -> list[dict]:
    rows = []
    for cb, o in zip(combats, results):
        if o.get("err"):
            rows.append({"game": cb.game, "turn": cb.turn, "error": o["err"]})
            continue
        r = cb.recorded
        d = {"winRate": round((o["w"] - (r.get("winRate") or 0)) * 100, 1),
             "tieRate": round((o["t"] - (r.get("tieRate") or 0)) * 100, 1),
             "lossRate": round((o["l"] - (r.get("lossRate") or 0)) * 100, 1)}
        rows.append({"game": cb.game, "turn": cb.turn, "recorded": r, "ours": o, "diffPts": d,
                     "maxDiff": max(abs(v) for v in d.values()),
                     "me": board_text(cb.me["board"]), "opp": board_text(cb.opp["board"]),
                     "oppHero": cb.opp.get("heroName")})
    return rows


def summarize(rows: list[dict]) -> dict:
    ok = [x for x in rows if "maxDiff" in x]
    if not ok:
        return {"n": 0, "errors": len(rows)}
    md = [x["maxDiff"] for x in ok]
    early = [x["maxDiff"] for x in ok if x["turn"] <= 6]
    late = [x["maxDiff"] for x in ok if x["turn"] > 6]
    return {"n": len(ok), "errors": len(rows) - len(ok),
            "within5": round(sum(v <= 5 for v in md) / len(md), 3),
            "within10": round(sum(v <= 10 for v in md) / len(md), 3),
            "within5Turn1to6": round(sum(v <= 5 for v in early) / len(early), 3) if early else None,
            "within5Turn7plus": round(sum(v <= 5 for v in late) / len(late), 3) if late else None,
            "meanAbsWinDiffPts": round(statistics.mean(abs(x["diffPts"]["winRate"]) for x in ok), 2),
            "meanSignedWinDiffPts": round(statistics.mean(x["diffPts"]["winRate"] for x in ok), 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdt", help="HDT 目录（里面有 BobsBuddy.dll）")
    ap.add_argument("--games", default=os.path.join(os.environ.get("APPDATA", ""), "HearthstoneDeckTracker",
                                                     "BGStateExporter", "games"))
    ap.add_argument("--limit", type=int, default=0, help="最多对比多少场真实战斗（0 = 全部）")
    ap.add_argument("--iterations", type=int, default=10000, help="对比时每场模拟次数")
    args = ap.parse_args()

    rep = Report()
    rep.log("=== 第 1 步：Bob's Buddy 独立运行检查（第二版：BGSimHost 批量接口）===")
    cpus = os.cpu_count() or 1
    rep.log(f"时间 {rep.data['time']}；Python {sys.version.split()[0]}；{platform.platform()}；CPU 逻辑核数 {cpus}；{platform.processor()}")
    rep.data.update(python=sys.version.split()[0], platform=platform.platform(), cpus=cpus, processor=platform.processor())

    hdt = find_hdt(args.hdt)
    if hdt is None:
        rep.log("没找到 HDT 目录（需要里面有 BobsBuddy.dll 和 HearthDb.dll）。请用 --hdt 指定。")
        print("\n报告：", rep.save())
        return 1
    rep.log(f"HDT 目录：{hdt}")
    rep.data["hdt"] = str(hdt)

    # 1. 加载 + 冒烟
    rep.log("\n--- 1. 加载 ---")
    try:
        bb = BobsBuddy(hdt)
        rep.log(bb.info)
        rep.data["host"] = bb.info
        def vanilla(n):
            return [{"id": "BG26_135", "atk": 3, "hp": 3, "tier": 1} for _ in range(n)]
        smoke = {"turn": 5, "races": [], "me": {"health": 30, "tier": 3, "board": vanilla(4)},
                 "opp": {"health": 30, "tier": 3, "board": vanilla(3)}}
        for mode in ("runner", "direct"):
            t0 = time.perf_counter()
            o = bb.simulate([smoke], iterations=2000, threads=1, mode=mode)[0]
            if o.get("err"):
                rep.log(f"冒烟测试（{mode}）出错：{o['err']}")
            else:
                rep.log(f"冒烟测试（{mode}）：4 个 3/3 打 3 个 3/3，胜 {o['w']:.1%} 平 {o['t']:.1%} 负 {o['l']:.1%}"
                        f"（{o['n']} 次，{time.perf_counter() - t0:.2f} 秒）")
            rep.data[f"smoke_{mode}"] = o
    except Exception:
        rep.log("加载失败：\n" + traceback.format_exc())
        print("\n报告：", rep.save())
        return 1

    # 2. 真实战斗对比
    rep.log(f"\n--- 2. 用真实战斗对比（对局记录：{args.games}）---")
    combats = extract_combats(args.games, rep.log) if os.path.isdir(args.games) else []
    if args.limit > 0:
        combats = combats[: args.limit]
    rep.log(f"共 {len(combats)} 场有 HDT 模拟结果的战斗，每场模拟 {args.iterations} 次")
    inputs = [c.to_input() for c in combats]
    best_rows: list[dict] = []
    for mode in ("runner", "direct"):
        t0 = time.perf_counter()
        try:
            results = bb.simulate(inputs, iterations=args.iterations, threads=cpus, max_ms=5000, mode=mode)
        except Exception:
            rep.log(f"{mode} 方式出错：\n" + traceback.format_exc())
            continue
        rows = compare(combats, results)
        s = summarize(rows)
        s["seconds"] = round(time.perf_counter() - t0, 1)
        rep.data[f"compare_{mode}"] = s
        rep.data[f"combats_{mode}"] = rows
        if s["n"]:
            rep.log(f"{mode}：对比 {s['n']} 场（出错 {s['errors']} 场，用时 {s['seconds']} 秒）："
                    f"胜/平/负都差 5 个百分点以内 {s['within5']:.0%}（第 1～6 回合 {s['within5Turn1to6'] or 0:.0%}，"
                    f"第 7 回合以后 {s['within5Turn7plus'] or 0:.0%}），10 个百分点以内 {s['within10']:.0%}；"
                    f"胜率平均差 {s['meanAbsWinDiffPts']} 点（带符号 {s['meanSignedWinDiffPts']:+} 点）")
        errs = [x for x in rows if "error" in x]
        for x in errs[:5]:
            rep.log(f"  出错：{x['game']} 第{x['turn']}回合：{x['error']}")
        if mode == "runner":
            best_rows = rows
    ok = [x for x in best_rows if "maxDiff" in x]
    if ok:
        rep.log("差距最大的 8 场（runner 方式；多半是附魔、成长计数、对手英雄技能和饰品这些对局记录里没有的信息）：")
        for x in sorted(ok, key=lambda x: -x["maxDiff"])[:8]:
            rep.log(f"  {x['game']} 第{x['turn']}回合 对{x['oppHero']}：HDT 胜{x['recorded']['winRate']:.0%} → 这里 胜{x['ours']['w']:.0%}"
                    f"（差 {x['maxDiff']} 点）")

    # 3. 测速
    rep.log("\n--- 3. 测速（不同战斗各模拟 1 次，训练环境的用法）---")
    speed: dict = {}
    try:
        pool = inputs or [smoke]
        n = 3000
        batch = [pool[i % len(pool)] for i in range(n)]
        t0 = time.perf_counter()
        payload = json.dumps(batch, separators=(",", ":"))
        speed["pythonJsonPerSec"] = round(n / max(1e-9, time.perf_counter() - t0))
        speed["buildPerSec"] = round(bb.bench_build(pool[:200], repeat=5))
        rep.log(f"Python 打包 JSON：每秒 {speed['pythonJsonPerSec']:,} 场；.NET 构造输入：每秒 {speed['buildPerSec']:,} 场（单线程）")
        del payload
        for mode, th in (("runner", cpus), ("direct", cpus), ("direct", max(1, cpus // 2))):
            bb.simulate(batch[:200], iterations=1, threads=th, mode=mode)        # 预热
            t0 = time.perf_counter()
            res = bb.simulate(batch, iterations=1, threads=th, mode=mode)
            dt = time.perf_counter() - t0
            errs = sum(1 for r in res if r.get("err"))
            key = f"{mode}_threads{th}"
            speed[key] = round(n / dt)
            rep.log(f"{mode}（{th} 线程）：每秒 {n / dt:,.0f} 场（含 JSON 往返{'，出错 %d 场' % errs if errs else ''}）")
        best = max(v for k, v in speed.items() if k.startswith(("runner_", "direct_")))
        per_game = 60    # 8 人一局约 15 轮 × 每轮 4 场战斗
        speed["estGamesPerSecCombatOnly"] = round(best / per_game, 1)
        rep.log(f"按一局 8 人约 {per_game} 场战斗估算，仅战斗部分每秒约 {best / per_game:,.1f} 局")
    except Exception:
        rep.log("测速出错：\n" + traceback.format_exc())
    rep.data["speed"] = speed

    path = rep.save()
    rep.log(f"\n报告已保存：{path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
