"""
第 1 步：验证 Bob's Buddy（HDT 自带的 BobsBuddy.dll）能否在 HDT 之外、直接在 Python 进程里调用，并测速。

做三件事：
  1. 加载 BobsBuddy.dll，列出它的主要类型和方法（写进报告，后面写模拟环境要用）
  2. 用你对局记录里的真实战斗重新模拟，和当时 HDT 里 Bob's Buddy 的胜率对比
  3. 测速：单场多次模拟、大量不同战斗各模拟一次（训练环境的用法）

用法（Windows，需要 Python 3.10～3.13 和 pythonnet）：
  python bb_check.py [--hdt D:\\claude\\HDT] [--games 对局记录目录] [--limit 0]
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
REPO = HERE.parent.parent                      # ...\BGStateExporter
REPORT_DIR = HERE.parent / "reports"


# ---------------------------------------------------------------- 报告

class Report:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.data: dict = {"time": datetime.now().isoformat(timespec="seconds")}

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


# ---------------------------------------------------------------- 从对局记录里取真实战斗（不依赖 .NET，可单独测试）

@dataclass
class Combat:
    game: str
    turn: int
    races: list[str]
    anomaly_dbf: int | None
    me: dict                       # 招募阶段最后一刻的 player（含 board、heroPowers、health、armor、tavernTier）
    opp: dict                      # 战斗开始时的 opponent
    recorded: dict                 # 当时 HDT 里 Bob's Buddy 的结果
    ours: dict = field(default_factory=dict)


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
            if not r or not live:
                continue
            me = dict(r["player"])
            if not me.get("board"):
                continue
            combats.append(Combat(
                game=os.path.basename(f), turn=turn,
                races=c.get("availableRaces") or [], anomaly_dbf=c.get("anomalyDbfId"),
                me=me, opp=c["opponent"],
                recorded={k: live.get(k) for k in ("winRate", "tieRate", "lossRate", "winLethal", "lossLethal", "simulationCount")},
            ))
            n += 1
        log(f"  [{i}/{len(files)}] {os.path.basename(f)}：{n} 场战斗")
    return combats


# ---------------------------------------------------------------- .NET / Bob's Buddy

def wait(task):
    """等 .NET Task 完成。time.sleep 会释放 GIL，模拟线程需要回调 Python 时不会卡死。"""
    while not task.IsCompleted:
        time.sleep(0)
    return task.Result


def find_hdt(arg: str | None) -> Path | None:
    cands: list[Path] = []
    if arg:
        cands.append(Path(arg))
    cands.append(REPO.parent / "HDT")
    local = os.environ.get("LOCALAPPDATA")
    if local:
        apps = sorted(Path(local, "HearthstoneDeckTracker").glob("app-*"),
                      key=lambda p: [int(x) if x.isdigit() else 0 for x in p.name[4:].split(".")], reverse=True)
        cands += apps
    for c in cands:
        if (c / "BobsBuddy.dll").exists() and (c / "HearthDb.dll").exists():
            return c.resolve()
    return None


class BB:
    """Bob's Buddy 的薄封装。"""

    def __init__(self, hdt: Path, rep: Report) -> None:
        self.rep = rep
        self.hdt = hdt
        from pythonnet import load
        load("netfx")                      # HDT 和 Bob's Buddy 都是 .NET Framework 4.x
        import clr                         # noqa: F401  (pythonnet)
        import System
        from System import AppDomain, ResolveEventHandler
        from System.Reflection import Assembly, AssemblyName
        self.System = System
        self.Assembly = Assembly

        # 依赖从 HDT 目录找（python.exe 没有 HDT 的 exe.config 里的版本重定向，靠这个回调兜底）。
        # 回调可能在模拟线程上触发，所以等待模拟结果时一律用 wait()，不直接取 .Result，
        # 否则主线程拿着 GIL 阻塞、模拟线程等 GIL 进回调，会互相卡死。
        def resolve(_sender, args):
            name = AssemblyName(args.Name).Name
            for ext in (".dll", ".exe"):
                p = hdt / (name + ext)
                if p.exists():
                    return Assembly.LoadFrom(str(p))
            return None
        self._handler = ResolveEventHandler(resolve)
        AppDomain.CurrentDomain.AssemblyResolve += self._handler
        sys.path.append(str(hdt))

        self.hearthdb = Assembly.LoadFrom(str(hdt / "HearthDb.dll"))
        self.asm = Assembly.LoadFrom(str(hdt / "BobsBuddy.dll"))
        clr.AddReference(str(hdt / "HearthDb.dll"))
        clr.AddReference(str(hdt / "BobsBuddy.dll"))
        rep.data["bobsBuddyVersion"] = str(self.asm.GetName().Version)
        rep.data["hearthDbVersion"] = str(self.hearthdb.GetName().Version)
        rep.log(f"BobsBuddy.dll 版本 {self.asm.GetName().Version}，HearthDb.dll 版本 {self.hearthdb.GetName().Version}")

        self.types = self._load_types()
        self.Simulator = self._py("Simulator")
        self.SimulationRunner = self._py("SimulationRunner")
        self.Input = self._py("Input")
        self.race_type = None
        self.List = None

    def _load_types(self) -> dict:
        try:
            types = list(self.asm.GetTypes())
        except Exception as ex:            # ReflectionTypeLoadException：部分类型加载失败
            types = [t for t in getattr(ex, "Types", []) if t is not None]
            self.rep.log(f"注意：部分类型加载失败：{ex}")
        by_name: dict = {}
        for t in types:
            if t.IsPublic or t.IsNestedPublic:
                by_name.setdefault(t.Name, []).append(t)
        return by_name

    def _py(self, name: str):
        """按类型名找到 Python 里可用的类（优先 BobsBuddy.Simulation 命名空间）。"""
        import importlib
        ts = self.types.get(name) or []
        if not ts:
            raise RuntimeError(f"BobsBuddy.dll 里找不到类型 {name}")
        ts = sorted(ts, key=lambda t: (t.Namespace != "BobsBuddy.Simulation", t.Namespace or ""))
        t = ts[0]
        mod = importlib.import_module(t.Namespace)
        return getattr(mod, t.Name)

    def preload_dependencies(self) -> None:
        """把 BobsBuddy 引用的程序集全部预先加载，尽量不在模拟中途触发加载。"""
        seen = set()
        stack = [self.asm, self.hearthdb]
        while stack:
            a = stack.pop()
            for ref in a.GetReferencedAssemblies():
                if ref.Name in seen:
                    continue
                seen.add(ref.Name)
                try:
                    stack.append(self.Assembly.Load(ref))
                except Exception:
                    for ext in (".dll", ".exe"):
                        p = self.hdt / (ref.Name + ext)
                        if p.exists():
                            try:
                                stack.append(self.Assembly.LoadFrom(str(p)))
                            except Exception:
                                pass
                            break

    # ---------- 反射工具

    @staticmethod
    def _member_type(obj, name: str):
        t = obj.GetType()
        f = t.GetField(name)
        if f is not None:
            return f.FieldType
        p = t.GetProperty(name)
        return p.PropertyType if p is not None else None

    def set(self, obj, name: str, value) -> bool:
        mt = self._member_type(obj, name)
        if mt is None:
            return False
        tn = mt.FullName
        if tn == "System.Boolean":
            value = bool(value)
        elif tn in ("System.Int32", "System.Int64", "System.Int16"):
            value = int(value)
        setattr(obj, name, value)
        return True

    def describe(self, names: list[str]) -> dict:
        """列出类型的公开成员（写进报告，后面写环境用）。"""
        from System.Reflection import BindingFlags
        flags = BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static | BindingFlags.DeclaredOnly
        out = {}
        for n in names:
            for t in self.types.get(n, [])[:2]:
                members = []
                for m in t.GetMembers(flags):
                    kind = str(m.MemberType)
                    if kind == "Method":
                        if m.IsSpecialName:
                            continue
                        ps = ", ".join(f"{p.ParameterType.Name} {p.Name}" for p in m.GetParameters())
                        members.append(f"{'static ' if m.IsStatic else ''}{m.ReturnType.Name} {m.Name}({ps})")
                    elif kind == "Field":
                        members.append(f"{'static ' if m.IsStatic else ''}field {m.FieldType.Name} {m.Name}")
                    elif kind == "Property":
                        members.append(f"prop {m.PropertyType.Name} {m.Name}")
                    elif kind == "Constructor":
                        ps = ", ".join(f"{p.ParameterType.Name} {p.Name}" for p in m.GetParameters())
                        members.append(f"ctor({ps})")
                out[f"{t.Namespace}.{t.Name}"] = sorted(set(members))
        return out

    # ---------- 构造输入

    def _races(self, names: list[str]):
        from System import Enum
        from System.Collections.Generic import List
        inp = self.Input()
        mt = self._member_type(inp, "availableRaces")
        if mt is None:
            return None
        race_t = mt.GetGenericArguments()[0]
        race_py = self._clr_to_py(race_t)
        lst = List[race_py]()
        for n in names:
            v = getattr(race_py, n.upper(), None)
            try:
                lst.Add(v if v is not None else Enum.Parse(race_t, n, True))
            except Exception:
                self.rep.data.setdefault("raceErrors", []).append(n)
        return lst

    @staticmethod
    def _clr_to_py(t):
        import importlib
        return getattr(importlib.import_module(t.Namespace), t.Name)

    def minion(self, sim, friendly: bool, c: dict):
        m = sim.MinionFactory.CreateFromCardId(c["cardId"], friendly)
        hp = c["health"]
        for name, v in (("baseAttack", c["attack"]), ("maxAttack", c["attack"]),
                        ("baseHealth", hp), ("maxHealth", hp),
                        ("taunt", c.get("taunt")), ("div", 1 if c.get("divineShield") else 0),
                        ("poisonous", c.get("poisonous")), ("venomous", c.get("venomous")),
                        ("windfury", c.get("windfury")), ("stealth", c.get("stealth")),
                        ("golden", c.get("golden")), ("tier", c.get("tier", 1)),
                        ("reborn", c.get("reborn"))):
            self.set(m, name, v or 0)
        return m

    def _hero_power(self, player, hp: dict, friendly: bool) -> bool:
        ms = [m for m in player.GetType().GetMethods() if m.Name == "AddHeroPower"]
        if not ms or not hp.get("cardId"):
            return False
        ps = ms[0].GetParameters()
        args = [hp["cardId"], friendly, bool(hp.get("used")), 0, 0, 0, None, hp.get("entityId", 0)][: len(ps)]
        try:
            player.AddHeroPower(*args)
            return True
        except Exception:
            return False

    def build(self, cb: Combat, hero_powers: bool = True):
        sim = self.Simulator()
        inp = self.Input()
        races = self._races(cb.races)
        if races is not None:
            inp.availableRaces = races
        unknown: list[str] = []
        for side, friendly, data in ((inp.Player, True, cb.me), (inp.Opponent, False, cb.opp)):
            hp_total = (data.get("health") or 0) + (data.get("armor") or 0)
            self.set(side, "Health", hp_total if (friendly or hp_total > 0) else 1000)
            self.set(side, "DamageTaken", 0)
            self.set(side, "Tier", data.get("tavernTier") or 1)
            if friendly and hero_powers:
                for hp in data.get("heroPowers") or []:
                    self._hero_power(side, hp, True)
            for c in sorted(data.get("board") or [], key=lambda x: x.get("position", 0)):
                try:
                    side.Side.Add(self.minion(sim, friendly, c))
                except Exception:
                    unknown.append(c.get("cardId") or "?")
        if cb.anomaly_dbf:
            try:
                from HearthDb import Cards
                card = Cards.GetFromDbfId(cb.anomaly_dbf, False)
                if card is not None:
                    inp.Anomaly = sim.AnomalyFactory.Create(card.Id)
            except Exception:
                pass
        inp.SetTurn(cb.turn)
        return inp, unknown

    def run(self, inp, iterations: int, threads: int, max_ms: int):
        return wait(self.SimulationRunner().SimulateMultiThreaded(inp, iterations, threads, max_ms))

    @staticmethod
    def summarize(o) -> dict:
        dmg = list(o.damageResults) if o.damageResults is not None else []
        return {"winRate": round(o.winRate, 4), "tieRate": round(o.tieRate, 4), "lossRate": round(o.lossRate, 4),
                "winLethal": round(o.theirDeathRate, 4), "lossLethal": round(o.myDeathRate, 4),
                "simulationCount": int(o.simulationCount),
                "avgDealt": round(statistics.mean([d for d in dmg if d > 0]), 2) if any(d > 0 for d in dmg) else 0,
                "avgTaken": round(-statistics.mean([d for d in dmg if d < 0]), 2) if any(d < 0 for d in dmg) else 0}


# ---------------------------------------------------------------- 主流程

def board_text(board: list[dict]) -> str:
    return " ".join(f"{c.get('name') or c.get('cardId')}{'(金)' if c.get('golden') else ''}{c['attack']}/{c['health']}"
                    for c in sorted(board, key=lambda x: x.get("position", 0)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdt", help="HDT 目录（里面有 BobsBuddy.dll）")
    ap.add_argument("--games", default=os.path.join(os.environ.get("APPDATA", ""), "HearthstoneDeckTracker",
                                                     "BGStateExporter", "games"))
    ap.add_argument("--limit", type=int, default=0, help="最多对比多少场真实战斗（0 = 全部）")
    ap.add_argument("--iterations", type=int, default=10000, help="对比时每场模拟次数")
    args = ap.parse_args()

    rep = Report()
    rep.log("=== 第 1 步：Bob's Buddy 独立运行检查 ===")
    rep.log(f"时间 {rep.data['time']}；Python {sys.version.split()[0]}；{platform.platform()}；CPU 逻辑核数 {os.cpu_count()}；{platform.processor()}")
    rep.data.update(python=sys.version.split()[0], platform=platform.platform(), cpus=os.cpu_count(),
                    processor=platform.processor())
    if sys.version_info < (3, 10):
        rep.log("需要 Python 3.10 或更新（推荐 3.12）。")
        rep.save()
        return 1

    hdt = find_hdt(args.hdt)
    if hdt is None:
        rep.log("没找到 HDT 目录（需要里面有 BobsBuddy.dll 和 HearthDb.dll）。请用 --hdt 指定，例如 --hdt D:\\claude\\HDT")
        rep.save()
        return 1
    rep.log(f"HDT 目录：{hdt}")
    rep.data["hdt"] = str(hdt)

    # 1. 加载
    rep.log("\n--- 1. 加载 BobsBuddy.dll ---")
    try:
        bb = BB(hdt, rep)
        rep.data["api"] = bb.describe(["Simulator", "SimulationRunner", "Input", "Output", "Player", "Minion",
                                       "MinionFactory", "AnomalyFactory", "EnchantmentFactory"])
        rep.log("已列出主要类型的公开成员（见 json 报告的 api 字段）")
        bb.preload_dependencies()
        rep.log("依赖已预加载")
    except Exception:
        rep.log("加载失败：\n" + traceback.format_exc())
        rep.data["loadOk"] = False
        print("\n报告：", rep.save())
        return 1
    rep.data["loadOk"] = True

    # 2. 冒烟测试：两个白板阵容
    rep.log("\n--- 2. 冒烟测试 ---")
    threads = max(1, (os.cpu_count() or 2) // 2)
    try:
        def vanilla(n, a, h):
            return [{"cardId": "BG26_135", "attack": a, "health": h, "tier": 1, "position": i + 1} for i in range(n)]
        smoke = Combat(game="smoke", turn=5, races=[], anomaly_dbf=None,
                       me={"health": 30, "armor": 0, "tavernTier": 3, "board": vanilla(4, 3, 3)},
                       opp={"health": 30, "armor": 0, "tavernTier": 3, "board": vanilla(3, 3, 3)},
                       recorded={})
        inp, unknown = bb.build(smoke)
        t0 = time.perf_counter()
        out = bb.summarize(bb.run(inp, 2000, threads, 5000))
        rep.log(f"4 个 3/3 打 3 个 3/3：胜 {out['winRate']:.1%} 平 {out['tieRate']:.1%} 负 {out['lossRate']:.1%}"
                f"（{out['simulationCount']} 次，{time.perf_counter() - t0:.2f} 秒）"
                + (f"；未识别卡牌 {unknown}" if unknown else ""))
        rep.data["smoke"] = out
        rep.data["smokeOk"] = out["simulationCount"] > 0
    except Exception:
        rep.log("冒烟测试失败：\n" + traceback.format_exc())
        rep.data["smokeOk"] = False
        print("\n报告：", rep.save())
        return 1

    # 3. 真实战斗对比
    rep.log(f"\n--- 3. 用真实战斗对比（对局记录：{args.games}）---")
    combats = extract_combats(args.games, rep.log) if os.path.isdir(args.games) else []
    if args.limit > 0:
        combats = combats[: args.limit]
    rep.log(f"共 {len(combats)} 场有 HDT 模拟结果的战斗，每场模拟 {args.iterations} 次")
    rows = []
    t_all = time.perf_counter()
    for i, cb in enumerate(combats, 1):
        try:
            inp, unknown = bb.build(cb)
            cb.ours = bb.summarize(bb.run(inp, args.iterations, threads, 4000))
        except Exception as ex:
            rows.append({"game": cb.game, "turn": cb.turn, "error": str(ex)})
            continue
        r, o = cb.recorded, cb.ours
        d = {k: round((o[k] - (r.get(k) or 0)) * 100, 1) for k in ("winRate", "tieRate", "lossRate")}
        rows.append({"game": cb.game, "turn": cb.turn, "recorded": r, "ours": o, "diffPts": d,
                     "maxDiff": max(abs(v) for v in d.values()), "unknownCards": unknown,
                     "me": board_text(cb.me["board"]), "opp": board_text(cb.opp["board"]),
                     "oppHero": cb.opp.get("heroName")})
        if i % 20 == 0:
            rep.log(f"  已完成 {i}/{len(combats)}")
    rep.data["combats"] = rows
    ok = [x for x in rows if "maxDiff" in x]
    if ok:
        md = [x["maxDiff"] for x in ok]
        within5 = sum(1 for v in md if v <= 5) / len(md)
        within10 = sum(1 for v in md if v <= 10) / len(md)
        mean_win = statistics.mean(abs(x["diffPts"]["winRate"]) for x in ok)
        rep.data["compare"] = {"n": len(ok), "errors": len(rows) - len(ok), "within5": within5,
                               "within10": within10, "meanAbsWinDiffPts": round(mean_win, 2),
                               "seconds": round(time.perf_counter() - t_all, 1)}
        rep.log(f"对比了 {len(ok)} 场（出错 {len(rows) - len(ok)} 场）：胜/平/负都差 5 个百分点以内 {within5:.0%}，"
                f"10 个百分点以内 {within10:.0%}；胜率平均差 {mean_win:.1f} 个百分点")
        rep.log("差距最大的 10 场（多半是附魔、成长计数、对手英雄技能这些对局记录里没有的信息）：")
        for x in sorted(ok, key=lambda x: -x["maxDiff"])[:10]:
            rep.log(f"  {x['game']} 第{x['turn']}回合 对{x['oppHero']}：HDT 胜{x['recorded']['winRate']:.0%} → 这里 胜{x['ours']['winRate']:.0%}"
                    f"（差 {x['maxDiff']} 点）")
            rep.log(f"     我方 {x['me']}")
            rep.log(f"     对手 {x['opp']}")
    errs = [x for x in rows if "error" in x]
    for x in errs[:5]:
        rep.log(f"  出错：{x['game']} 第{x['turn']}回合：{x['error']}")

    # 4. 测速
    rep.log("\n--- 4. 测速 ---")
    speed: dict = {}
    try:
        pool = [c for c in combats if len(c.me["board"]) >= 6 and len(c.opp["board"]) >= 6] or combats or [smoke]
        big = pool[len(pool) // 2]
        inp, _ = bb.build(big)
        for th in sorted({threads, os.cpu_count() or 1}):
            t0 = time.perf_counter()
            o = bb.run(inp, 20000, th, 20000)
            dt = time.perf_counter() - t0
            speed[f"oneCombat_threads{th}"] = round(o.simulationCount / dt)
            rep.log(f"同一场战斗反复模拟（{th} 线程）：每秒 {o.simulationCount / dt:,.0f} 次")

        n = 400
        src = [pool[i % len(pool)] for i in range(n)]
        t0 = time.perf_counter()
        inputs = [bb.build(c)[0] for c in src]
        t_build = time.perf_counter() - t0
        speed["buildPerSec"] = round(n / t_build)
        rep.log(f"从 Python 构造战斗输入：每秒 {n / t_build:,.0f} 场")

        t0 = time.perf_counter()
        for x in inputs[:150]:
            bb.run(x, 1, 1, 2000)
        dt = time.perf_counter() - t0
        speed["distinctSequentialPerSec"] = round(150 / dt)
        rep.log(f"不同战斗各模拟 1 次、逐场串行：每秒 {150 / dt:,.0f} 场")

        inputs = [bb.build(c)[0] for c in src]          # 重新构造，避免复用已模拟过的输入
        t0 = time.perf_counter()
        tasks = [bb.SimulationRunner().SimulateMultiThreaded(x, 1, 1, 5000) for x in inputs]
        for t in tasks:
            wait(t)
        dt = time.perf_counter() - t0
        speed["distinctParallelPerSec"] = round(n / dt)
        rep.log(f"不同战斗各模拟 1 次、{n} 场同时提交：每秒 {n / dt:,.0f} 场")

        best = max(speed["distinctParallelPerSec"], speed["distinctSequentialPerSec"])
        per_game = 60    # 8 人一局约 15 轮 × 每轮 4 场战斗
        rep.log(f"按一局 8 人约 {per_game} 场战斗估算，仅战斗部分每秒约 {best / per_game:,.1f} 局"
                f"（再加上 Python 构造输入：约 {1 / (per_game / best + per_game / speed['buildPerSec']):,.1f} 局/秒）")
        speed["estGamesPerSecCombatOnly"] = round(best / per_game, 1)
    except Exception:
        rep.log("测速出错：\n" + traceback.format_exc())
    rep.data["speed"] = speed

    path = rep.save()
    rep.log(f"\n报告已保存：{path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
