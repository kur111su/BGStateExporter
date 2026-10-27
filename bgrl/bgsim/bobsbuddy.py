"""Bob's Buddy 战斗模拟（通过 BGSimHost.dll 批量调用）。

用法：
    bb = BobsBuddy(hdt_dir)                       # 只能创建一次（一个进程只加载一次 .NET）
    results = bb.simulate([combat, ...], iterations=1)

combat 的格式见 bgrl/simhost/Host.cs 的 CombatIn：
    {"turn": 8, "races": ["BEAST", ...], "anomalyDbf": 0,
     "me":  {"health": 35, "tier": 4, "heroPowers": [{"id": "...", "used": false}], "trinkets": [...],
             "board": [{"id": "BG26_135", "atk": 3, "hp": 1, "tier": 1, "golden": false, "taunt": false, ...}]},
     "opp": {...}}
结果：{"w","t","l","wl","ll","dealt","taken","n","dmg","err"}，见 Host.cs 的 ResultOut。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

BGRL = Path(__file__).resolve().parent.parent
DEFAULT_HOST = BGRL / "simhost" / "bin" / "BGSimHost.dll"


def find_hdt(arg: str | None = None) -> Path | None:
    """依次找：参数、环境变量 HDT_PATH、与仓库同级的 HDT、%LocalAppData% 里最新的 HDT。"""
    cands: list[Path] = []
    for v in (arg, os.environ.get("HDT_PATH")):
        if v:
            cands.append(Path(v))
    cands.append(BGRL.parent.parent / "HDT")
    local = os.environ.get("LOCALAPPDATA")
    if local:
        cands += sorted(Path(local, "HearthstoneDeckTracker").glob("app-*"),
                        key=lambda p: [int(x) if x.isdigit() else 0 for x in p.name[4:].split(".")], reverse=True)
    for c in cands:
        if (c / "BobsBuddy.dll").exists() and (c / "HearthDb.dll").exists():
            return c.resolve()
    return None


class BobsBuddy:
    def __init__(self, hdt: Path, host_dll: Path = DEFAULT_HOST) -> None:
        if not host_dll.exists():
            raise FileNotFoundError(f"找不到 {host_dll}，请先编译 bgrl/simhost（step1_bb_check.bat 会自动编译）")
        from pythonnet import load
        load("netfx")                       # HDT 和 Bob's Buddy 都是 .NET Framework 4.x
        import clr
        from System import AppDomain, ResolveEventHandler
        from System.Reflection import Assembly, AssemblyName

        dirs = [Path(hdt), host_dll.parent]

        def resolve(_sender, args):
            name = AssemblyName(args.Name).Name
            for d in dirs:
                for ext in (".dll", ".exe"):
                    p = d / (name + ext)
                    if p.exists():
                        return Assembly.LoadFrom(str(p))
            return None

        # 加载阶段临时用 Python 回调解析依赖；Host.Init 之后改由 .NET 自己解析，
        # 这样模拟线程加载程序集时不会回调 Python（主线程在等结果、拿着 GIL，回调会卡死）。
        handler = ResolveEventHandler(resolve)
        AppDomain.CurrentDomain.AssemblyResolve += handler
        try:
            for f in ("HearthDb.dll", "Newtonsoft.Json.dll", "BobsBuddy.dll"):
                Assembly.LoadFrom(str(Path(hdt) / f))
            clr.AddReference(str(host_dll))
            from BGSimHost import Host     # type: ignore
            self.info = str(Host.Init(str(hdt)))
        finally:
            AppDomain.CurrentDomain.AssemblyResolve -= handler
        self._host = Host

    def simulate(self, combats: list[dict], iterations: int = 1, threads: int | None = None,
                 max_ms: int = 2000, mode: str = "direct") -> list[dict]:
        """批量模拟。mode：direct = 直接 Simulator.SimulateForInput（实测快约 25%，结果与 runner 相同）；
        runner = HDT 的用法（SimulationRunner）。"""
        if not combats:
            return []
        payload = json.dumps(combats, ensure_ascii=False, separators=(",", ":"))
        out = self._host.Simulate(payload, int(iterations), int(threads or os.cpu_count() or 1), int(max_ms), mode)
        return json.loads(str(out))

    def bench_build(self, combats: list[dict], repeat: int = 5) -> float:
        """只在 .NET 里构造输入（不模拟），返回每秒构造场数。"""
        payload = json.dumps(combats, ensure_ascii=False, separators=(",", ":"))
        return float(self._host.BenchBuild(payload, int(repeat)))


# ---------------- 从插件导出的局面（state.json / 对局记录）转成 CombatIn ----------------

def card_in(c: dict) -> dict:
    return {"id": c.get("cardId") or "", "atk": c.get("attack", 0), "hp": c.get("health", 0),
            "tier": c.get("tier") or 1, "golden": bool(c.get("golden")), "taunt": bool(c.get("taunt")),
            "ds": bool(c.get("divineShield")), "poison": bool(c.get("poisonous")), "venom": bool(c.get("venomous")),
            "wf": bool(c.get("windfury")), "stealth": bool(c.get("stealth")), "reborn": bool(c.get("reborn"))}


def side_in(p: dict, friendly: bool) -> dict:
    side = {"health": (p.get("health") or 0) + (p.get("armor") or 0), "tier": p.get("tavernTier") or 1,
            "board": [card_in(c) for c in sorted(p.get("board") or [], key=lambda x: x.get("position", 0))]}
    if friendly:
        side["heroPowers"] = [{"id": h["cardId"], "used": bool(h.get("used"))}
                              for h in p.get("heroPowers") or [] if h.get("cardId")]
        side["trinkets"] = [t["cardId"] for t in p.get("trinkets") or [] if t.get("cardId")]
    return side


def combat_from_states(turn: int, races: list[str], anomaly_dbf: int | None, me: dict, opp: dict) -> dict:
    return {"turn": turn, "races": races or [], "anomalyDbf": anomaly_dbf or 0,
            "me": side_in(me, True), "opp": side_in(opp, False)}
