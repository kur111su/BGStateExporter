"""按真人对局校准机器人强度（第二阶段训练用）。

问题：TavernGym 的脚本机器人后期很弱（第 13 回合阵容总攻防约 570，你遇到的真人约 4,700）。
模型在 v2 里学会了“一直待在 1–2 本、靠低本卡滚雪球”——对机器人很有效（考试 1.8 名），
但真人后期的阵容会把它打穿，影子模式里它也几乎从不建议升本。

做法：机器人局里，每个机器人在每回合招募开始时，如果场面总攻防低于“它这局的目标”，
就把每个随从的攻防按比例放大到目标（不改随从种类、关键词和等级）。目标取自真人画像
（rl/real_profile.json）同一回合的四分位：每局给每个机器人抽一个分位 u（0.1–0.9），在 Q1–Q3 之间
插值（u 越大越强），再乘强度 strength（训练前期从小到大爬升，避免一下子全输）。

只改机器人，不碰学习者；改动通过 BatchEnv.save / load（整局存档读档，精确）完成，
每局每回合最多一次，每次约 10 毫秒。
"""
from __future__ import annotations

import json
import math
import random
from collections import deque

import numpy as np

from tg.catalog import base_id, is_golden_id

try:                                   # orjson 快 5–10 倍（没装就用标准库）
    import orjson

    def _loads(s):
        return orjson.loads(s)

    def _dumps(o):
        return orjson.dumps(o).decode("utf-8")
except ImportError:                    # pragma: no cover
    _loads, _dumps = json.loads, lambda o: json.dumps(o, separators=(",", ":"))


def target_curve(real: dict, min_games: int = 5) -> dict[int, tuple[float, float, float]]:
    """真人画像 → {回合: (Q1, 中位数, Q3)}。样本少于 min_games 的回合不用（后期只剩几局时数字乱跳），
    而且只增不减（活到后期的对手场面不会比前一回合弱）。"""
    out: dict[int, tuple[float, float, float]] = {}
    prev = (0.0, 0.0, 0.0)
    for t, v in sorted((int(k), x) for k, x in (real.get("turns") or {}).items()):
        if (v.get("n") or 0) < min_games:
            continue
        cur = (float(v.get("boardQ1") or 0), float(v.get("boardMedian") or 0), float(v.get("boardQ3") or 0))
        prev = tuple(max(a, b) for a, b in zip(prev, cur))
        out[t] = prev
    return out


def target_for(curve: dict, turn: int, u: float) -> float:
    """第 turn 回合、分位 u（0–1）的目标总攻防：Q1 → 中位数 → Q3 按 u 插值；超出画像的回合用最后一回合。"""
    if not curve:
        return 0.0
    t = min(max(turn, min(curve)), max(curve))
    while t not in curve and t > min(curve):
        t -= 1
    q1, med, q3 = curve[t]
    if u <= 0.5:
        return q1 + (med - q1) * (u / 0.5)
    return med + (q3 - med) * ((u - 0.5) / 0.5)


class BotBooster:
    def __init__(self, real: dict, frac: float = 0.75, start_turn: int = 7, seed: int = 0, every: int = 2):
        self.curve = target_curve(real)
        self.frac = frac
        self.start_turn = start_turn
        self.every = max(1, every)
        self.rng = random.Random(seed)
        self.strength = 1.0
        self.plan: dict[int, tuple[int, list[float] | None]] = {}     # 对局 → (第几局, 每个座位的分位；None = 不加强)
        self.last_turn: dict[int, int] = {}
        self.n_boosts = 0
        self.added = 0.0                                                 # 这一轮总共加了多少攻防（日志用）

    def _seat_plan(self, g: int, episode: int):
        cur = self.plan.get(g)
        if cur is None or cur[0] != episode:
            us = [self.rng.uniform(0.1, 0.9) for _ in range(8)] if self.rng.random() < self.frac else None
            cur = (episode, us)
            self.plan[g] = cur
            self.last_turn.pop(g, None)
        return cur[1]

    def boost(self, env, features: np.ndarray, episodes=None) -> bool:
        """env：controlled=(0,) 的 BatchEnv（槽位 i = 对局 i）。features：env.observe()["features"]。
        episodes：每局是第几局（不给就按需问 env.episode_index）。
        返回这次有没有改动（改了的话调用方要重新 observe）。"""
        if self.frac <= 0 or not self.curve:
            return False
        turns = np.rint(features[:, 0] * 20).astype(int)                 # 观察向量第 0 个数 = 回合 / 20
        changed = False
        for g in range(env.n_games):
            t = int(turns[g])
            if t < self.start_turn or self.last_turn.get(g) == t or (t - self.start_turn) % self.every:
                continue
            us = self._seat_plan(g, int(episodes[g]) if episodes is not None else env.episode_index(g))
            self.last_turn[g] = t
            if us is None:
                continue
            s = _loads(env.save(g))
            game = s["game"]
            if game.get("turn") != t or str(game.get("phase")) != "Recruit":
                self.last_turn.pop(g, None)                              # 观察和存档不同步：下一步再试
                continue
            touched = False
            for seat in range(1, len(game["players"])):
                p = game["players"][seat]
                board = p.get("board") or []
                if not p.get("alive", True) or not board:
                    continue
                total = sum(m["atk"] + m["hp"] for m in board)
                goal = target_for(self.curve, t, us[seat]) * self.strength
                if total <= 0 or goal <= total:
                    continue
                k = goal / total
                for m in board:
                    na = max(1, int(round(m["atk"] * k)))
                    nh = max(1, int(round(m["hp"] * k)))
                    dh = nh - m["hp"]
                    m["atk"], m["hp"] = na, nh
                    m["max_hp"] = max(1, m.get("max_hp", m["hp"]) + dh)
                    m["peak_atk"] = max(m.get("peak_atk", 0), na)
                    m["peak_hp"] = max(m.get("peak_hp", 0), nh)
                self.added += goal - total
                touched = True
            if touched:
                env.load(g, _dumps(s))
                self.n_boosts += 1
                changed = True
        return changed

    def finish(self, g: int):
        """对局 g 打完了：返回这局有没有加强（True / False；没到加强回合就结束了返回 None），并清掉这局的计划。"""
        cur = self.plan.pop(g, None)
        self.last_turn.pop(g, None)
        return None if cur is None else cur[1] is not None

    def take_stats(self) -> dict:
        out = {"boosts": self.n_boosts, "boost_added_k": round(self.added / 1000, 1)}
        self.n_boosts, self.added = 0, 0.0
        return out


class AdaptiveStrength:
    """按学习者在加强局里的名次自动调加强幅度（第二阶段的课程）。

    v3 前 250 轮用固定爬坡：第 200 轮到 100% 后学习者在加强局里平均 4.5 名左右（8 人局随便打也是 4.5），
    名次基本由哪个机器人被放得更大决定，奖励里几乎没有信息，熵一路涨、无效操作变多。
    改成：最近 window 局加强局的平均名次好于 target - band 就加一点强度，差于 target + band 就减一点，
    让学习者一直处在“努力能赢”的难度。strength 本身就是进度：爬到 1.0 = 能和真人画像强度的机器人打平。"""

    def __init__(self, start: float = 0.4, target: float = 3.75, band: float = 0.25, step: float = 0.02,
                 lo: float = 0.15, hi: float = 1.0, window: int = 200, min_games: int = 60):
        self.strength = start
        self.target, self.band, self.step, self.lo, self.hi = target, band, step, lo, hi
        self.places: deque = deque(maxlen=window)
        self.min_games = min_games
        self.new = 0                                     # 上次调整后新打完的加强局

    def add(self, place: int):
        self.places.append(int(place))
        self.new += 1

    def mean(self):
        return round(float(np.mean(self.places)), 3) if self.places else None

    def update(self) -> float:
        """每轮调用一次；新结果不够 min_games 局不动（避免同一批结果反复调）。"""
        if self.new < self.min_games or len(self.places) < self.min_games:
            return self.strength
        m = float(np.mean(self.places))
        if m < self.target - self.band:
            self.strength = min(self.hi, self.strength + self.step)
        elif m > self.target + self.band:
            self.strength = max(self.lo, self.strength - self.step)
        self.new = 0
        return self.strength

    def to_json(self) -> dict:
        return {"strength": self.strength, "places": list(self.places)}

    def load_json(self, d: dict | None):
        if d:
            self.strength = float(d.get("strength", self.strength))
            self.places.extend(int(x) for x in d.get("places") or [])


def ramp(it: int, start_it: int, ramp_iters: int, lo: float = 0.3) -> float:
    """强度从 lo 线性爬到 1（第 start_it 轮开始，ramp_iters 轮爬满）。"""
    if ramp_iters <= 0:
        return 1.0
    x = (it - start_it) / ramp_iters
    return lo + (1 - lo) * min(1.0, max(0.0, x)) if not math.isnan(x) else 1.0


# ---------------- 高分真人阵容（第二阶段 · 换阵容版） ----------------

def load_pro_boards(path) -> dict:
    """tools/pro_library.py 生成的 runs/pro_boards.json。"""
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    return {int(t): v for t, v in (d.get("turns") or {}).items() if v}


def merge_pro_boards(paths) -> dict:
    """几个阵容库合在一起（训练 --pro-boards "a;b"、goldflow / comptest 用）：{回合: [阵容…]}，按给的顺序接起来。"""
    out: dict = {}
    for p in paths:
        for t, v in load_pro_boards(p).items():
            out.setdefault(t, []).extend(v)
    return out


class ProBoardBooster(BotBooster):
    """机器人局里，把机器人的场面直接换成同一回合 6500+ 对局里的真实阵容（卡牌、金色、攻 / 血），
    酒馆等级提到那位玩家当时的等级（不降）。之后机器人照常用这套阵容招募、战斗。

    为什么不用放大攻防（BotBooster）：放大的只是低本随从的身材，模型学会了“刷新买增益法术堆身材”
    就能对付，竞技场里升本慢、不会用高本怪。真实阵容带着高本随从的效果（亡语、复生、开战、光环），
    只堆身材打不过。

    strength = 每个机器人座位在每次加强时被换阵容的概率（AdaptiveStrength 按学习者名次调）。
    换阵容通过 Game.from_json + 调试命令（clear_zone / add_minion / set_player）完成，一局约 15 毫秒。"""

    def __init__(self, lib: dict, frac: float = 0.75, start_turn: int = 5, seed: int = 0, every: int = 2):
        super().__init__({"turns": {}}, frac, start_turn, seed, every)
        self.lib = lib
        self.turns = sorted(lib)
        self.injected = 0
        self.failed = 0
        self.late_from, self.late_strength = 0, 0.0   # 第 late_from 回合起换阵容的概率至少 late_strength（0 = 不分前后期）

    def strength_at(self, t: int) -> float:
        """第 t 回合每个机器人座位被换成高分阵容的概率：前期按自适应的 strength，后期（late_from 起）至少 late_strength。"""
        if self.late_from and t >= self.late_from:
            return max(self.strength, self.late_strength)
        return self.strength

    def _pick(self, t: int) -> dict:
        ok = [x for x in self.turns if x <= t]
        tt = ok[-1] if ok else self.turns[0]
        return self.rng.choice(self.lib[tt])

    def boost(self, env, features: np.ndarray, episodes=None) -> bool:
        if self.frac <= 0 or not self.lib or max(self.strength, self.late_strength if self.late_from else 0) <= 0:
            return False
        from taverngym import _core
        turns = np.rint(features[:, 0] * 20).astype(int)
        changed = False
        for g in range(env.n_games):
            t = int(turns[g])
            if t < self.start_turn or self.last_turn.get(g) == t or (t - self.start_turn) % self.every:
                continue
            us = self._seat_plan(g, int(episodes[g]) if episodes is not None else env.episode_index(g))
            self.last_turn[g] = t
            if us is None:
                continue
            s = _loads(env.save(g))
            game = s["game"]
            if game.get("turn") != t or str(game.get("phase")) != "Recruit":
                self.last_turn.pop(g, None)
                continue
            try:                                               # 引擎内部出错（PanicException）：这局这次不换
                G = None
                for seat in range(1, len(game["players"])):
                    p = game["players"][seat]
                    if not p.get("alive", True) or self.rng.random() >= self.strength_at(t):
                        continue
                    if G is None:
                        G = _core.Game.from_json(_dumps(game))
                    entry = self._pick(t)
                    try:
                        G.debug(_dumps({"cmd": "clear_zone", "p": seat, "zone": "board"}))
                    except Exception:                          # noqa: BLE001
                        self.failed += 1
                        continue
                    for cid, atk, hp in entry["board"][:7]:
                        gold = is_golden_id(cid)
                        try:
                            G.debug(_dumps({"cmd": "add_minion", "p": seat, "zone": "board",
                                            "card": base_id(cid), "golden": gold,
                                            "atk": max(0, int(atk)), "hp": max(1, int(hp))}))
                        except Exception:                      # noqa: BLE001
                            self.failed += 1
                    if int(entry.get("tier") or 1) > int(p.get("tier") or 1):
                        try:
                            G.debug(_dumps({"cmd": "set_player", "p": seat, "tier": int(entry["tier"])}))
                        except Exception:                      # noqa: BLE001
                            self.failed += 1
                    self.injected += 1
                if G is not None:
                    s["game"] = _loads(G.state_json())
                    env.load(g, _dumps(s))
                    self.n_boosts += 1
                    changed = True
            except BaseException as ex:                        # noqa: BLE001
                from tg.actions import is_panic
                if not is_panic(ex):
                    raise
                self.failed += 1
        return changed

    def take_stats(self) -> dict:
        out = {"boosts": self.n_boosts, "pro_injected": self.injected, "pro_failed": self.failed}
        self.n_boosts, self.injected, self.failed, self.added = 0, 0, 0, 0.0
        return out


class CompSeeder:
    """“成型以后怎么玩”的课程：一部分机器人局里，学习者（0 号座位）到了第 t 回合（turns 里随机一个）招募开始时，
    场面直接换成同一回合高分阵容库里一套已经成型的 S / A 级目标阵容（rl/comp.formed：核心卡 ≥ 2 张且 ≥ 2/3、
    种族阵容 ≥ 3 个本族随从；t < formed_from 时换成做到一半的：核心卡过半、场上 2 个本族随从，学怎么做完），而且这套在本局凑得成、场上每张卡的种族本局都开了；酒馆等级提到那位玩家当时的等级（不降）。
    之后学习者自己接着打：买配套卡、施法养场面、倒转经济、升本做后期。

    为什么：模型自己几乎走不到成型的局面（第 14350 轮阵容测试按严格标准 S / A 只有 3%），没见过就学不会成型以后怎么用；
    高手的场面第 9 / 10 / 11 回合分别有 28% / 46% / 51% 是成型的 S / A。
    换场面后这一局的阵容强度 / 目标阵容奖励从新场面重新起算（不把换来的强度算成学习者的功劳），见 ppo 的 seeded。"""

    def __init__(self, lib: dict, frac: float = 0.15, turns=(8, 9), seed: int = 0, formed_from: int = 8):
        from tg.catalog import catalog
        from .comp import comp_progress, formed, load_target_comps, reachable

        def half(ids, c):                                  # 做到一半：核心卡过半，种族阵容场上 2 个本族
            fr, k, sup = comp_progress(ids, c)
            return k >= 1 and fr >= 0.5 - 1e-9 and (not c.get("tribes") or sup >= 0.5)
        self.frac, self.turns = frac, tuple(int(t) for t in turns)
        self.rng = random.Random(seed + 4243)
        det = catalog()
        comps = [c for c in load_target_comps() if c["tier"] in "SA"]
        self.pool: dict[int, list] = {}
        for t in self.turns:
            out = []
            for e in lib.get(t, []):
                ids = [base_id(x[0]) for x in e["board"][:7]]
                hit = [c for c in comps if (formed(ids, c) if t >= formed_from else half(ids, c))]
                if not hit:
                    continue
                races = set()
                for cid in ids:
                    rs = set((det.get(cid) or {}).get("races") or [])
                    if rs and "ALL" not in rs:
                        races.add(frozenset(rs))
                out.append((e, hit, races))
            self.pool[t] = out
        self._reachable = reachable
        self.plan: dict[int, tuple[int, int | None]] = {}     # 对局 → (第几局, 第几回合换；None = 这局不换)
        self.n_seeded, self.n_missing = 0, 0
        self.seeded_slots: list[int] = []                      # 这次换过的对局（调用方据此重置塑形）
        self.names: dict[str, int] = {}
        self.comp_of: dict[int, str] = {}                      # 对局 → 换成的那套阵容（局末记名次用）

    def size(self) -> int:
        return sum(len(v) for v in self.pool.values())

    def _turn(self, g: int, episode: int):
        cur = self.plan.get(g)
        if cur is None or cur[0] != episode:
            t = self.rng.choice(self.turns) if self.rng.random() < self.frac else None
            cur = self.plan[g] = (episode, t)
        return cur[1]

    def seed(self, env, features: np.ndarray, episodes=None, skip=None) -> bool:
        """env：controlled=(0,) 的 BatchEnv（槽位 i = 对局 i，学习者是 0 号座位）。返回有没有改（改了要重新 observe）。
        skip：这些对局不换（阵容练习局，见 ShopPractice）。"""
        from tg.obs import lobby
        self.seeded_slots = []
        if self.frac <= 0 or not self.size():
            return False
        from taverngym import _core
        turns = np.rint(features[:, 0] * 20).astype(int)
        changed = False
        for g in range(env.n_games):
            t = int(turns[g])
            if t not in self.turns or (skip and g in skip):
                continue
            want = self._turn(g, int(episodes[g]) if episodes is not None else env.episode_index(g))
            if want != t:
                continue
            lob = lobby(features[g])
            ok = [x for x in self.pool.get(t, []) if all(r & lob for r in x[2])
                  and any(self._reachable(c, lob) for c in x[1])]
            self.plan[g] = (self.plan[g][0], -2)                  # 这局不再试（-2 = 试过没换成，-1 = 换了）
            if not ok:
                self.n_missing += 1
                continue
            s = _loads(env.save(g))
            game = s["game"]
            if game.get("turn") != t or str(game.get("phase")) != "Recruit":
                continue
            e, hit, _ = self.rng.choice(ok)
            try:
                G = _core.Game.from_json(_dumps(game))
                G.debug(_dumps({"cmd": "clear_zone", "p": 0, "zone": "board"}))
                for cid, atk, hp in e["board"][:7]:
                    G.debug(_dumps({"cmd": "add_minion", "p": 0, "zone": "board", "card": base_id(cid),
                                    "golden": is_golden_id(cid), "atk": max(0, int(atk)), "hp": max(1, int(hp))}))
                if int(e.get("tier") or 1) > int(game["players"][0].get("tier") or 1):
                    G.debug(_dumps({"cmd": "set_player", "p": 0, "tier": int(e["tier"])}))
            except BaseException as ex:                            # noqa: BLE001 —— 引擎内部出错（PanicException）：这局不换
                if isinstance(ex, (KeyboardInterrupt, SystemExit)):
                    raise
                self.n_missing += 1
                continue
            s["game"] = _loads(G.state_json())
            env.load(g, _dumps(s))
            self.n_seeded += 1
            self.plan[g] = (self.plan[g][0], -1)
            self.seeded_slots.append(g)
            for c in hit:
                self.names[c["name"]] = self.names.get(c["name"], 0) + 1
            best = max(hit, key=lambda c: ("SAB".find(c["tier"]) * -1, len(c["core"])))
            self.comp_of[g] = best["name"]
            changed = True
        return changed

    def finish(self, g: int):
        """对局 g 打完了：换过场面就返回换成的那套阵容的名字（非空字符串），没换返回空字符串。"""
        cur = self.plan.pop(g, None)
        name = self.comp_of.pop(g, "")
        return (name or "?") if cur is not None and cur[1] == -1 else ""

    def take_stats(self) -> dict:
        out = {"seeded": self.n_seeded, "seed_missing": self.n_missing}
        self.n_seeded, self.n_missing = 0, 0
        return out


class StartSeeder:
    """起步课程：一部分机器人局里，学习者（0 号座位）到了第 t 回合（turns 里随机一个）招募开始时，从本局凑得成的 S / A 阵容里
    均匀抽一套（每套机会一样，不是按高分阵容库里的多少），把场上最弱的几个随从换成这套阵容“现在就买得到”的起步卡：
    先是酒馆等级够得着的核心卡，再是配套卡，种族阵容再补本族随从（都按卡牌原始身材，不带高手的成长）。
    之后学习者自己接着打：要不要沿着这套做下去、怎么从起步走到成型，全靠它自己。

    为什么：成型阵容课程只给“做到一半 / 已经成型”的结果，模型学会了打成型的阵容（课程局平均 2.7–2.9 名），
    但自己的局里成型的 S / A 只有 1%——从第 4 回合到第 8 回合那段起步、过渡它几乎没走过；只靠奖励逼它做阵容名次会掉
    （第 16420–16850 轮）。老师模型（只模仿高手）单独打平均 7.98 名，从它开始练也不行。"""

    def __init__(self, frac: float = 0.2, turns=(4, 5, 6), seed: int = 0, n_plant: int = 3):
        from tg.catalog import catalog
        from .comp import load_target_comps, reachable
        self.frac, self.turns, self.n_plant = frac, tuple(int(t) for t in turns), int(n_plant)
        self.rng = random.Random(seed + 7717)
        det = catalog()
        self.det = det
        self.by_cid = {int(d["cid"]): k for k, d in det.items() if d.get("cid") is not None}
        self.comps = [c for c in load_target_comps() if c["tier"] in "SA"]
        pool = [k for k, d in det.items() if d.get("kind") == "minion" and d.get("in_pool") and not is_golden_id(k)]
        self.tribe_pool = {}                                    # 种族 → 池子里的随从
        for k in pool:
            for r in self.det[k].get("races") or []:
                self.tribe_pool.setdefault(r, []).append(k)
        self._reachable = reachable
        self.plan: dict[int, tuple[int, int | None]] = {}
        self.n_seeded, self.n_missing = 0, 0
        self.seeded_slots: list[int] = []
        self.names: dict[str, int] = {}

    def _minion_ok(self, cid: str, tier: int) -> bool:
        d = self.det.get(cid) or {}
        return d.get("kind") == "minion" and bool(d.get("in_pool")) and 0 < int(d.get("tier") or 99) <= tier

    def starters(self, comp: dict, tier: int, lob) -> list[str]:
        """这套阵容在酒馆等级 tier 时能放上场的起步卡（最多 n_plant 张，同一张不重复）：核心卡 → 配套卡 → 本族随从，
        每组里等级高的优先（同级随机）。"""
        out: list[str] = []
        groups = [sorted(comp["core"]), sorted(comp.get("support") or [])]
        tribes = sorted(comp.get("tribes") or [])
        if tribes:
            groups.append(sorted({k for r in tribes for k in self.tribe_pool.get(r, [])}))
        for g in groups:
            cand = [k for k in g if k not in out and self._minion_ok(k, tier)
                    and (not lob or not (self.det[k].get("races") or []) or any(r in lob or r == "ALL" for r in self.det[k]["races"]))]
            self.rng.shuffle(cand)
            cand.sort(key=lambda k: -int(self.det[k].get("tier") or 0))
            for k in cand:
                if len(out) >= self.n_plant:
                    return out
                out.append(k)
        return out

    def _turn(self, g: int, episode: int):
        cur = self.plan.get(g)
        if cur is None or cur[0] != episode:
            t = self.rng.choice(self.turns) if self.rng.random() < self.frac else None
            cur = self.plan[g] = (episode, t)
        return cur[1]

    def seed(self, env, features: np.ndarray, episodes=None, skip=None) -> bool:
        from tg.obs import lobby
        self.seeded_slots = []
        if self.frac <= 0 or not self.comps:
            return False
        from taverngym import _core
        turns = np.rint(features[:, 0] * 20).astype(int)
        changed = False
        for g in range(env.n_games):
            t = int(turns[g])
            if t not in self.turns:
                continue
            want = self._turn(g, int(episodes[g]) if episodes is not None else env.episode_index(g))
            if want != t:
                continue
            self.plan[g] = (self.plan[g][0], -2)
            lob = lobby(features[g])
            s = _loads(env.save(g))
            game = s["game"]
            if game.get("turn") != t or str(game.get("phase")) != "Recruit":
                continue
            me = game["players"][0]
            tier = int(me.get("tier") or 1)
            ok = [c for c in self.comps if self._reachable(c, lob)]
            self.rng.shuffle(ok)
            comp, plant = None, []
            for c in ok:                                         # 均匀抽一套；起步卡不到 2 张就换下一套
                plant = self.starters(c, tier, lob)
                if len(plant) >= 2:
                    comp = c
                    break
            if comp is None:
                self.n_missing += 1
                continue
            board = [m for m in me.get("board") or [] if not m.get("destroyed")]
            keep_n = max(0, min(len(board) - min(len(board), len(plant)), 7 - len(plant)))   # 换掉最弱的几张，总数不超过 7
            keep = sorted(board, key=lambda m: int(m.get("atk", 0)) + int(m.get("hp", 0)), reverse=True)[:keep_n]
            keep = sorted(keep, key=lambda m: board.index(m))
            try:
                G = _core.Game.from_json(_dumps(game))
                G.debug(_dumps({"cmd": "clear_zone", "p": 0, "zone": "board"}))
                for m in keep:
                    cid = self.by_cid.get(int(m.get("card", -1)))
                    if not cid:
                        continue
                    G.debug(_dumps({"cmd": "add_minion", "p": 0, "zone": "board", "card": base_id(cid),
                                    "golden": bool(m.get("golden")), "atk": max(0, int(m.get("atk", 0))),
                                    "hp": max(1, int(m.get("hp", 1)))}))
                for cid in plant:
                    d = self.det[cid]
                    G.debug(_dumps({"cmd": "add_minion", "p": 0, "zone": "board", "card": cid, "golden": False,
                                    "atk": max(0, int(d.get("atk") or 0)), "hp": max(1, int(d.get("hp") or 1))}))
            except BaseException as ex:                            # noqa: BLE001
                if isinstance(ex, (KeyboardInterrupt, SystemExit)):
                    raise
                self.n_missing += 1
                continue
            s["game"] = _loads(G.state_json())
            env.load(g, _dumps(s))
            self.n_seeded += 1
            self.plan[g] = (self.plan[g][0], -1)
            self.seeded_slots.append(g)
            self.names[comp["name"]] = self.names.get(comp["name"], 0) + 1
            changed = True
        return changed

    def finish(self, g: int):
        cur = self.plan.pop(g, None)
        return cur is not None and cur[1] == -1

    def take_stats(self) -> dict:
        out = {"started": self.n_seeded, "start_missing": self.n_missing}
        self.n_seeded, self.n_missing = 0, 0
        return out


class ShopPractice(StartSeeder):
    """阵容练习局：frac 的机器人局一开局就给学习者（0 号座位）定一套本局凑得成的 S / A 阵容（每套机会一样），
    之后 turns 里每回合招募开始时，把酒馆里 n_slots 个随从（没冻住的，先换和这套无关的）换成这套“现在买得到”的卡：
    核心卡（一半的机会）、配套卡（3 成）、本族随从（2 成），酒馆等级够得着、本局开了那个种族，原始身材。
    场面、手牌、金币一概不动：买不买、留不留、怎么打、做不做这套，全是模型自己决定——等于这局一直“刷到”这套阵容的牌。

    为什么（第 20020 轮）：成型阵容课程是直接把高手第 6–11 回合的场面给它，前面怎么一步步凑出来它没走过；
    起步课程（StartSeeder）把场上的随从换成起步卡，打乱了它自己的场面（那些局 3.40–3.58 名，比自己打还差），撤了。
    模型自己的局里 S / A 成型 0–1%、核心卡买 1–3%（高手 7–9%），从来没自己做出过一套，不知道做出来值多少。
    这里只是让对的牌常出现，选择权还在它手里；局末记“做成这一套了没有”和名次（practice_by_comp）。"""

    def __init__(self, frac: float = 0.25, turns=(7, 8, 9, 10, 11, 12, 13), seed: int = 0, n_slots: int = 2,
                 weights=(0.6, 0.4, 0.0)):
        super().__init__(frac=frac, turns=turns, seed=seed, n_plant=12)
        self.weights = tuple(float(x) for x in weights)         # 核心 / 配套 / 本族随从 各占多少机会
        self.rng = random.Random(seed + 9311)
        self.n_slots = int(n_slots)
        self.plan: dict[int, list] = {}                # 对局 → [第几局, 这局练的阵容（None = 普通局）, 最近换过的回合]
        self.n_applied = 0
        self._base: list[str] | None = None            # 卡牌序号 → 基础卡 id（expert 用）

    def _groups(self, comp: dict, tier: int, lob) -> list[list[str]]:
        def ok(k):
            if not self._minion_ok(k, tier):
                return False
            rs = self.det[k].get("races") or []
            return not lob or not rs or any(r in lob or r == "ALL" for r in rs)
        core = [k for k in sorted(comp["core"]) if ok(k)]
        sup = [k for k in sorted(comp.get("support") or []) if ok(k) and k not in comp["core"]]
        tribes = sorted(comp.get("tribes") or [])
        tri = sorted({k for r in tribes for k in self.tribe_pool.get(r, []) if ok(k)} - set(core) - set(sup))
        return [core, sup, tri]

    def offer(self, comp: dict, tier: int, lob, k: int, have=()) -> list[str]:
        """这回合往酒馆里放的 k 张：每张先按 weights（默认核心 0.6 / 配套 0.4 / 本族 0）抽一组（空的组跳过），再在组里随机一张。
        have：场上 + 手里已经有的基础卡——核心卡先给还没有的（都有了才重复，好凑三连）。"""
        groups = self._groups(comp, tier, lob)
        have = set(have)
        if groups[0] and any(x not in have for x in groups[0]):
            groups[0] = [x for x in groups[0] if x not in have]
        w = list(self.weights)
        out = []
        for _ in range(k):
            live = [(g, wi) for g, wi in zip(groups, w) if g and wi > 0]
            if not live:
                break
            x, acc = self.rng.random() * sum(wi for _, wi in live), 0.0
            for g, wi in live:
                acc += wi
                if x <= acc:
                    out.append(self.rng.choice(g))
                    break
        return out

    def active(self) -> set:
        """这一局在练阵容的对局（成型阵容课程不换它们的场面）。"""
        return {g for g, v in self.plan.items() if v[1] is not None}

    def is_active(self, g: int) -> bool:
        v = self.plan.get(g)
        return v is not None and v[1] is not None

    def apply(self, env, features: np.ndarray, episodes=None) -> bool:
        """每步 observe 之后调用：新的一局先决定练不练、练哪套；练的局在 turns 里每回合第一次看到招募阶段时换酒馆。
        返回有没有改（改了要重新 observe）。"""
        from tg.obs import lobby
        if self.frac <= 0 or not self.comps:
            return False
        from taverngym import _core
        turns = np.rint(features[:, 0] * 20).astype(int)
        changed = False
        for g in range(env.n_games):
            ep = int(episodes[g]) if episodes is not None else env.episode_index(g)
            cur = self.plan.get(g)
            if cur is None or cur[0] != ep:
                comp = None
                if self.rng.random() < self.frac:
                    lob = lobby(features[g])
                    ok = [c for c in self.comps if self._reachable(c, lob)]
                    comp = self.rng.choice(ok) if ok else None
                cur = self.plan[g] = [ep, comp, -1]
            t = int(turns[g])
            if cur[1] is None or t not in self.turns or cur[2] == t:
                continue
            s = _loads(env.save(g))
            game = s["game"]
            if game.get("turn") != t or str(game.get("phase")) != "Recruit":
                continue
            cur[2] = t
            me = game["players"][0]
            shop = me.get("shop") or []
            want = set(cur[1]["core"]) | set(cur[1].get("support") or [])
            idx = [k for k, sl in enumerate(shop) if "Minion" in (sl.get("item") or {}) and not sl.get("frozen")]
            if not idx:
                continue
            lob = lobby(features[g])
            have = [self.by_cid.get(int(x.get("card", -1)), "") for x in (me.get("board") or [])]
            have += [self.by_cid.get(int(((h.get("item") or {}).get("Minion") or {}).get("card", -1)), "")
                     for h in (me.get("hand") or []) if isinstance(h, dict)]
            cards = self.offer(cur[1], int(me.get("tier") or 1), lob, min(self.n_slots, len(idx)), have=have)
            if not cards:
                self.n_missing += 1
                continue
            mine = lambda k: self.by_cid.get(int(shop[k]["item"]["Minion"].get("card", -1)), "") in want   # noqa: E731
            idx.sort(key=lambda k: (mine(k), self.rng.random()))      # 先换和这套无关的
            try:
                G = _core.Game.from_json(_dumps(game))
                for k in sorted(idx[:len(cards)], reverse=True):
                    G.debug(_dumps({"cmd": "remove", "p": 0, "zone": "shop", "index": int(k)}))
                for cid in cards:
                    G.debug(_dumps({"cmd": "add_minion", "p": 0, "zone": "shop", "card": cid}))
            except BaseException as ex:                            # noqa: BLE001 —— 引擎内部出错：这回合不换
                if isinstance(ex, (KeyboardInterrupt, SystemExit)):
                    raise
                self.n_missing += 1
                continue
            s["game"] = _loads(G.state_json())
            env.load(g, _dumps(s))
            self.n_applied += 1
            changed = True
        return changed

    def _bases(self) -> list[str]:
        if self._base is None:
            from tg.catalog import base_of_index, n_cards
            self._base = [base_of_index(i) if i else "" for i in range(n_cards())]
        return self._base

    def rank(self, comp: dict, b: str) -> int:
        """这张卡（基础卡 id）对这套阵容：核心卡 3、配套卡 2、本族随从 1、无关 0。"""
        if not b:
            return 0
        if b in comp["core"]:
            return 3
        if b in (comp.get("support") or ()):
            return 2
        tribes = comp.get("tribes") or ()
        d = self.det.get(b) or {}
        if tribes and d.get("kind") == "minion" and any(r in tribes or r == "ALL" for r in (d.get("races") or [])):
            return 1
        return 0

    # 照着凑的升本节奏（回合 → 回合末至少到几本），按高手的平均：第 5 回合 3.2、第 7 回合 4.4、第 8 回合 5.0、第 9 回合 5.4
    CURVE = {1: 1, 2: 2, 3: 2, 4: 3, 5: 3, 6: 4, 7: 4, 8: 5, 9: 5}

    def expert(self, g: int, c, m, f=None) -> int:
        """练习局里“照着这套凑”的动作（没有就 -1，交给模型自己）。c：这一行的卡牌序号（obs["cards"]），m：合法动作，
        f：这一行的 features（看回合、酒馆等级；不给就不管升本）。按顺序：
          1 发现 / 选择：有这套的牌就选最相关的（核心 > 配套 > 本族）
          2 手里有这套的随从、场上有位置：打出来（放最右边）
          3 酒馆里有这套的核心卡、买得起：买
          4 酒馆等级落后于高手节奏（第 5 回合 3 本、第 6 回合 4 本、第 8 回合 5 本、第 10 回合 6 本）、升得起：升本
          5 酒馆里有配套卡 / 本族随从：场上 + 手里随从不满 7 个、或者场上有比它更没用的可以换，就买
          6 场上满了、手里压着更有用的随从：卖掉场上最没用的那个（无关 < 本族 < 配套，同样先卖等级低的）
          7 第 7 回合起、4 本以上、场上满了还有不属于这套的随从、酒馆里没有能换上的牌、刷得起：刷新（找核心卡）
        冻结、结束回合、法术、站位都不管。"""
        from tg.actions import (A_BUY, A_CHOOSE, A_PLAY, A_REFRESH, A_SELL, A_UPGRADE, N_BOARD, N_CHOICE, N_HAND, N_POS,
                                N_SHOP)
        from tg.obs import C_BOARD, C_CHOICE, C_HAND, C_SHOP
        cur = self.plan.get(g)
        if cur is None or cur[1] is None:
            return -1
        comp, base = cur[1], self._bases()
        cc = np.asarray(c).astype(np.int64)
        mm = np.asarray(m).astype(bool)

        def b_of(x):
            return base[int(x)] if 0 < int(x) < len(base) else ""

        def kind(x):
            return (self.det.get(b_of(x)) or {}).get("kind")

        def tier_of(x):
            return int((self.det.get(b_of(x)) or {}).get("tier") or 0)

        def rk(x):
            return self.rank(comp, b_of(x))
        ch = [(rk(cc[C_CHOICE + i]), -i) for i in range(N_CHOICE) if mm[A_CHOOSE + i]]
        if ch:
            r, ni = max(ch)
            return A_CHOOSE - ni if r > 0 else -1
        board = [int(x) for x in cc[C_BOARD:C_BOARD + N_BOARD]]
        hand = [int(x) for x in cc[C_HAND:C_HAND + N_HAND]]
        n_board = sum(x > 0 for x in board)
        n_min = n_board + sum(x > 0 and kind(x) == "minion" for x in hand)
        worst = min([(rk(x), tier_of(x)) for x in board if x > 0], default=(9, 0))   # 场上最没用的随从
        for h in range(N_HAND):
            if hand[h] > 0 and kind(hand[h]) == "minion" and rk(hand[h]) > 0:
                for p in range(N_POS - 1, -1, -1):
                    if mm[A_PLAY + h * N_POS + p]:
                        return A_PLAY + h * N_POS + p
        shop = [(rk(cc[C_SHOP + i]), i) for i in range(N_SHOP) if mm[A_BUY + i]]
        core = [i for r, i in shop if r == 3]
        if core:
            return A_BUY + core[0]
        if f is not None and mm[A_UPGRADE]:
            t, tr = int(round(float(f[0]) * 20)), None
            from tg.obs import tier as tier_f
            tr = int(tier_f(f))
            if tr < self.CURVE.get(t, 6):
                return A_UPGRADE
        for r_want in (2, 1):
            for r, i in shop:
                if r == r_want and (n_min < N_BOARD or (n_board >= N_BOARD and worst[0] < r)):
                    return A_BUY + i
        waiting = max([rk(x) for x in hand if x > 0 and kind(x) == "minion"], default=0)
        if n_board >= N_BOARD and waiting > 0:
            cand = [(rk(board[i]), tier_of(board[i]), i) for i in range(N_BOARD)
                    if board[i] > 0 and rk(board[i]) < waiting and mm[A_SELL + i]]
            if cand:
                return A_SELL + min(cand)[2]
        if f is not None and mm[A_REFRESH] and n_board >= N_BOARD and worst[0] < 2:
            from tg.obs import tier as tier_f
            if int(round(float(f[0]) * 20)) >= 7 and int(tier_f(f)) >= 4:
                return A_REFRESH
        return -1

    def finish(self, g: int) -> str:
        """对局 g 打完了：练阵容的局返回练的那套的名字，普通局返回空字符串。"""
        cur = self.plan.pop(g, None)
        return cur[1]["name"] if cur is not None and cur[1] is not None else ""

    def take_stats(self) -> dict:
        out = {"practice_shops": self.n_applied, "practice_missing": self.n_missing}
        self.n_applied, self.n_missing = 0, 0
        return out


class CardGifts:
    """送牌课程（第 20800 轮起）：frac 的机器人局里，turns 每回合招募开始时按 p 的机会往学习者手里放一张“高手抢着买、
    模型从来不买”的牌：rl/card_value.json 里高手挑中倍率 ≥ min_lift 的随从和法术（发现、免费刷新、鲜血宝石、抉择……；
    纯加金币的经济法术不送——那些模型自己会买，倍率比高手还高）。酒馆等级够得着、本局开了那个种族、手牌没满才送。
    场面、金币、酒馆不动：打不打、怎么选、留不留全看模型。

    为什么：模型买这些牌的倍率是 0.0–0.3（高手 2–4.5）——无畏的食客、刷新畸体、尖利的鲨鱼、圈套陷阱师、竞技表演者、
    热气球测绘员……高手靠它们每回合多拿宝石、刷新、发现、金色随从，把场面养大（第 9 回合卖牌收入 8 块，模型 2 块）。
    模型从来没拿过这些牌，价值网络不知道“手里有一张无畏的食客”值多少，所以买它永远不划算。送到手里以后它会打出来、
    做抉择、用宝石，价值网络学到这些局面值多少，酒馆里再看到这张牌时“买”的优势就是正的——不用改奖励、不逼动作。
    规则照着凑（ShopPractice.expert）那一版逼它卖牌、刷新，带坏了所有局；这里只是让它见过。"""

    def __init__(self, frac: float = 0.25, turns=(4, 5, 6, 7, 8, 9, 10, 11, 12), p: float = 0.6,
                 min_lift: float = 2.0, seed: int = 0, cards: list | None = None):
        import re
        from pathlib import Path
        from tg.catalog import catalog
        self.det = catalog()
        self.by_cid = {int(v["cid"]): k for k, v in self.det.items() if "cid" in v}
        self.frac, self.turns, self.p = float(frac), set(int(t) for t in turns), float(p)
        self.rng = random.Random(seed + 7717)
        if cards is None:
            cv = Path(__file__).with_name("card_value.json")
            ref = json.loads(cv.read_text(encoding="utf-8")).get("cards", {}) if cv.exists() else {}
            cards = [k for k, v in ref.items() if float(v.get("lift", 0)) >= min_lift]
        self.pool = []                                 # [(卡 id, 种类, 等级, 种族)]
        for k in cards:
            d = self.det.get(k) or {}
            kind = d.get("kind")
            if not d.get("in_pool") or kind not in ("minion", "tavern_spell"):
                continue
            if kind == "tavern_spell" and re.search(r"\bGold\b", d.get("text") or ""):
                continue
            self.pool.append((k, kind, int(d.get("tier") or 1), list(d.get("races") or [])))
        self.plan: dict[int, list] = {}                # 对局 → [第几局, 送不送, 最近送过的回合]
        self.n_given, self.n_full = 0, 0
        self.given: dict[str, int] = {}

    def is_active(self, g: int) -> bool:
        v = self.plan.get(g)
        return v is not None and bool(v[1])

    def active(self) -> set:
        return {g for g, v in self.plan.items() if v[1]}

    def options(self, tier: int, lob) -> list:
        return [x for x in self.pool if x[2] <= tier and
                (not lob or not x[3] or any(r in lob or r == "ALL" for r in x[3]))]

    def apply(self, env, features: np.ndarray, episodes=None) -> bool:
        """每步 observe 之后调用：新的一局先决定送不送；送的局在 turns 里每回合第一次看到招募阶段时按 p 送一张。
        返回有没有改（改了要重新 observe）。"""
        from tg.obs import lobby
        if self.frac <= 0 or not self.pool:
            return False
        from taverngym import _core
        turns = np.rint(features[:, 0] * 20).astype(int)
        changed = False
        for g in range(env.n_games):
            ep = int(episodes[g]) if episodes is not None else env.episode_index(g)
            cur = self.plan.get(g)
            if cur is None or cur[0] != ep:
                cur = self.plan[g] = [ep, self.rng.random() < self.frac, -1]
            t = int(turns[g])
            if not cur[1] or t not in self.turns or cur[2] == t:
                continue
            s = _loads(env.save(g))
            game = s["game"]
            if game.get("turn") != t or str(game.get("phase")) != "Recruit":
                continue
            cur[2] = t
            if self.rng.random() >= self.p:
                continue
            me = game["players"][0]
            if len(me.get("hand") or []) >= 10:
                self.n_full += 1
                continue
            opts = self.options(int(me.get("tier") or 1), lobby(features[g]))
            if not opts:
                continue
            k, kind = self.rng.choice(opts)[:2]
            try:
                G = _core.Game.from_json(_dumps(game))
                G.debug(_dumps({"cmd": "add_minion" if kind == "minion" else "add_spell", "p": 0, "zone": "hand",
                                "card": k}))
            except BaseException as ex:                            # noqa: BLE001 —— 引擎内部出错：这回合不送
                if isinstance(ex, (KeyboardInterrupt, SystemExit)):
                    raise
                continue
            s["game"] = _loads(G.state_json())
            env.load(g, _dumps(s))
            self.n_given += 1
            self.given[k] = self.given.get(k, 0) + 1
            changed = True
        return changed

    def finish(self, g: int) -> bool:
        """对局 g 打完了：返回这局是不是送牌局。"""
        cur = self.plan.pop(g, None)
        return bool(cur is not None and cur[1])

    def take_stats(self) -> dict:
        out = {"gifts": self.n_given, "gift_hand_full": self.n_full}
        self.n_given, self.n_full = 0, 0
        return out
