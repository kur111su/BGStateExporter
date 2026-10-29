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

    def _pick(self, t: int) -> dict:
        ok = [x for x in self.turns if x <= t]
        tt = ok[-1] if ok else self.turns[0]
        return self.rng.choice(self.lib[tt])

    def boost(self, env, features: np.ndarray, episodes=None) -> bool:
        if self.frac <= 0 or not self.lib or self.strength <= 0:
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
            G = None
            for seat in range(1, len(game["players"])):
                p = game["players"][seat]
                if not p.get("alive", True) or self.rng.random() >= self.strength:
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
                    gold = cid.endswith("_G")
                    try:
                        G.debug(_dumps({"cmd": "add_minion", "p": seat, "zone": "board",
                                        "card": cid[:-2] if gold else cid, "golden": gold,
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
        return changed

    def take_stats(self) -> dict:
        out = {"boosts": self.n_boosts, "pro_injected": self.injected, "pro_failed": self.failed}
        self.n_boosts, self.injected, self.failed, self.added = 0, 0, 0, 0.0
        return out
