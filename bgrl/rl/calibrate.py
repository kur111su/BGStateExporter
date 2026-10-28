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

    def take_stats(self) -> dict:
        out = {"boosts": self.n_boosts, "boost_added_k": round(self.added / 1000, 1)}
        self.n_boosts, self.added = 0, 0.0
        return out


def ramp(it: int, start_it: int, ramp_iters: int, lo: float = 0.3) -> float:
    """强度从 lo 线性爬到 1（第 start_it 轮开始，ramp_iters 轮爬满）。"""
    if ramp_iters <= 0:
        return 1.0
    x = (it - start_it) / ramp_iters
    return lo + (1 - lo) * min(1.0, max(0.0, x)) if not math.isnan(x) else 1.0
