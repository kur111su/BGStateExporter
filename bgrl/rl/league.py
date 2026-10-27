"""B1 对手库：按真实对局校准对手强度。

为什么要有：TavernGym 自带的脚本机器人前 7 回合和真实对局差不多，后期却弱得多（实测：第 11 回合
机器人阵容总攻防中位数约 290，真实对手约 1,600；第 13 回合约 570 对 4,700）。只和机器人打，学到的是
弱对手的后期打法。论文里单项提升最大的就是“按真实对局校准的对手库”。

做法：
  1. 现实画像（real_profile）：从你的对局记录统计每回合对手的阵容总攻防（交手时看到的）、酒馆等级、血量。
  2. 对手库：训练过程中定期存下的策略快照 + 脚本机器人。每个快照用自我对弈测一遍同样的画像。
  3. 开局时给 7 个对手座位抽对手：越接近现实画像的快照权重越高（按第 4–13 回合阵容强度的对数差）。
  4. 考试库和训练库分开：每隔几个快照留一个只用来考试（评估），训练对局里不出现，避免过拟合到特定对手。
"""
from __future__ import annotations

import json
import math
import random
import statistics as st
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

PROFILE_TURNS = range(1, 16)
CALIB_TURNS = range(4, 14)
REAL_PROFILE = Path(__file__).resolve().parent / "real_profile.json"


# ---------------- 画像 ----------------

def real_profile(files: list[str]) -> dict:
    """从对局记录统计：每回合对手阵容总攻防（中位数、四分位）、对手等级、对手血量+护甲、你的名次分布。"""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tg.logs import read_records, seen_boards

    board, tier, hp = defaultdict(list), defaultdict(list), defaultdict(list)
    places = []
    games = 0
    for f in files:
        r = read_records(f)
        if len(r) < 50:
            continue
        games += 1
        for t, (_h, b) in seen_boards(r).items():
            board[t].append(sum((c.get("attack") or 0) + (c.get("health") or 0) for c in b))
        last = {}
        for s in r:
            if s.get("phase") == "recruit" and s.get("leaderboard"):
                last[s["turn"]] = s
        for t, s in last.items():
            for e in s["leaderboard"]:
                if e.get("isPlayer") or e.get("dead"):
                    continue
                tier[t].append(e.get("tavernTier") or 0)
                hp[t].append((e.get("health") or 0) + (e.get("armor") or 0))
        fin = [s for s in r if s.get("leaderboard")]
        me = [e for e in (fin[-1]["leaderboard"] if fin else []) if e.get("isPlayer")]
        if me and me[0].get("place"):
            places.append(me[0]["place"])
    return {"source": "real", "games": games,
            "avgPlace": round(sum(places) / len(places), 2) if places else None, "turns": _summ(board, tier, hp)}


def _summ(board, tier, hp) -> dict:
    out = {}
    for t in PROFILE_TURNS:
        if not board.get(t) and not tier.get(t):
            continue
        b = sorted(board.get(t) or [0])
        out[str(t)] = {"n": len(board.get(t) or []), "boardMedian": st.median(b),
                       "boardQ1": b[len(b) // 4], "boardQ3": b[(3 * len(b)) // 4],
                       "tier": round(st.mean(tier[t]), 2) if tier.get(t) else None,
                       "hp": round(st.mean(hp[t]), 1) if hp.get(t) else None}
    return out


def sim_profile(policy=None, games: int = 32, seed: int = 0, config: dict | None = None) -> dict:
    """在 TavernGym 里测一个策略的画像（8 个座位都用它；policy=None 用脚本机器人）。
    policy(game, seat) -> 动作编号。统计每回合战斗时的阵容总攻防、等级、血量。"""
    import taverngym
    board, tier, hp = defaultdict(list), defaultdict(list), defaultdict(list)
    for k in range(games):
        g = taverngym.Game(seed=seed + k, config_json=json.dumps(config) if config else None)
        last = g.turn
        while not g.is_over():
            for p in range(g.num_players):
                if g.needs_action(p):
                    g.step(p, g.bot_action(p) if policy is None else policy(g, p))
            if g.turn != last:
                s = json.loads(g.state_json())
                for pl in s["players"]:
                    if pl["alive"]:
                        board[last].append(sum(m["atk"] + m["hp"] for m in pl["last_board"]))
                        tier[last].append(pl["tier"])
                        hp[last].append(pl["health"] + pl["armor"])
                last = g.turn
    return {"source": "sim", "games": games, "turns": _summ(board, tier, hp)}


def sim_profile_batched(policy_batch, games: int = 16, seed: int = 0, config: dict | None = None,
                        max_rounds: int = 20000) -> dict:
    """同 sim_profile，但所有对局、所有座位一起出动作：policy_batch(features[n,F], cards[n,C], mask[n,A]) -> 动作[n]。
    （逐个决策调用模型太慢：一局要几千次前向。）"""
    import numpy as np
    import taverngym
    gs = [taverngym.Game(seed=seed + k, config_json=json.dumps(config) if config else None) for k in range(games)]
    last = [g.turn for g in gs]
    board, tier, hp = defaultdict(list), defaultdict(list), defaultdict(list)
    for _ in range(max_rounds):
        todo = [(gi, p) for gi, g in enumerate(gs) if not g.is_over() for p in range(g.num_players) if g.needs_action(p)]
        if not todo:
            break
        fs, cs, ms = [], [], []
        for gi, p in todo:
            f, c = gs[gi].observe(p)
            fs.append(np.frombuffer(f, "<f4"))
            cs.append(np.frombuffer(c, "<i4"))
            ms.append(np.frombuffer(gs[gi].legal_mask(p), dtype=np.uint8).astype(bool))
        acts = policy_batch(np.stack(fs), np.stack(cs), np.stack(ms))
        for (gi, p), a in zip(todo, acts):
            g = gs[gi]
            if g.needs_action(p):
                g.step(p, int(a))
        for gi, g in enumerate(gs):
            if g.turn != last[gi]:
                st_ = json.loads(g.state_json())
                for pl in st_["players"]:
                    if pl["alive"]:
                        board[last[gi]].append(sum(m["atk"] + m["hp"] for m in pl["last_board"]))
                        tier[last[gi]].append(pl["tier"])
                        hp[last[gi]].append(pl["health"] + pl["armor"])
                last[gi] = g.turn
    return {"source": "sim", "games": games, "turns": _summ(board, tier, hp)}


def distance(p: dict, real: dict) -> float:
    """两个画像的距离：第 4–13 回合阵容总攻防中位数的对数差的均方根（等级差也算一点）。"""
    acc, n = 0.0, 0
    for t in CALIB_TURNS:
        a, b = p["turns"].get(str(t)), real["turns"].get(str(t))
        if not a or not b:
            continue
        acc += (math.log1p(a["boardMedian"]) - math.log1p(b["boardMedian"])) ** 2
        if a.get("tier") is not None and b.get("tier") is not None:
            acc += 0.25 * (a["tier"] - b["tier"]) ** 2
        n += 1
    return math.sqrt(acc / n) if n else float("inf")


def load_real_profile(path: str | Path = REAL_PROFILE) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------- 对手库 ----------------

@dataclass
class Member:
    name: str                       # "bot" 或检查点文件名
    path: str = ""                  # 检查点路径（bot 为空）
    profile: dict | None = None
    exam: bool = False              # 只用于考试
    dist: float = float("inf")


@dataclass
class League:
    """对手库。members 里的快照按和现实画像的距离加权抽样；exam 的只在评估时用。"""
    real: dict
    members: list[Member] = field(default_factory=list)
    temperature: float = 0.25
    exam_every: int = 4             # 每存 4 个快照留 1 个给考试库
    bot_floor: float = 0.05         # 机器人至少占这么多（保持对手多样）

    def add(self, name: str, path: str = "", profile: dict | None = None) -> Member:
        n_snap = sum(1 for m in self.members if m.name != "bot")
        exam = name != "bot" and n_snap % self.exam_every == self.exam_every - 1
        m = Member(name, path, profile, exam)
        if profile:
            m.dist = distance(profile, self.real)
        self.members.append(m)
        return m

    def training_weights(self) -> list[tuple[Member, float]]:
        pool = [m for m in self.members if not m.exam]
        if not pool:
            return []
        best = min(m.dist for m in pool)
        w = [math.exp(-(m.dist - best) / self.temperature) if math.isfinite(m.dist) else 0.0 for m in pool]
        if sum(w) == 0:
            w = [1.0] * len(pool)
        tot = sum(w)
        w = [x / tot for x in w]
        out = []
        for m, x in zip(pool, w):
            if m.name == "bot":
                x = max(x, self.bot_floor)
            out.append((m, x))
        tot = sum(x for _, x in out)
        return [(m, x / tot) for m, x in out]

    def sample(self, rng: random.Random, k: int = 7) -> list[Member]:
        ws = self.training_weights()
        return rng.choices([m for m, _ in ws], weights=[x for _, x in ws], k=k)

    def exam_members(self) -> list[Member]:
        return [m for m in self.members if m.exam] or [m for m in self.members if m.name == "bot"]

    def to_json(self) -> dict:
        return {"members": [{"name": m.name, "path": m.path, "exam": m.exam, "dist": m.dist,
                             "profile": m.profile} for m in self.members]}

    @classmethod
    def from_json(cls, d: dict, real: dict) -> "League":
        lg = cls(real)
        for x in d.get("members") or []:
            m = Member(x["name"], x.get("path", ""), x.get("profile"), x.get("exam", False), x.get("dist", float("inf")))
            lg.members.append(m)
        return lg
