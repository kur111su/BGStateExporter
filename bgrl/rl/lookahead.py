"""训练时用模拟器推演，产生比模型自己更好的学习目标（Config.look_*；思路同 AlphaZero 的“搜索 → 蒸馏”）。

后期（look_from 回合起）抽一部分学习者的决策，在这局的副本上推演：候选 = 模型分数最高的 k 个动作 + 结束回合，
每个候选在同一个副本上走一步，再让模型自己（贪心）把这一回合打完（最多 depth 步），回合结束时的场面拿同一组同期
高分阵容实打（BoardJudge），场面最强的那个就是这个局面的“推演答案”。只有明显比模型自己的首选强（margin）才换答案。

答案存进 LookBuffer（局面 + 答案），更新时和 PPO 一起学一项交叉熵（look_coef × -log π(答案)）。
推演只在训练里用：竞技场里的模型还是一次前向出动作，不需要模拟器。

副本的随机数换过：推演看不到真实的下一个酒馆（不然学到的是“偷看”）。考官和养场面 / 阵容强度奖励用的分开。
"""
from __future__ import annotations

import json

import numpy as np

from tg.actions import A_END


def search(game, seat: int, turn: int, m, lg, judge, lg_fn, k: int = 5, depth: int = 6, margin: float = 0.02,
           rng_seed: int = 0) -> tuple[int, int, float | None, float | None]:
    """game = 这一局（env.game(g) 拿到的，不会改它）。返回 (推演答案, 模型首选, 首选的场面强度, 答案的场面强度)；
    推演不出来（考官出错）强度是 None，答案 = 首选。"""
    from taverngym import _core
    m = np.asarray(m, dtype=bool)
    lg = np.asarray(lg, dtype=np.float64)
    legal = np.flatnonzero(m)
    order = legal[np.argsort(-lg[legal], kind="stable")]
    a0 = int(order[0])
    cand = [int(x) for x in order[:k]]
    if m[A_END] and A_END not in cand:
        cand.append(A_END)
    s = json.loads(game.state_json())
    r = np.random.default_rng(rng_seed)
    s["rng"]["s"] = [int(x) for x in r.integers(1, 2 ** 63, size=4)]
    base = _core.Game.from_json(json.dumps(s))
    runs = []
    for a in cand:
        G = base.copy()
        if a != A_END:
            try:
                G.step(int(seat), int(a))
            except ValueError:
                continue
        runs.append([a, G, a == A_END])
    for _ in range(depth):                                 # 每个候选让模型自己把这回合打完（遇到结束回合就停）
        live = [x for x in runs if not x[2]]
        if not live:
            break
        F, C, M = [], [], []
        for x in live:
            fb, cb = x[1].observe(int(seat))
            F.append(np.frombuffer(fb, dtype=np.float32))
            C.append(np.frombuffer(cb, dtype=np.int32))
            M.append(np.frombuffer(x[1].legal_mask(int(seat)), dtype=np.uint8).astype(bool))
        Mx = np.stack(M)
        ok = Mx.any(axis=1)
        for x, good in zip(live, ok):
            if not good:
                x[2] = True
        if not ok.any():
            break
        idx = np.flatnonzero(ok)
        L2 = lg_fn(np.stack(F)[idx], np.stack(C)[idx], Mx[idx])
        for x, l2, mm in zip([live[i] for i in idx], L2, Mx[idx]):
            b = int(np.argmax(np.where(mm, l2, -np.inf)))
            if b == A_END:
                x[2] = True
                continue
            try:
                x[1].step(int(seat), b)
            except ValueError:
                x[2] = True
    vals = {}
    for a, G, _ in runs:
        v = judge.judge(G, int(seat), int(turn))
        if v is not None:
            vals[a] = float(v)
    v0 = vals.get(a0)
    if v0 is None or not vals:
        return a0, a0, None, None
    best = max(vals, key=lambda a: (vals[a], a == a0))
    if vals[best] < v0 + margin:
        best = a0
    return int(best), a0, v0, vals[best]


class LookBuffer:
    """推演答案：最近 cap 条（局面压缩后的特征 float16、卡牌编号、合法动作、答案），满了覆盖最早的。"""

    def __init__(self, cap: int, seed: int = 0):
        self.cap, self.n, self.pos = int(cap), 0, 0
        self.f = self.c = self.m = None
        self.a = np.zeros(self.cap, dtype=np.int64)
        self.rng = np.random.default_rng(seed + 4049)

    def __len__(self) -> int:
        return self.n

    def add(self, f, c, m, a) -> None:
        f, c, m = np.asarray(f, dtype=np.float16), np.asarray(c, dtype=np.int32), np.asarray(m, dtype=bool)
        if self.f is None:
            self.f = np.zeros((self.cap, f.shape[-1]), dtype=np.float16)
            self.c = np.zeros((self.cap, c.shape[-1]), dtype=np.int32)
            self.m = np.zeros((self.cap, m.shape[-1]), dtype=bool)
        i = self.pos
        self.f[i], self.c[i], self.m[i], self.a[i] = f, c, m, int(a)
        self.pos = (i + 1) % self.cap
        self.n = min(self.cap, self.n + 1)

    def sample(self, k: int):
        j = self.rng.integers(0, self.n, size=min(int(k), self.n))
        return self.f[j], self.c[j], self.m[j], self.a[j]
