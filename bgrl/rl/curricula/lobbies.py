"""指定酒馆种族（课程之一；配方 lobby_*）：新开的一局按 lobby_frac 的概率换成必开 lobby_must 的一局，
其余几个种族从 lobby_pool 里随机抽（空 = 老师不会的 7 个种族，也就是野猪人 / 恶魔 / 元素一个都不开）。

用途：单种族专家（configs/exp/beast.json）——只在“必开野兽、没有野猪人 / 恶魔 / 元素”的酒馆里练，
逼模型在这些局里自己找阵容；练好以后再蒸馏回主模型。lobby_exam 打开时考试也在这种酒馆里考（守护按专家的本事挑 best）。

实现：另开一局 1 局的 BatchEnv（配置多一项 lobby_types，5 个种族全写死），存成 JSON 读进原来的位置
（BatchEnv.save / load）；episode 编号改成原来那个，之后接着开的局还是按原来的编号往下（种子不重复）。
第 28935 轮删掉的种族课程（TribeLobbies）用的是同一个办法，那时只强制 3 个、剩下 2 个随机，野猪人 / 恶魔 / 元素常常还在。
"""
from __future__ import annotations

import json

import numpy as np

from ..comp import WEAK

N_RACES = 5                                       # TavernGym 默认每局 5 个种族（RulesConfig.lobby_races）


def parse_types(spec: str) -> list[str]:
    from tg.obs import F_LOBBY
    out = [x.strip().upper() for x in str(spec or "").split(",") if x.strip()]
    bad = [x for x in out if x not in F_LOBBY]
    if bad:
        raise ValueError(f"不认识的种族 {bad}，可选 {sorted(F_LOBBY)}")
    return out


class LobbyForcer:
    def __init__(self, must, pool=None, frac: float = 1.0, rules: dict | None = None, seed: int = 0):
        self.must = list(must)
        self.pool = [t for t in (pool or WEAK) if t not in self.must]
        if len(self.must) > N_RACES or len(self.must) + len(self.pool) < N_RACES:
            raise ValueError(f"lobby_must {self.must} / lobby_pool {self.pool} 凑不出 {N_RACES} 个种族")
        self.frac, self.rules = float(frac), dict(rules or {})
        self.rng = np.random.default_rng(seed + 5557)
        self.n_applied = self.n_failed = 0

    def types(self) -> list[str]:
        k = N_RACES - len(self.must)
        rest = self.rng.choice(self.pool, size=k, replace=False).tolist() if k else []
        return sorted(self.must + rest)

    def apply(self, env, g: int) -> bool:
        """第 g 局刚开：按概率换成指定种族的一局。换了返回 True（调用方要重新 observe）。"""
        if self.frac <= 0 or self.rng.random() >= self.frac:
            return False
        from taverngym import BatchEnv
        try:
            gen = BatchEnv(1, seed=int(self.rng.integers(1, 2**62)), controlled=tuple(env.controlled), threads=1,
                           config={**self.rules, "lobby_types": self.types()})
            ep = json.loads(gen.save(0))
            if isinstance(ep, dict) and "index" in ep:
                ep["index"] = int(env.episode_index(g))
            env.load(g, json.dumps(ep))
        except Exception as ex:                         # noqa: BLE001 —— 换不了就打普通的一局
            self.n_failed += 1
            if self.n_failed <= 3:
                print(f"指定种族：换不了这一局（{type(ex).__name__}: {str(ex)[:120]}），打普通的", flush=True)
            return False
        self.n_applied += 1
        return True

    def apply_all(self, env) -> bool:
        changed = False
        for g in range(env.n_games):
            changed |= self.apply(env, g)
        return changed

    def take_stats(self) -> dict:
        out = {"lobby_forced": self.n_applied}
        if self.n_failed:
            out["lobby_failed"] = self.n_failed
        self.n_applied = self.n_failed = 0
        return out
