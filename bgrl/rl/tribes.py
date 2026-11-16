"""种族课程（rl/ppo.py 用；配方 tribe_*）：一部分机器人局开局时强制本局出现几个模型还不会的种族。

为什么：集中学老师以后（第 27116 轮），256 局阵容测试里模型的主种族是野猪人 48%、恶魔 23%、元素 11%，
野兽 / 亡灵 / 机械 / 鱼人 / 龙的阵容成型全是 0%；这三族都没开的局只能打“混合”，平均 5.25 名、吃鸡 0%。
随机的酒馆里这种局只有一成多，模型没有压力学别的。TavernGym 的 RulesConfig.lobby_types（“For debugging and
curricula”）可以指定本局必定出现的种族：每局 5 个种族里先放 tribe_force 个 tribe_pool 里的，剩下的随机抽。
tribe_force 3：剩下 2 个从其余 7 个里抽，野猪人 / 恶魔 / 元素一个都没有的局约 29%（随机酒馆约 8%）。

实现：新开的一局（上一局打完）按 tribe_frac 的概率换成另外开的一局（1 局的 BatchEnv，配置多一项 lobby_types），
存成 JSON 读进原来的位置（BatchEnv.save / load）；episode 编号改成原来那个，之后接着开的局还是普通的。
"""
from __future__ import annotations

import json

import numpy as np

STRONG = ("QUILBOAR", "DEMON", "ELEMENTAL")     # 老师（外部模型）会的三族
WEAK = ("BEAST", "DRAGON", "MECHANICAL", "MURLOC", "PIRATE", "UNDEAD", "ABERRATION")


def parse_types(spec: str) -> list[str]:
    from tg.obs import F_LOBBY
    out = [x.strip().upper() for x in str(spec or "").split(",") if x.strip()]
    bad = [x for x in out if x not in F_LOBBY]
    if bad:
        raise ValueError(f"不认识的种族 {bad}，可选 {sorted(F_LOBBY)}")
    return out


def lobby_has_any(f, types) -> np.ndarray:
    """[行]：本局开放的种族里有没有 types 里的任何一个（f 原始或压缩过的都行）。types 空 → 全是 True。"""
    from tg.obs import F_LOBBY, lobby_bits
    names = list(F_LOBBY)
    want = sum(1 << names.index(t) for t in types)
    bits = lobby_bits(f)
    return np.ones(len(bits), dtype=bool) if not want else (bits & want) > 0


class TribeLobbies:
    def __init__(self, frac: float, pool: list[str], force: int, rules: dict, seed: int = 0):
        self.frac, self.pool, self.force, self.rules = float(frac), list(pool), int(force), dict(rules or {})
        self.rng = np.random.default_rng(seed + 4049)
        self.active: dict[int, tuple] = {}         # 局 → 这局强制的种族
        self.n_applied = self.n_failed = 0

    def apply(self, env, g: int) -> bool:
        """第 g 局刚开：按概率换成指定种族的一局。换了返回 True（调用方要重新 observe）。"""
        self.active.pop(g, None)
        if self.frac <= 0 or not self.pool or self.rng.random() >= self.frac:
            return False
        from taverngym import BatchEnv
        k = min(self.force, len(self.pool))
        types = tuple(sorted(self.rng.choice(self.pool, size=k, replace=False).tolist()))
        try:
            gen = BatchEnv(1, seed=int(self.rng.integers(1, 2**62)), controlled=tuple(env.controlled), threads=1,
                           config={**self.rules, "lobby_types": list(types)})
            ep = json.loads(gen.save(0))
            if isinstance(ep, dict) and "index" in ep:
                ep["index"] = int(env.episode_index(g))   # 之后接着开的局按原来的编号往下（种子不重复）
            env.load(g, json.dumps(ep))
        except Exception as ex:                         # noqa: BLE001 —— 换不了就打普通的一局
            self.n_failed += 1
            if self.n_failed <= 3:
                print(f"种族课程：换不了这一局（{type(ex).__name__}: {str(ex)[:120]}），打普通的", flush=True)
            return False
        self.active[g] = types
        self.n_applied += 1
        return True

    def finish(self, g: int):
        """第 g 局打完：是种族课程的局就返回强制的种族（tuple），否则 None。"""
        return self.active.pop(g, None)

    def clear(self):
        self.active.clear()

    def take_stats(self) -> dict:
        out = {"tribe_new": self.n_applied}
        self.n_applied = 0
        return out
