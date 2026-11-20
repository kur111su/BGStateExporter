"""种族课程（课程之一，原 rl/tribes.py；配方 tribe_*）：一部分机器人局开局时强制本局出现几个模型还不会的种族。

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


def parse_comps(spec: str) -> list[str]:
    """“圣盾流龙族,喝茶流” → 名字列表；不认识的名字报错。"""
    from ..comp import load_target_comps
    names = [x.strip() for x in str(spec or "").split(",") if x.strip()]
    known = {c["name"] for c in load_target_comps()}
    bad = [x for x in names if x not in known]
    if bad:
        raise ValueError(f"不认识的阵容 {bad}，可选 {sorted(known)}")
    return names


def new_comps(comps: list[dict]) -> list[dict]:
    """老师不会的种族的 S / A 阵容：核心卡有种族、而且一个都不是野猪人 / 恶魔 / 元素。"""
    return [c for c in comps if c["tier"] in "SA" and c.get("tribes") and not (set(c["tribes"]) & set(STRONG))]


class TribeStarter:
    """种族课程局的起步牌（第 28049 轮起；配方 tribe_start_*）：p 的种族课程局在第 turn 回合招募开始时，
    往学习者手里放 cards 张一套新种族阵容的核心卡（本局凑得成、用到强制的种族之一；先给等级低的、酒馆等级 + 2 以内）。
    之后买什么、留不留、怎么做完全看模型。

    为什么：种族课程加了全强度以后练了 600 轮，这些局里成了新种族阵容的只有 1.5–2%（没成型 55%，其余是恶魔 / 元素 /
    野猪人）；成型阵容课程给的是做好的场面（平均 3.0 名，拿到会打），缺的是开头几步——第 3–6 回合认出这局能走野兽 /
    亡灵、去买去留。起步牌让它从“手里已经有两张”开始往下做；等没给牌的课程局也开始成型（tribe_newcomp_free 往上走），
    再把 turn 往前、cards 往少调（反向课程）。"""

    def __init__(self, p: float = 0.5, turn: int = 4, cards: int = 2, seed: int = 0, focus: list[str] | None = None):
        from .gifts import CompGifts
        self.cg = CompGifts(frac=0.0, seed=seed)          # 借它的卡牌表、next_card、_owned
        self.comps = [c for c in self.cg.comps if c["name"] in focus] if focus else new_comps(self.cg.comps)
        self.p, self.turn, self.cards = float(p), int(turn), int(cards)
        self.rng = np.random.default_rng(seed + 6151)
        self.plan: dict[int, dict] = {}               # 局 → {"forced", "done", "comp"}
        self.n_given = self.n_none = 0

    def begin(self, g: int, forced) -> bool:
        """种族课程局刚开：按 p 决定给不给起步牌。"""
        self.plan.pop(g, None)
        if self.p <= 0 or not self.comps or self.rng.random() >= self.p:
            return False
        self.plan[g] = {"forced": set(forced), "done": False, "comp": ""}
        return True

    def active(self) -> set:
        return set(self.plan)

    def clear(self):
        self.plan.clear()

    def apply(self, env, features: np.ndarray) -> bool:
        """每步 observe 之后调用：到了第 turn 回合的招募阶段，给一次。返回有没有改（改了要重新 observe）。"""
        if not self.plan:
            return False
        from taverngym import _core

        from tg.obs import lobby
        from ._json import _dumps, _loads
        from ..comp import reachable
        turns = np.rint(np.asarray(features[:, 0], dtype=np.float32) * 20).astype(int)
        changed = False
        for g, cur in self.plan.items():
            if cur["done"] or g >= len(turns) or int(turns[g]) < self.turn:
                continue
            s = _loads(env.save(g))
            game = s["game"]
            if str(game.get("phase")) != "Recruit":
                continue
            cur["done"] = True
            lob = lobby(features[g])
            ok = [c for c in self.comps if reachable(c, lob) and (not c.get("tribes") or set(c["tribes"]) & cur["forced"])]
            if not ok:
                self.n_none += 1
                continue
            me = game["players"][0]
            own, tier, hand = self.cg._owned(me), int(me.get("tier") or 1), len(me.get("hand") or [])
            # 随机挑一套“现在就有核心卡能给”的（核心卡等级 ≤ 酒馆等级 + 2）
            ok = [ok[j] for j in self.rng.permutation(len(ok))]
            comp = next((c for c in ok if self.cg.next_card(c, set(own), tier) in c["core"]), None)
            if comp is None:
                self.n_none += 1
                continue
            try:
                G = _core.Game.from_json(_dumps(game))
                given = 0
                for _ in range(self.cards):
                    k = self.cg.next_card(comp, own, tier)
                    if not k or k not in comp["core"] or hand + given >= 10:
                        break
                    G.debug(_dumps({"cmd": "add_minion", "p": 0, "zone": "hand", "card": k}))
                    own.add(k)
                    given += 1
            except BaseException as ex:                     # noqa: BLE001 —— 引擎内部出错：这局不给
                if isinstance(ex, (KeyboardInterrupt, SystemExit)):
                    raise
                continue
            if given:
                s["game"] = _loads(G.state_json())
                env.load(g, _dumps(s))
                cur["comp"] = comp["name"]
                self.n_given += given
                changed = True
        return changed

    def finish(self, g: int):
        """局打完：给过起步牌的局返回那套的名字（没给成 = ""），不是起步牌局返回 None。"""
        cur = self.plan.pop(g, None)
        return None if cur is None else cur["comp"]

    def take_stats(self) -> dict:
        out = {"tribe_start_cards": self.n_given, "tribe_start_none": self.n_none}
        self.n_given = self.n_none = 0
        return out
