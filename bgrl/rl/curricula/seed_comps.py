"""成型阵容课程（课程之一，原 rl/curricula/seed_comps.CompSeeder）：机器人局里学习者到了某一回合，场面换成库里成型的 S / A 阵容，学“成型以后怎么玩”。"""
from __future__ import annotations

import random

import numpy as np

from tg.catalog import base_id, is_golden_id

from ._json import _dumps, _loads


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
        from ..comp import comp_progress, formed, load_target_comps, reachable

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
        skip：这些对局不换（送牌局，见 CardGifts）。"""
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
