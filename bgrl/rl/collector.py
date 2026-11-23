"""采样（rl/ppo.Trainer 的一部分，作为 mixin 继承进去）：对局安排、学习者 / 对手快照前向、课程钩子、奖励塑形、局末记账、
环境出错时重建。collect() 一轮采满 steps_per_iter 步，返回给 update() 的数据。
"""
from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from tg.actions import (
    A_ACTIVATE,
    A_BUY,
    A_CHOOSE,
    A_SELL,
    is_noop,
    is_panic,
)
from tg.obs import C_BOARD, C_SHOP, F_BOARD, MF_ATK, MF_HP, MINION_F, N_SHOP

from .curricula import (
    BotBooster,
    ProBoardBooster,
    merge_pro_boards,
    ramp,
)
from .comp import board_scores, load_comp_refs
from .league import load_real_profile
from .model import build_model
from .rollout import (
    Policy,
    linear_place_reward,
)


def _board_stat_sum(f, slot: int | None = None) -> float:
    """场上随从攻 + 血的总和（slot 给了就只算那一格）。"""
    rng = range(7) if slot is None else (slot,)
    return float(sum(round(float(f[F_BOARD + MINION_F * s + MF_ATK]) * 10) + round(float(f[F_BOARD + MINION_F * s + MF_HP]) * 10)
                     for s in rng))


_ENGINE_IDX: dict = {}


def _engine_board(cards: np.ndarray, kind: str) -> np.ndarray:
    """每一行（卡牌编号）的场面是不是这种连招场面。rally：最左边是猎头狮鹫 / 囤食土狼，场上有逐亡陆行鸟（金色也算）。"""
    if kind != "rally":
        raise ValueError(f"trick_engine 只支持 rally，给的是 {kind!r}")
    if kind not in _ENGINE_IDX:
        from tg.catalog import base_id, card_id, n_cards
        ids = [base_id(card_id(i)) if i else "" for i in range(n_cards())]
        left = np.array([x in ("BG36_204", "BG36_210") for x in ids])
        need = np.array([x == "BG36_208" for x in ids])
        _ENGINE_IDX[kind] = (left, need)
    left, need = _ENGINE_IDX[kind]
    b = np.asarray(cards)[:, C_BOARD:C_BOARD + 7].astype(np.int64)
    return left[b[:, 0]] & need[b].any(axis=1)


def _commit_allowed(comp: str) -> np.ndarray:
    """定向局能买的卡（按卡牌编号）：这套阵容的核心 / 配套卡、野兽、不是随从的（法术、饰品）。"""
    from tg.catalog import base_id, card_id, catalog, n_cards
    from .comp import load_target_comps
    want = set().union(*(set(c["core"]) | set(c.get("support") or ()) for c in load_target_comps() if c["name"] == comp))
    if not want:
        raise ValueError(f"commit_comp：目标阵容里没有 {comp!r}")
    det = catalog()
    ok = np.ones(n_cards(), dtype=bool)
    for i in range(1, n_cards()):
        b = base_id(card_id(i))
        e = det.get(b) or {}
        if e.get("kind") == "minion" and b not in want and "BEAST" not in (e.get("races") or ()):
            ok[i] = False
    return ok


class CollectMixin:
    def _role_id(self, name: str) -> int:
        if name not in self.role_names:
            self.role_names.append(name)
        return self.role_names.index(name)

    def _rebuild_env(self, ei: int, ex: BaseException) -> None:
        """引擎出错（is_panic）：这一组对局全部作废，换一个新的环境接着练。没打完的局按“结束、奖励 0”收尾。"""
        from taverngym import BatchEnv
        cfg = self.cfg
        name, env, off = self.envs[ei]
        self.env_rebuilds = getattr(self, "env_rebuilds", 0) + 1
        self.mix["env_rebuild"] += 1
        if self.mix["env_rebuild"] > 20:
            raise RuntimeError(f"TavernGym 引擎一轮里出错 {self.mix['env_rebuild']} 次：{ex}")
        print(f"第 {self.iter} 轮：TavernGym 引擎出错（{str(ex)[:200]}），{name} 环境的 {env.n_games} 局作废、重新开局",
              flush=True)
        for i in range(env.n_slots):
            if self.role[off + i] == 0 and self.rollout.last[off + i] >= 0:
                self.rollout.reward(off + i, self._econ_close(off + i), True, 0)
        seed = (cfg.seed if name == "bot" else cfg.seed + 10**6) + 7919 * (self.iter + 1) + self.env_rebuilds
        new = BatchEnv(env.n_games, seed=seed, controlled=env.controlled, threads=cfg.threads, config=cfg.rules())
        if self.lobbies is not None:
            self.lobbies.apply_all(new)
        self.envs[ei] = (name, new, off)
        self.episode[name] = np.asarray([new.episode_index(i) for i in range(new.n_games)])
        for g in range(new.n_games):
            self._assign(name, new, off, g)
        if name == "bot" and self.booster is not None:
            self.booster.plan.clear()
            self.booster.last_turn.clear()
        if name == "bot" and self.seeder is not None:
            self.seeder.plan.clear()
        if name == "bot" and self.gifts is not None:
            self.gifts.plan.clear()
        self.obs[name] = new.observe()

    def _assign(self, name, env, off, g):
        k = len(env.controlled)
        self.role[off + g * k: off + (g + 1) * k] = 0
        if name == "bot":
            return
        # 每局只从“当前在用”的几个快照里抽：每一步每个不同的对手模型都要单独算一次，种类多了会很慢
        snaps = self.active_snaps
        for s in range(1, 8):
            if snaps and self.rng.random() < 0.75:
                m = self.rng.choices([x for x, _ in snaps], weights=[w for _, w in snaps])[0]
                self.role[off + g * k + s] = self._role_id(m.name)

    def _refresh_active(self):
        """每隔 10 轮按对手库权重重新挑最多 active_snapshots 个快照作为这段时间的对手。"""
        pool = [m for m in self.league.training_weights() if m[0].name != "bot" and m[1] > 0]
        if len(pool) > self.cfg.active_snapshots:
            pick = set()
            while len(pick) < self.cfg.active_snapshots:
                pick.add(self.rng.choices(range(len(pool)), weights=[w for _, w in pool])[0])
            pool = [pool[i] for i in sorted(pick)]
        self.active_snaps = pool

    def _policy(self, rid):
        if rid == 0:
            return self.learner
        name = self.role_names[rid]
        if name not in self.snap_cache:
            m = next(x for x in self.league.members if x.name == name)
            model = build_model(**self.mkw).to(self.rdev)
            model.load_state_dict(torch.load(m.path, map_location=self.rdev))
            model.eval()
            self.snap_cache[name] = Policy(model, self.rdev, self.amp)
            if len(self.snap_cache) > 12:
                self.snap_cache.pop(next(iter(self.snap_cache)))
        return self.snap_cache[name]

    def collect(self):
        cfg = self.cfg
        self.model.eval()
        if self.iter % 10 == 0 or not self.active_snaps:
            self._refresh_active()
        n_dec = 0
        self.n_noop = 0
        self.finished = 0
        self.ref_idx[:] = -1                           # 上一轮的步编号在这一轮无效
        t0 = time.time()
        t_env = t_net = 0.0
        self.t_infer, self.n_calls, self.rows_calls = 0.0, 0, 0   # t_net 里纯模型前向的部分（其余是塑形、记账、阵容强度……）
        for _ in range(cfg.steps_per_iter):
            # 先把所有环境里同一个模型要算的局面合在一起，一次前向（比每个环境各算一次快）
            ta = time.time()
            if cfg.commit_frac > 0:
                self._commit_mask()
            actions = {name: np.zeros(env.n_slots, dtype=np.int64) for name, env, _ in self.envs}
            groups = defaultdict(list)                 # 模型编号 -> [(环境下标, 槽位数组)]
            for ei, (name, env, off) in enumerate(self.envs):
                ready = np.flatnonzero(self.obs[name]["ready"])
                if len(ready):
                    roles = self.role[off + ready]
                    for rid in np.unique(roles):
                        groups[int(rid)].append((ei, ready[roles == rid]))
            for rid, parts in groups.items():
                cat = lambda key: np.concatenate([self.obs[self.envs[ei][0]][key][idx] for ei, idx in parts])  # noqa: E731
                f, c, m = cat("features"), cat("cards"), cat("action_mask")
                ti = time.perf_counter()
                res = self._policy(rid)(f, c, m, want_obs=(rid == 0))
                self.t_infer += time.perf_counter() - ti            # Policy 返回 numpy，已经等显卡算完
                self.n_calls += 1
                self.rows_calls += len(f)
                a = res[0]
                o = 0
                for ei, idx in parts:
                    name, env, off = self.envs[ei]
                    k = len(idx)
                    actions[name][idx] = a[o:o + k]
                    if rid == 0:
                        ak = a[o:o + k]
                        noop = is_noop(ak) | (ak == 7)                # 冻结、调站位、取消（打出 → 取消可以无限循环）
                        tk = np.rint(f[o:o + k, 0] * 20).astype(int)
                        r0 = -cfg.noop_penalty * noop.astype(np.float64)
                        self.audit.count(ak, tk)
                        self.audit.add("noop", r0, ak, tk)
                        r0 = self._shape(off + idx, f[o:o + k], ak, r0, c[o:o + k])
                        b = r0.copy()
                        r0 = self._combat_shape(env, off, idx, f[o:o + k], ak, r0)
                        self.audit.add("combat", r0 - b, ak, tk)
                        b = r0.copy()
                        r0 = self._grow_shape(env, off, idx, f[o:o + k], ak, r0)
                        self.audit.add("grow", r0 - b, ak, tk)
                        self.rollout.add(off + idx, res[3][o:o + k], c[o:o + k], m[o:o + k], ak,
                                         res[1][o:o + k], res[2][o:o + k], r0=r0)
                        self._after_add(off + idx, ak, np.rint(f[o:o + k, 0] * 20).astype(int))
                        if cfg.trick_bonus > 0:                # 招募阶段连招：发动 / 卖之前其余随从的身材总和
                            trig = ((ak >= A_ACTIVATE) & (ak < A_ACTIVATE + 7)) | ((ak >= A_SELL) & (ak < A_SELL + 7))
                            # 发动深潜狮子鱼要再选一张酒馆里的牌（选择动作）才真正触发：紧跟在发动 / 卖后面的选择也算
                            chain = self.trick_chain[off + idx]
                            sel = (trig | (chain & (ak >= A_CHOOSE) & (ak < A_CHOOSE + 16))) & (tk >= cfg.trick_from_turn)
                            self.trick_chain[off + idx] = trig | (chain & (ak >= A_CHOOSE) & (ak < A_CHOOSE + 16))
                            if cfg.trick_engine:
                                sel &= _engine_board(c[o:o + k], cfg.trick_engine)
                            for j in np.flatnonzero(sel):
                                st = _board_stat_sum(f[o + j])
                                if A_SELL <= int(ak[j]) < A_SELL + 7:
                                    st -= _board_stat_sum(f[o + j], int(ak[j]) - A_SELL)
                                self.trick_pre.setdefault(name, {})[int(idx[j])] = (st, int(tk[j]))
                        n_dec += k
                        self.n_noop += int(noop.sum())
                        self.mix["noop_move"] += int((ak >= 196).sum())        # 无效操作分开记：调站位 / 冻结 / 取消
                        self.mix["noop_freeze"] += int((ak == 2).sum())
                        self.mix["noop_cancel"] += int((ak == 7).sum())
                    o += k
            tb = time.time()
            for ei, (name, env, off) in enumerate(self.envs):
                try:
                    rewards, dones, info = env.step(actions[name])
                except BaseException as ex:                # noqa: BLE001
                    if not is_panic(ex):
                        raise
                    self._rebuild_env(ei, ex)
                    continue
                for i in np.flatnonzero((rewards != 0) | dones):
                    if self.role[off + i] == 0:
                        r_env = float(rewards[i])
                        if dones[i] and self.place_tab is not None and 1 <= int(info["placement"][i]) <= 8:
                            pl = int(info["placement"][i])         # 局末名次奖励换成表里的值（违规扣分留着）
                            r_env += float(self.place_tab[pl]) - linear_place_reward(pl)
                        if dones[i]:
                            r_ec, r_rep = self._econ_close(off + int(i)), self._shape_close(off + int(i))
                            self.audit.add_one("place", r_env)
                            self.audit.add_one("econ_close", r_ec)
                            self.audit.add_one("repay", r_rep)
                            self.audit.episode()
                            r_end = r_env + r_ec + r_rep
                        else:
                            self.audit.add_one("env", r_env)
                            r_end = r_env
                        fcomp = self._final_comp(off + int(i)) if dones[i] else None
                        self.rollout.reward(off + int(i), r_end, bool(dones[i]), info["placement"][i], fcomp)
                        if dones[i] and env.controlled[int(i) % len(env.controlled)] == 0:
                            seeded = name == "bot" and self.seeder is not None and self.seeder.finish(int(i))
                            gifted = name == "bot" and self.gifts is not None and self.gifts.finish(int(i))
                            committed = name == "bot" and cfg.commit_frac > 0 and self._committed(name, int(i))
                            if committed and not seeded and not gifted:   # 定向局：单独记名次
                                self.commit_places.append(int(info["placement"][i]))
                                if self.booster is not None:
                                    self.booster.finish(int(i))
                                seeded = True
                            elif gifted and not seeded:    # 送牌局：单独记名次
                                self.gift_places.append(int(info["placement"][i]))
                                if self.booster is not None:
                                    self.booster.finish(int(i))
                                seeded = True              # 不算进 place_bot / 加强强度
                            elif seeded:                   # 换过场面的局单独记，不算进 place_bot / 加强强度
                                self.seed_places.append(int(info["placement"][i]))
                                bst = self.booster.finish(int(i)) if self.booster is not None else None
                                # 拿着哪套成型阵容打、打了第几、对手是不是加强过（考试那轮汇总进日志）
                                self.seed_by.setdefault(str(seeded), []).append((int(info["placement"][i]), bool(bst)))
                                seeded = True
                            else:
                                self.stats.append((name, int(info["placement"][i])))
                            self.finished += 1
                            if not seeded and name == "bot" and self.booster is not None and self.booster.finish(int(i)):
                                self.boost_places.append(int(info["placement"][i]))
                                if self.adapt is not None:
                                    self.adapt.add(int(info["placement"][i]))
                self.obs[name] = env.observe()
                pend = self.trick_pre.pop(name, None) if cfg.trick_bonus > 0 else None
                if pend:                                       # 招募阶段连招：发动 / 卖以后其余随从涨了多少
                    fn = self.obs[name]["features"]
                    for i, (st0, t0_) in pend.items():
                        if dones[i] or int(np.rint(fn[i, 0] * 20)) != t0_:
                            continue
                        d = min(100.0, _board_stat_sum(fn[i]) - st0)
                        if d > 0:
                            r = cfg.trick_bonus * d / 10.0
                            self.rollout.reward(off + int(i), r, False, 0)
                            self.audit.add_one("trick", r)
                            self.trick_log[0] += 1
                            self.trick_log[1] += d
                if name == "bot" and self.booster is not None:   # 机器人加强到真人强度（见 rl/curricula/boost.py）
                    tc = time.time()
                    if self.adapt is None:
                        self.booster.strength = ramp(self.iter, 0, cfg.boost_ramp_iters)
                    try:
                        if self.booster.boost(env, self.obs[name]["features"]):
                            self.obs[name] = env.observe()
                    except BaseException as ex:            # noqa: BLE001
                        if not is_panic(ex):
                            raise
                        self._rebuild_env(ei, ex)
                        continue
                    self.t_boost += time.time() - tc
                broke = False
                if name == "bot" and self.gifts is not None:      # 送牌课程：往手里放一张（场面不动，不切 GAE）
                    try:
                        if self.gifts.apply(env, self.obs[name]["features"]):
                            self.obs[name] = env.observe()
                    except BaseException as ex:            # noqa: BLE001
                        if not is_panic(ex):
                            raise
                        self._rebuild_env(ei, ex)
                        continue
                skip = None
                if name == "bot":
                    skip = self.gifts.active() if self.gifts is not None else set()
                for sd in ((self.seeder,) if name == "bot" else ()):
                    if sd is None:                         # 成型阵容课程（换整个场面）
                        continue
                    try:
                        if sd.seed(env, self.obs[name]["features"], skip=skip):
                            self.obs[name] = env.observe()
                            for g in sd.seeded_slots:
                                self.rollout.cut(off + g)        # GAE 在换场面处切断（见 Rollout.cut）
                                self.cs_ep[off + g] = -1         # 阵容强度从新场面重新起算
                                self.gr_turn[off + g] = -1       # 养场面：这回合的“开始”在新场面上重新量
                    except BaseException as ex:            # noqa: BLE001
                        if not is_panic(ex):
                            raise
                        self._rebuild_env(ei, ex)
                        broke = True
                        break
                if broke:
                    continue
                if dones.any():                                # 新开的对局：重新抽对手
                    k = len(env.controlled)
                    swapped = False
                    for g in np.unique(np.flatnonzero(dones) // k):
                        e = env.episode_index(int(g))
                        if e != self.episode[name][g]:
                            if self.lobbies is not None and self.lobbies.apply(env, int(g)):
                                swapped = True             # 指定酒馆种族：换成必开那几个种族的一局（episode 编号不变）
                            self.episode[name][g] = env.episode_index(int(g))
                            self._assign(name, env, off, int(g))
                    if swapped:
                        self.obs[name] = env.observe()
            t_net += tb - ta
            t_env += time.time() - tb
        # 没打完的：用当前局面的价值自举
        last_values = np.zeros(self.n_slots, dtype=np.float32)
        for name, env, off in self.envs:
            obs = self.obs[name]
            live = np.flatnonzero(self.rollout.last[off:off + env.n_slots] >= 0)
            if len(live):
                m = obs["action_mask"][live].copy()
                m[~m.any(1)] = True
                _, _, v = self.learner(obs["features"][live], obs["cards"][live], m)
                last_values[off + live] = v
        batch = self.rollout.collect(cfg.gamma, cfg.lam, last_values)
        return batch, n_dec, time.time() - t0, t_env, t_net

    def _committed(self, name: str, g: int) -> bool:
        """机器人局的第 g 局（这一局的 episode）是不是定向局：按 (局号, episode) 定下来，整局不变。"""
        e = int(self.episode[name][g])
        return ((g * 7919 + e * 104729 + self.cfg.seed * 31) % 1000) < self.cfg.commit_frac * 1000

    def _commit_mask(self):
        """定向局里学习者的“买”只留这套阵容的牌、野兽和非随从（见 Config.commit_comp）。"""
        if self._commit_ok is None:
            self._commit_ok = _commit_allowed(self.cfg.commit_comp)
        for name, env, off in self.envs:
            if name != "bot":
                continue
            obs = self.obs[name]
            k = len(env.controlled)
            for i in np.flatnonzero(obs["ready"]):
                if self.role[off + i] != 0 or not self._committed(name, int(i) // k):
                    continue
                if int(np.rint(obs["features"][i, 0] * 20)) < self.cfg.commit_from_turn:
                    continue
                shop = obs["cards"][i, C_SHOP:C_SHOP + N_SHOP].astype(np.int64)
                m = obs["action_mask"][i]
                bad = ~self._commit_ok[shop] & (shop > 0)
                if bad.any():
                    m[A_BUY:A_BUY + N_SHOP] &= ~bad
                    self.commit_masked += int(bad.sum())

    def _make_lobbies(self, seed: int, frac: float | None = None):
        cfg = self.cfg
        from .curricula import LobbyForcer, parse_types
        return LobbyForcer(parse_types(cfg.lobby_must), parse_types(cfg.lobby_pool) or None,
                           cfg.lobby_frac if frac is None else frac, cfg.rules(), seed=seed)

    def _make_booster(self, frac: float, seed: int):
        cfg = self.cfg
        if cfg.boost_kind == "pro":
            from paths import RUNS
            paths = [Path(p) for p in str(cfg.pro_boards).split(";") if p.strip()] or [RUNS / "pro_boards.json"]
            have = [p for p in paths if p.exists()]
            path = ";".join(str(p) for p in have) or paths[0]
            if have:
                if not getattr(self, "_pro_lib", None):
                    self._pro_lib = merge_pro_boards(have)     # 几个阵容库（分号隔开）合在一起
                    print(f"高分阵容库：{sum(len(v) for v in self._pro_lib.values())} 个整板（{path}）", flush=True)
                return ProBoardBooster(self._pro_lib, frac, cfg.boost_from_turn, seed=seed, every=cfg.boost_every)
            print(f"没有高分阵容库 {path}（先运行 bg.py prolib），改用放大攻防", flush=True)
        return BotBooster(load_real_profile(), frac, cfg.boost_from_turn, seed=seed, every=cfg.boost_every)

    def _comp_scores(self, boards) -> dict:
        """最终阵容的配合程度（场上 ≥ 5 个随从的局；见 rl/comp.board_scores）：
        comp_coh 主种族纯度（万三 0.70）、comp_syn 高手常见搭配对数（2.07）、comp_key 关键卡张数（4.34）、comp_golden 金色随从数（1.85）。"""
        try:
            from tg.catalog import base_id, card_id, catalog, is_golden_id
            if not hasattr(self, "_comp_ref"):
                self._comp_ref = (*load_comp_refs(), catalog())
            pairs, key, det = self._comp_ref
            if not hasattr(self, "_comp_targets"):
                from .comp import load_target_comps
                self._comp_targets = [c for c in load_target_comps() if c["tier"] in "SA"]
            from .comp import formed
            rows, by, n_formed = [], {}, 0
            for b in boards:
                ids = [card_id(int(x)) for x in b if x > 0]
                if len(ids) < 5:
                    continue
                base = [base_id(x) for x in ids]
                sc = board_scores(base, det, pairs, key)
                sc["golden"] = sum(is_golden_id(x) for x in ids)
                rows.append(sc)
                hit = [c["name"] for c in self._comp_targets if formed(base, c)]
                n_formed += bool(hit)
                for nm in hit:
                    by[nm] = by.get(nm, 0) + 1
            if not rows:
                return {}
            out = {f"comp_{k}": round(float(np.mean([r[k] for r in rows])), 3) for k in ("coh", "syn", "key", "golden")}
            out["comp_formed"] = round(n_formed / len(rows), 3)             # 最终场面成了 S / A 阵容的比例（rl/comp.formed）
            out["comp_formed_by"] = {k: round(v / len(rows), 3) for k, v in sorted(by.items(), key=lambda kv: -kv[1])}
            return out
        except Exception as ex:                        # noqa: BLE001 —— 只是统计，出错不影响训练
            return {"comp_error": type(ex).__name__}

    def _apply_boost_floor(self, rec: dict):
        """前期强度下限（boost_min_strength）：下限从启动时的强度每轮最多抬 boost_min_step，自适应不再往下调过它。"""
        cfg = self.cfg
        if cfg.boost_min_strength <= 0 or self.adapt is None:
            return
        base = self.adapt.strength if self.boost_floor is None else self.boost_floor
        self.boost_floor = min(float(cfg.boost_min_strength), base + float(cfg.boost_min_step))
        self.adapt.lo = max(self.adapt.lo, self.boost_floor)
        if self.adapt.strength < self.boost_floor:
            self.adapt.strength = self.boost_floor
        self.booster.strength = self.adapt.strength
        rec["boost_floor"] = round(self.boost_floor, 3)

    def _make_seeder(self):
        """成型阵容课程（seed_comp_frac）：用和加强局同一个高分阵容库。"""
        from .curricula import CompSeeder
        cfg = self.cfg
        lib = getattr(self, "_pro_lib", None)
        if not lib:
            from paths import RUNS
            paths = [Path(x) for x in str(cfg.pro_boards).split(";") if x.strip()] or [RUNS / "pro_boards.json"]
            lib = merge_pro_boards(p for p in paths if p.exists())
        if not lib:
            print("没有高分阵容库（runs\\pro_boards.json），成型阵容课程不用", flush=True)
            return None
        turns = [int(x) for x in str(cfg.seed_comp_turns).split(",") if x.strip()]
        names = [x.strip() for x in str(cfg.seed_comp_names).split(",") if x.strip()] or None
        sd = CompSeeder(lib, cfg.seed_comp_frac, turns, seed=cfg.seed, formed_from=cfg.seed_comp_formed_from, names=names)
        print(f"成型阵容课程：{cfg.seed_comp_frac:.0%} 的机器人局在第 {cfg.seed_comp_turns} 回合把学习者的场面换成 "
              + ("、".join(names) if names else "S / A 阵容") +
              f"（第 {cfg.seed_comp_formed_from} 回合以前是做到一半的，之后是成型的；库里 {sd.size()} 个："
              + "、".join(f"第 {t} 回合 {len(v)}" for t, v in sorted(sd.pool.items())) + "）", flush=True)
        return sd if sd.size() else None

    def _final_comp(self, s: int) -> int:
        """这一局（槽位 s）最后成了哪一套：rl/target_comps.json 里的序号，没成 = K_MAX（辅助头的“没成”类）。"""
        from .model import K_MAX
        ep = int(self.rollout.ep[s])
        k = int(self.fc_idx[s]) if getattr(self, "fc_ep", None) is not None and self.fc_ep[s] == ep else -1
        return k if 0 <= k < K_MAX else K_MAX
