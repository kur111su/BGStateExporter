"""B3 PPO 训练（TavernGym BatchEnv + 实体 Transformer + 对手库）。

超参按论文：γ=1、λ=0.995、clip 0.2、目标 KL 0.05（超过就提前结束本轮更新）、熵 0.002、价值 0.5、
名次辅助 0.1、梯度裁剪 1.0、每轮 1 遍；新阶段开始时对上一阶段策略加 KL 锚定（系数 0.1，前 100 轮）。
规模按一张 3070（8GB）缩小：每轮约 4–5 万个决策；小批 4096，按显存分成若干份累积梯度（micro_batch）。

对局安排（每局重新抽）：
  - 一部分对局：0 号座位是学习者，其余 7 个是 TavernGym 脚本机器人（controlled=(0,) 的环境）；
  - 其余对局：8 个座位都由我们控制；0 号是学习者，其它座位按对手库权重抽（库里还没有快照时就是学习者
    自己 = 自我对弈）。学习者控制的座位都产生训练数据。
奖励：TavernGym 的名次奖励（第 1 名 1 … 第 8 名 -1），对局结束时给出。
"""
from __future__ import annotations

import json
import os
import random
import time
from collections import defaultdict, deque
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tg.actions import (
    is_noop,
    is_panic,  # noqa: F401 —— 引擎内部出错（第 5413 轮 engine.rs:925 召唤后下标越界，整个训练停了）
)
from tg.obs import C_BOARD

from .calibrate import (
    AdaptiveStrength,
    BotBooster,
    ProBoardBooster,
    merge_pro_boards,
    ramp,
)
from .comp import board_scores, load_comp_refs
from .config import Config
from .guard import GuardMixin
from .imitation import ImitationMixin
from .league import League, load_real_profile
from .model import build_model, n_params
from .rollout import (  # noqa: F401
    Policy,
    Rollout,
    grow_opt_state,
    linear_place_reward,
    parse_place_reward,
    place_score_equiv,
    seed_row,
)
from .shaping import ShapingMixin
from .teacher import TEACHER_FAMILIES, TeacherMixin, teacher_action_mask  # noqa: F401


class Trainer(ShapingMixin, ImitationMixin, TeacherMixin, GuardMixin):
    """训练主流程：采样（collect）、更新（update）、对手库快照、评估、检查点、train 循环。
    奖励塑形在 rl/shaping.py，模仿学习在 rl/imitation.py（作为 mixin 继承进来，Trainer._shape 之类的旧名字照样能用）。"""

    def __init__(self, cfg: Config):
        import taverngym
        from taverngym import BatchEnv
        self.cfg = cfg
        from .device import default_threads, resolve
        dname, rname = resolve(cfg.device, getattr(cfg, "rollout_device", "auto"))
        cfg.device, cfg.rollout_device = dname, rname
        if not cfg.threads:
            cfg.threads = default_threads()
        self.dev = torch.device(dname)
        self.rdev = torch.device(rname)
        if dname == "mps":                              # Mac：显存就是系统内存，而且不做半精度（占用约是 3070 上的两倍）：一份放小一半
            self._micro_recipe = cfg.micro_batch       # 半精度实测能用的话再放回去（内存减半）
            cfg.micro_batch = min(cfg.micro_batch, int(os.environ.get("BGRL_MPS_MICRO", "256")))
            cfg.teacher_chunk = min(cfg.teacher_chunk, cfg.micro_batch)
        print(f"设备：更新 {dname}，采样 {rname}，模拟器线程 {cfg.threads}" +
              (f"，每份 {cfg.micro_batch} 个样本" if dname == "mps" else ""), flush=True)
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.rng = random.Random(cfg.seed)
        self.run = Path(cfg.run_dir)
        (self.run / "snapshots").mkdir(parents=True, exist_ok=True)
        self.mkw = cfg.model_kwargs()
        self.model = build_model(**self.mkw).to(self.dev)
        ka = int(self.model.card_attr.shape[1])
        print(f"卡牌属性表：{ka} 维（data\\external\\model.pt）" if ka else
              "卡牌属性表：没有（data\\external\\model.pt 不在），只用原来的卡牌特征", flush=True)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, eps=1e-5)
        from .device import mps_amp
        self.amp = (cfg.amp and self.dev.type == "cuda") or (self.dev.type == "mps" and mps_amp(cfg.amp_mps))
        if self.dev.type == "mps":
            print(f"MPS 混合精度（bfloat16）：{'开' if self.amp else '关'}", flush=True)
        self.scaler = torch.amp.GradScaler("cuda", enabled=False)      # bfloat16 不需要缩放
        self.rollbacks = 0
        self.since_rollback = 0
        self.good = None                       # 最近一次健康时的模型和优化器（出问题时回退到这里）
        self.model._ver = 0
        self.learner = Policy(self.model, self.rdev, self.amp)
        self.envs = []                                 # (名字, 环境, 槽位起点)
        off = 0
        if cfg.games_bot:
            e = BatchEnv(cfg.games_bot, seed=cfg.seed, controlled=(0,), threads=cfg.threads, config=cfg.rules())
            self.envs.append(("bot", e, off))
            off += e.n_slots
        if cfg.games_league:
            e = BatchEnv(cfg.games_league, seed=cfg.seed + 10**6, controlled=range(8), threads=cfg.threads,
                         config=cfg.rules())
            self.envs.append(("league", e, off))
            off += e.n_slots
        self.n_slots = off
        self.booster = self._make_booster(cfg.boost_frac, cfg.seed) if cfg.boost_frac > 0 and cfg.games_bot else None
        if self.booster is not None and hasattr(self.booster, "late_from") and cfg.boost_late_from > 0:
            self.booster.late_from, self.booster.late_strength = int(cfg.boost_late_from), float(cfg.boost_late_strength)
            print(f"加强局后期全强度：第 {cfg.boost_late_from} 回合起机器人换成高分阵容的概率至少 {cfg.boost_late_strength:.0%}", flush=True)
        self.t_boost = 0.0
        self.adapt = (AdaptiveStrength(start=cfg.boost_start, target=cfg.boost_target_place, step=cfg.boost_step)
                      if self.booster is not None and cfg.boost_adapt else None)
        if self.booster is not None:
            self.booster.strength = self.adapt.strength if self.adapt else ramp(0, 0, cfg.boost_ramp_iters)
        self.boost_places: list[int] = []              # 这一轮打完的加强局名次（日志用）
        self.seeder = self._make_seeder() if cfg.seed_comp_frac > 0 and cfg.games_bot else None
        self.seed_places: list[int] = []
        self.seed_by: dict[str, list] = {}             # 成型阵容课程：阵容 → [(名次, 是不是加强局)]
        self.boost_floor: float | None = None          # 前期强度下限（boost_min_strength），从启动时的强度慢慢爬上去
        self.gifts = None
        if cfg.gift_frac > 0 and cfg.games_bot:
            from .calibrate import CardGifts
            g_turns = [int(x) for x in str(cfg.gift_turns).split(",") if x.strip()]
            self.gifts = CardGifts(cfg.gift_frac, g_turns, p=cfg.gift_p, min_lift=cfg.gift_min_lift, seed=cfg.seed)
            print(f"送牌课程：{cfg.gift_frac:.0%} 的机器人局，第 {cfg.gift_turns} 回合每回合 {cfg.gift_p:.0%} 的机会往手里放一张"
                  f"高手抢着买、模型不买的牌（{len(self.gifts.pool)} 种）", flush=True)
        self.gift_places: list[int] = []
        self.cgifts = None
        if cfg.cgift_frac > 0 and cfg.games_bot:
            from .calibrate import CompGifts
            cg_turns = [int(x) for x in str(cfg.cgift_turns).split(",") if x.strip()]
            self.cgifts = CompGifts(cfg.cgift_frac, cg_turns, p=cfg.cgift_p, seed=cfg.seed)
            print(f"送整套阵容：{cfg.cgift_frac:.0%} 的机器人局开局定一套本局凑得成的目标阵容（{len(self.cgifts.comps)} 套），"
                  f"第 {cfg.cgift_turns} 回合每回合 {cfg.cgift_p:.0%} 的机会送一张这套还没有的核心卡", flush=True)
        self.tribes = None
        if cfg.tribe_frac > 0 and cfg.games_bot:
            from .tribes import TribeLobbies, parse_types
            pool = parse_types(cfg.tribe_pool)
            self.tribes = TribeLobbies(cfg.tribe_frac, pool, cfg.tribe_force, cfg.rules(), seed=cfg.seed)
            print(f"种族课程：{cfg.tribe_frac:.0%} 的机器人局强制本局出现 {cfg.tribe_force} 个这些种族：{', '.join(pool)}"
                  + (f"；老师只在开了 {cfg.teacher_need_tribes} 之一的局面上教" if cfg.teacher_need_tribes else ""), flush=True)
        self.tribe_start = None
        if self.tribes is not None and cfg.tribe_start_p > 0:
            from .tribes import TribeStarter
            self.tribe_start = TribeStarter(cfg.tribe_start_p, cfg.tribe_start_turn, cfg.tribe_start_cards, seed=cfg.seed)
            print(f"起步牌：{cfg.tribe_start_p:.0%} 的种族课程局在第 {cfg.tribe_start_turn} 回合往手里放 {cfg.tribe_start_cards} 张"
                  f"新种族阵容的核心卡（{len(self.tribe_start.comps)} 套可选）", flush=True)
        self.tribe_new_start: list[bool] = []          # 给过起步牌的种族课程局：成了新种族阵容
        self.tribe_new_free: list[bool] = []           # 没给的：成了新种族阵容（真正要看的——自己从零做出来）
        self.tribe_places: list[int] = []
        self.tribe_new_comp: list[bool] = []           # 种族课程局：最后成了老师不会的种族的阵容（这一轮）
        self.tribe_comps: dict[str, int] = {}          # 种族课程局最后成了哪套（上次考试以来）
        self.tribe_by: dict[str, list] = {}            # 种族课程：种族 → [(名次)]（上次考试以来）
        self.cgift_by: dict[str, list] = {}            # 送整套阵容：阵容 → [(名次, 最后做成这套没有)]（上次考试以来）
        self.cgift_iter: list = []                     # 同上，这一轮的
        self.tg_rebase: set[int] = set()               # 换过场面的槽位：下次量目标阵容时从新场面起算
        self.eg_rebase: set[int] = set()               # 同上，引擎联动
        self.obs = {name: env.observe() for name, env, _ in self.envs}
        self.episode = {name: np.asarray([env.episode_index(i) for i in range(env.n_games)])
                        for name, env, _ in self.envs}
        self.role = np.zeros(off, dtype=np.int64)      # 0 = 学习者，k = self.role_names[k]
        self.role_names = ["learner"]
        self.snap_cache: dict[str, Policy] = {}
        self.league = League(load_real_profile())
        self.league.add("bot")
        self.rollout = Rollout(off)
        self.active_snaps = []
        self.iter = 0
        self.anchor = None
        self._init_imitation()                         # 示范数据和模仿强度（rl/imitation.py）
        self._init_shaping()                           # 升本曲线和每个座位的塑形状态（rl/shaping.py）
        self.teacher, self.teacher_t0 = self._load_teacher(), None
        self.repay_t0 = None                           # 局末扣回势能从哪一轮开始爬（存在 latest.pt 里）
        self.place_tab = parse_place_reward(getattr(cfg, "place_reward", ""))
        if self.place_tab is not None:
            print("名次奖励表：" + "、".join(f"{p} 名 {self.place_tab[p]:+.2f}" for p in range(1, 9)), flush=True)
        self.guard_base = {k: float(getattr(cfg, k)) for k in self._guard_keys()}
        self.guard_st = self._guard_new()
        self.mix = defaultdict(int)
        self.anchor_until = 0
        self.stats = deque(maxlen=2000)               # 学习者 0 号座位的名次（对机器人 / 对手库）
        self.log_f = open(self.run / "log.jsonl", "a", encoding="utf-8")
        (self.run / "config.json").write_text(json.dumps(asdict(cfg), indent=1), encoding="utf-8")
        for name, env, o in self.envs:
            for g in range(env.n_games):
                self._assign(name, env, o, g)
        print(f"模型 {cfg.size} {self.mkw}：参数 {n_params(self.model) / 1e6:.2f}M，设备 {self.dev}，"
              f"taverngym {taverngym.version()}，{self.n_slots} 个座位", flush=True)

    # ---------------- 对手 ----------------

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
        if name == "bot" and self.cgifts is not None:
            self.cgifts.plan.clear()
        if name == "bot" and self.tribes is not None:
            self.tribes.clear()
            if self.tribe_start is not None:
                self.tribe_start.clear()
            if getattr(self.booster, "full", None) is not None:
                self.booster.full.clear()
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

    # ---------------- 采样 ----------------

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
        for _ in range(cfg.steps_per_iter):
            # 先把所有环境里同一个模型要算的局面合在一起，一次前向（比每个环境各算一次快）
            ta = time.time()
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
                res = self._policy(rid)(f, c, m, want_obs=(rid == 0))
                a = res[0]
                o = 0
                for ei, idx in parts:
                    name, env, off = self.envs[ei]
                    k = len(idx)
                    actions[name][idx] = a[o:o + k]
                    if rid == 0:
                        ak = a[o:o + k]
                        noop = is_noop(ak) | (ak == 7)                # 冻结、调站位、取消（打出 → 取消可以无限循环）
                        r0 = self._shape(off + idx, f[o:o + k], ak, -cfg.noop_penalty * noop.astype(np.float64), c[o:o + k])
                        r0 = self._combat_shape(env, off, idx, f[o:o + k], ak, r0)
                        r0 = self._grow_shape(env, off, idx, f[o:o + k], ak, r0)
                        self.rollout.add(off + idx, res[3][o:o + k], c[o:o + k], m[o:o + k], ak,
                                         res[1][o:o + k], res[2][o:o + k], r0=r0)
                        self._after_add(off + idx, ak, np.rint(f[o:o + k, 0] * 20).astype(int))
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
                        r_end = r_env + ((self._econ_close(off + int(i)) + self._shape_close(off + int(i))) if dones[i] else 0.0)
                        fcomp = self._final_comp(off + int(i)) if dones[i] else None
                        self.rollout.reward(off + int(i), r_end, bool(dones[i]), info["placement"][i], fcomp)
                        if dones[i] and env.controlled[int(i) % len(env.controlled)] == 0:
                            seeded = name == "bot" and self.seeder is not None and self.seeder.finish(int(i))
                            gifted = name == "bot" and self.gifts is not None and self.gifts.finish(int(i))
                            cg = self.cgifts.finish(int(i)) if name == "bot" and self.cgifts is not None else ""
                            if cg and not seeded:          # 送整套阵容的局：单独记名次、最后做成这套没有
                                row = (int(info["placement"][i]), self._comp_name(fcomp) == cg)
                                self.cgift_by.setdefault(cg, []).append(row)
                                self.cgift_iter.append(row)
                                if self.booster is not None:
                                    self.booster.finish(int(i))
                                seeded = True              # 不算进 place_bot / 加强强度
                            tribal = self.tribes.finish(int(i)) if name == "bot" and self.tribes is not None else None
                            if tribal is not None and not seeded and not cg:
                                pl_t = int(info["placement"][i])
                                cn = self._comp_name(fcomp) if fcomp is not None else ""
                                ct = self._comp_tribes().get(cn, frozenset())
                                new = bool(ct) and not (ct & self._strong_tribes())
                                self.tribe_places.append(pl_t)
                                self.tribe_new_comp.append(new)
                                st = self.tribe_start.finish(int(i)) if self.tribe_start is not None else None
                                (self.tribe_new_free if st is None else self.tribe_new_start).append(new)
                                self.tribe_comps[cn or "没成型"] = self.tribe_comps.get(cn or "没成型", 0) + 1
                                for t in tribal:                # (名次, 成了用到这个种族的阵容)
                                    self.tribe_by.setdefault(t, []).append((pl_t, t in ct))
                                if self.booster is not None:
                                    self.booster.finish(int(i))
                                if gifted and self.gifts is not None:
                                    self.gift_places.append(pl_t)
                                seeded = True              # 不算进 place_bot / 加强强度
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
                if name == "bot" and self.booster is not None:   # 机器人加强到真人强度（见 rl/calibrate.py）
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
                if name == "bot" and self.cgifts is not None:     # 送整套阵容：往手里放一张核心卡（场面不动，不切 GAE）
                    try:
                        if self.cgifts.apply(env, self.obs[name]["features"]):
                            self.obs[name] = env.observe()
                    except BaseException as ex:            # noqa: BLE001
                        if not is_panic(ex):
                            raise
                        self._rebuild_env(ei, ex)
                        continue
                if name == "bot" and self.tribe_start is not None:  # 种族课程起步牌：往手里放新种族的核心卡（不切 GAE）
                    try:
                        if self.tribe_start.apply(env, self.obs[name]["features"]):
                            self.obs[name] = env.observe()
                    except BaseException as ex:            # noqa: BLE001
                        if not is_panic(ex):
                            raise
                        self._rebuild_env(ei, ex)
                        continue
                if name == "bot" and self.gifts is not None:      # 送牌课程：往手里放一张（场面不动，不切 GAE）
                    try:
                        if self.gifts.apply(env, self.obs[name]["features"], skip=self.cgifts.active() if self.cgifts else None):
                            self.obs[name] = env.observe()
                    except BaseException as ex:            # noqa: BLE001
                        if not is_panic(ex):
                            raise
                        self._rebuild_env(ei, ex)
                        continue
                skip = None
                if name == "bot":
                    skip = (self.gifts.active() if self.gifts is not None else set()) | \
                           (self.cgifts.active() if self.cgifts is not None else set())
                    if self.tribe_start is not None:
                        skip |= self.tribe_start.active()      # 起步牌局自己往下做，不换成型场面
                for sd in ((self.seeder,) if name == "bot" else ()):
                    if sd is None:                         # 成型阵容课程（换整个场面）
                        continue
                    try:
                        if sd.seed(env, self.obs[name]["features"], skip=skip):
                            self.obs[name] = env.observe()
                            for g in sd.seeded_slots:
                                self.rollout.cut(off + g)        # GAE 在换场面处切断（见 Rollout.cut）
                                self.cs_ep[off + g] = -1         # 阵容强度、目标阵容从新场面重新起算
                                self.gr_turn[off + g] = -1       # 养场面：这回合的“开始”在新场面上重新量
                                self.tg_rebase.add(off + g)
                                self.eg_rebase.add(off + g)
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
                            if name == "bot" and self.tribes is not None:
                                full = getattr(self.booster, "full", None)
                                if full is not None:
                                    full.discard(int(g))
                                if self.tribe_start is not None:
                                    self.tribe_start.plan.pop(int(g), None)
                                if self.tribes.apply(env, int(g)):
                                    swapped = True         # 种族课程：换成指定种族的一局（episode 编号不变）
                                    if self.tribe_start is not None:
                                        self.tribe_start.begin(int(g), self.tribes.active.get(int(g), ()))
                                    if cfg.tribe_full and full is not None:
                                        full.add(int(g))   # 对手全强度（和考试一样）
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

    # ---------------- 更新 ----------------


    def _probe_mps_amp(self, b) -> None:
        """Mac：实测 bfloat16 混合精度在这台机器上对不对、快不快（第一次更新前跑一次，几秒）。
        同一份样本算两遍：结果要和 float32 对得上（动作一样的 ≥ 98%、没有 NaN），而且快 15% 以上才启用。"""
        cfg = self.cfg
        A_N = self.model.L.actions
        j = np.arange(min(len(b["a"]), cfg.micro_batch))
        f = torch.as_tensor(b["f"][j], dtype=torch.float32).to(self.dev)
        c = torch.as_tensor(b["c"][j], dtype=torch.long).to(self.dev)
        m = torch.as_tensor(np.unpackbits(b["m"][j], axis=1)[:, :A_N].astype(bool)).to(self.dev)

        def run(amp: bool, reps: int):
            t0 = time.time()
            for _ in range(reps):
                self.opt.zero_grad(set_to_none=True)
                with torch.autocast("mps", dtype=torch.bfloat16, enabled=amp):
                    lg, v, _ = self.model(f, c, m, compressed=True)
                lp = F.log_softmax(lg.float(), -1)
                (-(lp.exp() * lp.masked_fill(~m, 0)).sum(-1).mean() + v.float().pow(2).mean()).backward()
                torch.mps.synchronize()
            self.opt.zero_grad(set_to_none=True)
            return (time.time() - t0) / reps, lg.detach().float()

        self.model.eval()
        try:
            run(False, 1)
            t32, l32 = run(False, 3)
            run(True, 1)
            t16, l16 = run(True, 3)
            same = float((l32.argmax(-1) == l16.argmax(-1)).float().mean())
            ok = bool(torch.isfinite(l16[m]).all()) and same >= 0.98
            use = ok and t16 < 0.85 * t32
            print(f"  混合精度实测：float32 {t32 * 1000:.0f} ms / 份，bfloat16 {t16 * 1000:.0f} ms / 份，动作一致 {same:.1%} → "
                  f"{'启用' if use else '不用'}", flush=True)
        except Exception as ex:                          # noqa: BLE001 —— 这台机器 / 这个 torch 不支持：不用
            use = False
            print(f"  混合精度实测：不支持（{type(ex).__name__}: {str(ex)[:80]}）→ 不用", flush=True)
        self.opt.zero_grad(set_to_none=True)
        torch.mps.empty_cache()
        if use:
            self.amp = True
            cfg.micro_batch = max(cfg.micro_batch, min(512, getattr(self, "_micro_recipe", cfg.micro_batch)))
            cfg.teacher_chunk = max(cfg.teacher_chunk, cfg.micro_batch)
            print(f"  半精度内存减半：每份放回 {cfg.micro_batch} 个样本", flush=True)
            self.learner.amp = True
            for pol in self.snap_cache.values():
                pol.amp = True

    def update(self, b):
        cfg = self.cfg
        if self.dev.type == "mps" and not self.amp and str(cfg.amp_mps).lower() == "auto" and not getattr(self, "_amp_probed", False):
            self._amp_probed = True
            self._probe_mps_amp(b)
        self.model.train()
        n = len(b["a"])
        adv = (b["adv"] - b["adv"].mean()) / (b["adv"].std() + 1e-8)
        idx = np.random.permutation(n)
        stats = defaultdict(list)
        use_anchor = self.anchor is not None and self.iter < self.anchor_until
        tcoef = self._teacher_coef()
        if tcoef > 0:
            stats["teacher_coef"].append(tcoef)
        stop = False
        A_N = self.model.L.actions
        hard = getattr(cfg, "teacher_mode", "hard") != "kl"
        t_ta = t_tp = None
        if tcoef > 0 and hard:                           # 老师的答案先算好（每轮一次，不是每个 epoch 每一份都算）
            t0_t = time.time()
            t_ta, t_tp = self._teacher_targets(b, A_N)
            stats["t_teacher"].append(time.time() - t0_t)
            stats["teacher_rows"].append(float((t_ta >= 0).mean()))
        if self.dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for _ in range(cfg.epochs):
            for s in range(0, n, cfg.minibatch):
                mb = idx[s:s + cfg.minibatch]
                self.opt.zero_grad(set_to_none=True)
                kls = []
                for u in range(0, len(mb), cfg.micro_batch):     # 分份算梯度再累加（省显存）
                    j = mb[u:u + cfg.micro_batch]
                    w = len(j) / len(mb)
                    t = lambda x, dt: torch.as_tensor(x, dtype=dt).to(self.dev, non_blocking=True)  # noqa: E731
                    f = t(b["f"][j], torch.float32)
                    c = t(b["c"][j], torch.long)
                    m = t(np.unpackbits(b["m"][j], axis=1)[:, :A_N].astype(bool), torch.bool)
                    a = t(b["a"][j], torch.long)
                    old = t(b["logp"][j], torch.float32)
                    A = t(adv[j], torch.float32)
                    R = t(b["ret"][j], torch.float32)
                    P = t(b["place"][j], torch.long)
                    with torch.autocast(self.dev.type, dtype=torch.bfloat16, enabled=self.amp):
                        logits, v, pl = self.model(f, c, m, compressed=True)
                    stats["bad_rows"].append(float(getattr(self.model, "last_bad", 0)))
                    logp_all = F.log_softmax(logits.float(), -1)
                    logp = logp_all.gather(1, a.unsqueeze(1)).squeeze(1)
                    ratio = torch.exp(logp - old)
                    pg = -torch.min(ratio * A, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * A).mean()
                    ent = -(logp_all.exp() * logp_all.masked_fill(~m, 0)).sum(-1).mean()
                    vf = F.mse_loss(v.float(), R)
                    has_p = P > 0
                    aux = F.cross_entropy(pl.float()[has_p], P[has_p] - 1) if has_p.any() else \
                        torch.zeros((), device=self.dev)
                    loss = pg + cfg.vf_coef * vf - cfg.ent_coef * ent + cfg.aux_coef * aux
                    if getattr(cfg, "aux_comp_coef", 0) > 0 and "comp" in b and self.model.last_comp_logits is not None:
                        K = t(b["comp"][j], torch.long)
                        has_k = K >= 0
                        if has_k.any():                # 体系感知辅助头：预测这局最后成了哪一套
                            aux_c = F.cross_entropy(self.model.last_comp_logits[has_k], K[has_k])
                            loss = loss + cfg.aux_comp_coef * aux_c
                            stats["aux_comp"].append(aux_c.item())
                            stats["comp_acc"].append(float((self.model.last_comp_logits[has_k].argmax(-1) == K[has_k]).float().mean()))
                    if use_anchor:
                        with torch.no_grad(), torch.autocast(self.dev.type, dtype=torch.bfloat16, enabled=self.amp):
                            ref = F.log_softmax(self.anchor(f, c, m, compressed=True)[0].float(), -1)
                        kl_anchor = (ref.exp() * (ref - logp_all)).masked_fill(~m, 0).sum(-1).mean()
                        loss = loss + cfg.anchor_coef * kl_anchor
                        stats["kl_anchor"].append(kl_anchor.item())
                    if tcoef > 0:
                        tw = self._teacher_rows(b["f"][j], b["c"][j])
                        if tw is not None:
                            stats["teacher_core"].append(float((tw > 1).mean()))
                        tw_t = t(tw, torch.float32) if tw is not None else torch.ones(len(j), device=self.dev)
                        if not hard:                                     # kl：整个分布向老师靠（要现算老师的分布）
                            with torch.no_grad(), torch.autocast(self.dev.type, dtype=torch.bfloat16, enabled=self.amp):
                                ch = int(getattr(cfg, "teacher_chunk", 512)) or f.shape[0]
                                tl = torch.cat([F.log_softmax(self.teacher(f[q:q + ch], c[q:q + ch], m[q:q + ch],
                                                                           compressed=True)[0].float(), -1)
                                                for q in range(0, f.shape[0], ch)])
                            kl_rows = (tl.exp() * (tl - logp_all)).masked_fill(~m, 0).sum(-1)
                            loss = loss + tcoef * (kl_rows * tw_t).mean()
                            stats["kl_teacher"].append(kl_rows.mean().item())
                        else:                                            # 只在老师有把握的局面，学老师最想做的那个动作
                            ta = t(t_ta[j], torch.long)
                            sure = ((t(t_tp[j], torch.float32) >= cfg.teacher_conf) & (ta >= 0)).float()
                            ta = ta.clamp(min=0)
                            nll_t = -logp_all.gather(1, ta.unsqueeze(1)).squeeze(1).clamp(max=20)
                            boost = 1.0 / min(1.0, max(0.05, float(getattr(cfg, "teacher_frac", 1.0))))   # 只算了一部分局面：按比例放大
                            loss = loss + tcoef * boost * (nll_t * sure * tw_t).mean()
                            stats["teacher_sure"].append(float(sure.mean()))
                            if sure.any():
                                agree = (logp_all.argmax(-1) == ta).float()
                                stats["teacher_agree"].append(float((agree * sure).sum() / sure.sum()))
                    self.scaler.scale(loss * w).backward()
                    with torch.no_grad():
                        kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                    kls.append(kl * w)
                    for k_, v_ in (("loss", loss.item()), ("pg", pg.item()), ("vf", vf.item()), ("ent", ent.item()),
                                   ("aux", aux.item()), ("kl", kl),
                                   ("clipfrac", ((ratio - 1).abs() > cfg.clip).float().mean().item())):
                        stats[k_].append(v_)
                if self.pro is not None and cfg.pro_coef > 0 and len(self.pro["train_idx"]):
                    pb = self.bc_loss(self.pro, self.pro_w, cfg.pro_batch)
                    self.scaler.scale(cfg.pro_coef * self.pro_scale * pb).backward()
                    stats["pro_bc"].append(pb.item())
                gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                if not torch.isfinite(gn):                # 梯度坏了：这一步不更新，记下来
                    stats["skipped"].append(1.0)
                    self.opt.zero_grad(set_to_none=True)
                    continue
                stats["skipped"].append(0.0)
                stats["grad_norm"].append(float(gn))
                self.opt.step()
                self.model._ver = getattr(self.model, "_ver", 0) + 1          # 采样用的副本下次出动作前同步
                if self.dev.type == "mps":                 # MPS 的缓存只涨不退：每个小批还一次
                    torch.mps.empty_cache()
                if sum(kls) > cfg.target_kl:
                    stop = True
                    break
            if stop:
                break
        out = {k: float(np.mean(v)) for k, v in stats.items()}
        out["bad_rows"] = int(sum(stats.get("bad_rows", [])))
        out["skipped"] = int(sum(stats.get("skipped", [])))
        out["early_stop"] = stop
        if self.dev.type == "cuda":
            out["gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        elif self.dev.type == "mps":                   # MPS 会把用过的显存一直留着（和系统内存是同一块）：每轮还回去
            out["gpu_mem_gb"] = round(torch.mps.driver_allocated_memory() / 2**30, 2)
            torch.mps.empty_cache()
        return out

    # ---------------- 对手库快照、评估 ----------------

    def snapshot(self):
        """存快照进对手库，并测它的画像（所有对局、座位一起成批出动作）。"""
        from .league import sim_profile_batched
        path = self.run / "snapshots" / f"it{self.iter:05d}.pt"
        torch.save(self.model.state_dict(), path)
        self.model.eval()

        def act(f, c, m):
            return self.learner(f, c, m)[0]
        prof = sim_profile_batched(act, games=self.cfg.profile_games, seed=10**7 + self.iter, config=self.cfg.rules())
        mem = self.league.add(path.stem, str(path), prof)
        (self.run / "league.json").write_text(json.dumps(self.league.to_json(), ensure_ascii=False, indent=1),
                                              encoding="utf-8")
        return mem

    def evaluate(self, games=None, seed=999, boosted=False):
        """考试：学习者（0 号座位）对 7 个脚本机器人，平均名次和第一名率。
        boosted=True：机器人按真人画像加强（全部对局、全强度），这是第二阶段主要看的成绩。"""
        from taverngym import BatchEnv
        games = games or self.cfg.eval_games
        env = BatchEnv(games, seed=seed, controlled=(0,), threads=self.cfg.threads, config=self.cfg.rules())
        booster = self._make_booster(1.0, seed) if boosted else None
        if booster is not None:
            booster.strength = 1.0
        obs = env.observe()
        places = {}
        last_board = {}                              # 每局最后一次招募时的场面（卡牌编号）：最终阵容
        self.model.eval()
        for _ in range(30000):                       # 保险：不管怎样都会结束
            if len(places) >= games:
                break
            ready = np.flatnonzero(obs["ready"])
            actions = np.zeros(env.n_slots, dtype=np.int64)
            if len(ready):
                a, _, _ = self.learner(obs["features"][ready], obs["cards"][ready], obs["action_mask"][ready])
                actions[ready] = a
                for i in ready:
                    if int(i) not in places:
                        last_board[int(i)] = obs["cards"][i][C_BOARD:C_BOARD + 7].copy()
            try:
                _, dones, info = env.step(actions)
            except BaseException as ex:                # noqa: BLE001
                if not is_panic(ex):
                    raise
                print(f"考试中 TavernGym 引擎出错（{str(ex)[:120]}），按已经打完的 {len(places)} 局算", flush=True)
                break
            for i in np.flatnonzero(dones):
                if i not in places:                  # 每个对局只算第一局
                    places[int(i)] = int(info["placement"][i])
            obs = env.observe()
            if booster is not None and booster.boost(env, obs["features"]):
                obs = env.observe()
        ps = [p for p in places.values() if p > 0]
        self.last_eval_places = ps
        self.last_comp = self._comp_scores([last_board[i] for i in places if i in last_board])
        return (round(float(np.mean(ps)), 3), round(float(np.mean([p == 1 for p in ps])), 3)) if ps else (None, None)

    def _comp_scores(self, boards) -> dict:
        """最终阵容的配合程度（场上 ≥ 5 个随从的局；见 rl/comp.board_scores）：
        comp_coh 主种族纯度（万三 0.70）、comp_syn 高手常见搭配对数（2.07）、comp_key 关键卡张数（4.34）、comp_golden 金色随从数（1.85）。"""
        try:
            from tg.catalog import base_id, card_id, catalog, is_golden_id
            if not hasattr(self, "_comp_ref"):
                self._comp_ref = (*load_comp_refs(), catalog())
            pairs, key, det = self._comp_ref
            rows = []
            for b in boards:
                ids = [card_id(int(x)) for x in b if x > 0]
                if len(ids) < 5:
                    continue
                sc = board_scores([base_id(x) for x in ids], det, pairs, key)
                sc["golden"] = sum(is_golden_id(x) for x in ids)
                rows.append(sc)
            if not rows:
                return {}
            return {f"comp_{k}": round(float(np.mean([r[k] for r in rows])), 3) for k in ("coh", "syn", "key", "golden")}
        except Exception as ex:                        # noqa: BLE001 —— 只是统计，出错不影响训练
            return {"comp_error": type(ex).__name__}

    # ---------------- 主循环 ----------------

    # ---------------- 健康检查 / 回退 ----------------


    # ---------------- 考试守护（guard_*） ----------------


    def _redraw_board(self):
        """后台线程重画训练看板；上一次还没画完就跳过。出错只打印一次，不影响训练。"""
        import threading
        th = getattr(self, "_board_th", None)
        if th is not None and th.is_alive():
            return

        def work():
            try:
                from .board import render
                render(self.run)
            except Exception as ex:                    # noqa: BLE001
                if not getattr(self, "_board_err", False):
                    self._board_err = True
                    print(f"训练看板没画成（不影响训练）：{type(ex).__name__}: {str(ex)[:160]}", flush=True)
        self._board_th = threading.Thread(target=work, daemon=True)
        self._board_th.start()


    def init_from(self, path):
        """用另一个检查点 / 快照的参数开始（不带优化器和对手库）。"""
        ck = torch.load(path, map_location=self.dev, weights_only=False)
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        if any(not torch.isfinite(v).all() for v in sd.values() if torch.is_tensor(v) and v.is_floating_point()):
            raise ValueError(f"{path} 里的参数有 NaN / inf，不能用")
        self.model.load_state_dict(sd)
        self.model._ver = getattr(self.model, "_ver", 0) + 1          # 采样用的副本下次出动作前同步
        print(f"从 {path} 的参数开始", flush=True)

    def save(self):
        torch.save({"model": self.model.state_dict(), "opt": self.opt.state_dict(), "iter": self.iter,
                    "league": self.league.to_json(), "model_kwargs": self.mkw,
                    "boost": {**self.adapt.to_json(), "kind": self.cfg.boost_kind} if self.adapt is not None else None, "tempo_t0": self.tempo_t0,
                    "pro_scale": self.pro_scale,
                    "teacher": {"t0": self.teacher_t0, "id": getattr(self, "teacher_id", None)},
                    "repay_t0": self.repay_t0, "guard": self.guard_st, "ent_coef": self.cfg.ent_coef,
                    "warm_t0": getattr(self, "warm_t0", None)}, self.run / "latest.pt")

    def load(self, path):
        ck = torch.load(path, map_location=self.dev, weights_only=False)
        if ck.get("model_kwargs") and ck["model_kwargs"] != self.mkw:
            raise ValueError(f"检查点的模型大小 {ck['model_kwargs']} 和这次的 {self.mkw} 不一样")
        self.model.load_state_dict(ck["model"])
        self.model._ver = getattr(self.model, "_ver", 0) + 1          # 采样用的副本下次出动作前同步
        self.opt.load_state_dict(grow_opt_state(self.opt, ck["opt"]))
        self.iter = ck["iter"]
        self.league = League.from_json(ck["league"], load_real_profile())
        self.league.relocate(self.run / "snapshots")
        gone = self.league.drop_missing()
        if gone:
            print(f"对手库：{gone} 个快照的文件已经不在（清理过 snapshots），从对手库里去掉，还剩 "
                  f"{sum(1 for m in self.league.members if m.name != 'bot')} 个", flush=True)
        self.tempo_t0 = ck.get("tempo_t0", self.tempo_t0)
        self.pro_scale = float(ck.get("pro_scale", self.pro_scale))
        tk = ck.get("teacher") or {}
        if getattr(self, "teacher", None) is not None and tk.get("t0") is not None and tk.get("id") == getattr(self, "teacher_id", None):
            self.teacher_t0 = int(tk["t0"])                 # 同一个老师：接着之前的爬坡，不是每次启动从 0 再爬（第 14240 轮那次重启又从 0 开始）
        if ck.get("repay_t0") is not None:
            self.repay_t0 = int(ck["repay_t0"])
        if isinstance(ck.get("guard"), dict):
            self.guard_st = {**self._guard_new(), **ck["guard"]}
            if "metric" not in ck["guard"]:
                self.guard_st["metric"] = "place"           # 第 18611 轮以前存的：按平均名次
            self._guard_check_metric()
            self._guard_apply()
        if getattr(self.cfg, "ent_target", 0) and ck.get("ent_coef") is not None:   # 调过的熵系数接着用
            self.cfg.ent_coef = float(min(self.cfg.ent_coef_max, max(self.cfg.ent_coef_min, ck["ent_coef"])))
        if ck.get("warm_t0") is not None:
            self.warm_t0 = int(ck["warm_t0"])
        if self.cfg.pro_gap_cap > 0:
            self.pro_scale = max(self.pro_scale, self.cfg.pro_scale_min)
        if self.adapt is not None:
            b = ck.get("boost") or {}
            if b.get("kind", "scale") == self.cfg.boost_kind:  # 换了加强方式：强度从头调（放大攻防的 0.88 和换阵容的 0.88 不是一回事）
                self.adapt.load_json(b)
            else:
                print(f"加强方式改为 {self.cfg.boost_kind}：强度从 {self.adapt.strength} 开始重新调", flush=True)
            self.booster.strength = self.adapt.strength

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
        from .calibrate import CompSeeder
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
        sd = CompSeeder(lib, cfg.seed_comp_frac, turns, seed=cfg.seed, formed_from=cfg.seed_comp_formed_from)
        print(f"成型阵容课程：{cfg.seed_comp_frac:.0%} 的机器人局在第 {cfg.seed_comp_turns} 回合把学习者的场面换成 S / A 阵容"
              f"（第 {cfg.seed_comp_formed_from} 回合以前是做到一半的，之后是成型的；库里 {sd.size()} 个："
              + "、".join(f"第 {t} 回合 {len(v)}" for t, v in sorted(sd.pool.items())) + "）", flush=True)
        return sd if sd.size() else None


    @torch.inference_mode()
    def _comp_name(self, k) -> str:
        """_final_comp 的序号 → 阵容名字（没成 → 空字符串）。"""
        if getattr(self, "_comp_names", None) is None:
            from .comp import load_target_comps
            self._comp_names = [c["name"] for c in load_target_comps()]
        return self._comp_names[int(k)] if k is not None and 0 <= int(k) < len(self._comp_names) else ""

    def _comp_tribes(self) -> dict:
        """阵容名字 → 它的核心卡用到的种族（rl/target_comps.json）。"""
        if getattr(self, "_comp_tribe_map", None) is None:
            from .comp import load_target_comps
            self._comp_tribe_map = {c["name"]: frozenset(c.get("tribes") or ()) for c in load_target_comps()}
        return self._comp_tribe_map

    @staticmethod
    def _strong_tribes() -> frozenset:
        from .tribes import STRONG
        return frozenset(STRONG)

    def _final_comp(self, s: int) -> int:
        """这一局（槽位 s）最后成了哪一套：rl/target_comps.json 里的序号，没成 = K_MAX（辅助头的“没成”类）。"""
        from .model import K_MAX
        ep = int(self.rollout.ep[s])
        k = int(self.fc_idx[s]) if getattr(self, "fc_ep", None) is not None and self.fc_ep[s] == ep else -1
        return k if 0 <= k < K_MAX else K_MAX


    def start_stage(self):
        """新阶段：以当前策略为锚，前 anchor_iters 轮加 KL 约束。"""
        import copy
        self.anchor = copy.deepcopy(self.model).eval()
        for p in self.anchor.parameters():
            p.requires_grad_(False)
        self.anchor_until = self.iter + self.cfg.anchor_iters

    def train(self, iters=None, on_log=None):
        cfg = self.cfg
        end = self.iter + (iters or cfg.iters)
        if self.good is None:
            self.mark_good()
        (Path(cfg.run_dir) / "STOP").unlink(missing_ok=True)     # 上次留下的停止请求不算
        while self.iter < end:
            self.learner.bad_rows = 0
            batch, n_dec, t_col, t_env, t_net = self.collect()
            if self.dev.type == "mps" or self.rdev.type == "mps":      # Mac：每一段完了报一下 GPU 占的内存（卡在哪一段看得出来）
                self._mps_n = getattr(self, "_mps_n", 0) + 1
                if self._mps_n <= 3:
                    print(f"  采样完（{t_col:.0f}s，{n_dec} 个决策），GPU 占用内存 {torch.mps.driver_allocated_memory() / 2**30:.1f} GB；开始更新……",
                          flush=True)
                torch.mps.empty_cache()
            self._apply_warmup()
            t1 = time.time()
            try:
                upd = self.update(batch) if batch is not None else {}
            except RuntimeError as ex:                  # 显存 / 内存不够：清缓存、每份减半，这一轮重算一次（不行再报错退出）
                if "out of memory" not in str(ex) or cfg.micro_batch <= 32:
                    raise
                import gc
                self.opt.zero_grad(set_to_none=True)
                gc.collect()
                if self.dev.type == "mps":
                    torch.mps.empty_cache()
                elif self.dev.type == "cuda":
                    torch.cuda.empty_cache()
                cfg.micro_batch = max(32, cfg.micro_batch // 2)
                cfg.teacher_chunk = min(cfg.teacher_chunk, cfg.micro_batch)
                print(f"  内存不够：清缓存，每份改成 {cfg.micro_batch} 个样本，这一轮的更新重算", flush=True)
                upd = self.update(batch)
            t_upd = time.time() - t1
            self.iter += 1
            problem = self.health(upd) or self._drift(upd.get("ent"), self.n_noop / max(1, n_dec))
            if not problem:
                self.since_rollback += 1
                self._adapt_ent(upd.get("ent"))
                g0 = self.opt.param_groups[0]["lr"]
                if getattr(self, "warm_t0", None) is None and g0 < cfg.lr and self.since_rollback >= cfg.lr_recover_iters:
                    for g in self.opt.param_groups:
                        g["lr"] = min(cfg.lr, g["lr"] * 2)
                    self.since_rollback = 0
            if problem:
                self.since_rollback = 0
                self.rollback(problem)
                rec = {"iter": self.iter, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "rollback": problem,
                       "lr": self.opt.param_groups[0]["lr"], "rollbacks": self.rollbacks}
                self.log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                self.log_f.flush()
                if on_log:
                    on_log(rec)
                continue
            rec = {"iter": self.iter, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "decisions": n_dec,
                   "games_done": self.finished, "dps": round(n_dec / max(t_col + t_upd, 1e-6)),
                   "t_collect": round(t_col, 1), "t_env": round(t_env, 1), "t_net": round(t_net, 1),
                   "t_update": round(t_upd, 1), "device": f"{self.dev.type}+{self.rdev.type}", "threads": cfg.threads,
                   "lr": self.opt.param_groups[0]["lr"], "ent_coef": round(cfg.ent_coef, 5),
                   "noop_frac": round(self.n_noop / max(1, n_dec), 3),
                   **{k: round(self.mix.get(k, 0) / max(1, n_dec), 3) for k in ("noop_move", "noop_freeze", "noop_cancel")}, **upd}
            mx, ends = self.mix, max(1, self.mix.get("end", 0))           # 每回合平均（学习者）
            rec.update(up_turn=round(mx["upgrade"] / ends, 3), ref_turn=round(mx["refresh"] / ends, 2),
                       buy_turn=round(mx["buy"] / ends, 2), sell_turn=round(mx["sell"] / ends, 3),
                       ref_waste=round(mx["refresh_waste"] / max(1, mx["refresh"]), 3),
                       tempo_bonus=round(self.tempo_now(), 4), tempo_paid=mx["tempo_paid"],
                       tier_gap=round(mx["gap_sum"] / max(1, mx["gap_n"]), 3),
                       up_afford=round(mx["aff_up"] / max(1, mx["aff_n"]), 3), up_t2=round(mx["t2_up"] / max(1, mx["t2_n"]), 3),
                       starve_turn=round(mx["starve"] / ends, 3), end_rich=round(mx["end_rich"] / ends, 3),
                       unspent=round(mx["unspent"] / ends, 3),
                       comp_q=round(mx["comp_q_sum"] / max(1, mx["comp_q_n"]), 3) if mx.get("comp_q_n") else None)
            if mx.get("grow_n"):                       # 养场面：第 grow_from_turn 回合起每回合结束 - 开始的场面强度
                rec["grow"] = round(mx["grow_sum"] / mx["grow_n"], 4)
                rec["grow_n"] = int(mx["grow_n"])
                rec["t_grow"] = round(mx.get("t_grow", 0.0), 2)
            if mx.get("cs_n"):                         # 阵容强度：对同期高分阵容的平均胜率（全部 / 第 9 回合起）
                rec.update(cs_wr=round(mx["cs_sum"] / mx["cs_n"], 3), cs_n=int(mx["cs_n"]), t_combat=round(mx["t_combat"], 1),
                           cs_late=round(mx["cs_late_sum"] / mx["cs_late_n"], 3) if mx.get("cs_late_n") else None)
            if mx.get("tgt_n"):                        # 结束回合时离目标阵容多近（S 级凑齐 = 1）
                rec["tgt_q"] = round(mx["tgt_sum"] / mx["tgt_n"], 3)
            if mx.get("eng_n"):                        # 结束回合时引擎联动的程度（一个引擎配齐 3 张燃料 = 1）
                rec["eng_q"] = round(mx["eng_sum"] / mx["eng_n"], 3)
            if mx.get("sells"):
                rec["keep_sold"] = round(mx.get("keep_sold", 0) / mx["sells"], 4)   # 卖牌里卖掉“该留的牌”的比例（高手 5.2%）
            if mx.get("econ_n"):                       # 结束回合时平均存下的钱（下回合额外金币 + 2 × 金币上限加成）
                rec["econ"] = round(mx["econ_sum"] / mx["econ_n"], 3)
            if mx.get("fin_n"):                        # 局末三项势能还挂着多少、这一轮扣回的比例
                rec["shape_fin"] = round(mx["fin_sum"] / mx["fin_n"], 3)
                rec["shape_repay"] = round(self._repay_now(), 3)
            if mx.get("env_rebuild"):
                rec["env_rebuild"] = mx["env_rebuild"]
            if mx.get("drift"):                        # 这一轮看起来在变乱（连续 drift_iters 轮就回退）
                rec["drift"] = mx["drift"]
            self.mix = defaultdict(int)
            for name in ("bot", "league"):
                ps = [p for e, p in self.stats if e == name and p > 0]
                if ps:
                    rec[f"place_{name}"] = round(float(np.mean(ps)), 3)
                    rec[f"top1_{name}"] = round(float(np.mean([p == 1 for p in ps])), 3)
            if self.booster is not None:
                rec.update(self.booster.take_stats(), t_boost=round(self.t_boost, 1),
                           boost_strength=round(self.booster.strength, 2))
                if self.boost_places:                  # 这一轮的加强局 / 普通机器人局分开看
                    rec["place_boost"] = round(float(np.mean(self.boost_places)), 3)
                    rec["boost_games"] = len(self.boost_places)
                self.boost_places = []
                if self.adapt is not None:
                    self.booster.strength = self.adapt.update()
                    rec["boost_place_avg"] = self.adapt.mean()
                    self._apply_boost_floor(rec)
                self.t_boost = 0.0
            if self.cgifts is not None:
                rec.update(self.cgifts.take_stats())
                cur, self.cgift_iter = self.cgift_iter, []
                if cur:
                    rec["place_cgift"] = round(float(np.mean([p for p, _ in cur])), 3)
                    rec["cgift_games"] = len(cur)
                    rec["cgift_formed"] = round(float(np.mean([f for _, f in cur])), 3)
            if self.gifts is not None:
                rec.update(self.gifts.take_stats())
                if self.gift_places:
                    rec["place_gift"] = round(float(np.mean(self.gift_places)), 3)
                    rec["gift_games"] = len(self.gift_places)
                self.gift_places = []
            if self.tribes is not None:
                rec.update(self.tribes.take_stats())
                if self.tribe_places:
                    rec["place_tribe"] = round(float(np.mean(self.tribe_places)), 3)
                    rec["tribe_games"] = len(self.tribe_places)
                    rec["tribe_newcomp"] = round(float(np.mean(self.tribe_new_comp)), 3)
                for k, v in (("tribe_newcomp_free", self.tribe_new_free), ("tribe_newcomp_start", self.tribe_new_start)):
                    if v:
                        rec[k] = round(float(np.mean(v)), 3)
                if self.tribe_start is not None:
                    rec.update(self.tribe_start.take_stats())
                self.tribe_places, self.tribe_new_comp, self.tribe_new_free, self.tribe_new_start = [], [], [], []
            if self.seeder is not None:
                rec.update(self.seeder.take_stats())
                if self.seed_places:
                    rec["place_seeded"] = round(float(np.mean(self.seed_places)), 3)
                    rec["seeded_games"] = len(self.seed_places)
                self.seed_places = []
            if cfg.snapshot_every and self.iter % cfg.snapshot_every == 0:
                m = self.snapshot()
                rec["snapshot"] = m.name
                rec["snapshot_dist"] = round(m.dist, 3)
                pt = (m.profile or {}).get("turns") or {}
                rec["snapshot_tier"] = "/".join(str(pt.get(str(t), {}).get("tier", "-")) for t in (5, 7, 9, 11, 13))
            if cfg.eval_every and self.iter % cfg.eval_every == 0:
                rec["eval_place"], rec["eval_top1"] = self.evaluate()
                if cfg.boost_frac > 0:
                    rec["eval_boost_place"], rec["eval_boost_top1"] = self.evaluate(seed=1999, boosted=True)
                    if self.cgift_by:                      # 送整套阵容：每套 [局数, 平均名次, 吃鸡率, 做成这套的比例]
                        rec["cgift_by_comp"] = {k: [len(v), round(float(np.mean([p for p, _ in v])), 2),
                                                    round(float(np.mean([p == 1 for p, _ in v])), 3),
                                                    round(float(np.mean([f for _, f in v])), 3)]
                                                for k, v in sorted(self.cgift_by.items())}
                        self.cgift_by = {}
                    if self.tribe_by:                      # 种族课程：每个强制的种族 [局数, 平均名次, 吃鸡率, 成了用这个种族的阵容的比例]
                        rec["tribe_by"] = {k: [len(v), round(float(np.mean([p for p, _ in v])), 2),
                                               round(float(np.mean([p == 1 for p, _ in v])), 3),
                                               round(float(np.mean([f for _, f in v])), 3)]
                                           for k, v in sorted(self.tribe_by.items())}
                        self.tribe_by = {}
                    if self.tribe_comps:                   # 种族课程局最后成了哪套（局数，多的在前）
                        rec["tribe_comps"] = dict(sorted(self.tribe_comps.items(), key=lambda kv: -kv[1]))
                        self.tribe_comps = {}
                    if self.seed_by:                       # 成型阵容课程：各套阵容拿到手以后打第几（上次考试以来）
                        rec["seeded_by_comp"] = {k: seed_row(v) for k, v in sorted(self.seed_by.items())}
                        self.seed_by = {}
                    eps = getattr(self, "last_eval_places", None) or []
                    if eps:
                        rec["eval_boost_top4"] = round(float(np.mean([p <= 4 for p in eps])), 3)
                        rec["eval_boost_score"] = round(place_score_equiv(eps, self.place_tab), 3)
                    rec.update(getattr(self, "last_comp", {}) or {})
                if self.pro is not None:
                    pe = self.demo_eval(self.pro, "pro")
                    self.adjust_pro(pe)
                    rec.update(pe, pro_scale=round(self.pro_scale, 3))
                if cfg.guard:
                    rec.update(self._guard(rec.get("eval_boost_score", rec.get("eval_boost_place")) if cfg.boost_frac > 0
                                           else rec.get("eval_place")))
            if cfg.save_every and self.iter % cfg.save_every == 0:
                self.save()
                if not getattr(self, "drift_n", 0):       # 正在变乱时不把现在当成回退点
                    self.mark_good()
            self.log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.log_f.flush()
            if cfg.board_every and self.iter % cfg.board_every == 0:
                self._redraw_board()
            if on_log:
                on_log(rec)
            stop = Path(cfg.run_dir) / "STOP"              # 面板的“停止训练”：这一轮跑完，存盘退出（见 panel/runner.py）
            if stop.exists():
                stop.unlink(missing_ok=True)
                print(f"收到停止请求：第 {self.iter} 轮已存盘，退出", flush=True)
                break
        self.save()
