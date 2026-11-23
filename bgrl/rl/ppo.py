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

from tg.actions import (
    is_panic,
)
from tg.obs import C_BOARD

from .audit import ShapeAudit
from .curricula import (
    AdaptiveStrength,
    ramp,
)
from .collector import CollectMixin
from .config import Config
from .guard import GuardMixin
from .imitation import ImitationMixin
from .league import League, load_real_profile
from .model import build_model, n_params
from .rollout import (
    Policy,
    Rollout,
    grow_opt_state,
    parse_place_reward,
    place_score_equiv,
    seed_row,
)
from .shaping import ShapingMixin
from .teacher import TeacherMixin
from .update import UpdateMixin


class Trainer(CollectMixin, UpdateMixin, ShapingMixin, ImitationMixin, TeacherMixin, GuardMixin):
    """训练主流程：组装各部分，管对手库快照、评估、检查点、train 循环。
    采样 rl/collector.py、更新 rl/update.py、奖励塑形 rl/shaping.py、模仿 rl/imitation.py、老师 rl/teacher.py、考试守护 rl/guard.py
    都作为 mixin 继承进来（Trainer.collect / Trainer._shape 之类照样能用）。"""

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
        self.lobbies = self._make_lobbies(cfg.seed) if cfg.lobby_frac > 0 else None
        if self.lobbies is not None:                   # 指定酒馆种族：第一批对局也换
            for _, e, _ in self.envs:
                self.lobbies.apply_all(e)
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
            from .curricula import CardGifts
            g_turns = [int(x) for x in str(cfg.gift_turns).split(",") if x.strip()]
            g_comps = [x.strip() for x in str(cfg.gift_comps).split(",") if x.strip()]
            g_cards = None
            if g_comps:
                from .comp import load_target_comps
                g_cards = sorted({cid for c in load_target_comps() if c["name"] in g_comps
                                  for cid in set(c["core"]) | set(c.get("support") or ())})
            self.gifts = CardGifts(cfg.gift_frac, g_turns, p=cfg.gift_p, min_lift=cfg.gift_min_lift, seed=cfg.seed,
                                   cards=g_cards)
            what = f"{'、'.join(g_comps)}的核心 / 配套卡" if g_comps else "高手抢着买、模型不买的牌"
            print(f"送牌课程：{cfg.gift_frac:.0%} 的机器人局，第 {cfg.gift_turns} 回合每回合 {cfg.gift_p:.0%} 的机会往手里放一张"
                  f"{what}（{len(self.gifts.pool)} 种）", flush=True)
        self.gift_places: list[int] = []
        self.trick_pre: dict = {}                      # 招募阶段连招奖励（trick_bonus）：环境 → {槽位: (发动 / 卖之前的身材, 回合)}
        self.trick_log = [0, 0.0]                      # [这一轮给了几次, 一共涨了多少身材]
        self.commit_places: list[int] = []            # 定向局（commit_comp）的名次
        self.commit_masked = 0                         # 定向局这一轮关掉了几个“买”
        self._commit_ok = None
        self._commit_force = None
        self.obs = {name: env.observe() for name, env, _ in self.envs}
        self.episode = {name: np.asarray([env.episode_index(i) for i in range(env.n_games)])
                        for name, env, _ in self.envs}
        self.role = np.zeros(off, dtype=np.int64)      # 0 = 学习者，k = self.role_names[k]
        self.trick_chain = np.zeros(off, dtype=bool)  # 招募阶段连招：上一步是不是发动 / 卖（紧跟着的选择也算）
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
        self.audit = ShapeAudit()                      # 塑形审计（C3）：每项塑形给了多少分
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

    # ---------------- 采样 ----------------

    # ---------------- 更新 ----------------

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
        if self.cfg.lobby_frac > 0 and self.cfg.lobby_exam:   # 专家：考试也在指定种族的酒馆里考（每次考试同一组酒馆）
            self._make_lobbies(seed, frac=1.0).apply_all(env)
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

    # ---------------- 主循环 ----------------

    # ---------------- 健康检查 / 回退 ----------------

    # ---------------- 考试守护（guard_*） ----------------

    def _redraw_board(self):
        """后台线程重写训练看板（dashboard.md）；上一次还没写完就跳过。出错只打印一次，不影响训练。"""
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
        return ck if isinstance(ck, dict) else {}

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
                   "t_infer": round(getattr(self, "t_infer", 0.0), 1), "rows_per_call": round(self.rows_calls / max(1, self.n_calls), 1),
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
                       unspent=round(mx["unspent"] / ends, 3))
            if mx.get("grow_n"):                       # 养场面：第 grow_from_turn 回合起每回合结束 - 开始的场面强度
                rec["grow"] = round(mx["grow_sum"] / mx["grow_n"], 4)
                rec["grow_n"] = int(mx["grow_n"])
                rec["t_grow"] = round(mx.get("t_grow", 0.0), 2)
            if mx.get("cs_n"):                         # 阵容强度：对同期高分阵容的平均胜率（全部 / 第 9 回合起）
                rec.update(cs_wr=round(mx["cs_sum"] / mx["cs_n"], 3), cs_n=int(mx["cs_n"]), t_combat=round(mx["t_combat"], 1),
                           cs_late=round(mx["cs_late_sum"] / mx["cs_late_n"], 3) if mx.get("cs_late_n") else None)
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
            rec.update(self.audit.take())              # sh_<项>：每局平均从这一项拿了多少分
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
            if self.gifts is not None:
                rec.update(self.gifts.take_stats())
                if self.gift_places:
                    rec["place_gift"] = round(float(np.mean(self.gift_places)), 3)
                    rec["gift_games"] = len(self.gift_places)
                self.gift_places = []
            if self.lobbies is not None:
                rec.update(self.lobbies.take_stats())
            if cfg.commit_frac > 0:
                rec["commit_masked"] = self.commit_masked
                if self.commit_places:
                    rec["place_commit"], rec["commit_games"] = round(float(np.mean(self.commit_places)), 3), len(self.commit_places)
                self.commit_places, self.commit_masked = [], 0
            if cfg.trick_bonus > 0:
                rec["trick_n"], rec["trick_stats"] = int(self.trick_log[0]), round(float(self.trick_log[1]), 1)
                self.trick_log = [0, 0.0]
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
                    mid = self.audit.take_mid()            # 塑形审计：第 6–8 回合每种动作每做一次从各项拿多少分
                    if mid:
                        rec["shape_mid"] = mid
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
