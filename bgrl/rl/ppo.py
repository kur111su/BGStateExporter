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
import random
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .league import League, load_real_profile
from .model import SIZES, build_model, n_params, soft_log
from tg.actions import A_BUY, A_END, A_REFRESH, A_SELL, A_UPGRADE, is_noop
from .calibrate import AdaptiveStrength, BotBooster, ProBoardBooster, load_pro_boards, ramp

# 观察（v1）里的几个全局数：第 0 个 回合/20，第 7 个 金币/10，第 10 个 酒馆等级/6，第 11 个 升本费/10（满级 0）
F_GOLD, F_TIER, F_UPCOST = 7, 10, 11


def is_panic(ex: BaseException) -> bool:
    """TavernGym 引擎（Rust）内部出错：pyo3 抛 PanicException（BaseException 的子类，except Exception 接不住）。
    第 5413 轮：engine.rs:925 召唤随从时任务奖励改了场面，下标越界，整个训练停了。"""
    return type(ex).__name__ == "PanicException"


@dataclass
class Config:
    run_dir: str = "runs/default"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0
    threads: int = 8
    games_bot: int = 192           # 和机器人打的并行对局数
    games_league: int = 48         # 8 座位都由我们控制的并行对局数（共 384 个座位）
    steps_per_iter: int = 128      # 每轮每个环境走多少步
    iters: int = 1000
    # 模型：size 选预设（small / medium / large），d 等单独给出时覆盖预设
    size: str = "medium"
    d: int = 0
    layers: int = 0
    heads: int = 0
    dc: int = 0
    # PPO
    gamma: float = 1.0
    lam: float = 0.995
    clip: float = 0.2
    target_kl: float = 0.05
    ent_coef: float = 0.005        # 论文 0.002；我们每轮样本少、噪声大，第一次训练第 330 轮策略塌缩（熵掉到 0.3），调高一点
    vf_coef: float = 0.5
    aux_coef: float = 0.1
    max_grad_norm: float = 1.0
    lr: float = 1e-4               # 3e-4 时第 310–325 轮更新过猛（KL 超标、名次头发散）
    minibatch: int = 4096
    micro_batch: int = 512         # 一次放进显卡的样本数（显存不够就调小；不影响结果，只影响速度）
    lr_recover_iters: int = 50     # 回退后连续这么多轮正常，学习率翻倍（最多回到 lr）
    # 每次调站位 / 冻结扣一点点奖励：这两个动作不推进游戏，第一版模型学会了一直点（影子模式的方案里 69% 是调站位）。
    # 30 次约等于 0.2 个名次，正常需要的一两次几乎不受影响
    noop_penalty: float = 0.003
    epochs: int = 1
    anchor_coef: float = 0.1
    anchor_iters: int = 100
    # 对手库 / 检查点
    snapshot_every: int = 50       # 每多少轮存一个快照进对手库
    profile_games: int = 16        # 测快照画像用多少局
    save_every: int = 10
    eval_every: int = 50
    eval_games: int = 128          # 64 局时平均名次的误差约 ±0.3，128 局约 ±0.2
    amp: bool = True               # 混合精度用 bfloat16（数值范围和 float32 一样，不会溢出；3070 支持）
    max_rollbacks: int = 5         # 连续出问题回退几次后停止训练
    # 每回合最多几个操作（第 1、2、3… 回合，最后一个数用于之后所有回合）。TavernGym 默认不限：
    # 调站位、冻结不推进游戏，策略可能一直点下去，对局永远结束不了（训练第 50 轮卡住就是这个原因）。
    # 按真实回合时长（约 45–90 秒）和每个操作 2–3 秒估。
    turn_caps: str = "14,18,22,26,30,34,38,42,45"
    active_snapshots: int = 1      # 同一时间最多用几个快照当对手（每多一个，每步要多算一次网络；3 个时采样慢一倍）
    # 第二阶段：按真人对局校准机器人强度（rl/calibrate.py）。v2 的模型学会了一直待在 1–2 本，对弱机器人有效、
    # 对真人后期无效；机器人局里把机器人场面按真人画像放大，逼模型学会升本、做后期。0 = 关闭
    boost_frac: float = 0.0        # 多少比例的机器人局加强
    boost_from_turn: int = 7       # 从第几回合开始（前 6 回合机器人和真人差不多）
    boost_every: int = 2           # 每几回合加强一次（每次要整局存档读档，约 7 毫秒）
    boost_ramp_iters: int = 200    # 不自适应时：加强幅度从 30% 爬到 100% 用多少轮
    # 自适应（默认）：按学习者在加强局的平均名次调幅度，见 rl/calibrate.py 的 AdaptiveStrength
    boost_adapt: bool = True
    boost_start: float = 0.4       # 自适应的起点（接着旧检查点练、检查点里没有记录时也用它）
    boost_target_place: float = 3.75   # 加强局的目标平均名次（8 人局随便打 = 4.5）
    boost_step: float = 0.02       # 每次调多少
    # 加强方式：scale = 放大机器人场面攻防（只是身材变大）；pro = 换成同一回合 6500+ 高分对局的真实阵容
    # （tools/pro_library.py 生成 runs/pro_boards.json，带高本随从和它们的效果）。pro 时 strength = 每个机器人被换的概率
    boost_kind: str = "scale"
    pro_boards: str = ""           # 空 = runs/pro_boards.json；几个库用分号隔开（例如加上 runs/h13_boards.json）
    # 升本节奏奖励的目标曲线：real = 你的对局（rl/real_profile.json）；pro = 6500+ 高分对局（rl/pro_profile.json）
    tempo_curve: str = "real"
    # 模仿你的操作（tools/make_demos.py 生成 runs/demos.npz）：每个小批另外抽 bc_batch 条示范算交叉熵，乘 bc_coef 加进损失。
    # 0 = 不模仿；有示范数据时日志照样记 human_top1 等（留出的验证局上模型和你的操作对得上多少）
    bc_coef: float = 0.0
    bc_data: str = ""              # 空 = runs/demos.npz
    bc_batch: int = 256
    # 按类别加权（竞技场里最缺的是升本、卖怪换怪、买随从）："upgrade:3,sell:3,buy:2"；没写的类别权重 1
    bc_weights: str = ""
    # 防止背下你的对局（示范只有十几局、两千来步）：每次考试看模型在训练用示范上的命中率，
    # 高于 bc_train_cap 就把模仿强度减半，低于 cap - 0.1 再加回来（最多到 bc_coef）。0 = 不控制
    # （第 2471–3047 轮：训练示范命中率 0.23 → 0.99，留出局只有 0.20，留出局交叉熵 3.2 → 8.2，完全是在背题）
    bc_train_cap: float = 0.0
    bc_smooth: float = 0.0         # 标签平滑：目标分一点给其它合法动作，避免对示范过分自信
    # 升本节奏奖励（课程，逐渐减到 0）：学习者升本时，如果升之前的等级低于真人同一回合的平均等级 + 0.5，
    # 当场加 tempo_bonus（一个名次约 0.29）。v3 在竞技场里第 9 回合平均 2.4 本（真人 5.0），77 个回合里
    # 有 56 个回合钱够却不升；奖励在 tempo_anneal_iters 轮内线性减到 0，之后只靠名次判断升本值不值
    tempo_bonus: float = 0.0
    tempo_anneal_iters: int = 1500
    # 落后扣分（不减弱）：每次结束回合时，估计等级（1 + 本局升本次数）比目标曲线 - 0.5 低多少，就扣 tempo_deficit × 差值。
    # 第 3041–3770 轮：升本奖励减到 0 以后，第 5/7/9/11/13 回合等级从 1.9/2.7/3.4/4.3/5.3 退回 1.8/2.5/3.3/3.9/4.7，
    # 只靠名次学不会按时升本；这一项一直在，逼它跟上 6500+ 的节奏（第 5 回合 3.7 本、第 9 回合 5.3 本）
    tempo_deficit: float = 0.0
    bc_scale_min: float = 0.01     # 模仿强度最低降到多少（见 bc_train_cap）
    # 模仿 6500+ 高分玩家（tools/make_pro_demos.py 从 Firestone 回放生成 runs/pro_demos/，约 10 万条决策）：
    # 和“模仿你的操作”分开算，有自己的系数、按类别权重和留出局指标（pro_top1 / pro_upgrade / pro_end …）
    pro_coef: float = 0.0
    pro_data: str = ""             # 空 = runs/pro_demos；几个目录用分号隔开（例如 "runs/pro_demos;runs/h13_demos"）
    pro_weights: str = ""          # 例如 "upgrade:3,end:2"
    pro_batch: int = 256
    # 无效刷新：刷新之后、下一次刷新或结束回合之前一个随从 / 法术都没买，就给那次刷新扣这么多
    # （竞技场 5 局：296 次刷新里 232 次之后什么都没买）
    refresh_waste_penalty: float = 0.0
    # 花掉升本的钱：落后于目标曲线（等级 < 目标 - 0.5）时，一步操作让金币从“够升本”掉到“不够升本”（不是升本本身），
    # 给这一步扣这么多（买随从扣一半）；结束回合时钱还够升本却没升，也扣一次。
    # 金币流向对比（361 局 6500+ vs 竞技场 5 局）：高手第 2 回合 86% 直接升本（钱够时 87%），第 4–8 回合钱够时
    # 51–78% 升本；模型第 2 回合 0% 升本——4 块钱先花 1 块发动“可疑的监狱守卫”（+3/+3），升本费不够了，
    # 再买 / 刷新；第 4–5 回合也是先发动、刷新，把升本拖到费用降到 1–2 块。落后扣分只在回合末按等级差扣，
    # 分不清是哪一步把升本的钱花掉的；这一项直接扣在那一步上
    upgrade_starve_penalty: float = 0.0
    # 上面那项里“买随从”按多少倍扣（1 = 和别的花法一样，0 = 买随从不扣）。
    # 第 4351–5054 轮（0.5）：第 2 回合升本率 4% → 51%，但买随从的钱第 5–8 回合只有高手的 1/4–1/3，阵容跟不上
    starve_buy_weight: float = 0.5
    # 回合末没花掉的钱：每剩 1 块扣这么多（第 2 回合起；高手几乎不剩，模型每回合剩 0.2–0.8 块）
    unspent_gold_penalty: float = 0.0
    # 防止背下高分示范：每次考试比较训练示范和留出局的命中率，差距大于 pro_gap_cap 就把模仿高手的强度减半，
    # 差距小于 pro_gap_cap - 0.04 再加回来（最多到 pro_coef）。0 = 不控制。
    # 第 4351–5054 轮（pro_coef 0.1）：训练示范命中率 0.44 → 0.70，留出局 0.45 → 0.41，交叉熵 2.0 → 2.33——在背题
    pro_gap_cap: float = 0.0

    def rules(self) -> dict:
        caps = [int(x) for x in str(self.turn_caps).replace(" ", "").split(",") if x]
        return {"turn_action_caps": caps} if caps else {}

    def model_kwargs(self) -> dict:
        kw = dict(SIZES[self.size])
        for k in ("d", "layers", "heads", "dc"):
            if getattr(self, k):
                kw[k] = getattr(self, k)
        return kw


class Policy:
    """一个可以出动作的策略：学习者或对手库里的快照。"""

    def __init__(self, model, device, amp=False):
        self.model, self.device = model, device
        self.amp = amp and device.type == "cuda"
        self.bad_rows = 0                      # 输出出现 NaN 的局面数（应该一直是 0）

    @torch.inference_mode()
    def __call__(self, f, c, m, greedy=False, want_obs=False):
        """want_obs=True 时另外返回压缩后的观察（float16，存训练数据用：压缩后最大十几，不会溢出）。"""
        f = torch.as_tensor(np.ascontiguousarray(f), dtype=torch.float32).to(self.device, non_blocking=True)
        c = torch.as_tensor(np.ascontiguousarray(c), dtype=torch.long).to(self.device, non_blocking=True)
        m = torch.as_tensor(np.ascontiguousarray(m), dtype=torch.bool).to(self.device, non_blocking=True)
        fz = soft_log(f)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.amp):
            logits, v, _ = self.model(fz, c, m, compressed=True)
        if greedy:
            a = logits.argmax(-1)
        else:
            a = torch.distributions.Categorical(logits=logits).sample()
        logp = torch.log_softmax(logits, -1).gather(1, a.unsqueeze(1)).squeeze(1)
        self.bad_rows += int(getattr(self.model, "last_bad", 0))
        out = (a.cpu().numpy(), logp.float().cpu().numpy(), v.float().cpu().numpy())
        if want_obs:
            out = out + (fz.half().cpu().numpy(),)
        return out


class Rollout:
    """学习者的决策按步成批存（numpy），名次奖励在对局结束时补到该座位最后一个决策上。
    槽位 = 全局座位编号（各环境的槽位依次排开）；ep = 这个槽位的第几局。"""

    def __init__(self, n_slots: int):
        self.n_slots = n_slots
        self.ep = np.zeros(n_slots, dtype=np.int64)
        self.place: dict[tuple, int] = {}
        self.reset()

    def reset(self):
        self.parts = defaultdict(list)
        self.r: list[float] = []
        self.done: list[bool] = []
        self.n = 0
        self.last = np.full(self.n_slots, -1, dtype=np.int64)

    def add(self, slots, f, c, m, a, logp, v, r0=None):
        k = len(slots)
        if not k:
            return
        slots = np.asarray(slots, dtype=np.int64)
        for name, x in (("slot", slots), ("ep", self.ep[slots]), ("f", np.asarray(f, dtype=np.float16)), ("c", c),
                        ("m", np.packbits(m, axis=1)), ("a", a), ("logp", logp), ("v", v)):
            self.parts[name].append(np.asarray(x))
        self.last[slots] = np.arange(self.n, self.n + k)
        self.r.extend([0.0] * k if r0 is None else [float(x) for x in r0])
        self.done.extend([False] * k)
        self.n += k

    def reward(self, slot, r, done, placement):
        i = self.last[slot]
        if i >= 0:
            self.r[i] += float(r)
            if done:
                self.done[i] = True
        if done:
            self.place[(int(slot), int(self.ep[slot]))] = int(placement)
            self.ep[slot] += 1
            self.last[slot] = -1

    def collect(self, gamma, lam, last_values: np.ndarray | None):
        """GAE（按槽位、按局分段）。没打完的用当前局面的价值自举。返回扁平化的数组。"""
        if self.n == 0:
            return None
        b = {k: np.concatenate(v) for k, v in self.parts.items()}
        r = np.asarray(self.r, dtype=np.float32)
        done = np.asarray(self.done, dtype=bool)
        v = b["v"].astype(np.float32)
        slot, ep = b["slot"], b["ep"]
        order = np.lexsort((np.arange(self.n), slot))           # 按槽位，槽位内按时间
        adv = np.zeros(self.n, dtype=np.float32)
        nxt_i = -1
        last = 0.0
        for pos in range(self.n - 1, -1, -1):
            i = order[pos]
            same = nxt_i >= 0 and slot[nxt_i] == slot[i] and ep[nxt_i] == ep[i]
            if done[i]:
                nv, last = 0.0, 0.0
            elif same:
                nv = v[nxt_i]
            else:                                              # 本轮这一段的最后一个决策
                last = 0.0
                live = last_values is not None and self.last[slot[i]] == i
                nv = float(last_values[slot[i]]) if live else 0.0
            delta = r[i] + gamma * nv - v[i]
            last = delta + gamma * lam * last
            adv[i] = last
            nxt_i = i
        b["adv"] = adv
        b["ret"] = adv + v
        b["place"] = np.asarray([self.place.get((int(s), int(e)), 0) for s, e in zip(slot, ep)], dtype=np.int64)
        self.place.clear()                 # 之后的决策只会属于还没结束的局，旧的名次用不到了
        self.reset()
        return b


class Trainer:
    def __init__(self, cfg: Config):
        import taverngym
        from taverngym import BatchEnv
        self.cfg = cfg
        self.dev = torch.device(cfg.device)
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.rng = random.Random(cfg.seed)
        self.run = Path(cfg.run_dir)
        (self.run / "snapshots").mkdir(parents=True, exist_ok=True)
        self.mkw = cfg.model_kwargs()
        self.model = build_model(**self.mkw).to(self.dev)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, eps=1e-5)
        self.amp = cfg.amp and self.dev.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=False)      # bfloat16 不需要缩放
        self.rollbacks = 0
        self.since_rollback = 0
        self.good = None                       # 最近一次健康时的模型和优化器（出问题时回退到这里）
        self.learner = Policy(self.model, self.dev, self.amp)
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
        self.t_boost = 0.0
        self.adapt = (AdaptiveStrength(start=cfg.boost_start, target=cfg.boost_target_place, step=cfg.boost_step)
                      if self.booster is not None and cfg.boost_adapt else None)
        if self.booster is not None:
            self.booster.strength = self.adapt.strength if self.adapt else ramp(0, 0, cfg.boost_ramp_iters)
        self.boost_places: list[int] = []              # 这一轮打完的加强局名次（日志用）
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
        self.demos = self._load_demos()
        self.bc_rng = np.random.default_rng(cfg.seed + 7)
        self.bc_w = self._bc_weights() if self.demos is not None else None
        self.pro = self._load_pro()
        self.pro_w = self._bc_weights(self.pro, cfg.pro_weights) if self.pro is not None else None
        self.bc_scale = 1.0                            # 模仿强度的倍数（见 bc_train_cap；存在检查点里）
        self.pro_scale = 1.0                           # 模仿高手强度的倍数（见 pro_gap_cap；存在检查点里）
        prof = load_real_profile().get("turns") or {}
        pro_prof = Path(__file__).resolve().parent / "pro_profile.json"
        if cfg.tempo_curve == "pro" and pro_prof.exists():
            prof = json.loads(pro_prof.read_text(encoding="utf-8")).get("turns") or {}
        self.tier_curve = {int(t): float(v.get("tier") or 0) for t, v in prof.items() if (v.get("n") or 0) >= 5}
        self.tempo_t0 = None                           # 升本奖励从哪一轮开始减（存在检查点里）
        self.upg_ep = np.full(self.n_slots, -1, dtype=np.int64)    # 每个槽位：这一局升了几次本
        self.upg_n = np.zeros(self.n_slots, dtype=np.int64)
        self.ref_idx = np.full(self.n_slots, -1, dtype=np.int64)   # 每个槽位：上一次还没买东西的刷新（本轮的步编号）
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
                self.rollout.reward(off + i, 0.0, True, 0)
        seed = (cfg.seed if name == "bot" else cfg.seed + 10**6) + 7919 * (self.iter + 1) + self.env_rebuilds
        new = BatchEnv(env.n_games, seed=seed, controlled=env.controlled, threads=cfg.threads, config=cfg.rules())
        self.envs[ei] = (name, new, off)
        self.episode[name] = np.asarray([new.episode_index(i) for i in range(new.n_games)])
        for g in range(new.n_games):
            self._assign(name, new, off, g)
        if name == "bot" and self.booster is not None:
            self.booster.plan.clear()
            self.booster.last_turn.clear()
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
            model = build_model(**self.mkw).to(self.dev)
            model.load_state_dict(torch.load(m.path, map_location=self.dev))
            model.eval()
            self.snap_cache[name] = Policy(model, self.dev, self.amp)
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
                        r0 = self._shape(off + idx, f[o:o + k], ak, -cfg.noop_penalty * noop.astype(np.float64))
                        self.rollout.add(off + idx, res[3][o:o + k], c[o:o + k], m[o:o + k], ak,
                                         res[1][o:o + k], res[2][o:o + k], r0=r0)
                        self._after_add(off + idx, ak)
                        n_dec += k
                        self.n_noop += int(noop.sum())
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
                        self.rollout.reward(off + int(i), rewards[i], bool(dones[i]), info["placement"][i])
                        if dones[i] and env.controlled[int(i) % len(env.controlled)] == 0:
                            self.stats.append((name, int(info["placement"][i])))
                            self.finished += 1
                            if name == "bot" and self.booster is not None and self.booster.finish(int(i)):
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
                if dones.any():                                # 新开的对局：重新抽对手
                    k = len(env.controlled)
                    for g in np.unique(np.flatnonzero(dones) // k):
                        e = env.episode_index(int(g))
                        if e != self.episode[name][g]:
                            self.episode[name][g] = e
                            self._assign(name, env, off, int(g))
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
                    self._pro_lib = {}
                    for p in have:                     # 几个阵容库（分号隔开）合在一起
                        for t, v in load_pro_boards(p).items():
                            self._pro_lib.setdefault(t, []).extend(v)
                    print(f"高分阵容库：{sum(len(v) for v in self._pro_lib.values())} 个整板（{path}）", flush=True)
                return ProBoardBooster(self._pro_lib, frac, cfg.boost_from_turn, seed=seed, every=cfg.boost_every)
            print(f"没有高分阵容库 {path}（先运行 bg.py prolib），改用放大攻防", flush=True)
        return BotBooster(load_real_profile(), frac, cfg.boost_from_turn, seed=seed, every=cfg.boost_every)

    # ---------------- 模仿你的操作 ----------------

    def _load_demos(self):
        from paths import RUNS
        path = Path(self.cfg.bc_data) if self.cfg.bc_data else RUNS / "demos.npz"
        if not path.exists():
            if self.cfg.bc_coef > 0:
                print(f"没有示范数据 {path}（先运行 bg.py demos），这次不模仿", flush=True)
            return None
        try:
            from tools.make_demos import load
            d = load(path)
        except Exception as ex:                        # noqa: BLE001
            print(f"示范数据 {path} 读不了（{type(ex).__name__}: {ex}），这次不模仿", flush=True)
            return None
        if d["m"].shape[1] != self.model.L.actions or d["f"].shape[1:] != self.obs[self.envs[0][0]]["features"].shape[1:]:
            print(f"示范数据 {path} 和当前环境的格式不一样（TavernGym 换过版本？），重新运行 bg.py demos", flush=True)
            return None
        d["train_idx"] = np.flatnonzero(d["split"] == 0)
        d["val_idx"] = np.flatnonzero(d["split"] == 1)
        print(f"示范数据：{d['meta'].get('games')} 局，训练 {len(d['train_idx'])} 条、验证 {len(d['val_idx'])} 条"
              f"（{d['meta'].get('created')}）；模仿系数 {self.cfg.bc_coef}", flush=True)
        return d

    def _load_pro(self):
        from paths import RUNS
        # 可以写几个目录，用分号隔开（例如 "runs\\pro_demos;runs\\h13_demos"），合在一起用
        paths = [Path(p) for p in str(self.cfg.pro_data).split(";") if p.strip()] or [RUNS / "pro_demos"]
        from tools.make_pro_demos import load_dir
        parts = []
        for path in paths:
            if not path.exists():
                if self.cfg.pro_coef > 0:
                    print(f"没有高分决策样本 {path}（先运行 bg.py prodemos / nativedemos），跳过", flush=True)
                continue
            try:
                x = load_dir(path)
            except Exception as ex:                    # noqa: BLE001
                print(f"高分决策样本 {path} 读不了（{type(ex).__name__}: {ex}）", flush=True)
                continue
            if x is not None:
                print(f"  {path}：{len(x['kind'])} 条（{x['meta'].get('source', '')}）", flush=True)
                parts.append(x)
        if not parts:
            return None
        d = parts[0]
        if len(parts) > 1:
            d = {k: np.concatenate([p[k] for p in parts]) for k in ("f", "c", "m", "cand", "kind", "turn", "split", "mmr")}
            d["compressed"] = True
            d["meta"] = {"games": sum(int(p["meta"].get("games") or 0) for p in parts)}
        path = ";".join(str(p) for p in paths)
        if d["m"].shape[1] != self.model.L.actions or d["f"].shape[1:] != self.obs[self.envs[0][0]]["features"].shape[1:]:
            print(f"高分决策样本 {path} 和当前环境的格式不一样，重新运行 bg.py prodemos", flush=True)
            return None
        d["train_idx"] = np.flatnonzero(d["split"] == 0)
        d["val_idx"] = np.flatnonzero(d["split"] == 1)
        print(f"高分决策样本：{d['meta'].get('games')} 局，训练 {len(d['train_idx'])} 条、验证 {len(d['val_idx'])} 条；"
              f"模仿系数 {self.cfg.pro_coef}", flush=True)
        return d

    def _demo_tensors(self, j, d=None):
        d = self.demos if d is None else d
        t = lambda x, dt: torch.as_tensor(x, dtype=dt).to(self.dev, non_blocking=True)  # noqa: E731
        return (t(d["f"][j], torch.float32), t(d["c"][j], torch.long), t(d["m"][j], torch.bool),
                t(d["cand"][j].astype(np.int64), torch.long))

    def _demo_logp(self, f, c, m, compressed=False):
        with torch.autocast(self.dev.type, dtype=torch.bfloat16, enabled=self.amp):
            logits = self.model(f, c, m, compressed=compressed)[0]
        return F.log_softmax(logits.float(), -1)

    def _bc_weights(self, d=None, spec=None):
        from tools.make_demos import KINDS
        d = self.demos if d is None else d
        spec = self.cfg.bc_weights if spec is None else spec
        w, tw = {}, []
        for part in str(spec or "").replace(" ", "").split(","):
            if ":" in part:
                k, v = part.split(":", 1)
                if k in KINDS:
                    w[KINDS.index(k)] = float(v)
                elif k[:1] == "t" and "turn" in d:              # 按回合再乘：t2:3 第 2 回合 ×3，t3-6:2 第 3–6 回合 ×2
                    lo, _, hi = k[1:].partition("-")
                    tw.append((int(lo), int(hi or lo), float(v)))
        out = np.ones(len(d["kind"]), dtype=np.float32)
        for k, v in w.items():
            out[d["kind"] == k] = v
        for lo, hi, v in tw:
            out[(d["turn"] >= lo) & (d["turn"] <= hi)] *= v
        return out

    def bc_loss(self, d=None, wts=None, batch=None):
        d = self.demos if d is None else d
        wts = self.bc_w if wts is None else wts
        batch = batch or self.cfg.bc_batch
        j = self.bc_rng.choice(d["train_idx"], size=min(batch, len(d["train_idx"])))
        f, c, m, cand = self._demo_tensors(j, d)
        lp_all = self._demo_logp(f, c, m, bool(d.get("compressed")))
        lp = lp_all.gather(1, cand.clamp(min=0)).masked_fill(cand < 0, float("-inf"))
        nll = -torch.logsumexp(lp, 1)                  # 候选有几个（磁力吸附目标不确定）时，任一个都算对
        eps = self.cfg.bc_smooth
        if eps > 0:                                    # 标签平滑：合法动作上的平均对数概率
            uni = -(lp_all.masked_fill(~m, 0).sum(-1) / m.sum(-1).clamp(min=1))
            nll = (1 - eps) * nll + eps * uni
        w = torch.as_tensor(wts[j], device=nll.device)
        return (nll * w).sum() / w.sum()

    def adjust_bc(self, he: dict) -> None:
        """按训练示范命中率调模仿强度（考试时调用）。"""
        cap = self.cfg.bc_train_cap
        tr = he.get("human_top1_train")
        if cap <= 0 or tr is None:
            return
        if tr > cap:
            self.bc_scale = max(self.cfg.bc_scale_min, self.bc_scale * 0.5)
        elif tr < cap - 0.1:
            self.bc_scale = min(1.0, self.bc_scale * 1.5)

    def adjust_pro(self, pe: dict) -> None:
        """按训练示范和留出局命中率的差距调模仿高手的强度（考试时调用）。"""
        cap = self.cfg.pro_gap_cap
        tr, va = pe.get("pro_top1_train"), pe.get("pro_top1")
        if cap <= 0 or tr is None or va is None:
            return
        if tr - va > cap:
            self.pro_scale = max(self.cfg.bc_scale_min, self.pro_scale * 0.5)
        elif tr - va < cap - 0.04:
            self.pro_scale = min(1.0, self.pro_scale * 1.5)

    def tempo_now(self) -> float:
        cfg = self.cfg
        if cfg.tempo_bonus <= 0:
            return 0.0
        if self.tempo_t0 is None:
            self.tempo_t0 = self.iter
        x = (self.iter - self.tempo_t0) / max(1, cfg.tempo_anneal_iters)
        return cfg.tempo_bonus * max(0.0, 1.0 - x)

    def _tier_target(self, turn: int) -> float:
        if not self.tier_curve:
            return 6.0
        t = min(max(turn, min(self.tier_curve)), max(self.tier_curve))
        while t not in self.tier_curve and t > min(self.tier_curve):
            t -= 1
        return self.tier_curve.get(t, 6.0)

    def _shape(self, slots, f, ak, r0):
        """升本节奏奖励（当场加在升本这一步上）；记下刷新，供 _after_add 判断是不是无效刷新。"""
        tb = self.tempo_now()
        turns = np.rint(f[:, 0] * 20).astype(int)
        dc = self.cfg.tempo_deficit
        for j in np.flatnonzero(ak == A_END):              # 结束回合：落后于目标曲线就扣分
            s = int(slots[j])
            ep = int(self.rollout.ep[s])
            tier = 1 + (int(self.upg_n[s]) if self.upg_ep[s] == ep else 0)
            gap = max(0.0, self._tier_target(int(turns[j])) - 0.5 - tier)
            self.mix["gap_sum"] += gap
            self.mix["gap_n"] += 1
            if dc > 0 and gap > 0:
                r0[j] -= dc * gap
        for j in np.flatnonzero(ak == A_UPGRADE):
            s = int(slots[j])
            ep = int(self.rollout.ep[s])
            if self.upg_ep[s] != ep:
                self.upg_ep[s], self.upg_n[s] = ep, 0
            if tb > 0 and 1 + self.upg_n[s] < self._tier_target(int(turns[j])) + 0.5:
                r0[j] += tb
                self.mix["tempo_paid"] += 1
            self.upg_n[s] += 1
        if f.shape[1] > F_UPCOST:
            self._gold_shape(slots, f, ak, r0)
        return r0

    def _gold_shape(self, slots, f, ak, r0):
        """金币流向（见 upgrade_starve_penalty）。观察里第 7 / 10 / 11 个数是 金币/10、等级/6、升本费/10。
        同时统计：回合开始时钱够升本的回合里升了本的比例（第 2–8 回合，up_afford；高手约 0.68）和第 2 回合升本率（up_t2）。"""
        pen = self.cfg.upgrade_starve_penalty
        if not hasattr(self, "gs"):
            n = len(self.rollout.last)
            self.gs = {k: np.full(n, -1, dtype=np.int64) for k in ("ep", "turn", "gold", "up", "tier", "act", "ts_turn",
                                                                  "ts_tier", "ts_aff")}
            self.gs["behind"] = np.zeros(n, dtype=bool)
        g = self.gs
        turns = np.rint(f[:, 0] * 20).astype(int)
        golds = np.rint(f[:, F_GOLD] * 10).astype(int)
        ups = np.rint(f[:, F_UPCOST] * 10).astype(int)
        tiers = np.rint(f[:, F_TIER] * 6).astype(int)
        for j in range(len(ak)):
            s, a, t = int(slots[j]), int(ak[j]), int(turns[j])
            ep = int(self.rollout.ep[s])
            gold, up, tier = int(golds[j]), int(ups[j]), int(tiers[j])
            same = g["ep"][s] == ep and g["turn"][s] == t
            # 上一步把金币从“够升本”花到“不够”（上一步不是升本、卖），而且当时落后
            if same and g["behind"][s] and g["act"][s] != A_UPGRADE and g["up"][s] > 0 \
                    and g["gold"][s] >= g["up"][s] > gold:
                self.mix["starve"] += 1
                prev = int(self.rollout.last[s])
                if pen > 0 and prev >= 0:
                    self.rollout.r[prev] -= pen * (self.cfg.starve_buy_weight if A_BUY <= g["act"][s] < A_SELL else 1.0)
            if not (g["ts_turn"][s] == t and g["ep"][s] == ep):         # 新回合的第一步
                g["ts_turn"][s], g["ts_tier"][s], g["ts_aff"][s] = t, tier, int(up > 0 and gold >= up)
            behind = tier < self._tier_target(t) - 0.5
            if a == A_END:
                if 2 <= t <= 8 and g["ts_aff"][s] == 1:
                    self.mix["aff_n"] += 1
                    self.mix["aff_up"] += int(tier > g["ts_tier"][s])
                if t == 2:
                    self.mix["t2_n"] += 1
                    self.mix["t2_up"] += int(tier > g["ts_tier"][s])
                if t >= 2 and gold > 0:
                    self.mix["unspent"] += gold
                    if self.cfg.unspent_gold_penalty > 0:
                        r0[j] -= self.cfg.unspent_gold_penalty * gold
                if behind and up > 0 and gold >= up:                    # 钱够升本、落后，却结束回合
                    self.mix["end_rich"] += 1
                    if pen > 0:
                        r0[j] -= pen
            g["ep"][s], g["turn"][s], g["gold"][s], g["up"][s], g["tier"][s], g["act"][s] = ep, t, gold, up, tier, a
            g["behind"][s] = behind
        return r0

    def _after_add(self, slots, ak):
        """无效刷新：上一次刷新之后没买任何东西就又刷新 / 结束回合 → 给那次刷新扣分（改已经存下的奖励）。"""
        pen = self.cfg.refresh_waste_penalty
        for j, a in enumerate(ak):
            s = int(slots[j])
            a = int(a)
            if a == A_REFRESH or a == A_END:
                prev = self.ref_idx[s]
                if prev >= 0:
                    self.mix["refresh_waste"] += 1
                    if pen > 0:
                        self.rollout.r[prev] -= pen
                self.ref_idx[s] = self.rollout.last[s] if a == A_REFRESH else -1
            elif A_BUY <= a < A_SELL:
                self.ref_idx[s] = -1
            kind = ("end" if a == A_END else "refresh" if a == A_REFRESH else "upgrade" if a == A_UPGRADE else
                    "buy" if A_BUY <= a < A_SELL else "sell" if A_SELL <= a < A_SELL + 7 else None)
            if kind:
                self.mix[kind] += 1

    @torch.no_grad()
    def human_eval(self, d=None, prefix: str = "human") -> dict:
        """留出的验证局上：模型最可能的动作是不是示范里的那个（{prefix}_top1），按类别的命中率，平均交叉熵。"""
        from tools.make_demos import KINDS
        d = self.demos if d is None else d
        self.model.eval()
        out = {}
        val = d["val_idx"]
        if len(val) > 6000:                            # 高分样本的验证集很大：固定抽 6000 条（每次同一批）
            val = val[np.random.default_rng(0).permutation(len(val))[:6000]]
        for tag, idx in (("", val), ("_train", d["train_idx"][:2000])):
            if not len(idx):
                continue
            hit, nll = [], []
            for s in range(0, len(idx), self.cfg.micro_batch):
                j = idx[s:s + self.cfg.micro_batch]
                f, c, m, cand = self._demo_tensors(j, d)
                lp = self._demo_logp(f, c, m, bool(d.get("compressed")))
                top = lp.argmax(1, keepdim=True)
                hit.append(((cand == top) & (cand >= 0)).any(1).cpu().numpy())
                nll.append((-torch.logsumexp(lp.gather(1, cand.clamp(min=0)).masked_fill(cand < 0, float("-inf")), 1))
                           .clamp(max=20).cpu().numpy())
            hit, nll = np.concatenate(hit), np.concatenate(nll)
            out[f"{prefix}_top1" + tag] = round(float(hit.mean()), 3)
            if tag == "":
                out[f"{prefix}_nll"] = round(float(nll.mean()), 3)
                kind = d["kind"][idx]
                for k in ("upgrade", "end", "buy", "refresh", "sell"):
                    sel = kind == KINDS.index(k)
                    if sel.any():
                        out[f"{prefix}_{k}"] = round(float(hit[sel].mean()), 3)
                mm = d.get("mmr")
                if mm is not None:                     # 几个来源合在一起时（--pro-data "a;b"）：万分以上的那部分单独看
                    hi = mm[idx] >= 10000
                    if hi.any() and not hi.all():
                        out[f"{prefix}_top1_hi"] = round(float(hit[hi].mean()), 3)
                        sel = hi & (kind == KINDS.index("upgrade"))
                        if sel.any():
                            out[f"{prefix}_upgrade_hi"] = round(float(hit[sel].mean()), 3)
        self.model.train()
        return out

    # ---------------- 更新 ----------------

    def update(self, b):
        cfg = self.cfg
        self.model.train()
        n = len(b["a"])
        adv = (b["adv"] - b["adv"].mean()) / (b["adv"].std() + 1e-8)
        idx = np.random.permutation(n)
        stats = defaultdict(list)
        use_anchor = self.anchor is not None and self.iter < self.anchor_until
        stop = False
        A_N = self.model.L.actions
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
                    if use_anchor:
                        with torch.no_grad(), torch.autocast(self.dev.type, dtype=torch.bfloat16, enabled=self.amp):
                            ref = F.log_softmax(self.anchor(f, c, m, compressed=True)[0].float(), -1)
                        kl_anchor = (ref.exp() * (ref - logp_all)).masked_fill(~m, 0).sum(-1).mean()
                        loss = loss + cfg.anchor_coef * kl_anchor
                        stats["kl_anchor"].append(kl_anchor.item())
                    self.scaler.scale(loss * w).backward()
                    with torch.no_grad():
                        kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                    kls.append(kl * w)
                    for k_, v_ in (("loss", loss.item()), ("pg", pg.item()), ("vf", vf.item()), ("ent", ent.item()),
                                   ("aux", aux.item()), ("kl", kl),
                                   ("clipfrac", ((ratio - 1).abs() > cfg.clip).float().mean().item())):
                        stats[k_].append(v_)
                if self.demos is not None and cfg.bc_coef > 0 and len(self.demos["train_idx"]):
                    bc = self.bc_loss()
                    self.scaler.scale(cfg.bc_coef * self.bc_scale * bc).backward()
                    stats["bc"].append(bc.item())
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
        self.model.eval()
        for _ in range(30000):                       # 保险：不管怎样都会结束
            if len(places) >= games:
                break
            ready = np.flatnonzero(obs["ready"])
            actions = np.zeros(env.n_slots, dtype=np.int64)
            if len(ready):
                a, _, _ = self.learner(obs["features"][ready], obs["cards"][ready], obs["action_mask"][ready])
                actions[ready] = a
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
        return (round(float(np.mean(ps)), 3), round(float(np.mean([p == 1 for p in ps])), 3)) if ps else (None, None)

    # ---------------- 主循环 ----------------

    # ---------------- 健康检查 / 回退 ----------------

    def health(self, upd: dict) -> str:
        """这一轮有没有出问题（返回问题描述，没有返回空串）。第一次训练第 310 轮起更新失控、
        第 455 轮起参数变成 NaN，之前的保险把 NaN 悄悄换成了随机动作，又白跑了 800 轮——现在直接回退。"""
        if any(not torch.isfinite(p).all() for p in self.model.parameters()):
            return "模型参数出现 NaN / inf"
        if self.learner.bad_rows or upd.get("bad_rows"):
            return f"网络输出出现 NaN（采样 {self.learner.bad_rows} 个、更新 {upd.get('bad_rows', 0)} 个局面）"
        if upd.get("skipped", 0) >= 2:
            return f"有 {upd['skipped']} 个小批的梯度是 NaN / inf"
        if upd.get("aux", 0) > 4.0:
            return f"名次头损失发散（{upd['aux']:.2f}）"
        return ""

    def mark_good(self):
        import copy
        self.good = (copy.deepcopy(self.model.state_dict()), copy.deepcopy(self.opt.state_dict()), self.iter)
        self.rollbacks = 0

    def rollback(self, problem: str):
        self.rollbacks += 1
        print(f"第 {self.iter} 轮出问题：{problem}。回退到第 {self.good[2] if self.good else '?'} 轮，学习率减半。", flush=True)
        if self.rollbacks > self.cfg.max_rollbacks or self.good is None:
            self.log_f.write(json.dumps({"iter": self.iter, "stopped": problem}, ensure_ascii=False) + "\n")
            self.log_f.flush()
            raise RuntimeError(f"连续回退 {self.rollbacks} 次仍然出问题（{problem}），训练停止。把 log.jsonl 发给 Claude。")
        sd, od, it = self.good
        self.model.load_state_dict(sd)
        self.opt.load_state_dict(od)
        for g in self.opt.param_groups:
            g["lr"] *= 0.5
        self.iter = it
        self.snap_cache.clear()

    def init_from(self, path):
        """用另一个检查点 / 快照的参数开始（不带优化器和对手库）。"""
        ck = torch.load(path, map_location=self.dev, weights_only=False)
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        if any(not torch.isfinite(v).all() for v in sd.values() if torch.is_tensor(v) and v.is_floating_point()):
            raise ValueError(f"{path} 里的参数有 NaN / inf，不能用")
        self.model.load_state_dict(sd)
        print(f"从 {path} 的参数开始", flush=True)

    def save(self):
        torch.save({"model": self.model.state_dict(), "opt": self.opt.state_dict(), "iter": self.iter,
                    "league": self.league.to_json(), "model_kwargs": self.mkw,
                    "boost": {**self.adapt.to_json(), "kind": self.cfg.boost_kind} if self.adapt is not None else None, "tempo_t0": self.tempo_t0,
                    "bc_scale": self.bc_scale, "pro_scale": self.pro_scale}, self.run / "latest.pt")

    def load(self, path):
        ck = torch.load(path, map_location=self.dev, weights_only=False)
        if ck.get("model_kwargs") and ck["model_kwargs"] != self.mkw:
            raise ValueError(f"检查点的模型大小 {ck['model_kwargs']} 和这次的 {self.mkw} 不一样")
        self.model.load_state_dict(ck["model"])
        self.opt.load_state_dict(ck["opt"])
        self.iter = ck["iter"]
        self.league = League.from_json(ck["league"], load_real_profile())
        self.tempo_t0 = ck.get("tempo_t0", self.tempo_t0)
        self.bc_scale = float(ck.get("bc_scale", self.bc_scale))
        self.pro_scale = float(ck.get("pro_scale", self.pro_scale))
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
        while self.iter < end:
            self.learner.bad_rows = 0
            batch, n_dec, t_col, t_env, t_net = self.collect()
            t1 = time.time()
            upd = self.update(batch) if batch is not None else {}
            t_upd = time.time() - t1
            self.iter += 1
            problem = self.health(upd)
            if not problem:
                self.since_rollback += 1
                g0 = self.opt.param_groups[0]["lr"]
                if g0 < cfg.lr and self.since_rollback >= cfg.lr_recover_iters:
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
                   "t_update": round(t_upd, 1), "lr": self.opt.param_groups[0]["lr"],
                   "noop_frac": round(self.n_noop / max(1, n_dec), 3), **upd}
            mx, ends = self.mix, max(1, self.mix.get("end", 0))           # 每回合平均（学习者）
            rec.update(up_turn=round(mx["upgrade"] / ends, 3), ref_turn=round(mx["refresh"] / ends, 2),
                       buy_turn=round(mx["buy"] / ends, 2), sell_turn=round(mx["sell"] / ends, 3),
                       ref_waste=round(mx["refresh_waste"] / max(1, mx["refresh"]), 3),
                       tempo_bonus=round(self.tempo_now(), 4), tempo_paid=mx["tempo_paid"],
                       tier_gap=round(mx["gap_sum"] / max(1, mx["gap_n"]), 3),
                       up_afford=round(mx["aff_up"] / max(1, mx["aff_n"]), 3), up_t2=round(mx["t2_up"] / max(1, mx["t2_n"]), 3),
                       starve_turn=round(mx["starve"] / ends, 3), end_rich=round(mx["end_rich"] / ends, 3),
                       unspent=round(mx["unspent"] / ends, 3))
            if mx.get("env_rebuild"):
                rec["env_rebuild"] = mx["env_rebuild"]
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
                self.t_boost = 0.0
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
                if self.demos is not None:
                    he = self.human_eval()
                    self.adjust_bc(he)
                    rec.update(he, bc_scale=round(self.bc_scale, 3))
                if self.pro is not None:
                    pe = self.human_eval(self.pro, "pro")
                    self.adjust_pro(pe)
                    rec.update(pe, pro_scale=round(self.pro_scale, 3))
            if cfg.save_every and self.iter % cfg.save_every == 0:
                self.save()
                self.mark_good()
            self.log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.log_f.flush()
            if on_log:
                on_log(rec)
        self.save()
