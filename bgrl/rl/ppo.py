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
from tg.actions import is_noop
from tg.obs import C_BOARD
from tg.actions import is_panic  # noqa: F401 —— 引擎内部出错（第 5413 轮 engine.rs:925 召唤后下标越界，整个训练停了）
from .calibrate import AdaptiveStrength, BotBooster, ProBoardBooster, merge_pro_boards, ramp
from .comp import board_scores, load_comp_refs
from .imitation import ImitationMixin
from .shaping import ShapingMixin


@dataclass
class Config:
    run_dir: str = "runs/default"
    # auto = 有 NVIDIA 显卡用 cuda，Apple 芯片用 mps，否则 cpu；跑过 bg.py bench-device 的按实测最快的选（rl/device.py）
    device: str = "auto"
    # 采样（模型出动作）用的设备：auto = 实测最快的 / 和 device 一样；也可以 cpu、mps、cuda。和 device 不同时采样用一份副本，每次更新后同步
    rollout_device: str = "auto"
    seed: int = 0
    threads: int = 0               # 模拟器线程数；0 = 自动（性能核数）
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
    # 优化器重新开始（--restore、考试守护换回 best.pt）后学习率从 10% 用 warmup_iters 轮爬到 lr（0 = 不爬）。
    # 第 15300 轮退回那次没有爬坡：新 Adam 头几步每个参数都走满一步，熵 12 轮里 0.97 → 1.34，之后一路涨到 1.6 以上
    warmup_iters: int = 0
    # 熵的目标（0 = 不管，ent_coef 固定）：每轮熵高于目标 ent_coef × (1 - ent_adapt)，低于 × (1 + ent_adapt)，
    # 限在 [ent_coef_min, ent_coef_max]。考试最好的几段熵在 1.05–1.16（第 13000–13500 轮 4.22–4.26），
    # 1.34–1.48 时 4.47–4.78；第 15300 轮退回后熵涨到 1.55–1.74，变乱检查连退 4 次、学习率 1.5e-4 → 1.9e-5
    ent_target: float = 0.0
    ent_adapt: float = 0.03
    ent_coef_min: float = 0.0005
    ent_coef_max: float = 0.01
    # 每次调站位 / 冻结扣一点点奖励：这两个动作不推进游戏，第一版模型学会了一直点（给出的方案里 69% 是调站位）。
    # 30 次约等于 0.2 个名次，正常需要的一两次几乎不受影响
    noop_penalty: float = 0.003
    epochs: int = 1
    anchor_coef: float = 0.1
    anchor_iters: int = 100
    # 对手库 / 检查点
    snapshot_every: int = 50       # 每多少轮存一个快照进对手库
    profile_games: int = 16        # 测快照画像用多少局
    save_every: int = 10
    # 训练看板（tools/board.py）：每这么多轮在后台重画 runs/<run>/dashboard.md + board/*.svg（0 = 不画）
    board_every: int = 0
    eval_every: int = 50
    eval_games: int = 128          # 64 局时平均名次的误差约 ±0.3，128 局约 ±0.2
    amp: bool = True               # 混合精度用 bfloat16（数值范围和 float32 一样，不会溢出；3070 支持）
    max_rollbacks: int = 5         # 连续出问题回退几次后停止训练
    # 策略慢慢变乱（不是 NaN 那种一下子坏掉）也算出问题：熵比最近 300 轮的中位数高出 drift_ent，或者无效操作
    # （冻结 / 换位 / 取消）比例超过 drift_noop，连续 drift_iters 轮 → 回退到变乱之前、学习率减半；变乱期间不更新回退点。
    # 第 12490–12700 轮：熵 1.0 → 1.8–2.0、无效操作 5% → 20–39%、白刷新 11% → 35–40%，加强局能承受的对手强度
    # 0.76 → 0.52，对高分阵容考试 4.1 → 4.6–4.9，原来的检查一样都没发现。0 = 不检查
    drift_ent: float = 0.35
    drift_noop: float = 0.15
    drift_iters: int = 10
    # 熵的绝对上限：连续 drift_iters 轮高于它也算变乱（0 = 不看）。上面的 drift_ent 跟着最近 300 轮的中位数走，
    # 熵慢慢涨的时候中位数也跟着涨（第 14240 轮中位数已经是 1.69），等它报警时回退点也早就乱了
    drift_ent_max: float = 0.0
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
    # 后期全强度（只对 boost_kind pro）：第 boost_late_from 回合起每个机器人座位被换成高分阵容的概率至少 boost_late_strength，
    # 前期照旧按自适应的强度。为什么：自适应把强度压在 0.6–0.7（加强局名次对着 3.75 调），后期一半机器人还是弱场面，
    # 前四名靠中期节奏阵容就够了，不需要能成长的阵容；考试是全强度，第 12 / 13 / 14 回合场面强度 0.49 / 0.39 / 0.28
    # （高手 0.67 / 0.60 / 0.68），名次就丢在这里。0 = 不分前后期
    boost_late_from: int = 0
    boost_late_strength: float = 1.0
    # 前期强度下限（第 19650 轮起）：自适应强度不低于 boost_min_strength；从启动时的强度每轮最多抬 boost_min_step 爬上去（不一下子变难）。
    # 为什么：第 17010 轮加了后期全强度以后，自适应把前期强度从 0.6–0.7 一路压到 0.26（加强局名次对着 3.75 调），训练对手越来越弱——
    # 机器人局吃鸡 20%，考试（全强度）只有 1–3%；靠前中期节奏就能在训练里赢，做阵容在训练里只值 +0.27 名。考试最好的一段
    # （第 11000–13500 轮 4.10–4.26）强度是 0.72–0.78。0 = 不设下限
    boost_min_strength: float = 0.0
    boost_min_step: float = 0.002
    # 加强方式：scale = 放大机器人场面攻防（只是身材变大）；pro = 换成同一回合 6500+ 高分对局的真实阵容
    # （tools/pro_library.py 生成 runs/pro_boards.json，带高本随从和它们的效果）。pro 时 strength = 每个机器人被换的概率
    boost_kind: str = "scale"
    pro_boards: str = ""           # 空 = runs/pro_boards.json；几个库用分号隔开（例如加上 runs/h13_boards.json）
    # 升本节奏奖励的目标曲线：real = 真人对局的汇总（rl/real_profile.json）；pro = 6500+ 高分对局（rl/pro_profile.json）
    tempo_curve: str = "real"
    # 模仿示范时的标签平滑（见 rl/imitation.py 的 bc_loss）
    bc_smooth: float = 0.0         # 标签平滑：目标分一点给其它合法动作，避免对示范过分自信
    # 升本节奏奖励（课程，逐渐减到 0）：学习者升本时，如果升之前的等级低于真人同一回合的平均等级 + 0.5，
    # 当场加 tempo_bonus（一个名次约 0.29）。v3 在竞技场里第 9 回合平均 2.4 本（真人 5.0），77 个回合里
    # 有 56 个回合钱够却不升；奖励在 tempo_anneal_iters 轮内线性减到 0，之后只靠名次判断升本值不值
    tempo_bonus: float = 0.0
    tempo_anneal_iters: int = 1500
    # 落后扣分（不减弱）：每次结束回合时，估计等级（1 + 本局升本次数）比目标曲线 - 0.5 低多少，就扣 tempo_deficit × 差值。
    # 第 3041–3770 轮：升本奖励减到 0 以后，第 5/7/9/11/13 回合等级从 1.9/2.7/3.4/4.3/5.3 退回 1.8/2.5/3.3/3.9/4.7，
    # 只靠名次学不会按时升本；这一项一直在，逼它跟上 6500+ 的节奏（第 5 回合 3.7 本、第 9 回合 5.3 本）
    # 第 6366–7237 轮（0.01）：第 2 回合升本率 0.95，但第 7/9/11 回合等级 3.6/4.0/4.9（高手 4.36/5.40/5.82），
    # 中期一直差一本多——第 4 回合起买随从不算“花掉升本的钱”，中期只剩这一项在推，太弱 → train.bat 用 0.03
    # 阵容测试（第 10350 轮，对高分阵容 256 局，平均名次 4.05）：第 8 / 9 回合升本率 31% / 26%（高手 64% / 36%），
    # 第 9 回合 4.46 本（高手 5.40）；对普通机器人第 8 回合升本 83%——被高分阵容压着时只买怪不升本 → 0.06
    tempo_deficit: float = 0.0
    # 模仿 6500+ 高分玩家（tools/make_pro_demos.py 从 Firestone 回放生成 runs/pro_demos/，约 10 万条决策）：
    # 有自己的系数、按类别权重和留出局指标（pro_top1 / pro_upgrade / pro_end …）
    pro_coef: float = 0.0
    pro_data: str = ""             # 空 = runs/pro_demos；几个目录用分号隔开（例如 "runs/pro_demos;runs/h13_demos"）
    pro_weights: str = ""          # 例如 "upgrade:3,end:2"
    pro_batch: int = 256
    # 无效刷新：刷新之后、下一次刷新或结束回合之前一个随从 / 法术都没买，就给那次刷新扣这么多
    # （竞技场 5 局：296 次刷新里 232 次之后什么都没买）
    refresh_waste_penalty: float = 0.0
    # 只在这一回合以前扣（含）。阵容测试（第 12370 轮）：第 8 / 9 回合刷新花 0.29 / 0.61 块，高手 1.70 / 2.62；
    # 金色 1.12 个（万三 1.85），第 10 回合起阵容强度落后——后期不刷就找不到三连和高本关键卡
    refresh_waste_until: int = 99
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
    # 从第几回合起买随从才按 starve_buy_weight 算（之前照常扣）。第 5900–6366 轮（买随从全程不扣）：第 2 回合
    # 升本率 0.45 → 0.06–0.14——模型学会了第 2 回合买随从来躲这一项（高手第 2 回合 86% 升本、只买 0.57 块）
    starve_buy_from_turn: int = 4
    # 黑暗之赐把升本的钱花掉时按多少倍扣。阵容测试（第 7920 轮，对高分阵容 64 局）：第 5 回合 97% 买黑暗之赐、
    # 升本率 20%（高手买黑暗之赐 22%、升本 67%），第 5 回合等级 2.56（高手 3.17）
    starve_gift_weight: float = 1.0
    # 回合末没花掉的钱：每剩 1 块扣这么多（第 2 回合起；高手几乎不剩，模型每回合剩 0.2–0.8 块）
    unspent_gold_penalty: float = 0.0
    # 阵容成型奖励（见 _comp_shape）：每次结束回合按场面配合程度加 comp_bonus × q（q 在 0–1）。0 = 不用
    comp_bonus: float = 0.0
    comp_from_turn: int = 6
    # 防止背下高分示范：每次考试比较训练示范和留出局的命中率，差距大于 pro_gap_cap 就把模仿高手的强度减半，
    # 差距小于 pro_gap_cap - 0.04 再加回来（最多到 pro_coef）。0 = 不控制。
    # 第 4351–5054 轮（pro_coef 0.1）：训练示范命中率 0.44 → 0.70，留出局 0.45 → 0.41，交叉熵 2.0 → 2.33——在背题
    pro_gap_cap: float = 0.0
    # 模仿高手强度最低降到多少倍（不让它完全消失）：第 5400–5770 轮降到 0.01 以后，第 2 回合升本率 0.47 → 0.24–0.40、
    # 第 5 回合等级 2.8 → 1.8–2.1，落后差距 0.18 → 0.37——升本节奏主要靠模仿撑着
    pro_scale_min: float = 0.25
    # 老师模型（bg.py teacher：只模仿高分示范、按留出局的交叉熵挑最好的那一版，存成 runs/teacher.pt）：
    # 在模型自己打出来的局面上加 KL(老师 ‖ 模型) × teacher_coef，开始用这个老师后的前 teacher_ramp 轮从 0 慢慢加上去
    # （记在 latest.pt 里，重启接着爬；换了老师 / --restore 从头爬）。
    # 和 pro_coef 的区别：pro_* 只在高手回放的那些局面上学，8.4M 参数背 10 万条——第 4450 轮起训练示范命中率
    # 0.47 → 0.70–0.82、留出局一直 0.37–0.42、交叉熵 2.0 → 2.4–2.9，模仿强度被 pro_gap_cap 压在最低 0.25；
    # 老师在模型自己的局面上给“高手这时候会怎么做”，背不了题。阵容测试（第 14140 轮）：高手抢着买的
    # 竞技表演者、安静的投递员、刷新畸体、圈套陷阱师、翩飞蝙蝠、狼宝宝倍率 0.0–0.3，第 9 回合卖随从 / 法术进账 2.3 块（高手 8.3）。
    # 空 / 0 = 不用
    teacher: str = ""
    teacher_coef: float = 0.0
    teacher_ramp: int = 200
    # 酒馆里有本局凑得成的 S / A 级目标阵容（rl/target_comps.json）的核心卡、或者正在做的 S / A 阵容的配套卡时，这一步的老师再乘这么多：
    # 该不该买那张核心卡是成型的关键一步，高手买卡里 16% 是 S / A 核心卡，模型 4%（第 14140 轮）。1 = 不加
    teacher_core_mult: float = 1.0
    # 老师怎么教：hard = 只在老师有把握（最想做的那个动作概率 ≥ teacher_conf）的局面上，让模型提高“做老师那个动作”的概率
    # （交叉熵，不逼模型把概率摊开）；kl = KL(老师 ‖ 模型)，把模型的整个概率分布往老师拉。
    # 第 14141–14240 轮用 kl：老师的分布本来就摊得开（只有 41% 猜中高手），系数才 0.005–0.015，熵 1.0 → 2.1、
    # 加强局承受的对手强度 0.69 → 0.32、考试 4.4 → 5.2，连续 3 次变乱退回——退回点本身已经乱了
    teacher_mode: str = "hard"
    # 体系感知辅助头（rl/model.py 的 comp_head）：预测这局最后成了哪一套 / 没成，交叉熵 × 这个系数。0 = 不训练这个头
    aux_comp_coef: float = 0.0
    # 别卖正在做的阵容里该留的牌（见 _keep_shape，rl/target_comps.json 的 playbook.keep）：每卖一张扣这么多。0 = 不扣
    keep_penalty: float = 0.0
    # 引擎联动（见 _engine_shape，rl/comp.ENGINE_RULES）：场上引擎卡配上燃料随从的程度涨多少加 engine_bonus × 涨的。0 = 不用
    engine_bonus: float = 0.0
    engine_from_turn: int = 5
    # “成型以后怎么玩”的课程（rl/calibrate.CompSeeder）：这么多比例的机器人局里，学习者在 seed_comp_turns 里随机一个回合
    # 招募开始时，场面换成同一回合高分阵容库里已经成型的 S / A 阵容（本局凑得成），之后自己接着打。0 = 不用。
    # 这些局不算进加强局的名次（不影响 boost_strength）和 place_bot，单独记 place_seeded
    seed_comp_frac: float = 0.0
    seed_comp_turns: str = "8,9"
    seed_comp_formed_from: int = 8     # 早于这一回合换的是“做到一半”的场面（核心卡过半、场上 2 个本族随从），学怎么做完
    teacher_conf: float = 0.5
    # 只学老师的某几类动作（hard 模式）：例如 "sell" = 只在老师有把握、而且它最想做的是“卖随从”的局面上学；可选
    # end / refresh / freeze / upgrade / buy / sell / play，逗号分开。空 = 全学。teacher_from_turn：早于这一回合的局面不学
    teacher_actions: str = ""
    teacher_from_turn: int = 0
    # 目标阵容各等级的权重（target_bonus 的 Φ），例如 "S:1,A:0.85,B:0.4"；空 = target_comps.json 里的 S 1 / A 0.8 / B 0.6
    target_weights: str = ""
    # 阵容成型奖励 q 里三项的权重：主种族纯度、万三常见搭配、关键卡（加起来是 1 就是满分 1）。
    # 阵容测试（第 11600 轮）：模型玩龙 / 海盗，万三那些搭配一对都用不上（0.02 对）；关键卡前四名的局 0.50 张、
    # 后四名 0.70 张，和名次反着走；纯度前四 0.66、后四 0.57 是对的方向
    comp_weights: str = "0.6,0.2,0.2"
    # 阵容强度奖励（rl/boardsim.py，见 _combat_shape）：结束回合时让引擎拿当前场面打 combat_boards 个同期高分阵容
    # （第 t … t-3 回合的，每个 combat_fights 场），胜率比这局上次量的时候涨了多少，就加 combat_bonus × 涨的。0 = 不用。
    # combat_frac：每次结束回合按这个概率量一次（一次约 2.5 毫秒，全量会让每轮多 9 秒左右）
    combat_bonus: float = 0.0
    combat_from_turn: int = 4
    combat_boards: int = 12
    combat_fights: int = 3
    combat_frac: float = 1.0
    # 后期阵容强度更值钱：第 combat_late_from 回合起 Φ 乘 combat_late_mult（势能 = 系数 × 胜率，仍是势能差）。
    # 第 14350 / 15180 轮阵容测试：第 3–9 回合模型的胜率和高手阵容一样甚至更高，第 12 / 13 回合 0.52 / 0.40（高手 0.67 / 0.60）。1 = 不加
    combat_late_from: int = 10
    combat_late_mult: float = 1.0
    # 养场面奖励（见 _grow_shape，第 19650 轮起）：第 grow_from_turn 回合起，学习者每回合第一次决策前用同一组同期高分阵容
    # （和 combat_* 同一把尺子）量一次场面强度，结束回合时再量一次，加 grow_bonus × (结束 - 开始)——这一回合的买、卖、
    # 打牌、法术、三连把场面养强了多少。不扣回、不和上回合比：对手每回合变强那一截不算模型的错，死得早也不会多拿。
    # grow_frac：每回合按这个概率量（一次开始 + 一次结束）。0 = 不用
    grow_bonus: float = 0.0
    grow_from_turn: int = 9
    grow_fights: int = 3
    grow_frac: float = 0.5
    # 经济奖励（见 _econ_shape）：结束回合时存下的资源（下回合额外金币 + 2 × 永久金币上限加成 + 手牌价值）每多 1 块加这么多，
    # 下回合钱到手 / 牌打出去时扣回来，一局结束时扣回最后的存款（纯势能差）。布莱恩·铜须让南海卖艺者、
    # 剃刀沼泽地卜师（鲜血宝石）、竞技表演者（发现法术）之类的战吼翻倍，“刷战吼牌倒转经济”在这里拿到当场的功劳
    econ_bonus: float = 0.0
    # 目标阵容奖励（见 _target_shape，rl/target_comps.json = bgbuddy.gg 的 S / A / B 级终局阵容核心卡）：
    # 结束回合时场面离目标阵容多近（等级权重 × 核心卡占比），涨多少加 target_bonus × 涨的，整局 = 最后的接近程度
    target_bonus: float = 0.0
    target_from_turn: int = 5
    # 模仿时把酒馆里随从的顺序随机打乱（买的那格跟着换）：酒馆顺序本来就不影响什么，背题的时候却会记住
    # “第几格买”——第 8000–11600 轮训练示范命中率 0.72 → 0.85，留出局一直 0.37
    demo_shop_shuffle: bool = False
    # 局末把势能型塑形（目标阵容 target_bonus、引擎联动 engine_bonus、阵容强度 combat_bonus）最后的 Φ 扣回来多少（0–1）。
    # 0 = 不扣（原来）：整局加起来 = 系数 × 最后的 Φ，等于在名次以外又多了一个“最后场面像不像阵容”的目标。
    # 第 16420–16850 轮这三项局末合计估计平均 0.7、最多 2.0（名次 1 → 8 一共也是 2.0）：tgt_q 0.07 → 0.14、eng_q 0 → 0.40，
    # 考试 4.44 → 4.75、加强局承受的对手强度 0.69 → 0.55——模型在拿“像阵容”的分，不是拿名次。
    # 1 = 纯势能差（整局和 = -第一次的 Φ，和动作无关）：做阵容那一步照样当场拿到功劳，但最后要靠名次兑现；
    # 前 shape_repay_ramp 轮从 0 慢慢加到这个数（价值头跟着改）
    shape_repay: float = 0.0
    shape_repay_ramp: int = 300
    # 名次奖励表（"1:1,2:0.55,…,8:-1"；空 = 引擎自带的线性 1 → -1，每差一名 0.29）。
    # 为什么（第 18611 轮起）：线性奖励下第 1 名和第 2 名只差 0.29，和第 4 / 5 名一样——稳定进前四、不冒险就是最优，
    # 正好是现在的打法：考试（全强度高分阵容）平均 4.2–4.6 名，吃鸡率只有 1.3–2.2%（8 个人平均 12.5%），
    # 第 12–14 回合场面 0.52 / 0.39 / 0.26（高手 0.67 / 0.60 / 0.68）。要吃鸡就得后期场面压过所有人，
    # 这一段线性奖励几乎不给额外的分。表换掉的只是局末那一下（引擎给的线性部分扣掉、换成表里的值），违规扣分照旧
    place_reward: str = ""
    # 送牌课程（rl/calibrate.CardGifts，第 20800 轮起）：gift_frac 的机器人局里，gift_turns 每回合招募开始时按 gift_p 的机会
    # 往学习者手里放一张“高手抢着买（rl/card_value.json 挑中倍率 ≥ gift_min_lift）、模型从来不买”的牌；纯加金币的法术不送。
    # 这些局不再换成型场面，名次单独记（place_gift），不算进 place_bot / 加强强度。0 = 不用
    gift_frac: float = 0.0
    gift_turns: str = "4,5,6,7,8,9,10,11,12"
    gift_p: float = 0.6
    gift_min_lift: float = 2.0
    # 送整套阵容（rl/calibrate.CompGifts，第 21800 轮起）：cgift_frac 的机器人局里，开局定一套本局凑得成的目标阵容，
    # cgift_turns 每回合招募开始时按 cgift_p 的机会把这套还没有的核心卡送一张到手里（齐了送配套卡）。
    # 这些局不再送零散的牌、不换成型场面，名次和“最后做成这套没有”单独记（place_cgift / cgift_formed）。0 = 不用
    cgift_frac: float = 0.0
    cgift_turns: str = "5,6,7,8,9,10,11"
    cgift_p: float = 0.7
    # 推演出学习目标（rl/lookahead.py，第 24150 轮起）：机器人局里第 look_from 回合起，每轮抽 look_per_iter 个学习者的决策，
    # 在副本上试模型最想走的 look_k 步 + 结束回合，各自让模型打完这回合（最多 look_depth 步），回合结束的场面打 look_boards
    # 个同期高分阵容 × look_fights 场；比模型首选强 look_margin 以上就换成那一步。答案存进最近 look_buffer 条的池子，
    # 每个小批另外抽 look_batch 条学交叉熵 look_coef × -log π(答案)（池子里有 look_min 条以后）。0 = 不用。
    # 推演只在训练里用，竞技场里的模型不需要模拟器。池子不存盘：重启后从头攒。
    look_coef: float = 0.0
    look_from: int = 9
    look_per_iter: int = 48
    look_k: int = 5
    look_depth: int = 6
    look_margin: float = 0.02
    look_boards: int = 8
    look_fights: int = 3
    look_buffer: int = 20000
    look_batch: int = 256
    look_min: int = 512
    # 扣回哪几项（target 目标阵容、engine 引擎联动、combat 阵容强度）。第 16045 轮起阵容强度不扣：它就是“结束回合时的场面打同期
    # 高分阵容的胜率”，和名次是一回事（comptest 前四名的局每回合都比后四名高）；扣掉以后后期养场面那几步几乎拿不到功劳
    # （离局末近，扣回那一下只打了 λ^k ≈ 0.8 的折扣），第 16040 轮第 12 / 13 / 14 回合阵容强度 0.52 / 0.38 / 0.28（高手 0.67 / 0.60 / 0.68）
    shape_repay_keys: str = "target,engine,combat"
    # 考试守护（三天无人看着跑用）：每次加强局考试（eval_every 轮一次、eval_games 局、全强度高分阵容）记下名次，
    # 最近 guard_window 次的平均比历史最好低（名次数字大）guard_margin 以上、连续 guard_patience 次 →
    # 换回最好的那一版参数（runs/<run>/best.pt，优化器重新开始），guard_keys 里的模仿 / 老师系数 × guard_decay。
    # 最多 guard_max 次，之后只记不改。平均变好时把当前参数存成 best.pt（竞技场 / comptest 用它）。
    # 拿第 12000–16850 轮的日志模拟（6 次平均、0.3、连续 3 次）：会在第 12500（变乱）、14700 / 15100（老师 0.02）、
    # 16700（引擎奖励）四个地方出手，正好是那几段变差的时候
    guard: bool = False
    guard_window: int = 6
    guard_margin: float = 0.3
    guard_patience: int = 3
    guard_decay: float = 0.7
    guard_max: int = 6
    guard_keys: str = "teacher_coef,pro_coef,look_coef"

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
    """一个可以出动作的策略：学习者或对手库里的快照。
    device 和模型所在的设备不同时（采样用 CPU、更新用 MPS / CUDA），在 device 上放一份副本，
    模型的版本号（_ver，训练器每次改参数时加 1）变了就同步一次。"""

    def __init__(self, model, device, amp=False, copy=False):
        self.src, self.device = model, device
        self.amp = amp and device.type == "cuda"
        self.bad_rows = 0                      # 输出出现 NaN 的局面数（应该一直是 0）
        if next(model.parameters()).device.type == device.type and not copy:
            self.model, self.ver = model, None
        else:
            import copy
            self.model = copy.deepcopy(model).to(device).eval()
            for q in self.model.parameters():
                q.requires_grad_(False)
            self.ver = getattr(model, "_ver", 0)

    def sync(self) -> None:
        if self.ver is not None and self.ver != getattr(self.src, "_ver", 0):
            with torch.no_grad():
                for a, b in zip(self.model.state_dict().values(), self.src.state_dict().values()):
                    a.copy_(b.detach().to(a.device))
            self.ver = getattr(self.src, "_ver", 0)

    @torch.inference_mode()
    def __call__(self, f, c, m, greedy=False, want_obs=False):
        """want_obs=True 时另外返回压缩后的观察（float16，存训练数据用：压缩后最大十几，不会溢出）。"""
        self.sync()
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


def parse_place_reward(spec: str):
    """"1:1,2:0.55,…,8:-1" → 长度 9 的数组（下标 = 名次，0 不用）；空 / 不全 → None（用引擎的线性奖励）。"""
    if not str(spec or "").strip():
        return None
    tab = np.full(9, np.nan)
    for part in str(spec).split(","):
        if ":" in part:
            k, v = part.split(":", 1)
            tab[int(k)] = float(v)
    if np.isnan(tab[1:]).any():
        raise ValueError(f"place_reward 要写全 1–8 名：{spec}")
    tab[0] = 0.0
    return tab


def linear_place_reward(p: int) -> float:
    """引擎自带的名次奖励：第 1 名 1，第 8 名 -1，线性。"""
    return 1.0 - 2.0 * (int(p) - 1) / 7.0


def seed_row(v) -> list:
    """成型阵容课程一套阵容的结果 [(名次, 加强局?)] → [局数, 平均名次, 吃鸡率, 其中加强局的局数, 平均名次, 吃鸡率]。"""
    ps = [int(p) for p, _ in v]
    bs = [int(p) for p, b in v if b]
    row = [len(ps), round(float(np.mean(ps)), 2), round(float(np.mean([p == 1 for p in ps])), 3), len(bs)]
    if bs:
        row += [round(float(np.mean(bs)), 2), round(float(np.mean([p == 1 for p in bs])), 3)]
    else:
        row += [0.0, 0.0]
    return row


def place_score_equiv(places, tab) -> float:
    """一组名次按奖励表算平均分，再换成“线性奖励下的等效名次”（4.5 - 3.5 × 平均分；线性表就是平均名次本身），
    考试守护拿它比，越小越好。"""
    if tab is None:
        return float(np.mean(places))
    return float(4.5 - 3.5 * np.mean([tab[int(p)] for p in places]))


TEACHER_FAMILIES = {"end": (0, 1), "refresh": (1, 2), "freeze": (2, 3), "upgrade": (3, 4), "buy": (8, 16), "sell": (16, 23),
                    "play": (46, 126)}


def teacher_action_mask(spec: str, n_actions: int):
    """"sell,buy" → [动作数] 的 0/1（这几类动作是 1）。空 → None（全学）。不认识的名字报错。"""
    names = [x.strip() for x in str(spec or "").split(",") if x.strip()]
    if not names:
        return None
    out = np.zeros(int(n_actions), dtype=np.float32)
    for nm in names:
        if nm not in TEACHER_FAMILIES:
            raise ValueError(f"teacher_actions 里不认识 {nm!r}，可选 {sorted(TEACHER_FAMILIES)}")
        a, b = TEACHER_FAMILIES[nm]
        out[a:b] = 1.0
    return out


def grow_opt_state(opt, od: dict) -> dict:
    """旧检查点的优化器状态比现在的模型少几个参数（第 15300 轮加了体系感知 comp_proj / comp_head，排在参数表最后）：
    把新参数的编号接到参数组后面（它们没有 Adam 状态，第一次更新时从 0 开始），旧参数的状态原样保留。
    数量一样时原样返回；新模型参数更少、或者有好几个参数组时也原样返回（让 load_state_dict 照常报错）。"""
    import copy
    groups = od.get("param_groups") or []
    if len(groups) != 1 or len(opt.param_groups) != 1:
        return od
    have, want = len(groups[0]["params"]), len(opt.param_groups[0]["params"])
    if have >= want:
        return od
    od = copy.deepcopy(od)
    ids = list(od["param_groups"][0]["params"])
    nxt = max(ids) + 1 if ids else 0
    od["param_groups"][0]["params"] = ids + list(range(nxt, nxt + want - have))
    print(f"优化器状态：检查点里 {have} 个参数，现在 {want} 个（新加的体系感知参数从 0 开始）", flush=True)
    return od


class Rollout:
    """学习者的决策按步成批存（numpy），名次奖励在对局结束时补到该座位最后一个决策上。
    槽位 = 全局座位编号（各环境的槽位依次排开）；ep = 这个槽位的第几局。"""

    def __init__(self, n_slots: int):
        self.n_slots = n_slots
        self.ep = np.zeros(n_slots, dtype=np.int64)
        self.place: dict[tuple, int] = {}
        self.comp: dict[tuple, int] = {}               # 局末成了哪一套（体系感知辅助头的标签）
        self.reset()

    def cut(self, slot: int):
        """成型阵容课程换场面之前调用：在这个座位最后一个决策后面切断 GAE。
        不切的话，换场面带来的价值跳变（课程局平均好将近 1 名）全算在换之前最后一步（上一回合的结束回合）头上——
        等于每回合早点结束回合、钱不花完也有奖励。切断后那一步用自己的价值自举（不换场面时下一步的价值≈这一步的），
        换之前那几步的阵容标签也不要（局末的阵容是换来的，不是它们做出来的）。"""
        i = int(self.last[slot])
        if i >= 0:
            self.cuts.add(i)
            self.seed_cut[(int(slot), int(self.ep[slot]))] = i

    def reset(self):
        self.parts = defaultdict(list)
        self.r: list[float] = []
        self.done: list[bool] = []
        self.n = 0
        self.last = np.full(self.n_slots, -1, dtype=np.int64)
        self.cuts: set[int] = set()
        self.seed_cut: dict[tuple, int] = {}

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

    def reward(self, slot, r, done, placement, comp=None):
        i = self.last[slot]
        if i >= 0:
            self.r[i] += float(r)
            if done:
                self.done[i] = True
        if done:
            self.place[(int(slot), int(self.ep[slot]))] = int(placement)
            if comp is not None:
                self.comp[(int(slot), int(self.ep[slot]))] = int(comp)
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
            elif i in self.cuts:                               # 换场面之前的最后一步：切断
                nv, last = float(v[i]), 0.0
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
        b["comp"] = np.asarray([self.comp.get((int(s), int(e)), -1) for s, e in zip(slot, ep)], dtype=np.int64)
        for (s_, e_), i_ in self.seed_cut.items():          # 换场面之前那几步不要阵容标签
            b["comp"][(slot == s_) & (ep == e_) & (np.arange(self.n) <= i_)] = -1
        self.place.clear()                 # 之后的决策只会属于还没结束的局，旧的名次用不到了
        self.comp.clear()
        self.reset()
        return b


class Trainer(ShapingMixin, ImitationMixin):
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
        print(f"设备：更新 {dname}，采样 {rname}，模拟器线程 {cfg.threads}", flush=True)
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
        self.amp = cfg.amp and self.dev.type == "cuda"
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
        self.cgift_by: dict[str, list] = {}            # 送整套阵容：阵容 → [(名次, 最后做成这套没有)]（上次考试以来）
        self.cgift_iter: list = []                     # 同上，这一轮的
        self.look_judge = None                         # 推演出学习目标（look_coef）：考官、答案池、抽样概率
        self.look_left, self.look_seen, self.look_p = 0, 0, 0.01
        self.look_rng = np.random.default_rng(cfg.seed + 977)
        if cfg.look_coef > 0 and cfg.games_bot:
            from .boardsim import BoardJudge
            from .lookahead import LookBuffer
            from paths import RUNS
            paths = [Path(p) for p in str(cfg.pro_boards).split(";") if p.strip()] or [RUNS / "pro_boards.json"]
            lib = merge_pro_boards(p for p in paths if p.exists())
            if lib:
                self.look_judge = BoardJudge(lib, k=cfg.look_boards, n=cfg.look_fights, seed=cfg.seed + 31)
                self.look_buf = LookBuffer(cfg.look_buffer, seed=cfg.seed)
                print(f"推演出学习目标：机器人局第 {cfg.look_from} 回合起每轮推演 {cfg.look_per_iter} 个决策（模型最想走的 "
                      f"{cfg.look_k} 步 + 结束回合，各打完这回合，场面打 {cfg.look_boards} 个同期高分阵容 × {cfg.look_fights} 场），"
                      f"答案池 {cfg.look_buffer} 条，系数 {cfg.look_coef}", flush=True)
            else:
                print("没有高分阵容库：推演出学习目标这次不用", flush=True)
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
        if self.look_judge is not None:               # 推演：按上一轮有多少个够格的决策定抽样概率，凑够 look_per_iter 个
            if self.look_seen:
                self.look_p = float(np.clip(1.3 * cfg.look_per_iter / self.look_seen, 1e-4, 1.0))
            self.look_left, self.look_seen = int(cfg.look_per_iter), 0
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
                        if name == "bot" and self.look_judge is not None and self.look_left > 0:
                            self._look(env, idx, f[o:o + k], c[o:o + k], m[o:o + k], res[3][o:o + k])
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
                    self._pro_lib = merge_pro_boards(have)     # 几个阵容库（分号隔开）合在一起
                    print(f"高分阵容库：{sum(len(v) for v in self._pro_lib.values())} 个整板（{path}）", flush=True)
                return ProBoardBooster(self._pro_lib, frac, cfg.boost_from_turn, seed=seed, every=cfg.boost_every)
            print(f"没有高分阵容库 {path}（先运行 bg.py prolib），改用放大攻防", flush=True)
        return BotBooster(load_real_profile(), frac, cfg.boost_from_turn, seed=seed, every=cfg.boost_every)

    # ---------------- 更新 ----------------

    def _bump(self) -> None:
        """模型参数变了：采样用的副本下次出动作前同步（Policy.sync）。"""
        self.model._ver = getattr(self.model, "_ver", 0) + 1

    def update(self, b):
        cfg = self.cfg
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
                        with torch.no_grad(), torch.autocast(self.dev.type, dtype=torch.bfloat16, enabled=self.amp):
                            tl = F.log_softmax(self.teacher(f, c, m, compressed=True)[0].float(), -1)
                        kl_rows = (tl.exp() * (tl - logp_all)).masked_fill(~m, 0).sum(-1)
                        kl_t = kl_rows.mean()
                        tw = self._teacher_rows(b["f"][j], b["c"][j])
                        if tw is not None:
                            stats["teacher_core"].append(float((tw > 1).mean()))
                        tw_t = t(tw, torch.float32) if tw is not None else torch.ones_like(kl_rows)
                        if getattr(cfg, "teacher_mode", "hard") == "kl":
                            loss = loss + tcoef * (kl_rows * tw_t).mean()
                        else:                                            # 只在老师有把握的局面，学老师最想做的那个动作
                            tp, ta = tl.max(-1)
                            sure = (tp.exp() >= cfg.teacher_conf).float()
                            want = teacher_action_mask(getattr(cfg, "teacher_actions", ""), logp_all.shape[1])
                            if want is not None:
                                sure = sure * t(want, torch.float32)[ta]
                            if getattr(cfg, "teacher_from_turn", 0) > 0:
                                sure = sure * (torch.round(f[:, 0].float() * 20) >= cfg.teacher_from_turn).float()
                            nll_t = -logp_all.gather(1, ta.unsqueeze(1)).squeeze(1).clamp(max=20)
                            loss = loss + tcoef * (nll_t * sure * tw_t).mean()
                            stats["teacher_sure"].append(float(sure.mean()))
                            if sure.any():
                                agree = (logp_all.argmax(-1) == ta).float()
                                stats["teacher_agree"].append(float((agree * sure).sum() / sure.sum()))
                        stats["kl_teacher"].append(kl_t.item())
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
                if self.look_judge is not None and cfg.look_coef > 0 and len(self.look_buf) >= cfg.look_min:
                    lf, lc, lm, la = self.look_buf.sample(cfg.look_batch)     # 推演答案的交叉熵（按条平均）
                    tt = lambda x, dt: torch.as_tensor(x, dtype=dt).to(self.dev, non_blocking=True)  # noqa: E731
                    la_t = tt(la, torch.long)
                    lp = self._demo_logp(tt(lf, torch.float32), tt(lc, torch.long), tt(lm, torch.bool), True)
                    lce = -lp.gather(1, la_t.unsqueeze(1)).squeeze(1).mean()
                    self.scaler.scale(cfg.look_coef * lce).backward()
                    stats["look_ce"].append(lce.item())
                    stats["look_agree"].append(float((lp.argmax(-1) == la_t).float().mean()))
                gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                if not torch.isfinite(gn):                # 梯度坏了：这一步不更新，记下来
                    stats["skipped"].append(1.0)
                    self.opt.zero_grad(set_to_none=True)
                    continue
                stats["skipped"].append(0.0)
                stats["grad_norm"].append(float(gn))
                self.opt.step()
                self._bump()
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

    def _drift(self, ent, noop) -> str:
        """策略慢慢变乱的检查（见 drift_ent / drift_noop）。返回问题描述，没有返回空串。"""
        cfg = self.cfg
        if not hasattr(self, "ent_hist"):
            self.ent_hist, self.drift_n = deque(maxlen=300), 0
        if ent is None or (cfg.drift_ent <= 0 and cfg.drift_noop <= 0 and getattr(cfg, "drift_ent_max", 0) <= 0):
            return ""
        base = float(np.median(self.ent_hist)) if len(self.ent_hist) >= 100 else None
        bad = []
        if cfg.drift_ent > 0 and base is not None and ent > base + cfg.drift_ent:
            bad.append(f"熵 {ent:.2f}（最近中位数 {base:.2f}）")
        if getattr(cfg, "drift_ent_max", 0) > 0 and ent > cfg.drift_ent_max:
            bad.append(f"熵 {ent:.2f}（上限 {cfg.drift_ent_max}）")
        if cfg.drift_noop > 0 and len(self.ent_hist) >= 100 and noop > cfg.drift_noop:
            bad.append(f"无效操作 {noop:.0%}")
        if not bad:
            self.drift_n = 0
            self.ent_hist.append(float(ent))
            return ""
        self.drift_n += 1
        self.mix["drift"] += 1
        if self.drift_n >= cfg.drift_iters:
            self.drift_n = 0
            return f"策略连续 {cfg.drift_iters} 轮变乱：" + "、".join(bad)
        return ""

    def restore(self, path):
        """接着训练目录练（对手库、加强强度、模仿强度都保留），但模型参数换成某个快照的、优化器重新开始，
        轮数退回快照那一轮，对手库里比它新的快照删掉（那是变坏以后的自己）。"""
        import re
        self.init_from(path)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr, eps=1e-5)
        m = re.search(r"it(\d+)", Path(path).name)
        if m:
            it = int(m.group(1))
            drop = [x for x in self.league.members if (re.search(r"it(\d+)", x.name or "") and
                                                      int(re.search(r"it(\d+)", x.name).group(1)) > it)]
            self.league.members = [x for x in self.league.members if x not in drop]
            self.iter = it
            print(f"退回第 {it} 轮的模型；对手库删掉 {len(drop)} 个更新的快照", flush=True)
        self.snap_cache.clear()
        self.active_snaps = []
        self.good = None
        self.teacher_t0 = None                         # 老师的作用从退回的那一轮重新慢慢加
        self.repay_t0 = None
        self.guard_st = self._guard_new()              # 考试守护从头记（best.pt 会被第一次凑满的平均覆盖）
        self._guard_apply()
        self._start_warmup()
        self._reset_drift()
        self.log_f.write(json.dumps({"iter": self.iter, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                                     "restored_from": str(path)}, ensure_ascii=False) + "\n")
        self.log_f.flush()

    # ---------------- 考试守护（guard_*） ----------------

    def _guard_keys(self) -> list[str]:
        return [k.strip() for k in str(getattr(self.cfg, "guard_keys", "") or "").split(",")
                if k.strip() and hasattr(self.cfg, k.strip())]

    @staticmethod
    def _guard_new() -> dict:
        return {"evals": [], "best": None, "best_iter": None, "bad": 0, "n": 0, "scale": 1.0, "metric": None}

    def _guard_metric(self) -> str:
        """守护比的是什么：平均名次（"place"），或者按名次奖励表算的等效名次（"score:<表>"）。"""
        spec = str(getattr(self.cfg, "place_reward", "") or "").replace(" ", "")
        return f"score:{spec}" if spec else "place"

    def _guard_check_metric(self):
        """换了名次奖励表：之前的最好成绩是按别的标准算的，不能比。best.pt 改名留着（best_<轮>.pt），守护从头记（缩放保留）。"""
        g, m = self.guard_st, self._guard_metric()
        if g.get("metric") in (None, m):
            g["metric"] = m
            return
        p = self.run / "best.pt"
        if p.exists():
            q = self.run / f"best_it{g.get('best_iter') or 0}.pt"
            p.replace(q)
            print(f"考试守护换了标准（{g.get('metric')} → {m}）：原来的 best.pt 改名为 {q.name}，从头记", flush=True)
        self.guard_st = {**self._guard_new(), "scale": g.get("scale", 1.0), "n": g.get("n", 0), "metric": m}

    def _guard_apply(self):
        """把守护的缩放（scale）乘到模仿 / 老师系数上（基准是启动时配方里的值）。"""
        sc = float(self.guard_st.get("scale", 1.0))
        for k, v in getattr(self, "guard_base", {}).items():
            setattr(self.cfg, k, v * sc)

    def _guard(self, x) -> dict:
        """每次加强局考试后调用。返回要写进日志的字段。"""
        cfg = self.cfg
        if not cfg.guard or x is None:
            return {}
        self._guard_check_metric()
        g = self.guard_st
        w = max(1, int(cfg.guard_window))
        g["evals"] = (g["evals"] + [float(x)])[-w:]
        out = {"guard_scale": round(float(g["scale"]), 3)}
        if len(g["evals"]) < w:
            return out
        m = float(np.mean(g["evals"]))
        out["guard_avg"] = round(m, 3)
        if g["best"] is None or m < g["best"]:
            g["best"], g["best_iter"], g["bad"] = m, int(self.iter), 0
            self._save_best(m)
            out["guard_best"] = round(m, 3)
            return out
        out["guard_ref"] = round(float(g["best"]), 3)
        if m < g["best"] + cfg.guard_margin:
            g["bad"] = 0
            return out
        g["bad"] += 1
        out["guard_bad"] = g["bad"]
        if g["bad"] < cfg.guard_patience:
            return out
        if g["n"] >= cfg.guard_max:
            out["guard_capped"] = True
            return out
        out["guard_fix"] = self._guard_fix(m)
        return out

    def _save_best(self, m: float):
        torch.save({"model": self.model.state_dict(), "model_kwargs": self.mkw, "iter": self.iter,
                    "guard": {"eval_avg": round(m, 3), "window": int(self.cfg.guard_window),
                              "time": time.strftime("%Y-%m-%d %H:%M:%S")}}, self.run / "best.pt")

    def _guard_fix(self, m: float) -> str:
        g, cfg = self.guard_st, self.cfg
        p = self.run / "best.pt"
        if not p.exists():
            g["bad"] = 0
            return "没有 best.pt，只记不改"
        sd = torch.load(p, map_location=self.dev, weights_only=False)["model"]
        self.model.load_state_dict(sd)
        self._bump()
        lr = self.opt.param_groups[0]["lr"]
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)
        self.snap_cache.clear()
        g["n"] += 1
        g["scale"] = float(g["scale"]) * float(cfg.guard_decay)
        g["evals"], g["bad"] = [], 0
        self._guard_apply()
        self._start_warmup()
        self._reset_drift()
        self.mark_good()
        msg = (f"第 {self.iter} 轮考试最近 {cfg.guard_window} 次平均 {m:.2f}，比最好的 {g['best']:.2f}（第 {g['best_iter']} 轮）"
               f"差 {cfg.guard_margin} 以上连续 {cfg.guard_patience} 次：换回 best.pt，"
               + "、".join(f"{k} → {getattr(cfg, k):.4g}" for k in self.guard_base) + f"（第 {g['n']} / {cfg.guard_max} 次）")
        print(msg, flush=True)
        return msg

    def mark_good(self):
        import copy
        self.good = (copy.deepcopy(self.model.state_dict()), copy.deepcopy(self.opt.state_dict()), self.iter)
        self.rollbacks = 0

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

    def _reset_drift(self):
        """换了一套参数（--restore、考试守护换回 best.pt）：变乱检查的“最近熵中位数”从头记。
        不清的话拿新参数的熵去比旧参数的中位数：第 17000 轮换回 best.pt（第 15450 轮那版熵 1.6）后，
        和之前的中位数 1.10 比一直算“变乱”，回退点又就是换回来的那版，连退 5 次训练停了。"""
        self.ent_hist, self.drift_n = deque(maxlen=300), 0

    def _start_warmup(self):
        """优化器刚重新开始：学习率从 10% 爬到 cfg.lr（warmup_iters 轮）。"""
        if getattr(self.cfg, "warmup_iters", 0) > 0:
            self.warm_t0 = int(self.iter)
            self._apply_warmup()

    def _apply_warmup(self):
        t0, n = getattr(self, "warm_t0", None), int(getattr(self.cfg, "warmup_iters", 0) or 0)
        if t0 is None or n <= 0:
            return
        k = (self.iter - t0) / n
        if k >= 1 or k < 0:
            self.warm_t0 = None
            k = 1.0
        for g in self.opt.param_groups:
            g["lr"] = self.cfg.lr * (0.1 + 0.9 * k)

    def _adapt_ent(self, ent) -> None:
        """熵对着 ent_target 调 ent_coef（见 Config.ent_target）。"""
        cfg = self.cfg
        if not getattr(cfg, "ent_target", 0) or ent is None:
            return
        f = (1 - cfg.ent_adapt) if ent > cfg.ent_target else (1 + cfg.ent_adapt)
        cfg.ent_coef = float(min(cfg.ent_coef_max, max(cfg.ent_coef_min, cfg.ent_coef * f)))

    def rollback(self, problem: str):
        self.rollbacks += 1
        self.warm_t0 = None
        print(f"第 {self.iter} 轮出问题：{problem}。回退到第 {self.good[2] if self.good else '?'} 轮，学习率减半。", flush=True)
        if self.rollbacks > self.cfg.max_rollbacks or self.good is None:
            self.log_f.write(json.dumps({"iter": self.iter, "stopped": problem}, ensure_ascii=False) + "\n")
            self.log_f.flush()
            raise RuntimeError(f"连续回退 {self.rollbacks} 次仍然出问题（{problem}），训练停止。把 log.jsonl 发给 Claude。")
        sd, od, it = self.good
        self.model.load_state_dict(sd)
        self._bump()
        self.opt.load_state_dict(od)
        for g in self.opt.param_groups:
            g["lr"] *= 0.5
        self.iter = it
        self.snap_cache.clear()
        if self.rollbacks >= 2 and "最近中位数" in problem and "上限" not in problem:
            # 连着第二次因为“熵比最近中位数高”回退：回退点自己的熵就在这个水平（比如换了名次奖励表以后，前 100 轮
            # 熵中位数还没攒够、检查没开，熵涨上去的那段也被当成了回退点），再退还是一样，第 5 次训练就停了。
            # 熵基准从头记（这 100 轮只看 drift_ent_max 上限），交给熵目标（ent_target）往回拉
            self._reset_drift()
            print("回退点本身的熵就在这个水平：变乱检查的熵基准从头记（只看上限），由熵目标往回拉", flush=True)

    def init_from(self, path):
        """用另一个检查点 / 快照的参数开始（不带优化器和对手库）。"""
        ck = torch.load(path, map_location=self.dev, weights_only=False)
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        if any(not torch.isfinite(v).all() for v in sd.values() if torch.is_tensor(v) and v.is_floating_point()):
            raise ValueError(f"{path} 里的参数有 NaN / inf，不能用")
        self.model.load_state_dict(sd)
        self._bump()
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
        self._bump()
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

    def _load_teacher(self):
        """老师模型（teacher / teacher_coef）：读进来、冻住。没有文件或格式不对就不用，训练照常。"""
        cfg = self.cfg
        if getattr(cfg, "teacher_coef", 0) <= 0 or not getattr(cfg, "teacher", ""):
            return None
        p = Path(str(cfg.teacher).replace("\\", "/"))
        if not p.exists():
            print(f"没有老师模型 {cfg.teacher}（先在面板“训练”里跑“老师模型”，约 15–25 分钟），这次不用", flush=True)
            return None
        from .model import load_model
        try:
            t, info = load_model(p, str(self.dev))
        except Exception as ex:                        # noqa: BLE001
            print(f"老师模型 {p} 读不了（{type(ex).__name__}: {ex}），这次不用", flush=True)
            return None
        if t.L.actions != self.model.L.actions:
            print(f"老师模型 {p} 和当前环境的格式不一样（TavernGym 换过版本？），重新跑一次面板里的“老师模型”", flush=True)
            return None
        for q in t.parameters():
            q.requires_grad_(False)
        if info.get("external"):                       # 外部模型（rl/extnet.py）：不 torch.load 它
            self.teacher_id = f"external:{p.name}:{int(info['mtime'])}"
            print(f"老师模型：外部模型 {p}（{info.get('license')}；只在本机用，不进 Git、不打包进竞技场）；"
                  f"系数 {cfg.teacher_coef}，前 {cfg.teacher_ramp} 轮慢慢加上，teacher_mode {cfg.teacher_mode}", flush=True)
            return t.eval()
        meta = torch.load(p, map_location="cpu", weights_only=False).get("teacher") or {}
        self.teacher_id = meta.get("created")
        val = meta.get("val") or {}
        print(f"老师模型：{p}（{meta.get('created', '?')}，留出局命中率 {val.get('pro_top1', '?')}、交叉熵 {val.get('pro_nll', '?')}）；"
              f"KL 系数 {cfg.teacher_coef}，前 {cfg.teacher_ramp} 轮慢慢加上", flush=True)
        return t.eval()

    @torch.inference_mode()
    def _logits_np(self, f, c, m) -> np.ndarray:
        """原始观察 → 当前模型的动作分数（推演里让模型自己打完这回合用）。"""
        f = soft_log(torch.as_tensor(np.ascontiguousarray(f), dtype=torch.float32).to(self.dev))
        c = torch.as_tensor(np.ascontiguousarray(c), dtype=torch.long).to(self.dev)
        m = torch.as_tensor(np.ascontiguousarray(m), dtype=torch.bool).to(self.dev)
        with torch.autocast(self.dev.type, dtype=torch.bfloat16, enabled=self.amp):
            lg = self.model(f, c, m, compressed=True)[0]
        return lg.float().cpu().numpy()

    def _look(self, env, idx, f, c, m, fz) -> None:
        """机器人局这一步里抽几个后期决策推演（见 Config.look_coef），答案存进池子。"""
        from tg.actions import action_family
        from .lookahead import search
        cfg = self.cfg
        t = np.rint(f[:, 0] * 20).astype(int)
        elig = np.flatnonzero((t >= cfg.look_from) & (np.asarray(m).sum(1) > 1))
        self.look_seen += len(elig)
        pick = elig[self.look_rng.random(len(elig)) < self.look_p][:self.look_left]
        if not len(pick):
            return
        t0 = time.time()
        L = self._logits_np(f[pick], c[pick], m[pick])
        for r, row in enumerate(pick):
            self.look_left -= 1
            g, seat = env.slot(int(idx[row]))
            try:
                best, a0, v0, vb = search(env.game(int(g)), int(seat), int(t[row]), m[row], L[r], self.look_judge,
                                          self._logits_np, cfg.look_k, cfg.look_depth, cfg.look_margin,
                                          rng_seed=int(self.look_rng.integers(1, 2 ** 31)))
            except BaseException as ex:                # noqa: BLE001 —— 引擎内部出错：这一个不算
                if not is_panic(ex) and not isinstance(ex, ValueError):
                    raise
                v0 = None
            if v0 is None:
                self.mix["look_fail"] += 1
                continue
            self.look_buf.add(fz[row], c[row], m[row], best)
            self.mix["look_n"] += 1
            if best != a0:
                self.mix["look_changed"] += 1
                self.mix["look_gain"] += float(vb - v0)
                self.mix["look_to_" + action_family(best)] += 1
        self.mix["t_look"] += time.time() - t0

    def _comp_name(self, k) -> str:
        """_final_comp 的序号 → 阵容名字（没成 → 空字符串）。"""
        if getattr(self, "_comp_names", None) is None:
            from .comp import load_target_comps
            self._comp_names = [c["name"] for c in load_target_comps()]
        return self._comp_names[int(k)] if k is not None and 0 <= int(k) < len(self._comp_names) else ""

    def _final_comp(self, s: int) -> int:
        """这一局（槽位 s）最后成了哪一套：rl/target_comps.json 里的序号，没成 = K_MAX（辅助头的“没成”类）。"""
        from .model import K_MAX
        ep = int(self.rollout.ep[s])
        k = int(self.fc_idx[s]) if getattr(self, "fc_ep", None) is not None and self.fc_ep[s] == ep else -1
        return k if 0 <= k < K_MAX else K_MAX

    def _teacher_rows(self, f, c):
        """每个局面的老师权重（体系选手）：先认出主选体系（S / A 里目标阵容得分最高、本局凑得成的那套）和 2 套备选；
        酒馆里有主选的核心卡 / 配套卡 → teacher_core_mult，有备选的 → (1 + teacher_core_mult) / 2，否则 1。
        还没有任何一套开头（前几回合）时不加权。不加权时返回 None。"""
        mult = float(getattr(self.cfg, "teacher_core_mult", 1.0))
        if mult == 1.0:
            return None
        if getattr(self, "_comp_tab", None) is None:
            from tg.catalog import base_of_index, n_cards
            from .comp import CompTables, load_target_comps
            self._comp_tab = CompTables(load_target_comps(), [base_of_index(i) if i else "" for i in range(n_cards())],
                                        tiers="SA")
        from tg.obs import lobby_bits
        tab = self._comp_tab
        main, backup = tab.main_backup(c, lobby_bits(f))
        hit_m = tab.shop_hit(c, main, tab.core) | tab.shop_hit(c, main, tab.supp)
        hit_b = tab.shop_hit(c, backup, tab.core) | tab.shop_hit(c, backup, tab.supp)
        hit_m |= Trainer._engine_hits(self, c)                  # 引擎联动：酒馆里有引擎（手里 / 场上已有 2 张燃料）或者场上有引擎、酒馆里有燃料
        return np.where(hit_m, mult, np.where(hit_b, (1.0 + mult) / 2, 1.0)).astype(np.float32)

    def _engine_hits(self, c) -> np.ndarray:
        """[行]：酒馆里有引擎卡、而且场上 + 手里已经有 ≥ 2 张它的燃料；或者场上有引擎、酒馆里有它的燃料（rl/comp.ENGINE_RULES）。"""
        from tg.obs import C_BOARD, C_HAND, C_SHOP, N_BOARD, N_HAND, N_SHOP
        tab = ShapingMixin._engine_tab(self)
        eng, fuel = tab["engine"], tab["fuel"]
        c = np.clip(np.asarray(c), 0, len(eng) - 1)
        board = c[:, C_BOARD:C_BOARD + N_BOARD]
        own = np.concatenate([board, c[:, C_HAND:C_HAND + N_HAND]], 1)
        shop = c[:, C_SHOP:C_SHOP + N_SHOP]
        nf = (fuel[own] & (own > 0)[:, :, None]).sum(1)
        on = (eng[board] & (board > 0)[:, :, None]).any(1)
        return ((eng[shop].any(1) & (nf >= 2)) | (on & fuel[shop].any(1))).any(1)

    def _teacher_coef(self) -> float:
        """这一轮的老师 KL 系数：每次启动后从 0 线性加到 teacher_coef（teacher_ramp 轮）。"""
        if getattr(self, "teacher", None) is None:
            return 0.0
        if self.teacher_t0 is None:
            self.teacher_t0 = self.iter
        ramp = max(1, int(self.cfg.teacher_ramp))
        return float(self.cfg.teacher_coef) * min(1.0, max(0.0, (self.iter - self.teacher_t0 + 1) / ramp))

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
            self._apply_warmup()
            t1 = time.time()
            upd = self.update(batch) if batch is not None else {}
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
            if mx.get("look_n") or mx.get("look_fail"):      # 推演出学习目标：推了几个、换了几个、换了以后场面强多少
                n = int(mx.get("look_n", 0))
                rec["look_n"] = n
                rec["look_fail"] = int(mx.get("look_fail", 0))
                rec["look_changed"] = round(mx.get("look_changed", 0) / max(1, n), 3)
                if mx.get("look_changed"):
                    rec["look_gain"] = round(mx["look_gain"] / mx["look_changed"], 4)
                    rec["look_to"] = {k[8:]: int(v) for k, v in mx.items() if k.startswith("look_to_")}
                rec["look_buf"] = len(self.look_buf)
                rec["t_look"] = round(mx.get("t_look", 0.0), 2)
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
