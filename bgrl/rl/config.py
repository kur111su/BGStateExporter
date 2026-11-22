"""训练的全部设置（Config）。配方文件 configs/v3.json 里的每一项对应这里的一个字段，说明写在字段旁边。"""
from __future__ import annotations

from dataclasses import dataclass

from .model import SIZES


@dataclass
class Config:
    run_dir: str = "runs/default"
    # 默认 mps（Mac 的 GPU）；没有 MPS 的机器（CI）自动改用 CPU。auto = 有 NVIDIA 显卡用 cuda、Apple 芯片用 mps、否则 cpu，跑过 bg.py bench-device 的按实测最快的选（rl/device.py）
    device: str = "mps"
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
    # 更新分段计时（t_data / t_fwd / t_bwd / t_bc / t_opt，秒/轮）：分段点要等显卡算完，会慢一点，只在测速时开
    profile_update: bool = False
    # 更新时每个小批里按“有内容的词数”排序再切成 micro：同一份里的行词数相近，打包截到的词少（Transformer 的计算和词数成正比，
    # 注意力和词数平方成正比）。梯度是各份加起来，和原来完全一样，只是算得快
    bucket_tokens: bool = True
    anchor_coef: float = 0.1
    anchor_iters: int = 100
    # 对手库 / 检查点
    snapshot_every: int = 50       # 每多少轮存一个快照进对手库
    profile_games: int = 16        # 测快照画像用多少局
    save_every: int = 10
    # 训练看板（tools/board.py）：每这么多轮在后台重写 runs/<run>/dashboard.md（0 = 不写）
    board_every: int = 0
    eval_every: int = 50
    eval_games: int = 128          # 64 局时平均名次的误差约 ±0.3，128 局约 ±0.2
    amp: bool = True               # 混合精度用 bfloat16（数值范围和 float32 一样，不会溢出；3070 支持）。只管 CUDA
    # Apple 芯片（MPS）上也用 bfloat16 混合精度：auto = 照 bench-device 实测（快才用），true / false 强制
    amp_mps: str = "auto"
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
    # 第二阶段：按真人对局校准机器人强度（rl/curricula/boost.py）。v2 的模型学会了一直待在 1–2 本，对弱机器人有效、
    # 对真人后期无效；机器人局里把机器人场面按真人画像放大，逼模型学会升本、做后期。0 = 关闭
    boost_frac: float = 0.0        # 多少比例的机器人局加强
    boost_from_turn: int = 7       # 从第几回合开始（前 6 回合机器人和真人差不多）
    boost_every: int = 2           # 每几回合加强一次（每次要整局存档读档，约 7 毫秒）
    boost_ramp_iters: int = 200    # 不自适应时：加强幅度从 30% 爬到 100% 用多少轮
    # 自适应（默认）：按学习者在加强局的平均名次调幅度，见 rl/curricula/boost.py 的 AdaptiveStrength
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
    # “成型以后怎么玩”的课程（rl/curricula/seed_comps.CompSeeder）：这么多比例的机器人局里，学习者在 seed_comp_turns 里随机一个回合
    # 招募开始时，场面换成同一回合高分阵容库里已经成型的 S / A 阵容（本局凑得成），之后自己接着打。0 = 不用。
    # 这些局不算进加强局的名次（不影响 boost_strength）和 place_bot，单独记 place_seeded
    seed_comp_frac: float = 0.0
    seed_comp_turns: str = "8,9"
    seed_comp_formed_from: int = 8     # 早于这一回合换的是“做到一半”的场面（核心卡过半、场上 2 个本族随从），学怎么做完
    teacher_conf: float = 0.5
    # 只学老师的某几类动作（hard 模式）：例如 "sell" = 只在老师有把握、而且它最想做的是“卖随从”的局面上学；可选
    # end / refresh / freeze / upgrade / buy / sell / play，逗号分开。空 = 全学。teacher_from_turn：早于这一回合的局面不学
    teacher_chunk: int = 512       # 老师一次算多少个局面（小 = 省内存；结果一样）
    # 每轮只让老师算这么大比例的局面（随机抽；损失按比例放大，力度不变）。老师一个局面要算 125 个词，全算很慢
    teacher_frac: float = 1.0
    teacher_actions: str = ""
    teacher_from_turn: int = 0
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
    # 模仿时把酒馆里随从的顺序随机打乱（买的那格跟着换）：酒馆顺序本来就不影响什么，背题的时候却会记住
    # “第几格买”——第 8000–11600 轮训练示范命中率 0.72 → 0.85，留出局一直 0.37
    demo_shop_shuffle: bool = False
    # 局末把阵容强度（combat_bonus）最后的 Φ 扣回来多少（0–1；第 33328 轮以前还有目标阵容、引擎联动两项）。
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
    # 送牌课程（rl/curricula/gifts.CardGifts，第 20800 轮起）：gift_frac 的机器人局里，gift_turns 每回合招募开始时按 gift_p 的机会
    # 往学习者手里放一张“高手抢着买（rl/card_value.json 挑中倍率 ≥ gift_min_lift）、模型从来不买”的牌；纯加金币的法术不送。
    # 这些局不再换成型场面，名次单独记（place_gift），不算进 place_bot / 加强强度。0 = 不用
    gift_frac: float = 0.0
    gift_turns: str = "4,5,6,7,8,9,10,11,12"
    gift_p: float = 0.6
    gift_min_lift: float = 2.0
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
    guard_keys: str = "teacher_coef,pro_coef"

    # 指定酒馆种族（rl/curricula/lobbies.py；单种族专家 configs/exp/beast.json 用）：新开的一局按 lobby_frac 的概率
    # 换成必开 lobby_must（逗号隔开）的一局，其余种族从 lobby_pool 里抽（空 = 野猪人 / 恶魔 / 元素以外的 7 个）。
    # 对机器人局和联赛都生效；lobby_exam 打开时考试也在这种酒馆里考。0 = 不用
    lobby_must: str = ""
    lobby_pool: str = ""
    lobby_frac: float = 0.0
    lobby_exam: bool = False

    def rules(self) -> dict:
        caps = [int(x) for x in str(self.turn_caps).replace(" ", "").split(",") if x]
        return {"turn_action_caps": caps} if caps else {}

    def model_kwargs(self) -> dict:
        kw = dict(SIZES[self.size])
        for k in ("d", "layers", "heads", "dc"):
            if getattr(self, k):
                kw[k] = getattr(self, k)
        return kw
