# bgrl：TavernGym 训练 + 影子模式 + 竞技场

在 TavernGym（MCG-NJU/AutoTavern，MIT，锁定的版本见 `taverngym/PIN.txt`）里用 PPO 训练一个实体 Transformer 策略。建议完全来自这个模型：影子模式把它的本回合方案显示在 http://localhost:8766/（插件 v0.10 起不再有自己的搜索建议器和网页面板）。影子模式只显示，不操作游戏；竞技场（TavernArena）只在官方服务器上的模拟对局里出动作。

需要 Python 3.10～3.13（推荐 3.12，安装时勾选 “Add python.exe to PATH”）和 NVIDIA 显卡驱动。

## 双击入口（13 个 .bat）

| 文件 | 做什么 | 结果在哪 |
| --- | --- | --- |
| `setup.bat` | 第一次用、或更新后运行：建 `.venv`，装 numpy、TavernGym 安装包、PyTorch（CUDA），跑测试和几轮极小训练。`setup.bat bench` 另外测三种模型大小的训练速度 | `reports\setup.txt` |
| `train.bat` | 训练 `runs\v3`（设置全在 `configs\v3.json`，见下面“训练配方”），关掉窗口再开会接着练；`runs\v3` 还没有模型时从 `runs\v2\latest.pt` 开始。每次启动先按新对局重建 `runs\demos.npz`。命令行再写的参数优先：`train.bat --threads 4` | `runs\v3\log.jsonl`、`latest.pt`、`snapshots\` |
| `shadow.bat` | 影子模式 http://localhost:8766/：模型的本回合方案、预计名次，加上插件的本场胜率、对位胜率、站位优化按钮。换局时自动换上最新模型。`shadow.bat --ckpt 某个快照.pt` 用指定模型 | `reports\shadow\日期.jsonl` |
| `goldflow.bat` | 前几回合的金币流向：模型（64 局对脚本机器人）和 6500+ 玩家并排比，每回合花在升本 / 买 / 刷新 / 发动 / 黑暗之赐上的钱、升本率。`--games 128`、`--ckpt …` | 屏幕；`runs\v3\goldflow.json` |
| `comptest.bat` | 阵容测试：同上，但对手是打高分真人阵容的机器人（`goldflow --boosted`），另外比最终阵容（主种族、常见搭配、关键卡、金色）、挑卡（关键卡 / 经济卡 / 引擎实测关键卡）和每回合的阵容强度（场面打同期高分阵容的胜率，前四 / 后四 / 高分阵容自己） | 屏幕；`runs\v3\comptest.json` |
| `comptest_seeded.bat` | 拿着成型阵容的阵容测试：和 `comptest.bat` 一样对全强度高分阵容打 256 局，但每局第 8 / 9 / 10 回合把模型的场面换成高手同一回合已经成型、本局凑得成的 S / A 阵容，之后模型自己打（`goldflow --boosted --seed-comps 8,9,10`）：按阵容列平均名次 / 吃鸡 / 前四，加上每回合阵容强度——模型拿到成型阵容以后能不能把场面养起来、吃鸡。两个测试都能加 `--late-ckpt runs\teacher.pt`（`--late-from 10`）：第 10 回合起换老师模型打，看后期换成模仿高手的打法场面能不能养起来（结果文件名后加 `_late`） | 屏幕；`runs\v3\comptest_seeded.json` |
| `comptest_lookahead.bat` | 推演测试：同一个模型、同一批 128 局阵容测试打两遍——先正常打，再从第 9 回合起每个决策先在这局的副本上（随机数换过，看不到真实的下一个酒馆）试模型最想走的 6 步 + 结束回合，各让模型打完这回合，拿 12 个同期高分阵容实打结束时的场面，选最强的那一步（比模型自己的选择强 0.02 以上才换）；最后并排打印名次、吃鸡、第 9–14 回合场面强度和养了多少。推演明显更强，就拿它当训练里的后期老师、影子模式的后期建议 | 屏幕；`runs\v3\comptest.json`、`comptest_look.json` |
| `prodemos.bat` | 从 `data\firestone` 里的高分对局生成决策样本：Firestone 6500+ 分享包的 zip（不用解压）、单局 HSReplay 回放（`.xml` / `.xml.zip`）、HDT 的 AI 复盘包（`*复盘包.zip` 或套着复盘包的 zip），可以混着放、可以放子文件夹；每次全部重新生成。约 10–20 分钟（加了选择示范后，慢一点的电脑 30–60 分钟） | `runs\pro_demos\`（只在本机） |
| `teacher.bat` | 老师模型：只模仿高分示范（`runs\pro_demos` + `runs\h13_demos`），从头学，留出局交叉熵最好的那一版存下来；之后 `train.bat` 自动用它（先关掉 train.bat；约 15–25 分钟） | `runs\teacher.pt`（只在本机） |
| `replay.bat` | 回放检查：把你对局里的每一步放进 TavernGym 重做，按卡统计不一致。`replay.bat --last 5` 只看最近 5 局 | `reports\a5_replay.txt` / `.json` |
| `tavernbench.bat` | 用 TavernBench（论文的 368 题评测集）测当前模型，和 TavernAgent、大语言模型、Jev 比。只用 CPU，训练开着也行 | `reports\tavernbench\<模型>\report.txt` |
| `arena.bat` | 竞技场：`check` 打包当前模型并核对竞技场契约、`practice` 一局不计分练习、`match` 一局计分对局。`check --ckpt …` 打包别的模型 | `arena\model_bundle\`、`arena\results\` |
| `arena_practice.bat` | 双击 = `arena.bat practice` | `arena\results\` |
| `arena_match.bat` | 双击 = `arena.bat match`（一局计分对局后停） | `arena\results\` |
| `arena_loop.bat` | 一局接一局打计分对局，直到关窗口或 Ctrl+C；`arena_loop.bat update` 每局前模型比打包的新 2 小时以上就重新打包 | `arena\results\history.jsonl`、`arena\records\` |
| `arena_view.bat` | 竞技场对局查看器 http://127.0.0.1:8767/：战绩、Elo、名次分布、每局每一步的选择和候选概率、正在打的对局 | — |

竞技场上的模型名是 `capri-it<训练轮数>`（`arena/export.py` 的 `NAME`，打包时写进 `arena\model_bundle\meta.json`，连接器用 `--model-version` 发出去）。

竞技场的 Agent Key：在窗口里设 `ARENA_API_KEY`，或者单独写进 `arena\key.txt`（不进 Git）。它不会被打印、记日志或出现在命令行上；连接器只上传动作，模型、代码、存档都不上传。连接器跑在单独的 `arena\.venv`（不装 TavernGym，`arena.bat check` 会建）。

## 命令行（`bg.py`）

所有入口都是 `bg.py` 的子命令（`.venv\Scripts\python bg.py <命令> --help` 看参数）：

| 命令 | 实现 | 说明 |
| --- | --- | --- |
| `train` | `tools/train.py` | `--recipe configs/v3.json` 按配方训练、`--run 目录 --resume` 接着练、`--init 检查点` 从别的模型开始、`--smoke` 冒烟、`--bench --size medium` 测速；其余参数和 `rl/ppo.py` 的 `Config` 一一对应，优先于配方 |
| `shadow` | `tools/shadow.py` | 影子模式服务 |
| `shadow-eval` | `tools/shadow_eval.py` | 影子模式评估：按模型版本统计模型方案和你实际操作对得上多少（同类 / 同一张卡 / 在方案里） → `reports\shadow_eval.json` |
| `goldflow` | `tools/goldflow.py` | 金币流向和阵容测试（`--boosted` 对高分阵容；`--pro data\firestone` 从回放重算高手那一列写进 `rl/pro_profile.json`） |
| `demos` | `tools/make_demos.py` | 你的对局记录 → 示范数据 `runs\demos.npz`（每步操作前的局面 + 你的动作，每回合末尾加“结束回合”；每 5 局留 1 局验证）。`--if-stale` 有新对局才重建 |
| `prolib` | `tools/pro_library.py` | Firestone 分享包的 `core/games.jsonl` → 高分阵容库 `runs\pro_boards.json`（只在本机）和汇总曲线 `rl/pro_profile.json` |
| `prodemos` | `tools/make_pro_demos.py` | Firestone 回放 → 6500+ 决策样本 `runs\pro_demos\`。回放 XML 由 `tg/hsreplay.py`、HDT 复盘包由 `tg/review_pack.py`（每步操作前后的局面；没有排行榜 / 对手阵容 / 英雄技能实体，升本费用按那一步的金币差）读成插件格式记录，之后和你的对局走同一条路（切动作 → TavernGym 重建 → 观察 + 动作编号） |
| `nativedemos` | `tools/native_demos.py` | 已经是 TavernGym 观察格式的别人对局（例如万三的 `human_native_v0.npz`）→ 决策样本 `runs\h13_demos\`、对手阵容库 `runs\h13_boards.json`、汇总 `rl/h13_profile.json`（升本节奏、最终阵容、常见搭配，只有统计数字） |
| `cardvalue` | `tools/cardvalue.py` | 高分示范里每次买卡时的选项 → 每张卡的挑中倍率、关键卡、经济卡、高手的买法比例 `rl/card_value.json`（只有统计数字） |
| `simcards` | `tools/simcards.py` | 引擎实测的卡牌价值和配合：高分阵容库里的场面在 TavernGym 里拿掉一张 / 两张卡再打同期高分阵容，看胜率掉多少 → `rl/sim_value.json`（关键卡、好搭配，只有统计数字；约 2–3 分钟） |
| `teacher` | `tools/teacher.py` | 老师模型（见上面 `teacher.bat`）：`--init` 从某个检查点开始（默认从头学：从训练中的模型开始时它已经背过示范，再学只会更差）、`--steps`、`--batch`、`--patience` |
| `playbook` | `tools/playbook.py` | 目标阵容的配套卡：高手正在做某一套（核心卡过半、本族 ≥ 2）时比平时多买 1.5 倍以上的卡 → `rl/target_comps.json` 的 `support`（只有统计数字；几秒） |
| `replay` | `tools/tg_replay.py` | A5 回放检查；不给文件时读插件的 games 目录 |
| `tavernbench` | `tools/tavernbench.py` | C3 TavernBench 评测；`--ckpt` 指定模型、`--smoke` 冒烟 |
| `bridge` | `tools/tg_bridge_check.py` | A4 局面重建检查 |
| `profile` | `tools/opp_profile.py` | B1 对手画像：真人对局 vs 脚本机器人（`--save` 更新 `rl/real_profile.json`） |
| `bench` | `tools/tg_bench.py` | A1 TavernGym 测速 |
| `test` | `tests/test_*.py` | 单元测试 |
| `info` | — | 环境、对局记录数、训练进度 |

竞技场的几个脚本不走 `bg.py`：`arena/export.py`（在训练环境里打包模型）、`arena/prep.py`（核对契约，生成卡牌编号对应表）、`arena/arena_client.py`（官方连接器）、`arena/agent.py`（连接器调用的策略）、`arena/viewer.py`（查看器），都由上面的 `arena*.bat` 调用。

## 训练配方

`configs/v3.json` 是 `train.bat` 的全部训练设置：`run`（训练目录）、`config`（`rl/ppo.py` 的 `Config` 字段和值）、`notes`（每组设置是干什么的）。优先级：`Config` 默认值 < 配方 < 命令行。要试别的设置，复制一份改名（例如 `configs/v4.json`，把 `run` 改成 `runs\v4`），然后 `bg.py train --recipe configs/v4.json --resume`。`tests/test_recipe.py` 检查配方里的字段都存在，并且 `train.bat` 得到的设置和改用配方之前那串命令行参数完全一样。

## 模块和依赖方向

依赖只往一个方向：`tg` ← `rl` ← `tools`（`tests/test_layout.py` 检查 `rl`、`tg` 不导入 `tools`，`tg` 不导入 `rl`）。`arena/agent.py` 按文件路径单独加载 `rl/model.py`，不依赖其它模块。

```
paths.py          路径和端口（runs/、reports/、默认训练目录 runs/v3、插件数据目录、8765 / 8766）
configs/v3.json   训练配方（train.bat）

tg/               TavernGym 桥接（不需要 torch）
  actions.py      245 个动作的编号、动作类别 action_family、引擎出错判断 is_panic
  obs.py          观察向量布局：回合 / 金币 / 等级 / 升本费的位置和缩放，场上 / 手牌 / 酒馆 / 选项切片
  catalog.py      卡牌目录：编号 → 卡牌 ID、金色 _G 处理（全项目一份缓存）
  logs.py         读插件对局记录：统一标签名、皮肤 → 英雄、切分动作；GameHistory（回放和影子模式共用）
  bridge.py       插件局面 → TavernGym 局面（随从、附魔、加成、饰品、黑暗之赐次数、对手阵容）
  replay.py       A5：执行你的动作并和记录比较
  hsreplay.py     回放 XML → 插件格式对局记录（按插件 StateReader.cs 的规则）
  supermask.py    没有合法动作表的数据：从观察推一个合法动作超集
  gametags.json、skins.json   标签名、不按后缀命名的英雄皮肤

rl/               模型和训练
  model.py        实体 Transformer：指针式动作头 + 价值头 + 名次头（small / medium / large）
  ppo.py          训练主流程（Trainer）：批量环境采样、GAE、PPO 更新、对手库快照、评估、健康检查和自动回滚、检查点
  shaping.py      奖励塑形（Trainer 继承）：升本节奏、金币流向、无效刷新、阵容成型，每个座位的塑形状态
  imitation.py    模仿学习（Trainer 继承）：读示范、按类别 / 回合 / 卡牌加权、损失、命中率、模仿强度自动调
  demos.py        示范数据格式：动作类别 KINDS、读写单文件和分片、soft_log、按买的卡挑示范（bought_in）
  comp.py         最终阵容打分：主种族纯度、常见搭配、关键卡（训练、goldflow、nativedemos 共用）；读 sim_value.json
  boardsim.py     阵容强度：用引擎的战斗模拟（combat_preview）量场面打同期高分阵容的胜率（阵容强度奖励、simcards、comptest 用）
  league.py       对手库：快照按“和真人强度画像的距离”加权；考试对手
  calibrate.py    机器人加强：按真人画像放大攻防（BotBooster）或换成高分真人阵容（ProBoardBooster），强度自适应
  advisor.py      影子模式的建议器：贪心走完本回合、跳过无效动作、第一步备选、预计名次
  *.json          只有汇总统计的参照数据：real_profile（你的对手画像）、pro_profile（6500+ 曲线）、
                  h13_profile（万三阵容和搭配）、card_value（关键卡、经济卡）、sim_value（引擎实测的关键卡和配合）

tools/            bg.py 各命令的实现（见上表）；make_fixtures.py、make_replay_fixture.py 生成 CI 样本
arena/            竞技场：export / prep / agent / arena_client / viewer
tests/            单元测试；fixtures/ 是 CI 用的局面、回放样本
```

## 数据流

```
插件 games\*.jsonl（你的对局）──▶ demos ──▶ runs\demos.npz ─────────────┐
                              ├─▶ replay / bridge（检查模拟器）          │
                              └─▶ profile ──▶ rl\real_profile.json ──┐   │
Firestone 6500+ 分享包 ──▶ prolib ──▶ runs\pro_boards.json、rl\pro_profile.json
                    └──▶ prodemos ──▶ runs\pro_demos\ ──────────────────┤
万三对局 .npz ──▶ nativedemos ──▶ runs\h13_demos\、runs\h13_boards.json、rl\h13_profile.json
高分示范 ──▶ cardvalue ──▶ rl\card_value.json                           │
                                                                       ▼
                     configs\v3.json ──▶ train（rl/ppo.py + shaping + imitation）──▶ runs\v3\latest.pt、snapshots\
                                                                       │
          ┌────────────────────────┬───────────────────────┬──────────┴──────────┐
          ▼                        ▼                       ▼                     ▼
  shadow（:8766，只显示）   goldflow / comptest     tavernbench          arena（export → 竞技场）
```

`runs\`、`data\`、`arena\model_bundle\`、`arena\key.txt` 都不进 Git：别人的对局（Firestone、万三）生成的样本和阵容库只在本机，仓库里只有汇总数字（`rl/*.json`）。

## CI

- `.github/workflows/taverngym.yml`：改 `taverngym/PIN.txt` 后编译 Windows / Linux 安装包，发布为预发布 `taverngym-<commit>`（插件自动更新不会取到）。
- `.github/workflows/bgrl-tg.yml`：A4 重建检查、A5 回放检查（用 `tests/fixtures` 里的样本）。
- `.github/workflows/bgrl-rl.yml`：单元测试（`bg.py test`）、TavernBench 冒烟、对手画像、冒烟训练（默认、加强对手、按配方、模仿你的操作、模仿高分玩家），CPU。`bgrl/` 下 rl、tg、tools、configs、tests、arena 有改动就跑。

样本更新：`tools/make_fixtures.py`（A4）、`tools/make_replay_fixture.py`（A5）从新的对局记录生成。

## 训练阶段

| 阶段 | 目录 | 对手 / 改动 | 看什么 |
| --- | --- | --- | --- |
| 第一阶段 | `runs\v2` | 脚本机器人 + 自己的旧版本 | 考试（对普通机器人）从 4.27 名到约 1.8 名；但学会了一直待在 1–2 本（第 13 回合平均 2.8 本，真人约 5.8 本） |
| 第二阶段 | `runs\v3` | 75% 的机器人局里机器人按真人强度加强 + 自己的旧版本；加强幅度按加强局名次自适应 | `boost_strength`、`place_boost`、`eval_boost_place`、`snapshot_tier` |
| + 模仿你的操作 | 第 2338 轮起 | 每个小批抽你的操作算交叉熵（`bc_coef`），训练示范命中率太高时自动减弱 | `human_top1`、`human_upgrade` / `human_end` |
| + 高分阵容 | 第 3048 轮起 | 机器人场面换成同一回合 6500+ 对局的真实阵容（`boost_kind pro`）；升本节奏按 6500+ 曲线（`tempo_curve pro`） | `boost_strength`、`eval_boost_place`、`snapshot_tier` 对比 `rl/pro_profile.json` |
| + 模仿高分玩家 | 第 4359 轮起 | 抽 6500+ 玩家的决策算交叉熵（`pro_coef 0.1`）；落后于等级曲线扣分（`tempo_deficit`） | `pro_top1`、`pro_upgrade` / `pro_end`、`tier_gap` |
| + 金币流向、万三 | 约第 5000 轮起（万三样本第 5100 轮起） | 落后时把升本的钱花在别处扣分（`upgrade_starve_penalty`）、剩钱扣分；加万三样本和阵容库；模仿强度按训练 / 留出命中率差自动调（`pro_gap_cap`） | `up_afford`、`up_t2`、`starve_turn`、`end_rich`、`unspent`、`pro_scale` |
| + 阵容成型 | 第 7950 轮起 | 买关键卡 / 经济卡 / 同种族 / 常见搭配的示范加权（`pro_weights`）；第 8661 轮起第 6 回合后结束回合按场面成型程度加分（`comp_bonus`，现为 0.08） | `comp_q`、`comp_coh`、`comp_syn`、`comp_key`，以及 `comptest.bat` |
| + 阵容强度 | 第 11600 轮起 | 结束回合时引擎拿场面打 12 个同期高分阵容，胜率涨多少加多少（`combat_bonus 0.3`，势能差）；阵容成型奖励只留主种族纯度（`comp_bonus 0.03`）；模仿时打乱酒馆顺序防背题（`demo_shop_shuffle`）；模仿权重加引擎实测关键卡 / 好搭配（`simkey`、`simsyn`）和引擎卡（`engine`：有布莱恩·铜须时买战吼随从）；经济奖励（`econ_bonus 0.01`：存下的下回合金币，布莱恩让南海卖艺者翻倍） | `cs_wr`、`cs_late`、`t_combat`、`econ`、`pro_top1` 和 `pro_top1_train` 的差距，以及 `comptest.bat` 的阵容强度表 |
| + 学得快一点、后期多刷 | 第 12370 轮起 | 学习率 1e-4 → 2e-4（每轮 KL 只有 0.002，第 5000 轮起成绩不动）；无效刷新只在第 6 回合以前扣（`refresh_waste_until`）；模仿权重加买引擎卡本身（`engcard`）、凑三连（`triple`） | `lr`、`kl`、`boost_strength`、`eval_boost_place`、`ref_turn`、`comp_golden`，以及 `comptest.bat` 的金色数、第 8–9 回合刷新、第 10–13 回合阵容强度 |
| 退回 + 变乱检查 | 第 12700 轮 → 退回第 12450 轮 | 第 12490 轮起策略变乱（熵 1.0 → 1.9、无效操作 5% → 20–39%、白刷新 35–40%，考试 4.1 → 4.6–4.9）：后期不扣刷新退回去，学习率 1.5e-4；新增变乱检查（`drift_ent`、`drift_noop`，连续 10 轮自动退回、学习率减半）；`train.bat --restore runs\v3\snapshots\it12450.pt` 把模型退回快照、对手库里更新的快照删掉 | `ent`、`noop_frac`、`ref_waste`、`drift`、`boost_strength`、`eval_boost_place` |
| + 目标阵容 | 第 12635 轮起 | `rl/target_comps.json`：bgbuddy.gg（补丁 255790）S / A / B 级 23 套终局阵容的核心卡。结束回合时离目标阵容多近（S 1 / A 0.8 / B 0.6 × 核心卡占比）涨多少加多少（`target_bonus 0.15`，整局 = 最后的接近程度）；模仿时买 S / A / B 核心卡 ×2 / ×1.5 / ×1.2、往同一套成型 ×1.5（`tgtS`、`tgtA`、`tgtB`、`tgtfit`）；comptest 统计最终阵容最像哪一套 | `tgt_q`，以及 `comptest.bat` 的“目标阵容”一行 |
| + 选择示范 + 老师模型 | 第 12630 / 14140 轮起 | 高分示范里加上“选择”（打出 / 施法之后选目标、发现：每个选项在引擎里试，和回放结果比出高手选的，14608 条；法术 / 宝石打给吃法术的随从再 ×3，`choose`、`spelltgt`）。老师模型（`teacher.bat` → `runs\teacher.pt`）：只模仿高分示范、在留出局交叉熵最好时停下的模型；训练时在模型自己的局面上加 KL(老师 ‖ 模型) × `teacher_coef 0.03`（每次启动后 200 轮慢慢加上）——直接模仿高手从第 4450 轮起在背题（训练示范命中率 0.7–0.8、留出局 0.37–0.42，强度一直压在最低 0.25） | `pro_choose`、`kl_teacher`、`teacher_coef`、`tgt_q`，comptest 的挑卡倍率和卖入 |
| + 必须学会 S / A 阵容 | 第 14140 轮起 | 目标阵容只算本局凑得成的（核心卡的种族本局开了，观察里的 `tg/obs.F_LOBBY`；每局平均 10 套 S / A 凑得成）；`target_bonus` 0.15 → 0.3，等级权重 S 1 / A 0.85 / B 0.4（`target_weights`）；酒馆里有凑得成的 S / A 核心卡时老师 KL ×3（`teacher_core_mult`）；模仿权重 `tgtS` ×3、`tgtA` ×2.5、`tgtfit` ×2（老师用新权重重学）；comptest 加“本局凑得成 S / A 的局里成了 S / A 的比例”和每套的成型率 | `tgt_q`、`teacher_core`，comptest 的“凑得成的局里成型的比例” |
| 退回 + 老师改成只教有把握的动作 | 第 14240 轮 → 退回第 14100 轮 | 第 14141–14240 轮老师用 KL：熵 1.0 → 2.1、加强局承受的对手强度 0.69 → 0.32、考试 5.2，连续 3 次变乱退回。老师改成只在它有把握（概率 ≥ 0.5）的局面教那个动作（`teacher_mode hard`、`teacher_conf 0.5`、`teacher_coef 0.02`、300 轮加上）；熵的绝对上限 1.6（`drift_ent_max`）；目标阵容 Φ = 权重 × 核心卡占比² × 本族随从 / 4（一张提图斯不再算半个畸变亡语）；comptest 按“核心卡 ≥ 2 张且 ≥ 2/3、种族阵容 ≥ 3 个本族随从”算成型 | `ent`、`teacher_sure`、`teacher_agree`、`tgt_q`、`boost_strength` |
| + 成型以后怎么用 | 第 14240 轮起 | `bg.py playbook`：从高分示范里找每套阵容的配套卡（高手做这套时比平时多买的卡，例如机械 APM 的扩音机、宰割流亡灵的友善的恶鬼 / 断手被遗忘者，高手买卡里 12.4% 是这种）；酒馆里有正在做的 S / A 阵容的配套卡时老师 ×3、模仿 `tgtsup` ×2.5；comptest 的买法里加“正在做 S / A 阵容时买它的配套卡” | comptest 的买法一行 |
| + 成型阵容课程 | 第 14350 轮起 | 15% 的机器人局在第 8 / 9 回合把学习者的场面换成高分阵容库里已经成型的 S / A 阵容（本局凑得成），之后自己接着打（`seed_comp_frac`、`seed_comp_turns`，`rl/calibrate.CompSeeder`）；换场面那一刻阵容强度 / 目标阵容奖励重新起算，这些局不算进 `place_bot` 和加强强度。老师的爬坡记进 `latest.pt`，重启不再从 0 爬 | `seeded`、`place_seeded`，comptest 的成型比例和第 10–13 回合阵容强度 |
| 后期加倍、老师减半 | 第 15180 轮起 | 第 15180 轮阵容测试：成型的 S / A 3% → 9%，成型的局平均 3.50 名、没成的 4.66 名；但熵 1.1 → 1.6、加强局对手强度 0.66 → 0.58、第 12 / 13 回合阵容强度 0.52 / 0.40（高手 0.67 / 0.60）。老师 0.02 → 0.01；老师 ×3 只在那套阵容已经开了头时；第 10 回合起阵容强度 Φ ×2（`combat_late_mult`） | `ent`、`boost_strength`、`cs_late`，comptest 第 10–13 回合阵容强度 |
| + 该留的牌别卖、配套卡联动 | 第 15180 轮起 | `bg.py playbook` 另外算出高手做某套时留着不卖的核心 / 配套卡（49 张；卖掉就触发效果的尖利的鲨鱼这类不算）；卖掉这种牌（没有第二张）扣 0.05（`keep_penalty`）；目标阵容 Φ 再乘配套卡配齐的程度 0.7–1；comptest 加“卖掉正在做的阵容里该留的牌”的比例（高手 5.2%） | `keep_sold`、`tgt_q`，comptest 的卖牌一行 |
| 成型阵容课程往前挪 | 第 15300 轮起 | 第 15300 轮阵容测试 4.07 名（第 11600 轮以来最好），卖掉该留的牌 2.8%（高手 5.2%）；但模型自己的局成型的 S / A 只有 6%、买配套卡 0%。课程改成 20% 的机器人局、第 7 / 8 / 9 回合：第 7 回合换做到一半的场面（核心卡过半、2 个本族随从，`seed_comp_formed_from 8`），学怎么做完 | `place_seeded`，comptest 的成型比例 |
| + 体系选手 | 第 15300 轮起 | 模型多了“体系感知”输入（`rl/model.py` 的 `CompSense`）：每套目标阵容本局凑不凑得成、核心卡占比、本族随从、配套卡、得分、主选 / 备选、酒馆里有没有它的牌，新参数 0 起步、旧检查点照常读；辅助头预测这局最后成了哪一套（`aux_comp_coef 0.05`）；老师在主选体系的牌出现时 ×3、备选 ×2；竞技场打包多一个 `comp.npz`；comptest 加“第 8 回合的主选 / 备选最后做成了哪个” | `aux_comp`、`comp_acc`，comptest 的体系一行 |
| 目标阵容加码 | 第 16420 轮起 | 体系感知上线 1000 多轮没起色（comptest 成型 S / A 6% → 2%，第 8 回合有主选 32%，tgt_q 一直 0.07）：开头一张核心卡几乎不值钱，模型不起步。目标阵容奖励 0.3 → 0.8，核心卡一项改成 (占比 + 占比²) / 2（开头就有分，凑齐更值钱）；成型阵容课程 30%、第 6–9 回合 | `tgt_q`，comptest 的成型比例、体系一行和平均名次 |
| + 引擎联动 | 第 16420 轮起 | 10 个引擎（布莱恩、提图斯、巴琳达、热气球测绘员、达卡莱附魔师、小瞎眼、逐亡陆行鸟、邪能野猪人、活化艾泽里特、火花破坏机）配上它放大的燃料随从：结束回合时每个引擎 min(1, 燃料 / 3) 加起来（最多 2），涨多少加 0.4 × 涨的（`engine_bonus`，`rl/comp.ENGINE_RULES`）；老师在引擎 / 燃料能配上时 ×3；模仿 `engfuel` ×3（高手买卡 17.8%）；comptest 列出每个引擎的上场率、燃料张数、配齐的局 | `eng_q`，comptest 的引擎联动一行 |
| 成型要配燃料 | 第 16600 轮起 | 第 16600 轮：成型的 S / A 2% → 18%（喝茶流 14%），布莱恩 / 提图斯上场 23% / 17%；但布莱恩平均只配 0.6 个战吼随从、巴琳达 0.4 个，平均名次 4.27 → 4.46、第 12 / 13 回合阵容强度 0.49 / 0.36。中立引擎阵容的“撑起来的程度”改成引擎核心卡配了几张燃料；目标阵容系数 0.8 → 0.6；日志把无效操作拆成调站位 / 冻结 / 取消 | `noop_move`、`noop_freeze`、`noop_cancel`，comptest 的成型比例和引擎燃料 |
| 三天无人值守 | 第 16850 轮起（从 it15300 退回） | 第 16420–16850 轮考试 4.44 → 4.75、加强局承受的对手强度 0.69 → 0.55，tgt_q / eng_q 却在涨：目标阵容、引擎联动、阵容强度三项局末不扣回，等于名次以外的第二个目标。改成局末扣回最后的 Φ（`shape_repay` 1，300 轮爬上去；做阵容那一步照样当场拿到功劳，整局和动作无关）；老师 0.01 → 0.004（0.017–0.02 时考试 4.63–4.72、0.0055 时 4.31、没老师时 4.17）；考试守护（`guard_*`）：最近 6 次考试平均比最好的差 0.3 以上连续 3 次 → 换回 `best.pt`、模仿 / 老师系数 × 0.7（最多 6 次），创新低时存 `best.pt`；`drift_noop` 0.15 → 0.2 | `shape_fin`、`shape_repay`、`guard_avg` / `guard_ref` / `guard_fix`，`runs\v3\best.pt` |
| 熵目标 + 爬坡 | 第 15558 轮起 | 第 15300 轮退回后熵 0.97 → 1.34（12 轮，新 Adam 头几步每个参数走满一步）→ 1.55–1.74，变乱检查连退 4 次、学习率 1.5e-4 → 1.9e-5；考试最好的几段熵 1.05–1.16。`ent_target` 1.1：熵高于目标 ent_coef × 0.97、低于 × 1.03（0.0005–0.01）；`warmup_iters` 50：优化器重开后学习率从 10% 爬上去；`drift_ent_max` 1.6 → 1.8。同时上线联动感知（SynergySense：每张卡 10 个标记，0 起步） | `ent_coef`、`lr` |
| 后期养场面 | 第 16045 轮起 | comptest 第 16040 轮 4.21 名，第 12 / 13 / 14 回合阵容强度 0.52 / 0.38 / 0.28（高手 0.67 / 0.60 / 0.68），成型的 S / A 1%。阵容强度不再局末扣回（`shape_repay_keys` target,engine：它就是场面打同期高分阵容的胜率，和名次一个方向；扣回后后期养场面几乎没功劳）；成型阵容课程 30% → 40%、加第 10 / 11 回合的成型场面（库里 238 / 266 个） | `cs_late`、`place_seeded`、`seeded` |
| 守护修复 | 第 17010 轮 | 第 16800–16950 轮考试 4.66–4.99，守护第 17000 轮换回 best.pt（第 15450 轮，熵 1.6）；变乱检查拿 1.6 比之前的中位数 1.10，连退 5 次训练停了。换回 / `--restore` 后熵中位数从头记；阵容强度重新局末扣回（`shape_repay_keys` 三项都扣，撤回第 16045 轮那条） | `guard_fix`、`ent` |
| 后期全强度 | 第 17010 轮起 | 自适应把加强局强度压在 0.6–0.7，后期还有三成机器人是弱场面，龙 / 海盗节奏阵容就能进前四，能成长的阵容在训练里不值钱；考试全强度，第 12–14 回合场面 0.49 / 0.39 / 0.28（高手 0.67 / 0.60 / 0.68）。`boost_late_from` 10、`boost_late_strength` 1.0：第 10 回合起机器人全换成同回合高分阵容，前期照旧自适应 | `boost_strength`（前期）、`place_boost` |
| 换场面切断 + 经济 | 第 17030 轮起 | 成型阵容课程换场面时 GAE 没切断：换场面的价值跳变（约 1 名）记在上一回合的结束回合上，等于奖励早结束回合、留钱（模型每回合剩 0.3–0.5 块，高手 0.01–0.26）；换之前的步数阵容标签也是错的。`Rollout.cut` 切断、标签置 -1；`unspent_gold_penalty` 0.005 → 0.015；`ent_target` 1.2、`ent_coef_min` 0.001；高手示范第 6–9 回合 ×1.5 | `unspent`、`comp_acc` |
| 起步课程 | 第 17790 轮起 | 老师模型单独打 7.98 名（只会模仿单步，中期就被打死），从它开始练不行；模型自己成型的 S / A 1%。`seed_start_*`：25% 的机器人局第 4–6 回合从本局凑得成的 S / A 里均匀抽一套，场上最弱的 3 个随从换成这套现在买得到的起步卡；成型阵容课程 40% → 30% | `started`、`place_started` |
| 关掉起步课程 | 第 18611 轮 | 起步课程的局 3.40–3.58 名，普通机器人局 3.09–3.20；成型的 S / A 2%，主种族还是龙 / 海盗；守护第 18450 轮第二次换回第 15450 轮那版。`seed_start_frac` 0 | |
| 为吃鸡打 | 第 18611 轮起 | 考试平均 4.2–4.6 名，吃鸡率第 12000 轮起一直 1.3–2.2%（平均 12.5%）：线性名次奖励下稳进前四就是最优。`place_reward` 1 名 +1、2 名 +0.55、3 名 +0.25、4 名 +0.05、5 名 -0.15、6 名 -0.4、7 名 -0.7、8 名 -1；守护按这张表的等效名次比（原来的 best.pt 改名 best_it15450.pt）；comptest 打印名次分布和吃鸡率 | `eval_boost_top1`、`eval_boost_score` |
| 训练看板 | 第 18611 轮起 | `tools/board.py`（`bg.py board`、`board.bat`）：读 log.jsonl 画 11 张曲线（考试、吃鸡率、训练名次、加强强度、熵、学习率、场面强度、经济、阵容、模仿、速度）+ 关键指标表 + 事件表 → `runs\v3\dashboard.md`、`runs\v3\board\*.svg`；训练时每 `board_every`（10）轮后台重画 | |
| 为什么总玩龙（诊断） | 第 18740 轮 | comptest 打印“模型走各主种族的结果”（每个主种族的局数、平均名次、吃鸡率、前四率）；成型阵容课程按阵容记“拿到手以后打第几”（`seeded_by_comp`，看板里有表）：看在模拟器里哪套阵容真能赢、模型自己走的龙是不是对它来说最好 | `seeded_by_comp` |
| 训练对手前期别太弱 + 拿着成型阵容打满强度（诊断） | 第 19650 轮起 | 加强局前期强度下限 0.7（`boost_min_strength`，从当时的 0.29 每轮抬 0.002）：自适应把前期强度压到 0.26，机器人局吃鸡 20%、考试 1–3%，节奏龙在训练里就能赢；`seeded_by_comp` 按加强局分开记（看板多三列）；`comptest_seeded.bat` 对全强度直接测“拿到成型阵容以后打第几” | `boost_floor`、`boost_strength`、`seeded_by_comp`、`comptest_seeded.json` |
| 养场面奖励 | 第 19650 轮起 | `comptest_seeded`：直接给高手成型的阵容，第 12–14 回合强度照样掉（0.55 / 0.41 / 0.38，高手 0.67 / 0.60 / 0.68），引擎拿到就卖；第 10 回合起换老师打更差。第 9 回合起每回合开始、结束各量一次同一组考官的强度，结束回合加 `grow_bonus 0.3` ×（结束 − 开始），不扣回：这回合自己养了多少（`rl/shaping.py` 的 `_grow_shape`）；comptest 阵容强度表多一列“养了多少” | `grow`、`t_grow`，comptest 的“养了多少” |
| 阵容练习局 | 第 20020 轮起 | 25% 的机器人局开局定一套本局凑得成的 S / A，第 2–9 回合每回合酒馆里换进 2 张这套的卡（核心 / 配套 / 本族），场面不动，做不做全看模型（`rl/calibrate.py` 的 `ShopPractice`，`practice_*`）；成型阵容课程跳过这些局；comptest 的强度表还多了一列“对手涨了” | `place_practice`、`practice_by_comp`（看板“阵容练习局”） |
| 练习局里凑阵容值分 | 第 20240 轮起 | 6554 局练习局只做成 0.5%：牌刷到了也不要。练习局里目标阵容奖励局末不扣回（`practice_keep_target 1`），凑成几成拿 0.6 × 几成；普通局照常扣回 | `practice_tgt`（看板关键指标“练习局局末阵容做到几成”） |
| 照着凑 | 第 20350 轮起 | 练习局第 7 回合起按规则给建议动作（发现选这套 → 打出这套的随从 → 买核心卡 → 跟上升本节奏 → 买配套 / 本族 → 满了卖杂牌 → 4 本以后满了有杂牌就刷新找核心卡），更新时加 `expert_coef 0.3` × 交叉熵；练习局换牌改到第 7–13 回合、只给核心 / 配套卡。启发式在引擎里试：只换牌 0%、按规则凑 50–67%、不换牌只靠刷新 25% 做成 | `expert_follow`、`expert_ce`、`expert_agree` |
| 关掉照着凑、换回第 20200 轮 | 第 20800 轮 | 照着凑让练习局做成 0.7% → 31–35%，但那些局 4.76 名（之前 3.44），卖牌 / 刷新带到了所有局：机器人局 3.4 → 4.5–4.9 名、考试 4.2–4.5 → 5.6–6.0。`expert_coef 0`，守护也缩它（`guard_keys` 加了 `expert_coef`）；`train.bat --restore runs\v3\best.pt` 回到第 20200 轮 | 考试回到 4.2–4.5 |
