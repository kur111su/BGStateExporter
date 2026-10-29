# bgrl：TavernGym 训练 + 影子模式

在 TavernGym（MCG-NJU/AutoTavern，MIT，锁定的版本见 `taverngym/PIN.txt`）里用 PPO 训练一个实体 Transformer 策略，再把它的建议显示在插件面板上。只显示，不操作游戏。

## 四个入口（Windows 双击）

| 文件 | 做什么 | 结果在哪 |
| --- | --- | --- |
| `setup.bat` | 第一次用、或更新后运行：建 `.venv`，装 numpy、TavernGym 安装包、PyTorch（CUDA），跑测试和几轮极小训练。`setup.bat bench` 另外测三种模型大小的训练速度 | `reports\setup.txt` |
| `train.bat` | 训练（第二阶段 `runs\v3`：机器人按真人强度加强，从第一阶段 `runs\v2` 的模型开始），关掉窗口再开会接着练。其它设置：`train.bat --help` | `runs\v3\log.jsonl`、`latest.pt`、`snapshots\` |
| `shadow.bat` | 影子模式：插件面板“模型建议”一栏、http://localhost:8766/ 。训练中换局时自动换上最新模型（当前训练目录还没有模型时，用最近更新的其它训练目录的）。`shadow.bat --ckpt 某个快照.pt` 用指定模型 | `reports\shadow\日期.jsonl` |
| `tavernbench.bat` | 用 TavernBench 测当前模型（约几分钟，只用 CPU，训练开着也行）。`tavernbench.bat --ckpt runs\v2\latest.pt` 测指定模型 | `reports\tavernbench\<模型>\report.txt` |
| `replay.bat` | 回放检查：把你对局里的每一步放进 TavernGym 重做，按卡统计不一致。`replay.bat --last 5` 只看最近 5 局 | `reports\a5_replay.txt` / `.json` |

需要 Python 3.10～3.13（推荐 3.12，安装时勾选 “Add python.exe to PATH”）和 NVIDIA 显卡驱动。

## 命令行

所有入口都是 `bg.py` 的子命令（`.venv\Scripts\python bg.py <命令> --help` 看参数）：

| 命令 | 实现 | 说明 |
| --- | --- | --- |
| `train` | `tools/train.py` | `--smoke` 冒烟、`--bench --size medium` 测速、`--run 目录 --resume` 正式训练；其余参数和 `rl/ppo.py` 的 `Config` 一一对应 |
| `shadow` | `tools/shadow.py` | 影子模式服务 |
| `replay` | `tools/tg_replay.py` | A5 回放检查；不给文件时读插件的 games 目录 |
| `shadow-eval` | `tools/shadow_eval.py` | 影子模式评估：按模型版本统计模型方案和你实际操作对得上多少（第一步同类 / 同一张卡 / 在方案里），结果写到 `reports\shadow_eval.json` |
| `tavernbench` | `tools/tavernbench.py` | C3：用 TavernBench（论文的 368 题评测集，`third_party/tavernbench`）测模型，和论文公布的 TavernAgent、大语言模型、Jev 比；`--ckpt` 指定模型。结果在 `reports\tavernbench\<模型>\report.txt` |
| `demos` | `tools/make_demos.py` | 把你的对局记录变成示范数据 `runs\demos.npz`（每步操作前的局面 + 你做的动作，每回合末尾加“结束回合”；每 5 局留 1 局验证）。`--if-stale` 有新对局才重新生成（train.bat 每次启动时用） |
| `bridge` | `tools/tg_bridge_check.py` | A4 局面重建检查 |
| `profile` | `tools/opp_profile.py` | B1 对手画像：真人对局 vs 脚本机器人（`--save` 更新 `rl/real_profile.json`） |
| `bench` | `tools/tg_bench.py` | A1 TavernGym 测速 |
| `test` | `tests/` | 单元测试 |
| `info` | — | 环境、对局记录数、训练进度 |

## 模块

```
paths.py          路径和端口（runs/、reports/、默认训练目录 runs/v2、插件数据目录、8765 / 8766）
tg/actions.py     245 个动作的编号（所有模块共用）
tg/logs.py        读插件对局记录：统一标签名、皮肤 → 英雄、切分动作；GameHistory 记对手上次阵容和已买的黑暗之赐（回放和影子模式共用）
tg/bridge.py      插件局面 → TavernGym 局面（随从、附魔、宝石 / 法术加成、饰品和随从计数、战败状态、黑暗之赐次数、对手阵容）
tg/replay.py      A5：执行你的动作并和记录比较（随机效果换种子、选项逐个试、动画延迟识别）
tg/skins.json     不按 _SKIN_ 后缀命名的英雄皮肤
rl/model.py       实体 Transformer：66 个实体 token，指针式动作头 + 价值头 + 名次头（small / medium / large）
rl/ppo.py         PPO 训练：批量环境采样、GAE、混合精度、健康检查和自动回滚、无效操作惩罚
rl/league.py      对手库：快照按“和真人强度画像的距离”加权；考试对手
rl/calibrate.py   第二阶段：机器人局里把机器人场面按真人画像（rl/real_profile.json）的四分位放大，逼模型学会升本、做后期
rl/advisor.py     影子模式的建议器：贪心走完本回合、跳过无效动作、第一步备选、预计名次
```

## CI

- `.github/workflows/taverngym.yml`：改 `taverngym/PIN.txt` 后编译 Windows / Linux 安装包，发布为预发布 `taverngym-<commit>`（插件自动更新不会取到）。
- `.github/workflows/bgrl-tg.yml`：A4 重建检查、A5 回放检查（用 `tests/fixtures` 里的样本）。
- `.github/workflows/bgrl-rl.yml`：单元测试、对手画像、冒烟训练（CPU）。

样本更新：`tools/make_fixtures.py`（A4）、`tools/make_replay_fixture.py`（A5）从新的对局记录生成。

## 训练阶段

| 阶段 | 目录 | 对手 | 结果 |
| --- | --- | --- | --- |
| 第一阶段 | `runs\v2` | 脚本机器人 + 自己的旧版本 | 考试（对普通机器人）从 4.27 名到约 1.8 名；但学会了一直待在 1–2 本（第 13 回合平均 2.8 本，真人约 5.8 本），影子模式里几乎从不建议升本 |
| 第二阶段 | `runs\v3` | 75% 的机器人局里机器人按真人强度加强（第 7 回合起每 2 回合一次）+ 自己的旧版本。加强幅度自适应：学习者在加强局里平均好于 3.5 名就加 2%，差于 4.0 名就减 2%（第 251 轮起；之前是 200 轮内固定爬到 100%，结果加强局名次和随便打一样、熵翻倍） | 看日志里的 `boost_strength`（爬到 1.0 = 能和真人画像强度的机器人打平）、`place_boost`（这一轮加强局的平均名次）、`eval_boost_place`（对 100% 加强机器人的考试）和 `snapshot_tier` |
| 第二阶段 + 模仿 | `runs\v3`（第 2338 轮起） | 同上；另外每个小批抽 256 条你的操作算交叉熵（`--bc-coef 0.02`）。自适应强度在 0.76–0.82 停了约 1000 轮、第 13 回合仍只有 3–4 本，靠奖励学不会按时升本 | 看 `human_top1`（留出局上模型首选 = 你的操作的比例）、`human_upgrade` / `human_end`（该升本 / 该结束时选对的比例），同时 `eval_place`、`boost_strength` 不能明显变差 |
