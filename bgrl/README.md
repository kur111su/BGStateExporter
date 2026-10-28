# bgrl：TavernGym 训练 + 影子模式

在 TavernGym（MCG-NJU/AutoTavern，MIT，锁定的版本见 `taverngym/PIN.txt`）里用 PPO 训练一个实体 Transformer 策略，再把它的建议显示在插件面板上。只显示，不操作游戏。

## 四个入口（Windows 双击）

| 文件 | 做什么 | 结果在哪 |
| --- | --- | --- |
| `setup.bat` | 第一次用、或更新后运行：建 `.venv`，装 numpy、TavernGym 安装包、PyTorch（CUDA），跑测试和几轮极小训练。`setup.bat bench` 另外测三种模型大小的训练速度 | `reports\setup.txt` |
| `train.bat` | 训练，关掉窗口再开会接着 `runs\v2\latest.pt` 练。其它设置：`train.bat --help` | `runs\v2\log.jsonl`、`latest.pt`、`snapshots\` |
| `shadow.bat` | 影子模式：插件面板“模型建议”一栏、http://localhost:8766/ 。训练中换局时自动换上最新模型。`shadow.bat --ckpt 某个快照.pt` 用指定模型 | `reports\shadow\日期.jsonl` |
| `replay.bat` | 回放检查：把你对局里的每一步放进 TavernGym 重做，按卡统计不一致。`replay.bat --last 5` 只看最近 5 局 | `reports\a5_replay.txt` / `.json` |

需要 Python 3.10～3.13（推荐 3.12，安装时勾选 “Add python.exe to PATH”）和 NVIDIA 显卡驱动。

## 命令行

所有入口都是 `bg.py` 的子命令（`.venv\Scripts\python bg.py <命令> --help` 看参数）：

| 命令 | 实现 | 说明 |
| --- | --- | --- |
| `train` | `tools/train.py` | `--smoke` 冒烟、`--bench --size medium` 测速、`--run 目录 --resume` 正式训练；其余参数和 `rl/ppo.py` 的 `Config` 一一对应 |
| `shadow` | `tools/shadow.py` | 影子模式服务 |
| `replay` | `tools/tg_replay.py` | A5 回放检查；不给文件时读插件的 games 目录 |
| `bridge` | `tools/tg_bridge_check.py` | A4 局面重建检查 |
| `profile` | `tools/opp_profile.py` | B1 对手画像：真人对局 vs 脚本机器人（`--save` 更新 `rl/real_profile.json`） |
| `bench` | `tools/tg_bench.py` | A1 TavernGym 测速 |
| `test` | `tests/` | 单元测试 |
| `info` | — | 环境、对局记录数、训练进度 |

## 模块

```
paths.py          路径和端口（runs/、reports/、默认训练目录 runs/v2、插件数据目录、8765 / 8766）
tg/actions.py     245 个动作的编号（所有模块共用）
tg/logs.py        读插件对局记录：统一标签名、皮肤 → 英雄、对手上次阵容、切分动作
tg/bridge.py      插件局面 → TavernGym 局面（随从、附魔、宝石 / 法术加成、饰品和随从计数、战败状态、对手阵容）
tg/replay.py      A5：执行你的动作并和记录比较（随机效果换种子、选项逐个试、动画延迟识别）
tg/skins.json     不按 _SKIN_ 后缀命名的英雄皮肤
rl/model.py       实体 Transformer：66 个实体 token，指针式动作头 + 价值头 + 名次头（small / medium / large）
rl/ppo.py         PPO 训练：批量环境采样、GAE、混合精度、健康检查和自动回滚、无效操作惩罚
rl/league.py      对手库：快照按“和真人强度画像的距离”加权；考试对手
rl/advisor.py     影子模式的建议器：贪心走完本回合、跳过无效动作、第一步备选、预计名次
```

## CI

- `.github/workflows/taverngym.yml`：改 `taverngym/PIN.txt` 后编译 Windows / Linux 安装包，发布为预发布 `taverngym-<commit>`（插件自动更新不会取到）。
- `.github/workflows/bgrl-tg.yml`：A4 重建检查、A5 回放检查（用 `tests/fixtures` 里的样本）。
- `.github/workflows/bgrl-rl.yml`：单元测试、对手画像、冒烟训练（CPU）。

样本更新：`tools/make_fixtures.py`（A4）、`tools/make_replay_fixture.py`（A5）从新的对局记录生成。
