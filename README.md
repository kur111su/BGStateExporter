# 酒馆战旗 AI：模拟训练 + 竞技场

在 TavernGym 模拟器（MCG-NJU/AutoTavern，MIT）里用 PPO 训练一个实体 Transformer 策略，练出来的模型打竞技场（TavernArena，官方服务器上的模拟对局）。

模型只在模拟器和竞技场的模拟对局里出动作，不操作真实的炉石客户端。

> 这个分支（`dev`）只有模型部分。原来的 HDT 插件（局面导出、Bob's Buddy 模拟）、影子模式、回放检查、模仿自己对局的示范已经去掉；插件留在 `main` 分支和已发布的版本里。

## 功能一览

| 功能 | 入口 |
| --- | --- |
| 环境安装、测试 | `bgrl\setup.bat` |
| 训练（断点续训、对手库、守护和自动回滚） | `bgrl\train.bat` |
| 训练看板（曲线、关键指标、事件 → `runs\v3\dashboard.md`） | `bgrl\board.bat` |
| 老师模型（只模仿高分示范） | `bgrl\teacher.bat` |
| 高分对局 → 决策样本和阵容库 | `bgrl\prodemos.bat`，`bg.py prolib` / `nativedemos` |
| 金币流向、阵容测试：模型和高分玩家并排比 | `bgrl\goldflow.bat`、`bgrl\comptest.bat`、`bgrl\comptest_seeded.bat` |
| TavernBench 评测 | `bgrl\tavernbench.bat` |
| 竞技场：打包模型、练习 / 计分对局、对局查看器 | `bgrl\arena*.bat` |

细节（每个命令、训练配方、训练记录）见 [bgrl/README.md](bgrl/README.md)。

## 目录

```
bgrl/                       全部代码（Python）
  *.bat                     双击入口（setup、train、board、teacher、goldflow、comptest、comptest_seeded、prodemos、tavernbench、arena×5）
  bg.py                     统一命令入口（python bg.py <命令>）
  paths.py                  路径
  configs/v3.json           训练配方（train.bat 的全部设置）
  tg/                       TavernGym 桥接：动作编号、观察布局、卡牌目录、高分回放解析、局面重建
  rl/                       模型、PPO 训练（奖励塑形、模仿学习分开放）、课程、示范格式、阵容打分、对手库
  tools/                    各命令的实现
  arena/                    竞技场：模型打包、契约核对、连接器策略、对局查看器
  tests/                    单元测试和 CI 用的局面样本
  taverngym/PIN.txt         锁定的 TavernGym 版本（改这里会重新编译安装包）
.github/workflows/          TavernGym 安装包、bgrl 检查
```

数据流：

```
高分示范和阵容库（Firestone 6500+、万三；只在本机）
        │  prolib / prodemos / nativedemos / cardvalue
        ▼
configs\v3.json ──▶ train（TavernGym 里 PPO 训练）──▶ runs\v3\latest.pt、best.pt、snapshots\
                                                        │
                        ┌───────────────┬───────────────┼───────────────┐
                        ▼               ▼               ▼               ▼
                  board（看板）   goldflow / comptest   tavernbench   arena（竞技场）
```

## 本机文件位置

| 内容 | 位置 |
| --- | --- |
| 训练进度、模型、对手库快照 | `bgrl\runs\v3\`（`log.jsonl`、`latest.pt`、`best.pt`、`snapshots\`） |
| 高分对局原始数据（不进 Git） | `bgrl\data\` |
| 竞技场打包的模型、密钥、对局结果（不进 Git） | `bgrl\arena\model_bundle\`、`bgrl\arena\key.txt`、`bgrl\arena\results\` |
| 评测报告、安装日志 | `bgrl\reports\` |

`runs\`、`data\`、竞技场密钥和模型都不进 Git：别人的对局（Firestone、万三）生成的样本和阵容库只在本机，仓库里只有汇总数字（`rl/*.json`）。
