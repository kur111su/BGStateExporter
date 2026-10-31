# BGStateExporter — 酒馆战旗局面导出、建议和 AI 训练

项目分两部分：

- **插件（`plugin/`）**：Hearthstone Deck Tracker（HDT）插件。实时读取酒馆战旗局面、用 HDT 自带的 Bob's Buddy 做对战模拟（本场胜率、对位胜率、站位优化），把局面通过本机接口 http://localhost:8765/state.json 提供出来，并把每局记录存下来。v0.10 起不再有自己的搜索建议器和网页面板。
- **模型（`bgrl/`）**：在 TavernGym 模拟器（MCG-NJU/AutoTavern，MIT）里用 PPO 训练一个实体 Transformer 策略。**建议完全来自这个模型**：影子模式（`bgrl\shadow.bat`）读插件的局面，在 http://localhost:8766/ 显示模型的本回合方案，同一页上也有插件的模拟结果和站位优化按钮。

两部分都**只读**：插件只读取 HDT 已经解析好的数据，不碰游戏进程，也不操作鼠标键盘；所有操作都由玩家自己完成。

## 功能一览

| 部分 | 功能 | 入口 |
| --- | --- | --- |
| 插件 | 局面导出（state.json、`/state.json`、`/events` SSE、每局 `games\*.jsonl`） | 装好即用 |
| 插件 | 本场胜率、对各对手胜率、站位优化（Bob's Buddy） | 影子模式页面 http://localhost:8766/ |
| 插件 | 自动更新（GitHub Releases） | HDT 启动时 |
| 模型 | 环境安装、测试 | `bgrl\setup.bat` |
| 模型 | 训练（断点续训、对手库、自动回滚） | `bgrl\train.bat` |
| 模型 | 影子模式：本回合方案、预计名次、第一步备选（只显示，不操作游戏） | `bgrl\shadow.bat` → http://localhost:8766/ |
| 模型 | 回放检查：用你的对局逐步检查模拟器 | `bgrl\replay.bat` |

插件细节（算法、权重、安装 / 编译 / 发布、JSON 格式）见 [docs/插件详细说明.md](docs/插件详细说明.md)；模型部分见 [bgrl/README.md](bgrl/README.md)。

## 安装插件

1. 到 [Releases](https://github.com/kur111su/BGStateExporter/releases/latest) 下载 `BGStateExporter-vX.Y.Z.zip`（不用解压）。
2. 打开 HDT → 选项 → 跟踪器 → 插件，把 zip **拖进插件列表**。
3. 勾选 **BG State Exporter**。看建议：运行 `bgrl\shadow.bat`，再点插件的「打开影子模式页面」（或浏览器打开 http://localhost:8766/）。

以后有新版本会自动下载，完全退出 HDT（托盘图标右键「退出」）再打开生效。

## 项目结构

```
plugin/                     HDT 插件（C#，.NET Framework 4.7.2）
  BGStateExporter.csproj    版本号在这里改；推送到 main 自动编译发布
  build.ps1                 本机编译并安装到 HDT 插件目录
  src/Core/                 插件入口、局面读取、JSON 模型、本机 HTTP 服务（/state.json、/events、/optimize）、自动更新
  src/Sim/                  Bob's Buddy 调用（反射）和模拟调度：本场胜率、对位胜率、站位优化
bgrl/                       模型：模拟环境桥接、训练、影子模式（Python）
  setup.bat train.bat shadow.bat replay.bat   四个双击入口
  bg.py                     统一命令入口（python bg.py <命令>）
  paths.py                  路径和端口
  tg/                       TavernGym 桥接：动作编号、对局记录解析、局面重建、回放检查
  rl/                       模型、PPO 训练、对手库、影子模式的建议器
  tools/                    各命令的实现
  tests/                    单元测试和 CI 用的局面样本
  taverngym/PIN.txt         锁定的 TavernGym 版本（改这里会重新编译安装包）
docs/                       插件详细说明、样例数据
examples/watch_state.py     订阅插件实时局面的最小示例
.github/workflows/          插件编译发布、TavernGym 安装包、bgrl 检查
```

数据流：

```
HDT ──读局面──▶ 插件 ──▶ :8765/state.json、state.json、games\*.jsonl（对局记录）
                                │                  │
                                ▼                  ├─▶ replay（检查模拟器）、demos（模仿你的操作）
             影子模式页面 :8766 ◀── 模型 runs\v3\latest.pt ◀── train（TavernGym 里 PPO 训练）
                                                  └─▶ 对手强度画像（校准训练对手）
```

## 本机文件位置

| 内容 | 位置 |
| --- | --- |
| 插件本体 | `%AppData%\HearthstoneDeckTracker\Plugins\BGStateExporter\` |
| 当前局面、对局记录 | `%AppData%\HearthstoneDeckTracker\BGStateExporter\`（`state.json`、`games\`；旧版留下的 `advisor_weights.json`、`training\` 已不再使用，可以删掉） |
| 训练进度、模型、对手库快照 | `bgrl\runs\v3\`（`log.jsonl`、`latest.pt`、`snapshots\`） |
| 回放报告、影子模式记录、安装日志 | `bgrl\reports\` |

v0.9.4 及以前插件里的搜索建议器（`plugin/src/Advisor/`）和网页面板（`plugin/www/index.html`）已移除，需要时可从提交 73fee12 取回（`git checkout 73fee12 -- plugin/src/Advisor plugin/www`）。

旧的自研模拟器 `bgrl/bgsim` 和 Bob's Buddy 批量模拟接口 `bgrl/simhost` 已被 TavernGym 取代、从仓库移除；需要时可从提交 9b60201 取回（`git checkout 9b60201 -- bgrl/bgsim bgrl/simhost`）。
