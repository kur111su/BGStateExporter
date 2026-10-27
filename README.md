# BG State Exporter — HDT 酒馆战旗局面导出 + 对战模拟插件

在 Hearthstone Deck Tracker（HDT）里实时读取酒馆战旗局面，并调用 HDT 自带的 Bob's Buddy 模拟器做对战模拟。输出方式：

| 输出 | 位置 | 用途 |
| --- | --- | --- |
| 网页面板 | http://localhost:8765/ | 实时可视化酒馆、场面、手牌、排行榜 |
| 实时推送 | http://localhost:8765/events（SSE）、/state.json | 给 Claude 教练、PPO 等程序订阅 |
| 站位优化 | http://localhost:8765/optimize?opponent=对手PLAYER_ID | 请求搜索更好的站位，结果出现在 `simulation.optimizer` |
| 文件 | `%AppData%\HearthstoneDeckTracker\BGStateExporter\state.json` | 当前局面，局面变化时原子覆盖 |
| 对局记录 | `...\BGStateExporter\games\yyyyMMdd_HHmmss.jsonl` | 每局每次变化一行，可做训练数据 |

插件只读取 HDT 已经解析好的数据，不碰游戏进程、不操作鼠标。

## 对战模拟（Bob's Buddy）

| 功能 | 什么时候 | 说明 |
| --- | --- | --- |
| 本场胜率 | 战斗阶段 | 直接读取 HDT 自己跑的 Bob's Buddy 结果（和 HDT 悬浮窗上的数字一致）；招募阶段显示上一场的赛前预测 |
| 对各对手胜率 | 招募阶段，场面停止变化 0.8 秒后自动跑 | 用你**当前**的场面，对每个还活着的对手**上次交手时**的阵容各模拟 3000 次，下个对手排第一 |
| 站位优化 | 点网页上的“优化站位”或调用 `/optimize` | 局部搜索：反复尝试把一个随从挪到另一个位置，提升超过 2 个百分点才接受；最多试 80 种，最后用 6000 次模拟复核当前和建议站位 |

注意：
- 对手阵容是“上次见到的”，对手这几回合可能已经变强，所以“对各对手胜率”和站位建议只能当参考，最有用的是**比较**（哪个站位更好、哪个对手更危险）。
- 对手的英雄技能、饰品等在招募阶段看不到，自己发起的模拟里不包含；HDT 本场战斗的结果是完整的。
- 一进入战斗，插件自己的模拟会立即停下，不和 HDT 的 Bob's Buddy 抢 CPU。
- Bob's Buddy 不支持的卡牌会在结果里显示原因。
- 用到了 HDT 的内部类（`BobsBuddyInvoker`、`BobsBuddyUtils.GetMinionFromEntity`），通过反射调用；HDT 大版本更新改了这些名字时，插件会退回到只设置随从基础属性的模式，或在 `simulation` 里报错，局面导出不受影响。

## 本回合建议（第一层：搜索 + 模拟）

招募阶段，局面停止变化 0.4 秒后自动计算，结果显示在网页面板最上方：推荐方案 + 两个备选，推荐方案要买的随从在酒馆里标绿、要卖的标红。也写在 JSON 的 `simulation.advisor` 里。

紧接着是**决策依据**表：每一行是一项特征（战力、升本、落后节奏、三连、对子、种族、剩余金币），每一格写着“权重 × 特征值 = 得分”，最下面一行是总分，一眼能看出推荐方案靠哪几项赢。右栏的“建议器权重”显示当前用的每个权重，竖线是默认值，自动学习调整过的会标出差值。

**怎么算的**
1. 酒馆随从按“身材估值 + 效果”挑前 4 个（能和已有随从凑对 / 三连的优先），买得起的酒馆法术挑前 2 个，枚举“升不升本 × 买其中 0～3 个随从 × 买不买一个法术”。
2. 每个组合里，手牌随从和新买的随从按估值从高到低尝试上场：场上有空位就上；没空位但比场上最弱的强，就卖掉最弱的换上；否则**留在手里**——不会为了打出手牌去卖掉更强的随从。买到第三张时合成金色。
3. 先用估值粗排，最好的 12 个（加上“什么都不买”作对照）用 Bob's Buddy 对**下个对手上次的阵容**各模拟 1000 次；下个对手没交过手时，用见过最强的阵容。
4. 总分 = Σ 权重 × 特征：战力（胜率 − 负率，按血量加权）、升本节奏、三连 / 对子、主种族、**随从效果**（买的随从有战吼 / 回合结束 / 复仇 / 战斗开始 / 集结 / 法术迸发 / 磁力等效果，按星级折算）、**购买**（买了几个）、**法术**、剩余金币能刷新几次，取前 3。
5. 酒馆还没刷出来（回合刚开始）时不计算，避免先闪出一个“什么都不做”。

购买、刷新、升本的价格从游戏里的按钮实体读取（面板上显示的“购买 3 金，刷新 1 金，升本 7 金”）。

**没算进去的**：战吼、回合结束、法术的**具体效果**模拟不了，用“随从效果 / 法术”两项特征近似，权重会被自动学习调整；英雄技能和刷新之后会出什么不考虑。

**默认升本节奏**来自 v0.7 时 11 局实际对局（前 2 名对局的中位数）：第 2 回合 2 本、第 5 回合 3 本、第 7 回合 4 本、第 8 回合 5 本、第 10 回合 6 本。旧的权重文件会自动升级（升本节奏没手动改过才会换成新默认值）。

**本局回合记录**：面板底部列出每个回合“酒馆刷出来后的第一份建议”和“你这回合实际做了什么”，标出一致 / 部分一致 / 不同，方便看建议器在哪些地方和你想法不一样。

**调权重**：`%AppData%\HearthstoneDeckTracker\BGStateExporter\advisor_weights.json`（第一次计算时自动生成，字段都是中文）。改完保存，下一次计算就会用新值，不用重启 HDT。常用的几个：

| 字段 | 默认 | 作用 |
| --- | --- | --- |
| 升本节奏 | 第 2 回合 2 本、第 5 回合 3 本、第 7 回合 4 本、第 8 回合 5 本、第 10 回合 6 本 | 落后时升本加分、不升本扣分 |
| 升本权重 / 不升本落后惩罚 | 0.35 / 0.12 | 越大越倾向升本 |
| 满血时战力系数 / 残血时战力系数 / 残血线 | 0.8 / 1.8 / 12 | 血越少越看重这回合赢 |
| 三连权重 / 对子权重 / 主种族权重 | 0.45 / 0.06 / 0.04 | |
| 随从效果权重 / 购买权重 / 法术权重 | 0.08 / 0.03 / 0.06 | 越大越倾向多买有效果的随从、多买法术 |
| 候选随从数 / 候选法术数 / 每方案最多购买 / 最多模拟方案数 | 4 / 2 / 3 / 12 | 搜索规模，越大越慢 |
| 每方案模拟次数 / 每方案模拟毫秒 | 1000 / 200 | |

每个方案下面那行小字（“战力 +0.48 · 升本 −0.12 …”）就是各项得分，调权重时看它。

## 自动学习权重

建议器每算一次，都会记下所有候选方案的特征。之后对比前后局面，判断你实际选了哪个方案（升没升本、买了谁、买没买法术）；这段时间你什么都没做（只刷新、只看）就不记录。每局结束时，按名次加权写入 `training\samples_v2.jsonl`：第 1 名权重 1.0，第 4 名 0.14，第 5～8 名不参与。

v0.8 新增了“随从效果 / 购买 / 法术”特征，候选方案的生成方式也改了，v0.7 及以前记录的 `training\samples.jsonl` 和新特征对不上，保留但不再使用，**局数从 0 重新累计**。

攒够 **30 局**（`自动学习最少局数`）后，每局结束会自动重新拟合一次：
- 模型：你选某个方案的概率 ∝ exp(β × Σ 权重 × 特征)。“战力”权重固定作为基准，β 单独拟合（代表你的选择有多稳定），其余权重向默认值收缩，局数越多越敢偏离（`自动学习强度` 越大越敢偏离）。
- 每 4 局取 1 局做验证。新权重在验证局上解释得更好才写入 `advisor_weights.json`，旧文件备份为 `advisor_weights.backup.json`，过程记录在 `training\fit_log.txt`。
- 面板上的“本回合建议”标题下会显示进度：“已收集 N/30 局”，以及上次拟合的结果。

注意：
- 学到的是**你自己打得好的对局里是怎么选的**，所以上限大致是你的水平；只调这 8 项特征之间的比例，不会学出新打法。
- 你的操作不在候选方案里时（比如买了估值排名靠后的随从、靠刷新后的新随从），那一步不参与学习。
- 想关掉：把 `自动学习` 改成 `false`。想恢复：把 `advisor_weights.backup.json` 改名覆盖回去，或者删掉 `advisor_weights.json` 恢复默认。
- 在用合成数据的测试里：数据中确实存在的偏好（比如更爱升本），大约 100 局能学到接近真实值；完全随机的选择会被验证集拒绝，权重保持不变。

## 安装

**直接用（推荐）**
1. 到 [Releases](https://github.com/kur111su/BGStateExporter/releases/latest) 下载 `BGStateExporter-vX.Y.Z.zip`（不用解压）。
2. 打开 HDT → 选项 → 跟踪器 → 插件，把 zip **拖进插件列表**，HDT 会自动安装，不用重启。
   注意是拖到 HDT 窗口的插件列表上；直接把 zip 复制进插件文件夹是不会被加载的。也可以手动解压，把 `BGStateExporter` 文件夹放进 `%AppData%\HearthstoneDeckTracker\Plugins\`，然后完全退出 HDT（托盘图标右键「退出」）再打开。
3. 勾选 **BG State Exporter**，点「打开网页面板」（或浏览器打开 http://localhost:8765/）。

以后有新版本时，插件启动会自动下载，重启 HDT（托盘图标右键「退出」后再打开）生效；网页面板顶部会提示。

**自己编译**
1. 装好 HDT 和 [.NET SDK](https://dotnet.microsoft.com/download)（6 或以上）。
2. 在本目录运行：`powershell -ExecutionPolicy Bypass -File build.ps1`（HDT 在别处时加 `-HDTPath "D:\你的\HDT"`）。
   脚本依次查找：`-HDTPath`、与本项目同级的 `..\HDT`、`%LocalAppData%\HearthstoneDeckTracker\app-最新版本`，
   编译后安装到 `%AppData%\HearthstoneDeckTracker\Plugins\BGStateExporter\`。HDT 目录下的 `Plugins` 只是镜像，HDT 启动时会用 AppData 里的内容覆盖它，不要把插件放那里。
3. 完全退出 HDT 再打开，勾选插件。

**发布新版本**：把 `BGStateExporter.csproj` 里的 `<Version>` 改成新版本号（例如 `0.3.1`），推送到 main。
GitHub Actions 会下载最新版 HDT 编译，这个版本还没发布过就自动创建 `v0.3.1` 标签和 Release；版本号没变时只编译不发布。
（直接推送 `v*` 标签也可以触发发布。）

不想安装也能先看面板效果：直接用浏览器打开 `www/index.html?demo`（招募阶段，可以点“优化站位”看效果）或 `www/index.html?demo=combat`（战斗阶段）。

## 文件结构

```
BGStateExporter.csproj    项目文件（引用 HDT 自带的 HearthstoneDeckTracker.exe / HearthDb.dll / BobsBuddy.dll / Newtonsoft.Json.dll）
BGStateExporterPlugin.cs  插件入口：每 ~100ms 读一次局面，有变化才输出
StateReader.cs            从 Core.Game 提取战旗局面
SimManager.cs             对战模拟调度：读 HDT 本场结果、自动对位模拟、站位优化
SimBridge.cs              调用 BobsBuddy.dll 和 HDT 内部工具（反射）
Advisor.cs                第一层建议器：枚举方案、模拟、打分
AdvisorWeights.cs         建议器权重（advisor_weights.json）
Learner.cs                自动学习：记录实际选择、按名次加权、拟合权重
UpdateChecker.cs          启动时检查 GitHub Releases，自动下载新版本
.github/workflows/        推送 v* 标签时自动编译并发布 Release
StateServer.cs            本机 HTTP 服务（只监听 localhost）+ SSE 推送
Models.cs                 JSON 数据结构
www/index.html            网页面板（编译时打包进 DLL）
tools/watch_state.py      Python 订阅示例，后续接 Claude / PPO 从这里开始
sample/state.json         样例数据
```

## JSON 格式（schema 1）

```jsonc
{
  "seq": 42,                    // 局面变化序号，每变一次 +1
  "ts": 1761300000000,          // 毫秒时间戳
  "inGame": true,
  "isBattlegrounds": true,
  "mode": "solo",               // solo | duos | none
  "phase": "recruit",           // menu | not_battlegrounds | hero_select | recruit | combat | game_over
  "turn": 9,
  "availableRaces": ["BEAST", "DRAGON", ...],   // 本局可用种族
  "anomalyDbfId": null,         // 本局畸变（如有）
  "player": {
    "heroCardId", "heroName", "health", "armor",
    "tavernTier",               // 酒馆等级
    "gold", "maxGold",          // 当前可用金币 / 本回合上限
    "extraGoldNextTurn", "leaderboardPlace",
    "heroPowers": [{ "entityId", "cardId", "name", "cost", "used" }],
    "shop": { "buyCost": 3, "refreshCost": 1, "freeRefreshes": 0, "canUpgrade": true, "upgradeCost": 7, ... },
    "board": [卡牌],            // 按位置从左到右
    "hand": [卡牌],
    "trinkets": [卡牌], "questRewards": [卡牌]
  },
  "tavern": [卡牌],             // 仅招募阶段：鲍勃酒馆里的随从和法术
  "opponent": {                 // 仅战斗阶段：本回合对手
    "heroCardId", "heroName", "health", "armor", "tavernTier", "board": [卡牌]
  },
  "leaderboard": [{ "place", "playerId", "heroCardId", "heroName", "health", "armor", "tavernTier", "isPlayer", "dead" }],
  "nextOpponentPlayerId": 12,   // 下回合对手（对应 leaderboard[].playerId）
  "simulation": {
    "available": true,
    "live": { "turn", "state", "errorState", "hasResult", ...模拟结果 },     // HDT 本场/上一场
    "matchups": { "status": "done", "turn": 9, "results": [
      { "opponentPlayerId", "heroCardId", "heroName", "isNextOpponent", "boardSeenTurn", "opponentMinionCount", "error", ...模拟结果 }
    ]},
    "advisor": { "status": "done", "turn": 9, "targetHeroName", "targetBoardSeenTurn", "plansConsidered", "plansSimulated", "elapsedMs", "message",
      "plans": [ { "rank": 1, "score": 0.62, "goldLeft": 2, "finalBoard": [entityId...], "sim": 模拟结果,
                   "steps": [ { "type": "sell|buy|upgrade|play|refresh|freeze|pass", "text": "买酒馆第 3 个：千折纸龙（2/3）", "entityId": 203, "gold": 5 } ],
                   "reasons": [...], "scoreParts": { "战力": 0.48, "升本": -0.12 } } ] },
    "optimizer": { "status": "done", "opponentPlayerId", "opponentHeroName", "boardSeenTurn",
                   "current": 模拟结果, "best": 模拟结果,
                   "currentOrder": [entityId...], "bestOrder": [entityId...], "bestOrderNames": [...],
                   "evaluated": 49, "message": "调整后（胜率 - 负率）提升 12.2 个百分点" }
  }
}
```

模拟结果：

```jsonc
{ "winRate": 0.583, "tieRate": 0.071, "lossRate": 0.346,
  "winLethal": 0.0,  "lossLethal": 0.0,        // 打死对手 / 被打死的概率
  "avgDamageDealt": 10.2, "avgDamageTaken": 8.3, // 赢时平均造成 / 输时平均承受
  "simulationCount": 3000 }
```

`status` 取值：`idle` 未开始、`waiting` 场面刚变化等待稳定、`running` 计算中、`done` 完成、`error` 出错（看 `error` / `message`）。

卡牌：

```jsonc
{ "entityId": 103, "cardId": "BG21_014", "dbfId": 0, "name": "奖品促销龙",
  "type": "minion",           // minion | spell | other
  "position": 3, "attack": 14, "health": 12, "tier": 3, "cost": 3,
  "races": ["DRAGON"],
  "golden": true, "taunt": false, "divineShield": false, "poisonous": false, "venomous": false,
  "reborn": false, "windfury": false, "stealth": false, "deathrattle": false, "frozen": false }
```

`entityId` 在一局内唯一，后续发操作指令（买第几个、把谁放到哪）时用它定位卡牌。

## 已知限制 / 需要实机确认

- 编译环境里拿不到 .NET 编译器，代码是对照 HDT 和 HearthDb 最新源码（2026-10-21）写的，Bob's Buddy 的接口按 HDT 源码里的用法调用；首次编译若有报错，把错误贴回来即可。
- 招募阶段酒馆取自鲍勃（对手位）的场面，这和 HDT 自己的悬浮提示用法一致。
- 双人模式下暂不导出队友信息。
- 端口 8765 被占用时，HDT 日志会出现 `[BGStateExporter] HTTP 服务启动失败`，文件输出不受影响。
- 卡图来自 art.hearthstonejson.com，离线时只显示边框和数字。
