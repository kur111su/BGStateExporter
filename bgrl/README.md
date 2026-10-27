# bgrl：酒馆战旗模拟环境 + 强化学习

方案见项目文档“酒馆战旗完整模拟方案”。这里按步骤逐步加代码，目前完成第 1 步。

## 第 1 步：检查 Bob's Buddy 能否脱离 HDT 运行

双击 `step1_bb_check.bat`。

- 需要 Python 3.10～3.13（推荐 3.12，安装时勾选 “Add python.exe to PATH”）。
- 第一次运行会在本目录建 `.venv` 并安装 pythonnet，失败会自动改用清华镜像。
- 默认读取 `..\..\HDT`（即 `D:\claude\HDT`）里的 BobsBuddy.dll，和 `%AppData%\HearthstoneDeckTracker\BGStateExporter\games` 里的对局记录。
  其它位置可以这样指定：`step1_bb_check.bat --hdt D:\你的\HDT --games D:\对局记录`
- HDT 开着也没关系，脚本只读 DLL 和对局记录，不改任何东西。

脚本会：
1. 在 Python 进程里加载 BobsBuddy.dll，列出主要类型和方法；
2. 用你对局记录里的每场真实战斗重新模拟，和当时 HDT 里 Bob's Buddy 的胜率对比；
3. 测速（同一场战斗反复模拟；大量不同战斗各模拟一次——训练环境就是这种用法）。

结果写到 `reports\bb_check_时间.txt`（给人看）和 `.json`（给程序用）。

对比时差距大的场次，多半是因为对局记录里没有附魔、成长计数、对手英雄技能这类信息，
不代表 Bob's Buddy 算错；训练环境里这些信息都由环境自己维护，不存在这个问题。
