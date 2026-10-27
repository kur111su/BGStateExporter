# bgrl：酒馆战旗模拟环境 + 强化学习

方案见项目文档“酒馆战旗完整模拟方案”。这里按步骤逐步加代码，目前完成第 1 步。

## 第 1 步：检查 Bob's Buddy 能否脱离 HDT 运行

双击 `step1_bb_check.bat`。

- 需要 .NET SDK（编译插件时装过的那个）和 Python 3.10～3.13（推荐 3.12，安装时勾选 “Add python.exe to PATH”）。
- 脚本先编译 `simhost\BGSimHost.dll`（Bob's Buddy 批量模拟接口），再建 `.venv`、装 pythonnet（失败会自动改用清华镜像），最后运行检查。
- 默认读取 `..\..\HDT`（即 `D:\claude\HDT`）里的 BobsBuddy.dll，和 `%AppData%\HearthstoneDeckTracker\BGStateExporter\games` 里的对局记录。
  HDT 在别处时，先在命令行 `set HDT_PATH=D:\你的\HDT` 再运行。
- HDT 开着也没关系，脚本只读 DLL 和对局记录，不改任何东西。

脚本会：
1. 加载 BGSimHost.dll 和 BobsBuddy.dll，跑冒烟测试；
2. 用你对局记录里的每场真实战斗重新模拟，和当时 HDT 里 Bob's Buddy 的胜率对比（两种调用方式各一次）；
3. 测速：大量不同战斗各模拟一次（训练环境就是这种用法）。

结果写到 `reports\bb_check_时间.txt`（给人看）和 `.json`（给程序用）。

对比时第 7 回合以后差距大的场次，多半是因为对局记录里没有附魔、成长计数、对手英雄技能和饰品这类信息，
不代表 Bob's Buddy 算错；训练环境里这些信息都由环境自己维护，不存在这个问题。

## 目录

```
simhost/      BGSimHost（C#）：Python 一次传一批战斗 JSON，在 .NET 里构造输入并行模拟
bgsim/        Python 模拟环境（目前只有 bobsbuddy.py：调用 BGSimHost 的封装）
tools/        检查和分析脚本
reports/      运行报告（不进仓库）
```
