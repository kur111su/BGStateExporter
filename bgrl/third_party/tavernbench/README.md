# TavernBench（第三方，MIT）

来自 [MCG-NJU/AutoTavern](https://github.com/MCG-NJU/AutoTavern) 提交 fb61201（TavernGym and TavernBench v1.0.0），和我们锁定的 TavernGym 是同一个提交。版权和许可见 LICENSE、NOTICE。

只拷了用得到的部分：
- `tavernbench/`：engine（局面、执行选项）、score、chance、report（打分和结果表）。没改动。
- `data/eval/`：论文用的 368 题评测集（题目、答案、模拟器局面）。
- `results/eval/`：论文公布的各系统答案（TavernAgent、Jev 和 10 个大语言模型；kimi-k2.6 的文件 2 MB，没拷）。

我们自己的模型用 `bgrl/tools/tavernbench.py` 答题（`bg.py tavernbench` / `tavernbench.bat`）。
