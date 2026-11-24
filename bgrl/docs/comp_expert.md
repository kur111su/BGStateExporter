# 流派专家：单独练一套阵容，再让主模型向它学

第一个例子是龙虾流野兽（configs/exp/beast.json、runs/beast，2026-11-22 到 11-24，约 2700 轮）。
下赛季换阵容时，照这份文档走：`bg.py comp-expert` 生成配方，按下面的步骤训练、测量、蒸馏。
完整的试错过程在 docs/recipe_history.md 的“单种族专家 + 蒸馏”一节。

## 为什么要专家

主模型在随机酒馆里练，靠“野猪人 / 恶魔 / 元素”这几套现成打法就能拿名次，从来不去试新阵容。
龙虾流的例子：主模型在野兽酒馆里龙虾流成型 0%，龙虾 / 狮子鱼 / 陆行鸟的购买倍率全是 0.0。
原因是这套阵容做到一半时打不过别的，而策略梯度只会学“当下更好”的东西：它从没把这套做完过，就不知道做完以后有多强。
专家在一个“只有这套能做”的世界里练（固定酒馆 + 定向局），先学会这套；再在主模型训练里，
只在“这套能做、而且已经在做 / 酒馆里出了这套的牌”的局面上当老师。

## 第 0 步：这套阵容在模拟器里强不强

```
bg.py compstrength --comps <阵容名>
```

高手库里这套已经成型的场面，打同一回合的随机高手场面，看胜率（0.5 = 五五开）。
龙虾流实测：第 8–10 回合 0.41–0.46、第 13–14 回合 0.55–0.59；野猪人·邪能野猪人 0.63–0.73、元素·活化艾泽里特 0.63–0.88、
炫彩机器人流机械 0.69–0.79。**先练这里强的阵容回报最高**。

注意：高手库只有卡牌和身材，没有玩家身上的计数（龙虾加成、甲虫成长……），靠计数的阵容会被低估
（甲虫流 0.09–0.52、深潜巨兽 0.11–0.28）。`--lobster-bonus 0,10,40` 可以给龙虾流的场面补上加成再测：
第 8 回合 0.46 → 0.83，第 14 回合几乎不变。

## 第 1 步：生成配方、看起点

```
bg.py comp-expert --comp <阵容名> [--tribe BEAST] [--counter lobster_bonus] [--trick rally]
bg.py comp-expert --stats          # 列出玩家 stats 里的计数，挑专属奖励用的
bg.py goldflow --boosted --games 256 --lobby <种族> --ckpt runs/v3/best.pt   # 起点
```

生成的 configs/exp/<run>.json 里有什么（每一项都是龙虾流试出来的，括号里是没有它时的问题）：

| 配置 | 作用 |
|---|---|
| lobby_must / lobby_frac 1 / lobby_exam | 所有训练局和考试都开这个种族，其余从弱种族里抽（主模型有退路就不学新的） |
| teacher_coef 0 | 关掉外部老师（它只会三族，会往回拉） |
| gift_comps | 一半的局第 3–9 回合往手里送这套的核心 / 配套卡，按酒馆等级送（从没拿过的牌，价值网络不知道它值多少） |
| seed_comp_names | 成型阵容课程只换成这一套 |
| pro_weights comp-<名字>:5 | 高手示范里买这套的牌 ×5 |
| commit_* | 定向局，见下 |
| lobster_coef + counter_key | 专属奖励：玩家计数每涨 1 加多少（龙虾流真正学会是靠它） |
| trick_bonus + trick_engine | 招募阶段连招奖励，只在进击流场面上算 |

生成以后**一定要按打法改**：哪些牌必买（commit_force，默认核心卡；龙虾流还加了囤食土狼、深潜狮子鱼）、
专属奖励用哪个计数、站位和连招。

## 第 2 步：练专家

```
bg.py train --recipe configs/exp/<run>.json --init runs/v3/best.pt
bg.py train --recipe configs/exp/<run>.json --resume                 # 中断以后
bg.py train --recipe configs/exp/<run>.json --resume --restore runs/<run>/snapshots/itNNNNN.pt   # 退回某个快照
```

每轮一行里要看的：`定向局 X 名`、`龙虾加成 +N（最高 M）`（专属计数）、`连招 N 次`、考试的 `成型 S/A`。

### 定向局（commit_*）

机器人局里 commit_frac 的局，从第 3 回合起：
1. 只能买这套的核心 / 配套卡、本族随从、法术和饰品（`commit_comp`）；
2. commit_force 里的牌：在手里、场上有空位就只能先打出来；酒馆里有、买得起就只能买；场上的不能卖；
3. 升本节奏 commit_tiers（3:5,4:7,5:8,6:10 = 第 5 回合 3 本……），到了回合还没到、升得起就只能升；
4. 找后期核心卡 commit_hunt（5:提图斯;6:提图斯,陆行鸟 = 5 本以后缺提图斯、6 本以后缺任一张，钱 ≥ 4 就刷新）。

比例：开始 30%，模型学会这套值钱以后降到 10%（逼久了会拖累自由打法）。

### 踩过的坑

| 现象 | 原因 | 怎么改的 |
|---|---|---|
| 不限场面的连招奖励：连招 236 → 1900 次 / 轮，考试 4.23 → 4.70 | 模型找到别的“发动 / 卖就涨身材”的打法刷奖励 | trick_engine：只在这套的场面上算；退回快照（--restore） |
| 只限“能买什么”：拿龙虾的只有 9/128 局 | 改买法术、靠发现 / 黑暗之赐拿别的随从 | commit_force 必买 |
| 必买以后龙虾加成还是 0 | 买了放在手里不打出来 | 必打 |
| 必打以后定向测试 5.6 名、提图斯 3% | 升本太慢，5 / 6 本的核心卡刷不到 | commit_tiers 升本节奏 |
| 找陆行鸟以后第 9 回合刷新 5.5 次、升本 1% | 陆行鸟是 6 本卡，5 本时刷不到 | commit_hunt 按等级分档 |
| 送牌 + 示范 ×5 以后购买倍率还是 0.0 | 只做到一半打不过别的，推一下推不动 | 定向局 + 专属奖励 |
| 强制 1400 轮定向测试没进步 | 战斗里攒的龙虾加成没有奖励信号 | lobster_coef（专属计数奖励）——这一步之后考试里龙虾流第一次自己成型 |

## 第 3 步：测量

```
# 会不会打（和定向局一样的限制）
bg.py goldflow --boosted --games 128 --lobby <种族> --commit <阵容名> --commit-force … --commit-tiers … --commit-hunt "…" --ckpt runs/<run>/snapshots/itNNNNN.pt
# 会不会自己选（不加限制）
bg.py goldflow --boosted --games 256 --lobby <种族> --ckpt runs/<run>/snapshots/itNNNNN.pt
```

输出里看：
- 目标阵容那段：这套的成型率、成型的局平均名次；
- 最终阵容：走这个种族的局和混合阵容的局的名次；
- 挑卡：核心卡的购买倍率（0 = 不碰）；
- “龙虾流”一段（进击流专用）：最左边是不是进击随从、陆行鸟 / 提图斯在不在、每回合发动狮子鱼几次；
- “龙虾加成”一段：第一次拿到在第几回合、按回合的加成和总身材、早拿 / 晚拿 / 没拿的名次。

专家**合格**的标准：自由测试里走这套的局名次不比别的局差，成型的局明显更好。
龙虾流第 2700 轮：野兽局 4.32 = 混合 4.32，龙虾流成型 25%、这些局 3.58 名；拿过龙虾的局第 14 回合 4178 身材、加成 33。

注意 runs/<run>/best.pt 是守护按考试平均存的，练这套的过程中考试可能先变差，best.pt 可能是很早的版本——
蒸馏用测量过的快照（runs/<run>/snapshots/itNNNNN.pt），不要直接用 best.pt。

## 第 4 步：蒸馏到主模型（第二阶段）

```
bg.py comp-expert --comp <阵容名> --distill        # 生成 configs/exp/distill_<run>.json
# 打开它，把 experts 里的 ckpt 改成第 3 步选好的快照
bg.py train --recipe configs/exp/distill_<run>.json --resume
```

主模型照常训练，另外：
- 15% 的局强制开专家的种族、不开它排除的种族（lobby_frac 0.15，考试不强制），保证有专家擅长的局面；
- 在专家擅长的局面上（酒馆条件 + 第 3 回合起 + 场上 / 手里 / 酒馆有这套的牌），专家最想做的动作概率 ≥ conf，
  加 coef × −log π(专家的动作)（rl/experts.py；前 expert_ramp 轮慢慢加上）。
每轮一行里 `专家 局面 X% 一致 Y%`：X = 这一轮多少局面归专家管，Y = 主模型和专家有把握的动作一致的比例（应该慢慢升）。
多个专家：experts 是 JSON 列表，每个流派一项，各管各的局面。

蒸馏前后对比主模型 best.pt：
```
bg.py goldflow --boosted --games 256 --lobby <种族> --ckpt runs/v3/best.pt     # 这种酒馆
bg.py goldflow --boosted --games 256 --ckpt runs/v3/best.pt                     # 随机酒馆（不能变差）
```

## 下赛季换卡池时

1. 更新 rl/target_comps.json（bgbuddy.gg 的 S / A / B 阵容表）和高手库（prolib / prodemos）；
2. `bg.py compstrength` 看哪套在模拟器里强，挑 1–2 套；
3. `bg.py comp-expert --stats` 找这套的专属计数（没有计数的阵容，专属奖励先不开）；
4. 按上面 1–4 步走；每一步的结果记进 docs/recipe_history.md。
