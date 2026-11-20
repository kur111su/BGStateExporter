# 实验配方

一个实验一个文件，只写和正式配方不同的几项，例如 `configs/exp/lam098.json`：

```json
{
 "base": "v3",
 "description": "C1：λ 0.995 → 0.98",
 "config": {"lam": 0.98}
}
```

`bg.py train --recipe configs/exp/lam098.json --resume` 先读 `configs/v3.json`，再用这里的 `config` 覆盖；不写 `run` 就接着练 `runs/v3`。
实验做完：留下的改动写回 `configs/v3.json`，结果记进 `docs/recipe_history.md`，这个文件删掉。
