"""B2/B3 冒烟测试（CI 用，CPU 即可）：模型形状、动作屏蔽、动作头顺序、参数量、GAE。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import taverngym  # noqa: E402
import taverngym._core as core  # noqa: E402
from rl.model import build_model, n_params  # noqa: E402
from rl.ppo import Rollout  # noqa: E402


def main() -> int:
    # 动作头顺序和 TavernGym 一致
    names = [core.action_name(a) for a in range(core.N_ACTIONS)]
    assert names[4] == "HeroPower(0)" and names[6] == "DarkGift" and names[7] == "Cancel"
    assert names[46 + 3 * 8 + 5] == "Play { hand: 3, pos: 5 }"
    assert names[126 + 4 * 7 + 2] == "Magnetize { hand: 4, board: 2 }"
    assert names[196 + 5 * 7 + 1] == "Move { from: 5, to: 1 }"

    from rl.model import SIZES
    for name, kw in SIZES.items():
        print(f"{name}: {n_params(build_model(**kw)) / 1e6:.2f}M")
    m = build_model(**SIZES["small"])
    print(("::notice::" if os.environ.get("GITHUB_ACTIONS") else "") + f"默认模型参数 {n_params(m) / 1e6:.2f}M")
    assert 1.5e6 < n_params(m) < 6e6

    env = taverngym.BatchEnv(n_games=8, seed=1, controlled=range(8), threads=2)
    small = build_model(64, 2, 2).eval()
    steps = 0
    for _ in range(300):
        o = env.observe()
        f, c, mk = (torch.as_tensor(o["features"]), torch.as_tensor(o["cards"]).long(), torch.as_tensor(o["action_mask"]))
        logits, v, pl = small(f, c, mk)
        assert logits.shape == (env.n_slots, core.N_ACTIONS) and v.shape == (env.n_slots,) and pl.shape[1] == 8
        a, logp, _ = small.act(f, c, mk)
        ready = o["ready"]
        assert all(o["action_mask"][i, a[i]] for i in np.flatnonzero(ready)), "选了不合法的动作"
        _, _, info = env.step(a.numpy())
        assert not info["rejected"][ready].any()
        steps += int(ready.sum())
    print(f"随机初始化模型走了 {steps} 个决策，没有不合法动作")

    # GAE：γ=1、λ=1 时优势 = 最终奖励 - 价值；另一个槽位没打完，用自举价值
    r = Rollout(2)
    for i in range(3):
        r.add([0, 1], np.zeros((2, 3), np.float32), np.zeros((2, 2), np.int32), np.zeros((2, 8), bool),
              np.zeros(2, np.int64), np.zeros(2, np.float32), np.full(2, 0.1 * i, np.float32))
    r.reward(0, 1.0, True, 1)
    b = r.collect(1.0, 1.0, np.asarray([0.0, 0.5], np.float32))
    s0, s1 = b["slot"] == 0, b["slot"] == 1
    assert np.allclose(b["adv"][s0], [1.0, 0.9, 0.8]) and np.allclose(b["ret"][s0], 1.0), b
    assert np.allclose(b["ret"][s1], 0.5) and list(b["place"][s0]) == [1, 1, 1] and list(b["place"][s1]) == [0, 0, 0]
    print("GAE 检查通过")

    # 混合精度：GPU 上用 float16，CPU 上用 bfloat16 走一遍前向和反向（之前 float16 下屏蔽值溢出）
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float16 if dev == "cuda" else torch.bfloat16
    small = small.to(dev).train()
    o = env.observe()
    f, c, mk = (torch.as_tensor(o["features"], device=dev), torch.as_tensor(o["cards"], device=dev).long(),
                torch.as_tensor(o["action_mask"], device=dev))
    with torch.autocast(dev, dtype=dt):
        logits, v, pl = small(f, c, mk)
    loss = torch.nn.functional.log_softmax(logits, -1).logsumexp(-1).mean() + v.mean() + pl.mean()
    loss.backward()
    assert torch.isfinite(loss), loss
    print(f"混合精度（{dev} {dt}）前向反向通过")
    print("TEST OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
