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

    m = build_model()
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

    # GAE：γ=1、λ=1 时优势 = 最终奖励 - 价值
    r = Rollout()
    for i in range(3):
        r.add(("x", 0), f=np.zeros(3, np.float16), c=np.zeros(2, np.int32), m=np.zeros(1, np.uint8), a=0, logp=0.0, v=0.1 * i)
    r.reward(("x", 0), 1.0, True, 1)
    b = r.collect(1.0, 1.0, {})
    assert np.allclose(b["adv"], [1.0, 0.9, 0.8]) and np.allclose(b["ret"], 1.0) and list(b["place"]) == [1, 1, 1]
    print("GAE 检查通过")
    print("TEST OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
