"""训练里的奖励调整（rl/ppo.py）：无效刷新扣分、升本节奏奖励只在落后于真人等级曲线时给、逐渐减到 0。"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rl.ppo import Trainer  # noqa: E402
from tg.actions import A_BUY, A_END, A_REFRESH, A_UPGRADE  # noqa: E402


def main() -> int:
    # 无效刷新：刷新 → 刷新（第一次无效）→ 买（第二次有效）→ 刷新 → 结束（第三次无效）
    roll = SimpleNamespace(r=[], last=np.full(2, -1, dtype=np.int64), ep=np.zeros(2, dtype=np.int64))
    t = SimpleNamespace(cfg=SimpleNamespace(refresh_waste_penalty=0.01), ref_idx=np.full(2, -1, dtype=np.int64),
                        rollout=roll, mix=defaultdict(int))
    for a in (A_REFRESH, A_REFRESH, A_BUY, A_REFRESH, A_END):
        roll.r.append(0.0)
        roll.last[0] = len(roll.r) - 1
        Trainer._after_add(t, np.array([0]), np.array([a]))
    assert roll.r == [-0.01, 0.0, 0.0, -0.01, 0.0], roll.r
    assert t.mix["refresh_waste"] == 2 and t.mix["refresh"] == 3 and t.mix["buy"] == 1 and t.mix["end"] == 1

    # 升本奖励：第 3 回合真人约 2.05 本，门槛 2.55 → 1→2、2→3 给奖励（升之前 1、2 本），3→4 不给（升之前 3 本）
    t2 = SimpleNamespace(cfg=SimpleNamespace(tempo_bonus=0.03, tempo_anneal_iters=100), tempo_t0=None, iter=0,
                         tier_curve={1: 1.0, 3: 2.05, 9: 5.04}, upg_ep=np.full(1, -1, dtype=np.int64),
                         upg_n=np.zeros(1, dtype=np.int64), rollout=roll, mix=defaultdict(int))
    t2.tempo_now = lambda: Trainer.tempo_now(t2)
    t2._tier_target = lambda turn: Trainer._tier_target(t2, turn)
    f = np.zeros((1, 4), dtype=np.float32)
    f[0, 0] = 3 / 20
    r = Trainer._shape(t2, np.array([0]), f, np.array([A_UPGRADE]), np.zeros(1))
    assert abs(r[0] - 0.03) < 1e-9, r
    r = Trainer._shape(t2, np.array([0]), f, np.array([A_UPGRADE]), np.zeros(1))
    assert abs(r[0] - 0.03) < 1e-9, r
    r = Trainer._shape(t2, np.array([0]), f, np.array([A_UPGRADE]), np.zeros(1))
    assert r[0] == 0.0, r
    assert t2.mix["tempo_paid"] == 2
    t2.iter = 50
    assert abs(Trainer.tempo_now(t2) - 0.015) < 1e-9
    t2.iter = 500
    assert Trainer.tempo_now(t2) == 0.0
    assert Trainer._tier_target(t2, 5) == 2.05 and Trainer._tier_target(t2, 99) == 5.04
    print("shaping OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
