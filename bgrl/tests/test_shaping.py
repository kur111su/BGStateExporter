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
    t2 = SimpleNamespace(cfg=SimpleNamespace(tempo_bonus=0.03, tempo_anneal_iters=100, tempo_deficit=0.0), tempo_t0=None, iter=0,
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
    # 落后扣分：第 9 回合目标 5.04，估计等级 1 + 已升 2 次 = 3 → 差 1.54 → 扣 0.0154；升本奖励已减到 0 也照扣
    t4 = SimpleNamespace(cfg=SimpleNamespace(tempo_bonus=0.0, tempo_deficit=0.01), tempo_t0=None, iter=0,
                         tier_curve={1: 1.0, 3: 2.05, 9: 5.04}, upg_ep=np.zeros(1, dtype=np.int64),
                         upg_n=np.array([2]), rollout=roll, mix=defaultdict(int))
    t4.tempo_now = lambda: Trainer.tempo_now(t4)
    t4._tier_target = lambda turn: Trainer._tier_target(t4, turn)
    roll.ep[0] = 0
    f9 = np.zeros((1, 4), dtype=np.float32)
    f9[0, 0] = 9 / 20
    r = Trainer._shape(t4, np.array([0]), f9, np.array([A_END]), np.zeros(1))
    assert abs(r[0] + 0.0154) < 1e-9, r
    t4.upg_n[0] = 5                                         # 6 本：不落后，不扣
    assert Trainer._shape(t4, np.array([0]), f9, np.array([A_END]), np.zeros(1))[0] == 0.0

    # 花掉升本的钱：第 2 回合 4 块、升本 4 块、1 本（目标 1.86 - 0.5 > 1 → 落后）
    #   发动 → 选（这一步花了 1 块：4 → 3，扣在“选”上）→ 买（3 → 0，之前就不够升本，不扣）→ 结束
    roll5 = SimpleNamespace(r=[], last=np.full(1, -1, dtype=np.int64), ep=np.zeros(1, dtype=np.int64))
    t5 = SimpleNamespace(cfg=SimpleNamespace(upgrade_starve_penalty=0.04, starve_buy_weight=0.5, unspent_gold_penalty=0.0,
                                            starve_buy_from_turn=4, starve_gift_weight=2.0), tier_curve={1: 1.0, 2: 1.86, 5: 3.17},
                         rollout=roll5, mix=defaultdict(int))
    t5._tier_target = lambda turn: Trainer._tier_target(t5, turn)

    def step(a, turn, gold, up, tier):
        fx = np.zeros((1, 16), dtype=np.float32)
        fx[0, 0], fx[0, 7], fx[0, 10], fx[0, 11] = turn / 20, gold / 10, tier / 6, up / 10
        r = Trainer._gold_shape(t5, np.array([0]), fx, np.array([a]), np.zeros(1))
        roll5.r.append(float(r[0]))
        roll5.last[0] = len(roll5.r) - 1
    for a, gold, up, tier in ((23, 4, 4, 1), (30, 4, 4, 1), (A_BUY, 3, 4, 1), (A_END, 0, 4, 1)):
        step(a, 2, gold, up, tier)
    assert roll5.r == [0.0, -0.04, 0.0, 0.0], roll5.r
    assert t5.mix["starve"] == 1 and t5.mix["t2_n"] == 1 and t5.mix["t2_up"] == 0 and t5.mix["aff_n"] == 1
    # 下一局：先升本（4 → 0，升本本身不扣）再结束 → 第 2 回合升本率 1/2
    roll5.ep[0] = 1
    roll5.r.clear()
    step(A_UPGRADE, 2, 4, 4, 1)
    step(A_END, 2, 0, 6, 2)
    assert roll5.r == [0.0, 0.0] and t5.mix["t2_up"] == 1 and t5.mix["aff_up"] == 1 and t5.mix["starve"] == 1
    # 第 5 回合 2 本、7 块、升本 6 块（目标 3.17 - 0.5 > 2 → 落后）：买随从（7 → 4）扣一半；剩 6 块直接结束也扣
    roll5.ep[0] = 2
    roll5.r.clear()
    step(A_BUY, 5, 7, 6, 2)
    step(A_END, 5, 4, 6, 2)
    assert roll5.r == [-0.02, 0.0], roll5.r
    roll5.r.clear()
    step(A_END, 6, 9, 6, 2)
    assert roll5.r == [-0.04] and t5.mix["end_rich"] == 1, roll5.r
    # 不扣买随从（starve_buy_weight 0）；剩下的钱按块扣（unspent_gold_penalty）
    t5.cfg.starve_buy_weight, t5.cfg.unspent_gold_penalty = 0.0, 0.005
    roll5.ep[0] = 3
    roll5.r.clear()
    step(A_BUY, 5, 7, 6, 2)
    step(A_END, 5, 2, 6, 2)
    assert roll5.r == [0.0, -0.01], roll5.r
    # 第 2 回合买随从把升本的钱花掉：照常扣（starve_buy_from_turn 4）
    roll5.ep[0] = 4
    roll5.r.clear()
    step(A_BUY, 2, 4, 4, 1)
    step(A_END, 2, 1, 4, 1)
    assert roll5.r == [-0.04, -0.005], roll5.r
    # 黑暗之赐把升本的钱花掉：×2（starve_gift_weight）
    from tg.actions import A_DARK_GIFT
    roll5.ep[0] = 5
    roll5.r.clear()
    step(A_DARK_GIFT, 5, 7, 6, 2)
    step(A_END, 5, 4, 6, 2)
    assert roll5.r == [-0.08, -0.02], roll5.r

    # 模仿高手的强度：训练示范比留出局高出 pro_gap_cap 以上就减半，差距小了再加回来
    t6 = SimpleNamespace(cfg=SimpleNamespace(pro_gap_cap=0.12, pro_scale_min=0.25), pro_scale=1.0)
    for tr, va, want in ((0.70, 0.41, 0.7), (0.60, 0.42, 0.49), (0.60, 0.42, 0.343), (0.60, 0.42, 0.25), (0.60, 0.42, 0.25),
                         (0.50, 0.42, 0.25), (0.46, 0.42, 0.3125)):
        Trainer.adjust_pro(t6, {"pro_top1_train": tr, "pro_top1": va})
        assert abs(t6.pro_scale - want) < 1e-9, (tr, va, t6.pro_scale)

    # 高分示范按类别 + 按回合加权：t2:2 第 2 回合 ×2，t3-4:1.5 第 3–4 回合 ×1.5
    from tools.make_demos import KINDS
    d = {"kind": np.array([KINDS.index(k) for k in ("upgrade", "buy", "upgrade", "end")]), "turn": np.array([2, 3, 5, 4])}
    w = Trainer._bc_weights(SimpleNamespace(cfg=SimpleNamespace(bc_weights=""), demos=None), d, "upgrade:3,t2:2,t3-4:1.5")
    assert w.tolist() == [6.0, 1.5, 3.0, 1.5], w

    # 模仿强度：训练示范命中率高于上限减半（最低 0.05），低于上限 - 0.1 回升
    t3 = SimpleNamespace(cfg=SimpleNamespace(bc_train_cap=0.6, bc_scale_min=0.05), bc_scale=1.0)
    for tr, want in ((0.99, 0.5), (0.95, 0.25), (0.55, 0.25), (0.4, 0.375), (0.3, 0.5625)):
        Trainer.adjust_bc(t3, {"human_top1_train": tr})
        assert abs(t3.bc_scale - want) < 1e-9, (tr, t3.bc_scale, want)
    for _ in range(10):
        Trainer.adjust_bc(t3, {"human_top1_train": 0.99})
    assert t3.bc_scale == 0.05
    print("shaping OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
