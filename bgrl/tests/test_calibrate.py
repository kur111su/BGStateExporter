"""第二阶段的机器人加强（rl/calibrate.py）：目标曲线插值；加强后机器人的场面达到目标、学习者不变、读档后能接着打。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rl.calibrate import AdaptiveStrength, BotBooster, ramp, target_curve, target_for  # noqa: E402

REAL = json.loads((Path(__file__).resolve().parent.parent / "rl" / "real_profile.json").read_text(encoding="utf-8"))


def main() -> int:
    import taverngym
    c = target_curve(REAL)
    q1, med, q3 = c[11]
    assert target_for(c, 11, 0.0) == q1 and target_for(c, 11, 0.5) == med and target_for(c, 11, 1.0) == q3
    assert target_for(c, 99, 0.5) == c[max(c)][1]
    assert abs(ramp(0, 0, 200) - 0.3) < 1e-9 and ramp(200, 0, 200) == 1.0 and ramp(5, 0, 0) == 1.0

    a = AdaptiveStrength(start=0.4, target=3.75, band=0.25, step=0.05, window=20, min_games=10)
    for _ in range(9):
        a.add(2)
    assert a.update() == 0.4                                    # 新结果不够，不动
    a.add(2)
    assert abs(a.update() - 0.45) < 1e-9                        # 赢多了：加强
    for _ in range(20):
        a.add(7)
    assert abs(a.update() - 0.40) < 1e-9                        # 输多了：减弱
    a2 = AdaptiveStrength()
    a2.load_json(a.to_json())
    assert a2.strength == a.strength and list(a2.places) == list(a.places)

    env = taverngym.BatchEnv(8, seed=5, controlled=(0,), threads=1, config={"turn_action_caps": [14, 18, 22, 26, 30]})
    b = BotBooster(REAL, frac=1.0, start_turn=4, seed=1, every=1)
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(2500):
        o = env.observe()
        eps = [env.episode_index(i) for i in range(env.n_games)]
        before = {g: json.loads(env.save(g))["game"] for g in range(env.n_games)} if checked < 3 else None
        if b.boost(env, o["features"], eps):
            o = env.observe()
            if before is not None:
                for g in range(env.n_games):
                    after = json.loads(env.save(g))["game"]
                    if after["players"][1]["board"] != before[g]["players"][1]["board"]:
                        t = after["turn"]
                        assert after["players"][0]["board"] == before[g]["players"][0]["board"], "学习者不该被改"
                        for seat in range(1, 8):
                            p = after["players"][seat]
                            if p["board"] and p.get("alive", True):
                                tot = sum(m["atk"] + m["hp"] for m in p["board"])
                                goal = target_for(c, t, b.plan[g][1][seat])
                                assert tot >= goal * 0.9, (t, seat, tot, goal)
                        checked += 1
        m, r = o["action_mask"], o["ready"]
        acts = np.array([rng.choice(np.flatnonzero(mm)) if rr and mm.any() else 0 for mm, rr in zip(m, r)], dtype=np.int64)
        env.step(acts)
    g0 = next(iter(b.plan))
    assert b.finish(g0) is True and b.finish(g0) is None and g0 not in b.last_turn
    st = b.take_stats()
    assert checked >= 1 and st["boosts"] > 0, (checked, st)
    print("calibrate OK", st, "checked", checked)
    return 0


if __name__ == "__main__":
    sys.exit(main())
