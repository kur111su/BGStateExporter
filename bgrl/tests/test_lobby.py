"""本局开了哪些种族（rl/comp.lobby_has_any）、老师不会的种族的 S / A 阵容（rl/comp.new_comps）：高手示范的 lobnew / tgtnew 标签用。只要 TavernGym。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from taverngym import BatchEnv                         # noqa: E402

from rl.comp import STRONG, WEAK, load_target_comps, lobby_has_any, new_comps   # noqa: E402
from rl.curricula import LobbyForcer, parse_types   # noqa: E402
from tg.obs import lobby                               # noqa: E402


def main():
    env = BatchEnv(4, seed=5, controlled=(0,), threads=1, config={"lobby_types": ["BEAST", "UNDEAD", "MURLOC"]})
    f = env.observe()["features"]
    for i in range(4):
        lob = lobby(f[i])
        assert {"BEAST", "UNDEAD", "MURLOC"} <= lob and len(lob) == 5, lob
        assert bool(lobby_has_any(f[i:i + 1], list(STRONG))[0]) == bool(lob & set(STRONG))
    assert lobby_has_any(f, ["BEAST"]).all() and lobby_has_any(f, []).all()
    assert lobby_has_any(f, ["BEAST"]).dtype == bool and len(lobby_has_any(f, ["BEAST"])) == 4
    assert not set(STRONG) & set(WEAK)
    nc = new_comps(load_target_comps())
    assert nc and all(c["tier"] in "SA" and c["tribes"] and not set(c["tribes"]) & set(STRONG) for c in nc)
    # 指定酒馆种族（单种族专家）：必开野兽、其余从老师不会的种族里抽 → 一定有野兽、没有野猪人 / 恶魔 / 元素；局号不变
    env = BatchEnv(6, seed=11, controlled=(0,), threads=1)
    eps = [env.episode_index(g) for g in range(6)]
    lf = LobbyForcer(parse_types("beast"), None, 1.0, {}, seed=3)
    assert lf.apply_all(env) and [env.episode_index(g) for g in range(6)] == eps
    f = env.observe()["features"]
    seen = set()
    for g in range(6):
        lob = lobby(f[g])
        assert "BEAST" in lob and not lob & set(STRONG) and len(lob) == 5, lob
        seen |= lob
    assert len(seen) > 5                                  # 其余 4 个是随机抽的，不是每局一样
    env.step(np.zeros(env.n_slots, dtype=np.int64))       # 换过的局能接着打
    assert lf.take_stats() == {"lobby_forced": 6}
    league = BatchEnv(2, seed=12, controlled=range(8), threads=1)   # 联赛（8 个座位都是学习者）也能换
    assert LobbyForcer(["BEAST", "UNDEAD"], ["MURLOC", "PIRATE", "DRAGON"], 1.0, {}, seed=1).apply_all(league)
    assert all({"BEAST", "UNDEAD", "MURLOC", "PIRATE", "DRAGON"} == lobby(league.observe()["features"][8 * g]) for g in range(2))
    assert not LobbyForcer(["BEAST"], None, 0.0, {}).apply_all(env)
    try:
        LobbyForcer(["BEAST"], ["UNDEAD"], 1.0, {})
        raise AssertionError("凑不出 5 个种族应该报错")
    except ValueError:
        pass
    print("lobby OK", len(nc), "套新种族阵容")


if __name__ == "__main__":
    main()
