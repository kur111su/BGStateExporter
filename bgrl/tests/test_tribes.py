"""种族课程（rl/tribes.py）：换进来的局真的开了指定的种族、编号接得上、之后的局恢复随机；老师按种族筛局面。只要 TavernGym。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                     # noqa: E402
from taverngym import BatchEnv                         # noqa: E402

from rl.tribes import STRONG, WEAK, TribeLobbies, lobby_has_any, parse_types   # noqa: E402
from tg.obs import lobby                               # noqa: E402


def main():
    assert parse_types("beast, Undead") == ["BEAST", "UNDEAD"]
    try:
        parse_types("NAGA")
        raise AssertionError("不认识的种族没报错")
    except ValueError:
        pass
    env = BatchEnv(4, seed=5, controlled=(0,), threads=1, config={})
    tl = TribeLobbies(1.0, list(WEAK), 3, {}, seed=1)
    idx0 = env.episode_index(2)
    assert tl.apply(env, 2)
    forced = tl.active[2]
    assert len(forced) == 3 and set(forced) <= set(WEAK)
    obs = env.observe()
    lob = lobby(obs["features"][2])
    assert set(forced) <= lob and len(lob) == 5, (forced, lob)
    assert env.episode_index(2) == idx0
    assert lobby_has_any(obs["features"][2:3], list(forced)).all()
    has_strong = lobby_has_any(obs["features"], list(STRONG))
    assert has_strong.dtype == bool and len(has_strong) == 4
    assert lobby_has_any(obs["features"], []).all()
    # 打完这局：编号往下走，之后开的局是普通的（不一定含指定种族）；finish 交回强制的种族
    seen = []
    for _ in range(20000):
        o = env.observe()
        a = np.zeros(env.n_slots, dtype=np.int64)
        for i in np.flatnonzero(o["ready"]):
            m = o["action_mask"][i]
            a[i] = 0 if m[0] else int(np.flatnonzero(m)[0])
        _, dones, _ = env.step(a)
        if dones[2]:
            assert tl.finish(2) == forced and tl.finish(2) is None
            assert env.episode_index(2) == idx0 + 1
            seen.append(lobby(env.observe()["features"][2]))
            break
    assert seen, "这局没打完"
    # 概率 0：不换
    assert not TribeLobbies(0.0, list(WEAK), 3, {}).apply(env, 1)
    print("tribes ok", forced, sorted(lob))


if __name__ == "__main__":
    main()
