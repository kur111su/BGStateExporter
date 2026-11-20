"""本局开了哪些种族（rl/comp.lobby_has_any）、老师不会的种族的 S / A 阵容（rl/comp.new_comps）：高手示范的 lobnew / tgtnew 标签用。只要 TavernGym。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from taverngym import BatchEnv                         # noqa: E402

from rl.comp import STRONG, WEAK, load_target_comps, lobby_has_any, new_comps   # noqa: E402
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
    print("lobby OK", len(nc), "套新种族阵容")


if __name__ == "__main__":
    main()
