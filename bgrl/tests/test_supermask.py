"""tg/supermask.py：从观察推出的动作集合必须包含所有真正合法的动作（随机打 TavernGym 对局检查）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tg.supermask import super_mask  # noqa: E402


def main(rounds: int = 600) -> int:
    from taverngym import BatchEnv
    env = BatchEnv(8, seed=5, controlled=tuple(range(8)), threads=2)
    obs = env.observe()
    rng = np.random.default_rng(1)
    F, C, M = [], [], []
    for _ in range(rounds):
        r = np.flatnonzero(obs["ready"])
        a = np.zeros(env.n_slots, dtype=np.int64)
        for i in r:
            m = obs["action_mask"][i]
            F.append(obs["features"][i].copy())
            C.append(obs["cards"][i].copy())
            M.append(m.astype(bool))
            legal = np.flatnonzero(m)
            nm = [x for x in legal if x < 196]
            a[i] = rng.choice(nm if nm and rng.random() < 0.9 else legal)
        env.step(a)
        obs = env.observe()
    F, C, M = np.array(F), np.array(C), np.array(M)
    S = super_mask(F, C)
    viol = M & ~S
    bad = {int(x): int(viol[:, x].sum()) for x in np.flatnonzero(viol.sum(0))}
    print(f"supermask：{len(F)} 个局面，合法 {M.sum(1).mean():.1f} 个 / 推出 {S.sum(1).mean():.1f} 个，漏掉 {bad}")
    assert viol.sum() <= len(F) * 0.0005, bad

    # tools/native_demos.py：把这些局面当成“别人的对局”转一遍（动作都是合法的，应该全部保留）
    import tempfile
    from tools.make_demos import KINDS
    from tools.native_demos import convert
    A = np.array([int(rng.choice(np.flatnonzero(m))) for m in M[:400]])
    turn = np.rint(F[:400, 0] * 20).astype(np.int16) * 2 - 1
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "x.npz"
        np.savez(p, features=F[:400].astype(np.float16), cards=C[:400].astype(np.int16), actions=A.astype(np.int16),
                 game_idx=np.arange(400, dtype=np.int16) // 50, turn=np.where(turn > 0, turn, -1),
                 synthetic=np.zeros(400, bool))
        data, boards, st, n_games, _ = convert(p, 13000)
    kept = int((turn > 0).sum()) - st["跳过：选择 / 取消"]
    assert len(data["kind"]) == kept and n_games == 8 and not st["跳过：动作和观察对不上"], (len(data["kind"]), kept, st)
    assert data["m"].shape[1] == (245 + 7) // 8 and set(data["cand"][:, 0].tolist()) <= set(A[turn > 0].tolist())
    # 训练里“买关键卡”的示范加权（rl/ppo.py Trainer._bought_in）：买的那格是关键卡才算
    import json as _json
    import taverngym
    from rl.ppo import Trainer
    key = _json.loads((Path(__file__).resolve().parent.parent / "rl" / "card_value.json").read_text(encoding="utf-8"))["key"]
    k_idx = next(i for i in range(1, taverngym.N_CARDS) if taverngym.card_info(i)[0] == key[0])
    other = next(i for i in range(1, taverngym.N_CARDS) if taverngym.card_info(i)[0] not in set(key)
                 and taverngym.card_info(i)[0].startswith("BG"))
    c = np.zeros((3, 124), np.int32)
    c[:, 36], c[:, 37] = k_idx, other
    dd = {"kind": np.array([KINDS.index("buy")] * 2 + [KINDS.index("end")], np.int8), "c": c,
          "cand": np.array([[8] + [-1] * 7, [9] + [-1] * 7, [0] + [-1] * 7], np.int16)}
    assert Trainer._bought_in(dd, "key").tolist() == [True, False, False]
    print("nativedemos OK", {KINDS[k]: int(v) for k, v in zip(*np.unique(data["kind"], return_counts=True))})
    return 0


if __name__ == "__main__":
    sys.exit(main(1500 if "--long" in sys.argv else 600))
