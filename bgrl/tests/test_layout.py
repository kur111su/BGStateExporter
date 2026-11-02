"""观察 / 动作 / 卡牌目录的共用常量（tg/obs.py、tg/actions.py、tg/catalog.py）和 TavernGym 实际布局一致。

TavernGym 换版本、观察布局变了的时候这里先报错，而不是训练 / 统计脚本悄悄读错位置。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    import taverngym
    from rl.model import Layout
    from tg import actions as A
    from tg import obs as O
    from tg.catalog import base_id, base_of_index, card_id, catalog, is_golden_id, n_cards

    L = dict(taverngym.layout())
    lay = Layout.from_taverngym()
    fs, cs = lay.fslices(), lay.cslices()

    # 浮点：场上随从、选项从哪开始，每格多长
    assert O.F_BOARD == fs["board"][0] and O.F_CHOICE == fs["choice"][0], (fs, O.F_BOARD, O.F_CHOICE)
    assert O.MINION_F == L["MINION_F"] and O.ITEM_F == L["ITEM_F"]
    # 卡牌编号：技能 / 场上 / 手牌 / 酒馆 / 选项
    assert (O.C_POWERS, O.C_BOARD, O.C_HAND, O.C_SHOP, O.C_CHOICE) == tuple(
        cs[k][0] for k in ("powers", "board", "hand", "shop", "choice")), cs
    assert (O.N_BOARD, O.N_HAND, O.N_SHOP, O.N_CHOICE, O.N_POS) == tuple(
        L[k] for k in ("N_BOARD", "N_HAND", "N_SHOP", "N_CHOICE", "N_POS"))
    # 动作编号
    for k, v in L.items():
        if k.startswith("A_"):
            assert getattr(A, k) == v, (k, getattr(A, k), v)
    assert A.N_ACTIONS == L["N_ACTIONS"] == taverngym.N_ACTIONS

    fam = [A.action_family(a) for a in range(A.N_ACTIONS)]
    assert fam[A.A_END] == "end" and fam[A.A_UPGRADE] == "upgrade" and fam[A.A_HERO_POWER + 1] == "hero_power"
    assert set(fam[A.A_BUY:A.A_SELL]) == {"buy"} and set(fam[A.A_SELL:A.A_ACTIVATE]) == {"sell"}
    assert set(fam[A.A_PLAY:A.A_MAGNETIZE]) == {"play"} and set(fam[A.A_MOVE:]) == {"move"}

    # 观察里读出的回合 / 金币 / 等级和引擎状态对得上
    import json
    g = taverngym.Game(seed=3)
    for _ in range(400):
        if g.is_over():
            break
        f, _c = g.observe(0)
        f = np.frombuffer(f, "<f4")
        s = json.loads(g.state_json())
        me = s["players"][0]
        assert int(O.tier(f)) == me["tier"], (int(O.tier(f)), me["tier"])
        assert int(O.gold(f)) == me["gold"], (int(O.gold(f)), me["gold"])
        for p in range(g.num_players):
            if g.needs_action(p):
                g.step(p, g.bot_action(p))

    # 卡牌目录
    det = catalog()
    assert n_cards() == taverngym.N_CARDS and len(det) > 100
    for i in range(1, n_cards()):
        cid = card_id(i)
        assert cid == taverngym.card_info(i)[0]
        assert base_of_index(i) == base_id(cid) and (base_id(cid) != cid) == is_golden_id(cid)
    assert base_id("BG_X_G") == "BG_X" and base_id("BG_X") == "BG_X" and is_golden_id("BG_X_G")
    print("布局 / 动作 / 卡牌目录一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
