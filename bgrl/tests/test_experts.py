"""rl/experts.py：配置解析、酒馆条件（不用开模拟器的部分）。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl.experts import _lobby_ok, parse_experts  # noqa: E402
from tg.obs import F_LOBBY, LOBBY_IDX  # noqa: E402


def test_parse():
    e = parse_experts('[{"ckpt": "runs/beast/best.pt", "must": "BEAST"}]')
    assert e[0]["name"] == "beast" and e[0]["coef"] == 0.05 and e[0]["from_turn"] == 3
    assert parse_experts("") == []
    try:
        parse_experts('[{"must": "BEAST"}]')
        raise AssertionError("缺 ckpt 应该报错")
    except ValueError:
        pass


def test_lobby():
    f = np.zeros((3, max(LOBBY_IDX) + 1), np.float32)
    on = lambda r, t: f.__setitem__((r, LOBBY_IDX[F_LOBBY.index(t)]), 1.0)  # noqa: E731
    on(0, "BEAST"); on(0, "MURLOC")
    on(1, "BEAST"); on(1, "DEMON")
    on(2, "MURLOC")
    ok = _lobby_ok(f, "BEAST", "QUILBOAR,DEMON,ELEMENTAL")
    assert ok.tolist() == [True, False, False], ok


if __name__ == "__main__":
    test_parse()
    test_lobby()
    print("test_experts 通过")
