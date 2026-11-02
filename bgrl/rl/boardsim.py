"""阵容强度：拿引擎的战斗模拟（TavernGym 调试命令 combat_preview）量“我的场面打同一回合高分阵容能赢几成”。

为什么要它：主种族纯度、万三常见搭配、关键卡这些都是从真人数据里数出来的，模型在模拟器里玩的是龙 / 海盗，
那些搭配一对都用不上，关键卡多的局反而名次更差（阵容测试第 11600 轮：前四 0.50 张、后四 0.70 张）。
直接让引擎打一遍最实在：亡语、复生、开战效果、光环、金色、站位、两张卡之间的配合，战斗里该算的都算了。

  BoardJudge.judge(game, seat)    这个座位当前场面对“同一回合的一组高分阵容”的平均胜率（平局算半场）
  lab_game() / put_board()        单独的实验局，给 tools/simcards.py 摆任意阵容
  strength(game, a, b, n)         a 打 b 的胜率

高分阵容来自 runs/pro_boards.json（6500+）和 runs/h13_boards.json（万三），格式 {回合: [{"tier", "board": [[卡牌ID, 攻, 血], …]}]}；
只有卡牌、攻 / 血、金色（附魔没有），和训练里给机器人换阵容（rl/calibrate.ProBoardBooster）是同一份。
每回合固定抽 k 个当“考官”（同一个尺子，前后回合的分数才能比），每个打 n 场。不需要 torch。
"""
from __future__ import annotations

import json
import random

from tg.catalog import base_id, is_golden_id

_dumps = json.dumps


def put_board(game, seat: int, board: list, tier: int | None = None) -> int:
    """把 seat 的场面换成 board（[[卡牌ID, 攻, 血], …]，最多 7 个），返回放上去的随从数。放不上的卡跳过。"""
    game.debug(_dumps({"cmd": "clear_zone", "p": seat, "zone": "board"}))
    n = 0
    for cid, atk, hp in board[:7]:
        try:
            game.debug(_dumps({"cmd": "add_minion", "p": seat, "zone": "board", "card": base_id(cid),
                               "golden": is_golden_id(cid), "atk": max(0, int(atk)), "hp": max(1, int(hp))}))
            n += 1
        except Exception:                          # noqa: BLE001 —— 这个版本的引擎里没有这张卡
            pass
    if tier:
        try:
            game.debug(_dumps({"cmd": "set_player", "p": seat, "tier": int(tier)}))
        except Exception:                          # noqa: BLE001
            pass
    return n


def strength(game, a: int, b: int, n: int = 4) -> float:
    """a 打 b 的胜率（平局算 0.5），引擎在副本上打 n 场，不改 game。"""
    r = json.loads(game.debug(_dumps({"cmd": "combat_preview", "a": a, "b": b, "n": int(n), "details": True})))
    return float(r["win_a"]) + 0.5 * float(r["tie"])


def lab_game(seed: int = 0, turn: int = 8):
    """空的实验局：8 个座位都在，场面都空，回合设成 turn。"""
    import taverngym
    g = taverngym.Game(seed=seed)
    try:
        g.debug(_dumps({"cmd": "set_turn", "turn": int(turn)}))
    except Exception:                              # noqa: BLE001
        pass
    return g


class BoardJudge:
    """Φ = 场面对同一回合 k 个高分阵容的平均胜率。每回合的 k 个阵容开始时抽好，之后不变。"""

    def __init__(self, lib: dict, k: int = 8, n: int = 4, seed: int = 0, lag: int = 3):
        """k 个考官平均分到第 t、t-1 … t-lag 回合（只用同一回合的话，后期大多数场面一个都打不过，分数全是 0，
        看不出好坏；带上前几回合的阵容，“落后高手两回合”和“落后四回合”也分得开）。"""
        self.lib = {int(t): v for t, v in lib.items() if v}
        self.turns = sorted(self.lib)
        self.k, self.n, self.lag = k, n, lag
        self.seed = seed
        self.panel: dict[int, list] = {}
        self.calls = 0
        self.failed = 0

    def _near(self, t: int) -> int:
        ok = [x for x in self.turns if x <= t]
        return ok[-1] if ok else self.turns[0]

    def boards(self, turn: int) -> list:
        if not self.turns:
            return []
        turn = self._near(int(turn))
        if turn not in self.panel:
            rng = random.Random(self.seed * 1000 + turn)
            per = [self.k // (self.lag + 1) + (1 if i < self.k % (self.lag + 1) else 0) for i in range(self.lag + 1)]
            out = []
            for d, m in enumerate(per):
                src = self.lib[self._near(max(self.turns[0], turn - d))]
                out += rng.sample(src, min(m, len(src)))
            self.panel[turn] = out
        return self.panel[turn]

    def judge(self, game, seat: int, turn: int | None = None) -> float | None:
        """game 会被改（对手座位的场面换掉）：传 env.game(i) 拿到的副本。出错返回 None。"""
        boards = self.boards(int(game.turn if turn is None else turn))
        if not boards:
            return None
        opp = (seat + 1) % game.num_players
        try:
            out = []
            for e in boards:
                if put_board(game, opp, e["board"]):
                    out.append(strength(game, seat, opp, self.n))
            self.calls += 1
            return sum(out) / len(out) if out else None
        except BaseException as ex:                # noqa: BLE001 —— 引擎内部出错（PanicException）：这次不算
            from tg.actions import is_panic
            if not is_panic(ex) and not isinstance(ex, ValueError):
                raise
            self.failed += 1
            return None
