"""阵容强度（rl/boardsim.py）、阵容强度 / 经济奖励（rl/shaping.py _combat_shape / _econ_shape）、模仿时打乱酒馆顺序（rl/demos.permute_shop）、引擎卡（rl/demos.bought_in engine）。"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _game_at(turn: int, seed: int = 3):
    import taverngym
    g = taverngym.Game(seed=seed)
    while g.turn < turn and not g.is_over():
        for p in range(g.num_players):
            if g.needs_action(p):
                g.step(p, g.bot_action(p))
    return g


def _obs(g, p=0):
    f, c = g.observe(p)
    return (np.frombuffer(f, "<f4").copy(), np.frombuffer(c, "<i4").copy(),
            np.frombuffer(g.legal_mask(p), dtype=np.uint8).astype(bool))


def check_permute_shop():
    """打乱后的观察 = 在引擎里把酒馆列表换成同样顺序以后再 observe() 的结果。"""
    import taverngym
    from rl.demos import permute_shop
    from rl.model import Layout
    from tg.actions import A_BUY
    lay = Layout.from_taverngym()
    shop_f = lay.fslices()["shop"][0]
    checked = 0
    for seed in range(6):
        g = _game_at(4 + seed % 3, seed)
        p = next((q for q in range(g.num_players) if g.needs_action(q)), None)
        if p is None:
            continue
        f, c, m = _obs(g, p)
        n = int((c[36:44] > 0).sum())
        if n < 3:
            continue

        class R:                                       # 让 permute_shop 用指定的顺序：倒过来
            def random(self, shape):
                return np.tile(np.linspace(0.9, 0.1, shape[1]), (shape[0], 1))
        cand = np.array([[A_BUY + 0, A_BUY + n - 1, 0, -1, -1, -1, -1, -1]], np.int16)
        f2, c2, m2, cand2 = permute_shop(f[None], c[None], m[None], cand, R(), shop_f, lay.IF, lay.NS)
        s = json.loads(g.state_json())
        shop = s["players"][p]["shop"]
        s["players"][p]["shop"] = shop[:n][::-1] + shop[n:]
        h = taverngym.Game.from_json(json.dumps(s))
        fe, ce, me = _obs(h, p)
        assert np.array_equal(c2[0], ce), (c[36:44], c2[0, 36:44], ce[36:44])
        assert np.allclose(f2[0], fe), np.flatnonzero(~np.isclose(f2[0], fe))[:10]
        assert np.array_equal(m2[0], me)
        assert cand2[0, :3].tolist() == [A_BUY + n - 1, A_BUY + 0, 0], cand2
        checked += 1
    assert checked >= 3, checked
    return checked


def check_judge():
    """强的场面 > 弱的场面 > 空场面；副本被改不影响原局。"""
    from rl.boardsim import BoardJudge, put_board
    from tg.catalog import catalog
    det = catalog()
    mins = sorted((x for x in det.values() if x.get("kind") == "minion" and x.get("tier") in (1, 2)
                   and not x.get("races") == ["ALL"]), key=lambda x: x["id"])[:30]
    lib = {t: [{"tier": 2, "board": [[x["id"], 3 + t, 3 + t] for x in mins[i:i + 5]]} for i in range(0, 25, 5)]
           for t in range(1, 12)}
    J = BoardJudge(lib, k=4, n=3, seed=0, lag=1)
    g = _game_at(6)
    before = g.state_json()
    strong, weak, empty = g.copy(), g.copy(), g.copy()
    put_board(strong, 0, [[x["id"], 30, 30] for x in mins[:7]])
    put_board(weak, 0, [[mins[0]["id"], 1, 1]])
    empty.debug(json.dumps({"cmd": "clear_zone", "p": 0, "zone": "board"}))
    vs, vw, ve = J.judge(strong, 0), J.judge(weak, 0), J.judge(empty, 0)
    assert vs > 0.9 and vw < vs and ve <= vw, (vs, vw, ve)
    assert g.state_json() == before
    return vs, vw, ve


def check_combat_shape():
    """势能差：整局加起来 = 系数 × (最后一次 - 第一次)；换了一局重新开始算。"""
    from rl.ppo import Trainer
    from tg.actions import A_BUY, A_END

    class FakeJudge:
        def __init__(self):
            self.v = iter([0.2, 0.5, 0.4, 0.9, 0.3])

        def judge(self, game, seat, turn=None):
            return next(self.v)

    class FakeEnv:
        def slot(self, i):
            return i, 0

        def game(self, g):
            return None

    t = SimpleNamespace(cfg=SimpleNamespace(combat_bonus=0.5, combat_from_turn=4, combat_frac=1.0), judge=FakeJudge(),
                        rollout=SimpleNamespace(ep=np.zeros(2, dtype=np.int64)), mix=defaultdict(float),
                        cs_ep=np.full(2, -1, dtype=np.int64), cs_prev=np.zeros(2))
    f = np.zeros((1, 16), np.float32)
    tot = []
    for turn in (3, 4, 5, 6, 7):                       # 第 3 回合不量
        f[0, 0] = turn / 20
        tot.append(float(Trainer._combat_shape(t, FakeEnv(), 0, np.array([0]), f, np.array([A_END]), np.zeros(1))[0]))
    assert tot[0] == 0.0 and abs(sum(tot) - 0.5 * (0.9 - 0.2)) < 1e-9, tot
    t.rollout.ep[0] = 1                                # 新的一局：第一次量只记下，不给分
    f[0, 0] = 5 / 20
    assert Trainer._combat_shape(t, FakeEnv(), 0, np.array([0]), f, np.array([A_END]), np.zeros(1))[0] == 0.0
    assert Trainer._combat_shape(t, FakeEnv(), 0, np.array([0]), f, np.array([A_BUY]), np.zeros(1))[0] == 0.0
    return tot


def check_econ():
    """布莱恩·铜须让南海卖艺者的“下回合获得 1 枚铸币”触发两次：观察里读出的下回合额外金币 1 → 2；
    经济奖励按存款的势能差给；买战吼随从时场上有布莱恩算 engine。"""
    from rl.demos import KINDS, bought_in
    from rl.ppo import Trainer
    from tg.actions import A_BUY, A_END
    from tg.catalog import card_id, n_cards
    from tg.obs import gold_next

    def busker(brann: bool):
        g = _game_at(3, 5)
        for z in ("board", "hand"):
            g.debug(json.dumps({"cmd": "clear_zone", "p": 0, "zone": z}))
        if brann:
            g.debug(json.dumps({"cmd": "add_minion", "p": 0, "zone": "board", "card": "BG_LOE_077"}))
        g.debug(json.dumps({"cmd": "add_minion", "p": 0, "zone": "hand", "card": "BG26_135"}))
        m = np.frombuffer(g.legal_mask(0), dtype=np.uint8)
        g.step(0, next(a for a in range(46, 126) if m[a]))
        return int(gold_next(_obs(g)[0]))
    assert (busker(False), busker(True)) == (1, 2), (busker(False), busker(True))

    t = SimpleNamespace(cfg=SimpleNamespace(econ_bonus=0.01), rollout=SimpleNamespace(ep=np.zeros(1, dtype=np.int64)), _hv=None,
                        mix=defaultdict(float), ec_ep=np.full(1, -1, dtype=np.int64), ec_prev=np.zeros(1))
    f = np.zeros((1, 16), np.float32)
    got = []
    for gn in (0, 2, 0, 1):                            # 存了 2 块 → 花掉 → 存 1 块
        f[0, 9] = gn / 5
        got.append(round(float(Trainer._econ_shape(t, np.array([0]), f, np.array([A_END]), np.zeros(1))[0]), 6))
    assert got == [0.0, 0.02, -0.02, 0.01], got

    assert abs(Trainer._econ_close(t, 0) + 0.01) < 1e-12 and Trainer._econ_close(t, 0) == 0.0   # 一局结束扣回最后的 1 块

    # 手牌也算存款：有布莱恩时“获取 / 发现”的战吼多给一份
    def hand_e(card: str, brann: bool) -> float:
        g = _game_at(3, 5)
        for z in ("board", "hand"):
            g.debug(json.dumps({"cmd": "clear_zone", "p": 0, "zone": z}))
        if brann:
            g.debug(json.dumps({"cmd": "add_minion", "p": 0, "zone": "board", "card": "BG_LOE_077"}))
        g.debug(json.dumps({"cmd": "add_minion", "p": 0, "zone": "hand", "card": card}))
        m = np.frombuffer(g.legal_mask(0), dtype=np.uint8)
        g.step(0, next(a for a in range(46, 126) if m[a]))
        for _ in range(4):                             # 发现：选第一个
            m = np.frombuffer(g.legal_mask(0), dtype=np.uint8)
            ch = [a for a in range(30, 46) if m[a]]
            if not ch:
                break
            g.step(0, ch[0])
        f, c, _ = _obs(g)
        return float(Trainer._econ_value(t, f[None], c[None])[0])
    gems = (hand_e("BG20_100", False), hand_e("BG20_100", True))         # 剃刀沼泽地卜师：2 / 4 张鲜血宝石
    fort = (hand_e("BG27_000", False), hand_e("BG27_000", True))         # 钢铁场地管理员：2 / 4 张强固（1 块一张）
    disc = (hand_e("BG28_550", False), hand_e("BG28_550", True))         # 竞技表演者：发现 1 / 2 张酒馆法术
    assert gems == (1.0, 2.0) and fort == (2.0, 4.0) and disc[1] > disc[0] > 0, (gems, fort, disc)

    idx = {card_id(i): i for i in range(1, n_cards())}
    c = np.zeros((2, 124), np.int32)
    c[:, 36] = idx["BG26_135"]                         # 酒馆第 1 格：南海卖艺者（战吼）
    c[0, 19] = idx["BG_LOE_077"]                       # 第 1 局场上有布莱恩
    c[1, 19] = idx["BG26_135"]
    d = {"kind": np.array([KINDS.index("buy")] * 2, np.int8), "c": c, "cand": np.array([[A_BUY] + [-1] * 7] * 2, np.int16)}
    assert bought_in(d, "engine").tolist() == [True, False]
    assert bought_in(d, "triple").tolist() == [False, True]          # 第 2 局场上已有一张南海卖艺者
    c2 = c.copy()
    c2[:, 36] = idx["BG_LOE_077"]
    assert bought_in({**d, "c": c2}, "engcard").tolist() == [True, True]

    # 后期刷新不扣“无效刷新”（refresh_waste_until）
    from tg.actions import A_REFRESH
    for turn, want in ((5, -0.015), (8, 0.0)):
        roll = SimpleNamespace(r=[0.0], last=np.zeros(1, dtype=np.int64), ep=np.zeros(1, dtype=np.int64))
        tt = SimpleNamespace(cfg=SimpleNamespace(refresh_waste_penalty=0.015, refresh_waste_until=6), rollout=roll,
                             ref_idx=np.full(1, -1, dtype=np.int64), mix=defaultdict(int))
        Trainer._after_add(tt, np.array([0]), np.array([A_REFRESH]), np.array([turn]))
        Trainer._after_add(tt, np.array([0]), np.array([A_END]), np.array([turn]))
        assert abs(roll.r[0] - want) < 1e-12, (turn, roll.r)
    return got, gems, fort, disc


def check_drift():
    """策略慢慢变乱的检查（Trainer._drift）：前 100 轮只记熵；熵高出中位数 0.35 或无效操作超过 15%，连续 10 轮才报。"""
    from rl.ppo import Trainer
    t = SimpleNamespace(cfg=SimpleNamespace(drift_ent=0.35, drift_noop=0.15, drift_iters=10), mix=defaultdict(int))
    rng = np.random.default_rng(0)
    for _ in range(150):
        assert Trainer._drift(t, 1.0 + rng.normal(0, 0.05), 0.05) == ""
    for k in range(9):
        assert Trainer._drift(t, 1.6, 0.05) == ""
    assert Trainer._drift(t, 1.0, 0.05) == "" and t.drift_n == 0          # 中间好了一轮：重新数
    msgs = [Trainer._drift(t, 1.0, 0.3) for _ in range(10)]
    assert msgs[-1].startswith("策略连续 10 轮变乱") and not any(msgs[:-1]), msgs
    # 熵的绝对上限：中位数慢慢涨上去也拦得住（第 14240 轮中位数已经 1.69）
    t2 = SimpleNamespace(cfg=SimpleNamespace(drift_ent=0.35, drift_noop=0.15, drift_iters=3, drift_ent_max=1.6), mix=defaultdict(int))
    for k in range(150):
        assert Trainer._drift(t2, 1.0 + k * 0.004, 0.05) == ""                # 1.0 → 1.6 慢慢涨，跟着中位数走的检查不报
    m2 = [Trainer._drift(t2, 1.7, 0.05) for _ in range(3)]
    assert m2[-1].endswith("熵 1.70（上限 1.6）") and not any(m2[:-1]), m2
    return msgs[-1]


def check_targets():
    """目标阵容（rl/target_comps.json）：核心卡识别、最像哪一套、势能差奖励、模仿时按 S / A / B 挑示范。"""
    from rl.comp import load_target_comps, nearest_comp, target_score
    from rl.demos import KINDS, bought_in
    from rl.ppo import Trainer
    from tg.actions import A_BUY, A_END
    from tg.catalog import card_id, n_cards
    comps = load_target_comps()
    assert len(comps) == 23 and {c["tier"] for c in comps} == {"S", "A", "B"}
    tea = next(c for c in comps if c["name"] == "喝茶流")
    assert {"BG_LOE_077"} <= tea["core"]                                 # 布莱恩·铜须
    c, fr = nearest_comp(list(tea["core"]) + ["BG26_135"], comps)
    assert c["name"] == "喝茶流" and fr == 1.0
    assert nearest_comp(["BG26_135"], comps)[0] is None
    q, c2, _ = target_score(["BG_LOE_077"], comps)                       # 只有布莱恩：喝茶流三张里的一张 → (1/3)²
    assert abs(q - 1 / 9) < 1e-9 and c2["name"] == "喝茶流"

    idx = {card_id(i): i for i in range(1, n_cards())}
    t = SimpleNamespace(cfg=SimpleNamespace(target_bonus=0.15, target_from_turn=5), mix=defaultdict(float), _tgt=None,
                        rollout=SimpleNamespace(ep=np.zeros(1, dtype=np.int64)),
                        tg_ep=np.full(1, -1, dtype=np.int64), tg_prev=np.zeros(1))
    f = np.zeros((1, 64), np.float32)                                   # 本局种族全没开：只剩喝茶流（中立 + 全种族）凑得成
    c = np.zeros((1, 124), np.int64)
    got = []
    for turn, board in ((4, ["BG_LOE_077"]), (6, ["BG_LOE_077"]), (7, list(tea["core"])), (8, ["BG26_135"])):
        f[0, 0] = turn / 20
        c[0, 19:26] = 0
        c[0, 19:19 + len(board)] = [idx[x] for x in board]
        got.append(round(float(Trainer._target_shape(t, np.array([0]), f, c, np.array([A_END]), np.zeros(1))[0]), 6))
    assert got == [0.0, 0.016667, 0.133333, -0.15], got                    # 第 4 回合不算；整局和 = 0.15 × 最后的 0

    cc = np.zeros((2, 124), np.int32)
    cc[:, 36] = idx["BG_LOE_077"]
    cc[1, 19] = idx[sorted(tea["core"] - {"BG_LOE_077"})[0]]             # 第 2 局场上已有喝茶流的另一张核心卡
    d = {"kind": np.array([KINDS.index("buy")] * 2, np.int8), "c": cc, "cand": np.array([[A_BUY] + [-1] * 7] * 2, np.int16)}
    assert bought_in(d, "tgtS").tolist() == [True, True] and bought_in(d, "tgtB").tolist() == [True, True]
    assert bought_in(d, "tgtfit").tolist() == [False, True]

    # 本局凑不成的阵容不算：没开野猪人时邪能野猪人那套（S）不追
    from rl.comp import reachable
    boar = next(c for c in comps if c["name"] == "野猪人·邪能野猪人")
    apm = next(c for c in comps if c["name"] == "机械 APM")
    assert reachable(tea, frozenset()) and not reachable(apm, frozenset({"BEAST"})) and reachable(apm, frozenset({"MECHANICAL"}))
    assert reachable(boar, frozenset({"DEMON"})) and not reachable(boar, frozenset({"BEAST"}))   # 邪能野猪人是恶魔 / 野猪人
    assert target_score(list(apm["core"]), comps, frozenset({"BEAST"}))[0] == 0.0
    assert target_score(list(apm["core"]), comps, frozenset({"MECHANICAL"}))[0] == 0.5   # 2 张核心卡都在，但只有 2 个机械（满 4 个）
    from rl.comp import formed
    from tg.catalog import catalog
    det = catalog()
    extra = next(k for k, v in det.items() if "MECHANICAL" in (v.get("races") or []) and k not in apm["core"]
                 and v.get("kind") == "minion" and not k.endswith("_G"))
    assert not formed(list(apm["core"]), apm) and formed(list(apm["core"]) + [extra], apm)    # 3 个机械才算撑起来
    titus = next(c for c in comps if c["name"] == "畸变亡语")
    assert not formed(["BG25_354"], titus)                                # 只放一张提图斯不算畸变亡语
    assert nearest_comp(["BG25_354", "BG26_135"], comps, strict=True)[0] is None
    w = load_target_comps(weights="S:1,A:0.85,B:0.4")
    assert {c["tier"]: c["w"] for c in w} == {"S": 1.0, "A": 0.85, "B": 0.4}
    return got


def check_lobby():
    """观察里本局开放的种族（tg/obs.F_LOBBY）和 view()["lobby_races"] 一致；老师 KL 在“酒馆里有凑得成的 S / A 核心卡”时加权。"""
    import taverngym
    from rl.ppo import Trainer
    from tg.catalog import card_id, n_cards
    from tg.obs import F_LOBBY, lobby, lobby_bits
    for seed in range(12):
        g = taverngym.Game(seed=seed)
        f = _obs(g)[0]
        assert lobby(f) == frozenset(json.loads(g.view(0))["lobby_races"]), seed
        assert lobby(np.log1p(f)) == lobby(f)                         # 训练数据里压缩过（soft_log）也认得
    idx = {card_id(i): i for i in range(1, n_cards())}
    f = np.zeros((3, 64), np.float32)
    f[0, F_LOBBY["MECHANICAL"]] = 1.0
    f[2, F_LOBBY["MECHANICAL"]] = 0.69
    c = np.zeros((3, 124), np.int32)
    from rl.comp import load_target_comps
    apm = next(x for x in load_target_comps() if x["name"] == "机械 APM")
    core = sorted(apm["core"])[0]
    c[:, 37] = idx[core]                                                # 酒馆第 2 格：机械 APM 的核心卡
    t = SimpleNamespace(cfg=SimpleNamespace(teacher_core_mult=3.0))
    w = Trainer._teacher_rows(t, f, c)
    assert w.tolist() == [3.0, 1.0, 3.0], w                            # 第 2 行本局没开机械：凑不成，不加权
    assert Trainer._teacher_rows(SimpleNamespace(cfg=SimpleNamespace(teacher_core_mult=1.0)), f, c) is None
    # 配套卡（bg.py playbook）：正在做机械 APM（两张核心卡都在场上），酒馆里有它的配套卡 → 老师加权、模仿 tgtsup
    from rl.demos import KINDS, bought_in
    from tg.actions import A_BUY
    sup = sorted(apm["support"])[0]
    c2 = np.zeros((2, 124), np.int32)
    c2[:, 36] = idx[sup]
    c2[0, 19:21] = [idx[x] for x in sorted(apm["core"])]                 # 第 1 行在做机械 APM，第 2 行场上空的
    f2 = np.zeros((2, 64), np.float32)
    f2[:, F_LOBBY["MECHANICAL"]] = 1.0
    assert Trainer._teacher_rows(SimpleNamespace(cfg=SimpleNamespace(teacher_core_mult=3.0)), f2, c2).tolist() == [3.0, 1.0]
    d = {"kind": np.array([KINDS.index("buy")] * 2, np.int8), "c": c2, "f": f2, "cand": np.array([[A_BUY] + [-1] * 7] * 2, np.int16)}
    assert bought_in(d, "tgtsup").tolist() == [True, False]
    assert lobby_bits(f).tolist()[1] == 0
    return True


def check_choose():
    """选择类示范：spelltgt 认出“宝石 / 法术打给恶魔干扰者这类吃法术的随从”；打乱酒馆时选择类的行不动。"""
    from rl.demos import KINDS, bought_in, permute_shop
    from tg.actions import A_BUY, A_CHOOSE
    from tg.catalog import card_id, n_cards
    idx = {card_id(i): i for i in range(1, n_cards())}
    c = np.zeros((2, 124), np.int32)
    c[:, 44], c[:, 45] = idx["BG26_135"], idx["BG36_762"]               # 选项：南海卖艺者、恶魔干扰者
    c[:, 36:39] = [idx["BG26_135"], idx["BG36_762"], idx["BG20_100"]]
    d = {"kind": np.array([KINDS.index("choose")] * 2, np.int8), "c": c,
         "cand": np.array([[A_CHOOSE + 1] + [-1] * 7, [A_CHOOSE] + [-1] * 7], np.int16)}
    assert bought_in(d, "spelltgt").tolist() == [True, False]
    f = np.zeros((2, 3227), np.float32)
    m = np.zeros((2, 245), bool)
    m[:, A_BUY:A_BUY + 3] = True

    class R:
        def random(self, shape):
            return np.tile(np.linspace(0.9, 0.1, shape[1]), (shape[0], 1))
    _, c2, _, _ = permute_shop(f, c, m, d["cand"], R(), 685, 40, 8, rows=np.array([True, False]))
    assert c2[0, 36:39].tolist() == c[0, 36:39][::-1].tolist() and c2[1, 36:39].tolist() == c[1, 36:39].tolist()
    return True


def check_teacher():
    """老师 KL 系数每次启动后从 0 线性加上去；没有老师文件时训练照常（不用老师）。"""
    from rl.ppo import Trainer
    t = SimpleNamespace(teacher=object(), teacher_t0=None, iter=10, cfg=SimpleNamespace(teacher_coef=0.03, teacher_ramp=4))
    got = []
    for it in (10, 11, 13, 20):
        t.iter = it
        got.append(round(Trainer._teacher_coef(t), 4))
    assert got == [0.0075, 0.015, 0.03, 0.03], got
    assert Trainer._teacher_coef(SimpleNamespace(teacher=None)) == 0.0
    t.iter = 5                                                          # 回退到爬坡开始之前：系数是 0，不是负的
    assert Trainer._teacher_coef(t) == 0.0
    miss = SimpleNamespace(cfg=SimpleNamespace(teacher="runs\\_no_such_teacher.pt", teacher_coef=0.03, teacher_ramp=4))
    assert Trainer._load_teacher(miss) is None
    assert Trainer._load_teacher(SimpleNamespace(cfg=SimpleNamespace(teacher="x.pt", teacher_coef=0.0))) is None
    return got


def check_seeder():
    """成型阵容课程（rl/calibrate.CompSeeder）：学习者到了那一回合，场面换成库里成型的 S / A 阵容（喝茶流：布莱恩、巴琳达、守门融合怪）。"""
    from taverngym import BatchEnv
    from rl.calibrate import CompSeeder
    from rl.comp import formed, load_target_comps
    from tg.catalog import base_id, card_id
    from tg.obs import C_BOARD, N_BOARD
    tea = next(c for c in load_target_comps() if c["name"] == "喝茶流")
    board = [[x, 10, 10] for x in sorted(tea["core"])]
    env = BatchEnv(4, seed=5, controlled=(0,), threads=1)
    rng = np.random.default_rng(1)
    for _ in range(3000):
        obs = env.observe()
        if (np.rint(obs["features"][:, 0] * 20) >= 3).all():
            break
        m = obs["action_mask"].astype(bool)
        env.step(np.array([rng.choice(np.flatnonzero(r)) for r in m]))
    t = int(np.rint(obs["features"][0, 0] * 20))
    sd = CompSeeder({t: [{"tier": 4, "board": board}]}, frac=1.0, turns=(t,), seed=0)
    assert sd.size() == 1
    assert sd.seed(env, obs["features"]) and 0 in sd.seeded_slots
    obs = env.observe()
    ids = [base_id(card_id(int(x))) for x in obs["cards"][0][C_BOARD:C_BOARD + N_BOARD] if x > 0]
    assert formed(ids, tea) and sorted(ids) == sorted(tea["core"]), ids
    assert not sd.seed(env, obs["features"])                            # 一局只换一次
    assert sd.finish(0) and sd.take_stats()["seeded"] >= 1
    return len(sd.seeded_slots)


def main() -> int:
    n = check_permute_shop()
    j = check_judge()
    tot = check_combat_shape()
    ec = check_econ()
    dr = check_drift()
    tg = check_targets()
    check_choose()
    check_teacher()
    check_lobby()
    check_seeder()
    print(f"boardsim OK：打乱酒馆 {n} 个局面和引擎一致；强 / 弱 / 空场面 {j[0]:.2f} / {j[1]:.2f} / {j[2]:.2f}；势能差 {tot}；"
          f"布莱恩 + 南海卖艺者 下回合 2 块，经济奖励 {ec[0]}；手牌存款（无 / 有布莱恩）鲜血宝石 {ec[1]}、强固 {ec[2]}、发现 {ec[3]}；{dr}；目标阵容奖励 {tg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
