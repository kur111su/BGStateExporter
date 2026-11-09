"""前几回合的金币流向：模型 vs 6500+ 高手。

每回合：开局金币、花在 升本 / 买 / 刷新 / 发动（含选目标）/ 黑暗之赐 / 英雄技能 / 其它 上的钱、卖怪收入、
回合末剩下的钱、回合末等级、升本率、“回合开始时钱够升本”的回合里升了本的比例。

  bg.py goldflow                        用 runs/v3/latest.pt 在 TavernGym 里打 64 局（0 号座位是模型，其余脚本机器人），
                                        和 rl/pro_profile.json 里的高手数字并排打印，结果写 runs/<run>/goldflow.json
  bg.py goldflow --ckpt X --games 128 --sample
  bg.py goldflow --pro data\\firestone   从高分回放算高手那一列，写进 rl/pro_profile.json 的 gold_flow（只有汇总数字）

模型这边按竞技场的方式选动作（最大概率，--sample 改成按概率抽）。金币、等级、升本费直接读观察（第 7 / 10 / 11 个数）。
高手那边用回放里每一步前后的金币差（tg.logs.segment）；黑暗之赐在回放里看不到是哪一步花的，算在“其它”里。
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paths import ROOT, RUNS  # noqa: E402
from tg.actions import A_BUY, A_END, A_SELL, action_family  # noqa: E402
from tg.catalog import base_id, card_id, catalog, is_golden_id  # noqa: E402
from tg.obs import C_BOARD, C_SHOP, N_BOARD, N_SHOP, gold, lobby, tier, turn, up_cost  # noqa: E402

PROFILE = ROOT / "rl" / "pro_profile.json"
COLS = ("upgrade", "buy", "refresh", "activate", "dark_gift", "hero_power", "other")
ZH = {"upgrade": "升本", "buy": "买", "refresh": "刷新", "activate": "发动", "dark_gift": "黑暗之赐",
      "hero_power": "技能", "other": "其它"}


def kind_of(a: int) -> str:
    """记账类别：冻结 / 取消 / 换位置不花钱，记成 noop；其它就是动作类别（tg.actions.action_family）。"""
    fam = action_family(a)
    return "noop" if fam in ("freeze", "cancel", "move") else fam


class TurnLog:
    """一个槽位（一局）的逐回合记录。选项（发现 / 目标）里花的钱记在发起它的那一步上（例如“发动 → 选目标”记在发动上）。"""

    def __init__(self):
        self.turns: dict[int, dict] = {}
        self.prev = None                     # (回合, 金币, 等级, 记账类别)
        self.src = "other"

    def see(self, turn: int, gold: int, tier: int, up: int, a: int):
        d = self.turns.get(turn)
        if d is None:
            d = self.turns[turn] = {"gold": gold, "tier0": tier, "afford": int(up > 0 and gold >= up),
                                    "spend": Counter(), "income": 0, "left": gold, "tier": tier}
        if self.prev and self.prev[0] == turn:
            diff = self.prev[1] - gold
            if diff > 0:
                d["spend"][self.prev[3]] += diff
            elif diff < 0:
                d["income"] += -diff
        d["left"], d["tier"] = gold, tier
        k = kind_of(a)
        if k != "choose":
            self.src = k if k in COLS else "other"
        self.prev = (turn, gold, tier, self.src)


def _mean(rows, f):
    return round(st.mean([f(r) for r in rows]), 2) if rows else None


def summarize(games: list[dict], max_turn: int = 10) -> dict:
    out = {}
    for t in range(1, max_turn + 1):
        rows = [g[t] for g in games if t in g]
        if len(rows) < 3:
            continue
        aff = [r for r in rows if r["afford"]]
        out[str(t)] = {"n": len(rows), "gold": _mean(rows, lambda r: r["gold"]),
                       **{k: _mean(rows, lambda r, k=k: r["spend"].get(k, 0)) for k in COLS},
                       "income": _mean(rows, lambda r: r["income"]), "left": _mean(rows, lambda r: r["left"]),
                       "tier": _mean(rows, lambda r: r["tier"]),
                       "up": round(st.mean([r["tier"] > r["tier0"] for r in rows]), 2),
                       "afford": round(len(aff) / len(rows), 2),
                       "up_afford": round(st.mean([r["tier"] > r["tier0"] for r in aff]), 2) if aff else None}
        if any("gift" in r for r in rows):
            out[str(t)]["gift"] = round(st.mean([bool(r.get("gift")) for r in rows]), 2)
    return out


# ---------------- 模型 ----------------

def train_rules(ckpt: str) -> dict:
    """训练时的规则（每回合动作数上限）：读检查点旁边的 config.json，没有就用默认。"""
    from rl.ppo import Config
    cfg = Config()
    cp = next((d / "config.json" for d in (Path(ckpt).resolve().parent, Path(ckpt).resolve().parent.parent)
               if (d / "config.json").exists()), None)
    if cp is not None:
        try:
            cfg.turn_caps = json.loads(cp.read_text(encoding="utf-8")).get("turn_caps", cfg.turn_caps)
        except (OSError, ValueError):
            pass
    return cfg.rules()


def model_policy(ckpt: str, device: str, greedy: bool):
    """检查点 → (选动作的函数 act(features, cards, mask) -> 动作数组, 检查点信息)。"""
    import torch
    from rl.advisor import load_model
    model, info = load_model(ckpt, device)

    def act(f, c, m):
        with torch.no_grad():
            mm = torch.as_tensor(m, device=device)
            lg = model(torch.as_tensor(f, dtype=torch.float32, device=device),
                       torch.as_tensor(c, dtype=torch.long, device=device), mm)[0].float()
            lg = lg.masked_fill(~mm, float("-inf"))
            a = lg.argmax(-1) if greedy else torch.distributions.Categorical(logits=lg).sample()
        return a.cpu().numpy()

    def logits(f, c, m):                               # 推演用：每个动作的分数（不合法 = -inf）
        with torch.no_grad():
            mm = torch.as_tensor(np.asarray(m, dtype=bool), device=device)
            lg = model(torch.as_tensor(np.asarray(f, dtype=np.float32), device=device),
                       torch.as_tensor(np.asarray(c, dtype=np.int64), device=device), mm)[0].float()
            return lg.masked_fill(~mm, float("-inf")).cpu().numpy()
    info["name"] = Path(ckpt).name
    info["logits"] = logits
    return act, info


def play_model(ckpt: str, games: int, seed: int, threads: int, greedy: bool, device: str, boosted: bool = False,
               policy=None, seed_comps: str = "", seed_lib: dict | None = None, late=None, late_from: int = 10,
               look_from: int = 0, look_k: int = 6, look_depth: int = 8, judge_lib: dict | None = None):
    """boosted：7 个机器人从第 5 回合起每 2 回合换成高分对局里同一回合的真实阵容（runs/pro_boards.json +
    runs/h13_boards.json，和训练时的考试 eval_boost_place 一样，全强度）。policy：测试用，代替检查点。
    seed_comps（例如 "8,9,10"）：每局在其中随机一个回合招募开始时，模型的场面换成高手同一回合已经成型的 S / A 阵容
    （rl/calibrate.CompSeeder，本局凑得成），看模型拿着成型阵容能打成什么样。seed_lib：测试用，代替高分阵容库。
    late：(act, info)，第 late_from 回合起换它来打（例如老师模型），看后期换个打法场面能不能养起来。
    look_from > 0：第 look_from 回合起每个决策先推演（见 lookahead）：模型最想走的 look_k 步 + 结束回合，各在这局的副本上走一步、
    再让模型自己把这回合打完（最多 look_depth 步，不结束回合），拿同一组同期高分阵容量结束时的场面，选最强的那一步
    （比模型自己的选择强 0.02 以上才换）。副本的随机数先换掉，推演看不到真实的下一个酒馆。judge_lib：测试用，代替高分阵容库。"""
    import time as _t
    from taverngym import BatchEnv
    from tools.cardvalue import PickStats
    print(f"读模型 {ckpt} …", flush=True)
    act, info = policy if policy is not None else model_policy(ckpt, device, greedy)
    # 和训练一样给每回合动作数设上限（turn_caps）：不设的话模型取最大概率时可能在两个动作之间来回绕
    # （例如换位置换回来），这一局永远打不完——第 57 局卡住就是这样
    env = BatchEnv(games, seed=seed, controlled=(0,), threads=threads, config=train_rules(ckpt))
    booster = None
    if boosted:
        from rl.calibrate import ProBoardBooster, merge_pro_boards
        lib = merge_pro_boards(p for p in (RUNS / "pro_boards.json", RUNS / "h13_boards.json") if p.exists())
        if lib:
            booster = ProBoardBooster(lib, frac=1.0, start_turn=5, seed=seed, every=2)
            booster.strength = 1.0
            print(f"对手：高分阵容 {sum(len(v) for v in lib.values())} 个（第 5 回合起，全强度）", flush=True)
        else:
            print("没有 runs/pro_boards.json / runs/h13_boards.json，对手是普通脚本机器人", flush=True)
    # 阵容强度（rl/boardsim.py，和训练的 combat_bonus 同一把尺子）：模型每次结束回合时的场面打同期高分阵容的胜率
    judge, lib = None, None
    from rl.calibrate import merge_pro_boards as _merge
    lib = judge_lib if judge_lib is not None else _merge(p for p in (RUNS / "pro_boards.json", RUNS / "h13_boards.json")
                                                       if p.exists())
    if lib:
        from rl.boardsim import BoardJudge
        judge = BoardJudge(lib, k=12, n=3, seed=0)
    look = None                                        # 推演：(每个动作分数的函数, 统计)
    if look_from > 0:
        lg_fn = (info or {}).get("logits") if isinstance(info, dict) else None
        if judge is None or lg_fn is None:
            print("推演要高分阵容库（runs/pro_boards.json）和模型的动作分数，这次不推演", flush=True)
        else:
            from rl.boardsim import BoardJudge
            # 推演用另一组考官（种子 1）：阵容强度表（种子 0 那组）还是独立的检查，不是“对着考题练”
            look = (lg_fn, {"n": 0, "changed": 0, "gain": 0.0, "by_kind": {}}, BoardJudge(lib, k=12, n=3, seed=1))
            print(f"推演：第 {look_from} 回合起每个决策试模型最想走的 {look_k} 步 + 结束回合，各自打完这回合（最多 {look_depth} 步），"
                  f"选结束时场面最强的", flush=True)
    seeder, seeded_comp = None, {}                     # 拿着成型阵容打：槽位 → 换成的那套阵容（空 = 没换成）
    if seed_comps:
        from rl.calibrate import CompSeeder
        ts = tuple(int(x) for x in str(seed_comps).split(",") if x.strip())
        slib = seed_lib if seed_lib is not None else lib
        if ts and slib:
            seeder = CompSeeder(slib, frac=1.0, turns=ts, seed=seed, formed_from=min(ts))
            print(f"拿着成型阵容打：每局在第 {','.join(map(str, ts))} 回合（随机一个）招募开始时，场面换成高手同一回合已经成型的 "
                  f"S / A 阵容（本局凑得成）；库里 " + "、".join(f"第 {t} 回合 {len(v)} 个" for t, v in sorted(seeder.pool.items())),
                  flush=True)
            if not seeder.size():
                seeder = None
                print("  库里没有成型的 S / A 阵容，不换", flush=True)
        else:
            print("没有高分阵容库（runs/pro_boards.json），不换成型阵容", flush=True)
    phis: dict[int, list] = {}                         # 槽位 → [(回合, Φ)]
    grows: dict[int, list] = {}                        # 槽位 → [(回合, 结束 - 开始)]：这回合自己把场面养强了多少
    g_start: dict[int, tuple] = {}                     # 槽位 → (回合, 这回合开始时的 Φ)
    starts: dict[int, list] = {}                       # 槽位 → [(回合, 开始时的 Φ)]：和上回合结束比，就是对手这回合涨了多少
    picks = PickStats()
    sells: dict[str, int] = {}
    buy_rows: list = []                                # 每次买：(当时的卡牌编号, 动作)，算“围绕种族 / 搭配”的买法
    sell_rows: list = []                               # 每次卖：(观察, 卡牌编号, 动作)，算“卖掉正在做的阵容里该留的牌”
    logs = [TurnLog() for _ in range(env.n_slots)]
    done = [False] * env.n_slots
    places = {}
    last_cards = [None] * env.n_slots                  # 每局最后一次招募时的场面（最终阵容）
    last_lobby = [None] * env.n_slots                  # 每局开放的种族（哪几套目标阵容凑得成）
    plan8 = [None] * env.n_slots                       # 第 8 回合结束时的主选 / 备选体系（S / A）
    from rl.comp import CompTables, load_target_comps
    from tg.catalog import base_of_index, n_cards
    from tg.obs import lobby_bits
    _ctab = CompTables(load_target_comps(), [base_of_index(i) if i else "" for i in range(n_cards())], tiers="SA")
    obs = env.observe()
    t0, shown = _t.time(), 0
    print(f"开始打 {games} 局（每打完 8 局报一次进度）…", flush=True)
    last_progress, step_no, quiet = 0, 0, 0
    for _ in range(200000):
        if all(done):
            break
        step_no += 1
        if len(places) != last_progress:
            last_progress, quiet = len(places), step_no
        elif step_no - quiet > 20000:
            print(f"  剩下 {games - len(places)} 局 2 万步没有进展，不等了（按已打完的 {len(places)} 局统计）", flush=True)
            break
        ready = [i for i in np.flatnonzero(obs["ready"]) if not done[i]]
        # 已经打完的槽位会自动开新局：随便给个合法动作让它走（不算数）
        actions = obs["action_mask"].astype(bool).argmax(1).astype(np.int64)
        if ready:
            f = obs["features"][ready]
            m = obs["action_mask"][ready].astype(bool)
            if late is None:
                a = np.asarray(act(f, obs["cards"][ready], m)).astype(np.int64)
            else:                                      # 第 late_from 回合起换另一个模型打
                cr = obs["cards"][ready]
                lo = np.rint(f[:, 0] * 20).astype(int) >= late_from
                a = np.zeros(len(ready), dtype=np.int64)
                if (~lo).any():
                    a[~lo] = np.asarray(act(f[~lo], cr[~lo], m[~lo]))
                if lo.any():
                    a[lo] = np.asarray(late[0](f[lo], cr[lo], m[lo]))
            if look is not None:                       # 推演：第 look_from 回合起每个决策在副本上试几步
                tt = np.rint(f[:, 0] * 20).astype(int)
                rows = [k for k in range(len(ready)) if tt[k] >= look_from and m[k].sum() > 1]
                if rows:
                    L = look[0](f[rows], obs["cards"][ready][rows], m[rows])
                    for r, k in enumerate(rows):
                        a[k] = lookahead(env, int(ready[k]), int(tt[k]), m[k], L[r], int(a[k]), look[2], look[0],
                                         look_k, look_depth, look[1], rng_seed=seed * 1000003 + step_no * 31 + k)
            for k, i in enumerate(ready):
                fi = f[k]
                ci = obs["cards"][i]
                last_cards[i] = ci[C_BOARD:C_BOARD + N_BOARD].copy()
                last_lobby[i] = lobby(fi)
                ak = int(a[k])
                if A_BUY <= ak < A_BUY + N_SHOP:
                    picks.add(ci[C_SHOP:C_SHOP + N_SHOP], ak - A_BUY)
                    buy_rows.append((ci.copy(), ak))
                elif A_SELL <= ak < A_SELL + N_BOARD and ci[C_BOARD + ak - A_SELL] > 0:
                    sid = picks._id(int(ci[C_BOARD + ak - A_SELL]))
                    sells[sid] = sells.get(sid, 0) + 1
                    sell_rows.append((fi[:64].copy(), ci.copy(), ak))
                logs[i].see(int(turn(fi)), int(gold(fi)), int(tier(fi)), int(up_cost(fi)), int(a[k]))
                if judge is not None and not done[i] and int(turn(fi)) >= 3 and g_start.get(int(i), (None,))[0] != int(turn(fi)):
                    g_, seat_ = env.slot(int(i))               # 这回合第一次决策：先量开始时的强度
                    g_start[int(i)] = (int(turn(fi)), judge.judge(env.game(int(g_)), int(seat_), int(turn(fi))))
                    if g_start[int(i)][1] is not None:
                        starts.setdefault(int(i), []).append(g_start[int(i)])
                if ak == A_END and int(turn(fi)) == 8 and _ctab.comps:
                    mn, bk = _ctab.main_backup(ci[None], lobby_bits(fi[None]))
                    plan8[i] = ([_ctab.comps[k]["name"] for k in np.flatnonzero(mn[0])],
                                [_ctab.comps[k]["name"] for k in np.flatnonzero(bk[0])])
                if judge is not None and ak == A_END and int(turn(fi)) >= 3 and not done[i]:   # 打完的槽位自动开的新局不算
                    g_, seat_ = env.slot(int(i))
                    v = judge.judge(env.game(int(g_)), int(seat_), int(turn(fi)))
                    if v is not None:
                        phis.setdefault(int(i), []).append((int(turn(fi)), v))
                        st0 = g_start.get(int(i))
                        if st0 and st0[0] == int(turn(fi)) and st0[1] is not None:
                            grows.setdefault(int(i), []).append((int(turn(fi)), v - st0[1]))
            actions[ready] = a
        _, dones, inf = env.step(actions)
        for i in np.flatnonzero(dones):
            if not done[i]:
                done[i] = True
                places[int(i)] = int(inf["placement"][i])
                if seeder is not None:
                    seeded_comp[int(i)] = seeder.finish(int(i))
        if len(places) >= shown + 8 or (len(places) == games and shown < games):
            shown = len(places)
            ps = [p for p in places.values() if p > 0]
            el = _t.time() - t0
            print(f"  {shown}/{games} 局，平均名次 {np.mean(ps):.2f}，{el / 60:.1f} 分钟"
                  + (f"，预计还要 {el / shown * (games - shown) / 60:.0f} 分钟" if shown < games else ""), flush=True)
        obs = env.observe()
        if booster is not None and booster.boost(env, obs["features"]):
            obs = env.observe()
        if seeder is not None and seeder.seed(env, obs["features"]):
            obs = env.observe()
    print(f"模型 {info.get('iter')} 轮（{info.get('name', Path(ckpt).name)}）：{len(places)} 局，平均名次 "
          f"{np.mean([p for p in places.values() if p > 0]):.2f}（{'对高分阵容' if booster else '对脚本机器人'}）", flush=True)
    pv = [p for p in places.values() if p > 0]
    if pv:
        dist = "、".join(f"{k} 名 {np.mean([p == k for p in pv]):.0%}" for k in range(1, 9))
        print(f"名次分布：{dist}；吃鸡 {np.mean([p == 1 for p in pv]):.1%}、前四 {np.mean([p <= 4 for p in pv]):.0%}", flush=True)
    info["picks"], info["sells"], info["buy_rows"], info["sell_rows"] = picks, sells, buy_rows, sell_rows
    finals = []
    for i, row in enumerate(last_cards):
        if row is None:
            continue
        ids = [card_id(int(x)) for x in row if x > 0]
        finals.append({"board": [base_id(x) for x in ids], "golden": sum(is_golden_id(x) for x in ids),
                       "place": places.get(i, 0), "lobby": last_lobby[i], "plan8": plan8[i]})
    info["finals"] = finals
    info["phis"] = {i: v for i, v in phis.items()}
    info["grows"] = {i: v for i, v in grows.items()}
    info["starts"] = {i: v for i, v in starts.items()}
    info["places"] = places
    info["judge"], info["lib"] = judge, lib
    if look is not None:
        st = look[1]
        print(f"\n推演：{st['n']} 个决策，换掉模型选择的 {st['changed'] / max(1, st['n']):.0%}，"
              f"换的那些平均让结束时的场面强 {st['gain'] / max(1, st['changed']):+.3f}；换成："
              + "、".join(f"{k} {v}" for k, v in sorted(st["by_kind"].items(), key=lambda kv: -kv[1])), flush=True)
        info["look"] = {k: v for k, v in st.items()}
    if seeder is not None:
        info["seeded"] = show_seeded(places, seeded_comp, seed_comps)
    return [x.turns for x in logs], info


def lookahead(env, i: int, t: int, m, lg, a0: int, judge, lg_fn, k: int = 6, depth: int = 8, st=None,
              margin: float = 0.02, rng_seed: int = 0) -> int:
    """槽位 i 这个决策的推演（见 play_model 的 look_from）：返回要走的动作。
    候选 = 模型自己的选择 + 分数最高的 k 个 + 结束回合；每个候选在同一个副本（随机数换过）上走一步，模型接着打完这回合
    （遇到结束回合就停、最多 depth 步），然后拿同一组同期高分阵容实打（judge），比结束时的场面。"""
    import json as _json
    from taverngym import _core
    from tg.actions import A_END, action_family
    legal = np.flatnonzero(m)
    order = legal[np.argsort(-np.asarray(lg)[legal], kind="stable")]
    cand = [int(a0)] + [int(x) for x in order[:k] if int(x) != int(a0)]
    if m[A_END] and A_END not in cand:
        cand.append(A_END)
    g, seat = env.slot(int(i))
    s = _json.loads(env.game(int(g)).state_json())
    r = np.random.default_rng(rng_seed)
    s["rng"]["s"] = [int(x) for x in r.integers(1, 2 ** 63, size=4)]   # 推演看不到真实的下一个酒馆
    base = _core.Game.from_json(_json.dumps(s))
    runs = []
    for a in cand:
        G = base.copy()
        if a != A_END:
            try:
                G.step(int(seat), int(a))
            except ValueError:
                continue
        runs.append([a, G, a == A_END])
    for _ in range(depth):
        live = [x for x in runs if not x[2]]
        if not live:
            break
        F, C, M = [], [], []
        for x in live:
            fb, cb = x[1].observe(int(seat))
            F.append(np.frombuffer(fb, dtype=np.float32))
            C.append(np.frombuffer(cb, dtype=np.int32))
            M.append(np.frombuffer(x[1].legal_mask(int(seat)), dtype=np.uint8).astype(bool))
        Mx = np.stack(M)
        if not Mx.any(axis=1).all():
            for x, mm in zip(live, Mx):
                if not mm.any():
                    x[2] = True
            continue
        L2 = lg_fn(np.stack(F), np.stack(C), Mx)
        for x, l2, mm in zip(live, L2, Mx):
            b = int(np.argmax(np.where(mm, l2, -np.inf)))
            if b == A_END:
                x[2] = True
                continue
            try:
                x[1].step(int(seat), b)
            except ValueError:
                x[2] = True
    best_a, best_v, v0 = int(a0), None, None
    for a, G, _ in runs:
        v = judge.judge(G, int(seat), int(t))
        if v is None:
            continue
        if a == int(a0):
            v0 = v
        if best_v is None or v > best_v:
            best_v, best_a = v, a
    if v0 is None or best_v is None or best_v < v0 + margin:
        best_a = int(a0)
    if st is not None:
        st["n"] += 1
        if best_a != int(a0):
            st["changed"] += 1
            st["gain"] += float(best_v - v0)
            kd = action_family(best_a)
            st["by_kind"][kd] = st["by_kind"].get(kd, 0) + 1
    return best_a


def show_seeded(places: dict, comp_of: dict, turns: str = "") -> dict:
    """拿着成型阵容打（--seed-comps）：换成了的局 / 没换成的局各打成什么样，按阵容分开。"""
    def fmt(v):
        return (f"{len(v)} 局 平均 {np.mean(v):.2f} 名 吃鸡 {np.mean([p == 1 for p in v]):.0%} "
                f"前四 {np.mean([p <= 4 for p in v]):.0%}")
    by: dict[str, list] = {}
    for i, p in places.items():
        if p > 0:
            by.setdefault(comp_of.get(i) or "", []).append(int(p))
    got = [p for k, v in by.items() if k for p in v]
    miss = by.get("", [])
    print(f"\n拿着成型阵容打（第 {turns} 回合换成高手的成型场面，之后模型自己打）："
          + (fmt(got) if got else "一局都没换成") + (f"；没换成的（本局凑不成 / 没打到那一回合）{fmt(miss)}" if miss else ""),
          flush=True)
    rows = sorted(((k, v) for k, v in by.items() if k), key=lambda kv: (np.mean(kv[1]), -len(kv[1])))
    for k, v in rows:
        print(f"  {k}：{fmt(v)}", flush=True)
    return {"turns": turns, "seeded": got, "missing": miss,
            "by_comp": {k: [len(v), round(float(np.mean(v)), 2), round(float(np.mean([p == 1 for p in v])), 3),
                            round(float(np.mean([p <= 4 for p in v])), 3)] for k, v in rows}}


def show_strength(info: dict, upto: int = 14) -> dict | None:
    """每回合结束时的阵容强度 Φ（打同期 12 个高分阵容的平均胜率）：模型 / 前四名的局 / 后四名的局 / 高分阵容自己。
    高分阵容那一列：库里同一回合的场面摆到实验局里打同一组考官（没有英雄技能、饰品，和模型那列差这一点）。"""
    judge, phis, places = info.get("judge"), info.get("phis") or {}, info.get("places") or {}
    if judge is None or not phis:
        return None
    import random as _r
    from rl.boardsim import put_board
    from tools.simcards import _lab
    lab = _lab(0)
    rng = _r.Random(1)
    rows = {}
    for t in range(3, upto + 1):
        allv = [v for i, xs in phis.items() for tt, v in xs if tt == t]
        if len(allv) < 8:
            continue
        top = [v for i, xs in phis.items() for tt, v in xs if tt == t and 0 < places.get(i, 0) <= 4]
        bot = [v for i, xs in phis.items() for tt, v in xs if tt == t and places.get(i, 0) > 4]
        pool = [e for e in (info["lib"].get(t) or []) if e.get("board")]
        ref = []
        for e in rng.sample(pool, min(40, len(pool))):
            g = lab.copy()
            if put_board(g, 0, e["board"]):
                v = judge.judge(g, 0, t)
                if v is not None:
                    ref.append(v)
        gv = [v for xs in (info.get("grows") or {}).values() for tt, v in xs if tt == t]
        fv = []                                        # 对手涨了多少：同一个场面，上回合结束时 - 这回合开始时（考官换成新一回合的）
        for i, xs in (info.get("starts") or {}).items():
            ends = {tt: v for tt, v in phis.get(i, [])}
            fv += [ends[t - 1] - v for tt, v in xs if tt == t and (t - 1) in ends]
        rows[t] = {"model": round(float(np.mean(allv)), 3), "n": len(allv),
                   "grow": round(float(np.mean(gv)), 3) if gv else None,
                   "field": round(float(np.mean(fv)), 3) if fv else None,
                   "top4": round(float(np.mean(top)), 3) if top else None,
                   "bottom4": round(float(np.mean(bot)), 3) if bot else None,
                   "pro": round(float(np.mean(ref)), 3) if ref else None}
    if not rows:
        return None
    f = lambda x: "  -  " if x is None else f"{x:.2f}"   # noqa: E731
    g = lambda x: "   -  " if x is None else f"{x:+.3f}"   # noqa: E731
    print("\n阵容强度（结束回合时的场面打同期 12 个高分阵容的胜率，第 t … t-3 回合各 3 个；引擎实打，平局算半场）")
    print("养了多少 = 同一组考官，结束回合时 - 这回合开始时：这回合的买、卖、打牌、法术、三连让场面强了多少")
    print("对手涨了 = 同一个场面，上回合结束时 - 这回合开始时（考官换成这一回合的）：养了多少比它小，场面就一回合比一回合落后")
    print("回合 | 模型 | 前四名的局 | 后四名的局 | 高分阵容 | 养了多少 | 对手涨了")
    for t, r in rows.items():
        print(f"{t:4d} | {f(r['model'])} |       {f(r['top4'])} |       {f(r['bottom4'])} |     {f(r['pro'])} |   {g(r.get('grow'))} |   {g(r.get('field'))}")
    return rows


# ---------------- 高手（回放） ----------------

def _pro_one(job):
    from tg.hsreplay import parse
    from tg.logs import load_records, segment
    from tools.make_pro_demos import _xml
    try:
        states = load_records(parse(_xml(job)), "pro")
        acts = segment(states)
    except Exception:                                   # noqa: BLE001
        return None
    T = {}
    for s in states:
        t = int(s["turn"])
        p = s["player"]
        sh = p.get("shop") or {}
        if t not in T:
            uc = sh.get("upgradeCost")
            T[t] = {"gold": p["gold"], "tier0": p["tavernTier"], "spend": Counter(), "income": 0,
                    "afford": int(bool(sh.get("canUpgrade")) and uc is not None and p["gold"] >= uc), "gift": False}
        T[t]["left"], T[t]["tier"] = p["gold"], p["tavernTier"]
        if (s.get("_dark_gifts") or {}).get("used_turn"):
            T[t]["gift"] = True
    for a in acts:
        d = T.get(int(a.turn))
        if d is None:
            continue
        g0 = a.before["player"]["gold"]
        g1 = (a.after or a.before)["player"]["gold"]
        k = a.kind if a.kind in COLS else "other"
        if g0 > g1:
            d["spend"][k] += g0 - g1
        elif g1 > g0:
            d["income"] += g1 - g0
    for d in T.values():                                # 回放里看不到是哪一步花的钱（主要是黑暗之赐）算“其它”
        miss = d["gold"] + d["income"] - d["left"] - sum(d["spend"].values())
        if miss > 0:
            d["spend"]["other"] += miss
    return T


def pro_table(src: Path) -> dict:
    from multiprocessing import Pool
    from tools.make_pro_demos import _jobs
    jobs = list(_jobs(src))
    if not jobs:
        raise SystemExit(f"{src} 里没有回放")
    with Pool() as pool:
        res = [r for r in pool.map(_pro_one, jobs, chunksize=4) if r]
    return {"games": len(res), "created": time.strftime("%Y-%m-%d"), "turns": summarize(res)}


# ---------------- 打印 ----------------

def show(model: dict, pro: dict | None, upto: int = 9):
    hdr = "回合 | 谁   | 金币 | " + " ".join(f"{ZH[k]:>4}" for k in COLS) + " | 卖入 | 剩 | 回合末等级 | 升本率 | 钱够时升本"
    print(hdr)
    print("-" * len(hdr.encode("gbk", "replace")))
    for t in range(1, upto + 1):
        for who, tab in (("模型", model), ("高手", (pro or {}).get("turns") or {})):
            r = tab.get(str(t))
            if not r:
                continue
            ua = f"{r['up_afford'] * 100:.0f}%" if r.get("up_afford") is not None else "-"
            print(f"{t:>4} | {who} | {r['gold']:>4} | " + " ".join(f"{(r.get(k) or 0):>6.2f}" for k in COLS) +
                  f" | {r['income']:>4} | {r['left']:>3} | {r['tier']:>10} | {r['up'] * 100:>5.0f}% | {ua:>8}")
    if pro:
        print("（高手一列：回放里认不出“发动”和黑暗之赐是哪一步花的钱，都算在“其它”里）")


def show_comps(finals: list[dict]) -> dict | None:
    """模型的最终阵容（场上 ≥ 5 个随从的局）和 rl/h13_profile.json（万三分段）对比。"""
    from rl.comp import ZH_RACE, comp_stats
    fb = [x for x in finals if len(x["board"]) >= 5]
    if not fb:
        return None
    m = comp_stats([x["board"] for x in fb], [x["golden"] for x in fb], [])
    hp = ROOT / "rl" / "h13_profile.json"
    h = json.loads(hp.read_text(encoding="utf-8"))["final"] if hp.exists() else None
    print(f"\n最终阵容（场上 ≥5 个随从的 {m['n']} 局）：金色 {m['golden']} 个" + (f"（万三 {h['golden']}）" if h else ""))
    fmt = lambda lst: "、".join(f"{ZH_RACE.get(k, k)} {v:.0%}" for k, v in lst[:6])   # noqa: E731
    print("  模型主种族：", fmt(m["main"]))
    if h:
        print("  万三主种族：", fmt(h["main"]))
    # 模型走每个主种族时打得怎么样：它偏爱龙 / 海盗，是因为在模拟器里对它来说确实更好，还是没试过别的
    from collections import defaultdict
    from rl.comp import main_tribe
    from tg.catalog import catalog as _cat
    det_ = _cat()
    by = defaultdict(list)
    for x in fb:
        if x.get("place"):
            by[main_tribe(x["board"], det_)].append(int(x["place"]))
    if by:
        rows_ = sorted(by.items(), key=lambda kv: -len(kv[1]))
        print("  模型走各主种族的结果：" + "；".join(
            f"{ZH_RACE.get(k, k)} {len(v)} 局 平均 {np.mean(v):.2f} 名 吃鸡 {np.mean([p == 1 for p in v]):.0%} "
            f"前四 {np.mean([p <= 4 for p in v]):.0%}" for k, v in rows_))
    print("  模型常用卡：", "、".join(f"{n} {v:.0%}" for _, n, v in m["cards"][:12]))
    if h:
        print("  万三常用卡：", "、".join(f"{n} {v:.0%}" for _, n, v in h["cards"][:12]))
    # 配合程度（rl/comp.board_scores）
    from rl.comp import board_scores, load_comp_refs
    pairs, key = load_comp_refs()
    det = catalog()
    sc = [board_scores(x["board"], det, pairs, key) for x in fb]
    m["scores"] = {k: round(float(np.mean([r[k] for r in sc])), 3) for k in ("coh", "syn", "key")}
    hs = (h or {}).get("scores") or {}
    print(f"  配合：主种族纯度 {m['scores']['coh']:.2f}（万三 {hs.get('coh', '-')}）、高手常见搭配 {m['scores']['syn']:.2f} 对"
          f"（{hs.get('syn', '-')}）、关键卡 {m['scores']['key']:.2f} 张（{hs.get('key', '-')}）")
    by_place = {}
    for x, r in zip(fb, sc):
        by_place.setdefault("前四" if 0 < x["place"] <= 4 else "后四", []).append(r)
    for k, rows in sorted(by_place.items()):
        print(f"  {k}名的局（{len(rows)}）：纯度 {np.mean([r['coh'] for r in rows]):.2f}、搭配 {np.mean([r['syn'] for r in rows]):.2f}、"
              f"关键卡 {np.mean([r['key'] for r in rows]):.2f}")
    # 哪些卡在前四的最终阵容里比后四多（模型自己的局；局数少时只是参考）
    top = [x for x in fb if 0 < x["place"] <= 4]
    bot = [x for x in fb if x["place"] > 4]
    if len(top) >= 8 and len(bot) >= 8:
        from collections import Counter as _C
        ct, cb = _C(c for x in top for c in set(x["board"])), _C(c for x in bot for c in set(x["board"]))
        diff = sorted(((ct[c] / len(top) - cb[c] / len(bot), c) for c in set(ct) | set(cb) if ct[c] + cb[c] >= 6), reverse=True)
        nm = lambda c: (det.get(c) or {}).get("name_zh", c)              # noqa: E731
        print("  前四更常有：" + "、".join(f"{nm(c)} {ct[c] / len(top):.0%}/{cb[c] / len(bot):.0%}" for d_, c in diff[:8] if d_ > 0))
        print("  后四更常有：" + "、".join(f"{nm(c)} {ct[c] / len(top):.0%}/{cb[c] / len(bot):.0%}" for d_, c in diff[::-1][:8] if d_ < 0))
    return m


def buy_style(buy_rows: list) -> dict:
    """模型的买法：买的随从和场上 ≥2 个随从同种族的比例（tribe）、和场上随从是高手常见搭配的比例（syn）、
    买关键卡 / 经济卡的比例；和高手示范同样的算法（rl/demos.bought_in）。高手：0.36 / 0.07 / 0.59 / 0.11。"""
    if not buy_rows:
        return {}
    from rl.demos import KINDS, bought_in
    n = len(buy_rows)
    d = {"kind": np.full(n, KINDS.index("buy"), np.int8), "c": np.stack([r[0] for r in buy_rows]).astype(np.int64),
         "cand": np.array([[a] + [-1] * 7 for _, a in buy_rows], np.int16)}
    out = {}
    for w in ("tribe", "syn", "key", "eco", "simkey", "simsyn", "engine", "engcard", "triple", "tgtS", "tgtA", "tgtB", "tgtfit", "tgtsup", "engfuel"):
        sel = bought_in(d, w)
        if sel is not None:
            out[w] = round(float(sel.mean()), 3)
    return out


def keep_sold(sell_rows: list) -> dict | None:
    """模型的卖牌里，卖掉“正在做的 S / A 阵容里该留的牌”（rl/comp.CompTables.keep_sold，训练的 keep_penalty 同一个判断）的比例。"""
    if not sell_rows:
        return None
    from collections import Counter
    from rl.comp import CompTables, load_target_comps
    from tg.catalog import base_of_index, catalog, n_cards
    comps = load_target_comps()
    if not any(c.get("keep") for c in comps):
        return None
    ids = [base_of_index(i) if i else "" for i in range(n_cards())]
    tab = CompTables(comps, ids, tiers="SA")
    f = np.stack([r[0] for r in sell_rows])
    c = np.stack([r[1] for r in sell_rows]).astype(np.int64)
    a = np.array([r[2] for r in sell_rows], dtype=np.int64)
    bad = tab.keep_sold(c, f, a)
    det = catalog()
    names = Counter((det.get(ids[int(c[i, C_BOARD + a[i] - A_SELL])]) or {}).get("name_zh", "?") for i in np.flatnonzero(bad))
    return {"share": round(float(bad.mean()), 4), "n": int(bad.sum()), "top": names.most_common(6)}


def show_picks(picks, sells: dict) -> dict | None:
    """模型买卡的挑中倍率和高手比（rl/card_value.json）：高手抢着买、模型不买的卡；模型偏爱、高手基本不买的卡；经济卡。"""
    cv = ROOT / "rl" / "card_value.json"
    if not cv.exists() or not picks.n:
        return None
    ref = json.loads(cv.read_text(encoding="utf-8"))
    hc = ref["cards"]
    mt = picks.table(15)
    both = [(k, mt[k]["lift"], hc[k]["lift"]) for k in mt if k in hc]
    if not both:
        return None
    name = lambda k: hc[k]["name"]                          # noqa: E731
    miss = sorted((x for x in both if x[2] >= 1.6), key=lambda x: x[1] - x[2])[:12]
    over = sorted((x for x in both if x[2] < 0.8), key=lambda x: x[2] - x[1])[:10]
    print(f"\n挑卡（模型买了 {picks.n} 次；倍率 = 实际买的次数 / 随便挑的期望，>1 是偏爱）")
    print("  高手抢着买、模型买得少：" + "、".join(f"{name(k)} {m:.1f}（高手 {p:.1f}）" for k, m, p in miss))
    print("  模型偏爱、高手很少买：" + "、".join(f"{name(k)} {m:.1f}（高手 {p:.1f}）" for k, m, p in over))
    eco = [k for k in ref.get("eco", []) if k in mt]
    if eco:
        print("  经济卡：" + "、".join(f"{name(k)} {mt[k]['lift']:.1f}（高手 {hc[k]['lift']:.1f}）" for k in eco))
    keyset = set(ref.get("key", []))
    kb = sum(mt[k]["picked"] for k in mt if k in keyset)
    print(f"  买的卡里关键卡占 {kb / max(1, picks.n):.0%}")
    corr = float(np.corrcoef([x[1] for x in both], [x[2] for x in both])[0, 1]) if len(both) > 2 else None
    if corr is not None:
        print(f"  和高手挑卡倍率的相关系数 {corr:.2f}（1 = 完全一样的偏好）")
    top_sold = sorted(sells.items(), key=lambda x: -x[1])[:10]
    if top_sold:
        print("  模型最常卖：" + "、".join(f"{(hc.get(k) or {}).get('name', k)} {v}" for k, v in top_sold))
    return {"key_share": round(kb / max(1, picks.n), 3), "corr": round(corr, 3) if corr is not None else None,
            "miss": [[name(k), m, p] for k, m, p in miss], "over": [[name(k), m, p] for k, m, p in over]}


def show_targets(finals: list[dict]) -> dict | None:
    """最终阵容（场上 ≥ 4 个随从）各自最像 rl/target_comps.json 里的哪一套（核心卡占一半以上才算），
    每套的局数比例和这些局的平均名次；S / A / B 合计。"""
    from rl.comp import load_target_comps, nearest_comp, target_score
    comps = load_target_comps()
    fb = [x for x in finals if len(x["board"]) >= 4]
    if not comps or not fb:
        return None
    rows, phis = {}, []
    for x in fb:
        c, fr = nearest_comp(x["board"], comps, strict=True)
        phis.append(target_score(x["board"], comps)[0])
        key = f"{c['tier']} {c['name']}" if c else "不像任何一套"
        r = rows.setdefault(key, {"n": 0, "places": [], "tier": c["tier"] if c else "-"})
        r["n"] += 1
        if x.get("place"):
            r["places"].append(x["place"])
    n = len(fb)
    tiers = {t: sum(r["n"] for r in rows.values() if r["tier"] == t) / n for t in "SAB"}
    order = sorted(rows.items(), key=lambda kv: ("SAB-".index(kv[1]["tier"]), -kv[1]["n"]))
    print(f"  目标阵容（bgbuddy.gg S / A / B；核心卡至少 2 张、2/3，种族阵容还要 3 个本族随从才算）：S {tiers['S']:.0%}、A {tiers['A']:.0%}、"
          f"B {tiers['B']:.0%}，平均接近程度 {np.mean(phis):.2f}（S 级凑齐、本族 4 个 = 1）")
    print("    " + "、".join(f"{k} {r['n'] / n:.0%}" + (f"（平均 {np.mean(r['places']):.2f} 名）" if r["places"] else "")
                           for k, r in order))
    # 只看本局凑得成的：核心卡的种族本局开了（训练的目标阵容奖励也只算这些）
    from rl.comp import formed, reachable
    reach = {}
    can_sa, got_sa, sa_places, other_places = 0, 0, [], []
    for x in fb:
        lob = x.get("lobby")
        ok = [c for c in comps if reachable(c, lob)]
        for c in ok:
            r = reach.setdefault(f"{c['tier']} {c['name']}", [0, 0])
            r[0] += 1
            r[1] += formed(x["board"], c)
        if any(c["tier"] in "SA" for c in ok):
            can_sa += 1
            c, _ = nearest_comp(x["board"], comps, lob=lob, strict=True)
            hit = c is not None and c["tier"] in "SA"
            got_sa += hit
            if x.get("place"):
                (sa_places if hit else other_places).append(x["place"])
    sa = {"games": can_sa, "formed": round(got_sa / max(1, can_sa), 3),
          "place_formed": round(float(np.mean(sa_places)), 2) if sa_places else None,
          "place_other": round(float(np.mean(other_places)), 2) if other_places else None}
    print(f"  本局凑得成 S / A 阵容的局 {can_sa / n:.0%}；这些局里最终成了 S / A 的 {sa['formed']:.0%}"
          + (f"（平均 {sa['place_formed']:.2f} 名，没成的 {sa['place_other']:.2f} 名）" if sa_places and other_places else ""))
    top = sorted(((k, v) for k, v in reach.items() if k[0] in "SA"), key=lambda kv: ("SA".index(kv[0][0]), -kv[1][0]))
    print("    凑得成的局里成型的比例：" + "、".join(f"{k} {v[1] / max(1, v[0]):.0%}（{v[0]} 局能凑）" for k, v in top))
    # 体系选手：第 8 回合的主选 / 备选，最后做成了哪个
    pl = [x for x in fb if x.get("plan8")]
    plan = None
    if pl:
        cnt = {"main": 0, "backup": 0, "other": 0, "none": 0, "no_plan": 0}
        for x in pl:
            mn, bk = x["plan8"]
            c, _ = nearest_comp(x["board"], comps, lob=x.get("lobby"), strict=True)
            if not mn:
                cnt["no_plan"] += 1
            elif c is None:
                cnt["none"] += 1
            else:
                cnt["main" if c["name"] in mn else "backup" if c["name"] in bk else "other"] += 1
        n8 = len(pl)
        withp = n8 - cnt["no_plan"]
        plan = {k: round(v / n8, 3) for k, v in cnt.items()}
        print(f"  体系：第 8 回合有主选 S / A 体系的局 {withp / n8:.0%}；这些局最后做成主选 {cnt['main'] / max(1, withp):.0%}、"
              f"备选 {cnt['backup'] / max(1, withp):.0%}、别的 {cnt['other'] / max(1, withp):.0%}、没成型 {cnt['none'] / max(1, withp):.0%}")
    return {"share": {t: round(v, 3) for t, v in tiers.items()}, "q": round(float(np.mean(phis)), 3), "sa_reach": sa, "plan8": plan,
            "reach": {k: {"games": v[0], "formed": round(v[1] / max(1, v[0]), 3)} for k, v in top},
            "comps": {k: {"share": round(r["n"] / n, 3), "place": round(float(np.mean(r["places"])), 2) if r["places"] else None}
                      for k, r in order}}


def show_engines(finals: list[dict]) -> dict | None:
    """引擎卡（布莱恩·铜须：战吼触发两次；提图斯·瑞文戴尔：亡语多触发）上场的比例，以及有它的局里场上能被放大的随从有几个。"""
    from rl.demos import ENGINES, _effects
    from tg.catalog import catalog
    fb = [x for x in finals if len(x["board"]) >= 5]
    if not fb:
        return None
    det = catalog()
    hp = ROOT / "rl" / "h13_profile.json"
    h13 = {r[0]: r[2] for r in (json.loads(hp.read_text(encoding="utf-8"))["final"]["cards"] if hp.exists() else [])}
    out, parts = {}, []
    for kind, ids, zh, eff in (("battlecry", ENGINES["battlecry"], "布莱恩·铜须", "战吼"),
                               ("deathrattle", ENGINES["deathrattle"], "提图斯·瑞文戴尔", "亡语")):
        base = {i[:-2] if i.endswith("_G") else i for i in ids}
        with_e = [x for x in fb if base & set(x["board"])]
        n_amp = [sum(kind in _effects(det.get(c) or {}) for c in x["board"]) for x in with_e]
        ref = max((h13.get(i, 0) for i in base), default=0)
        out[kind] = {"share": round(len(with_e) / len(fb), 3), "amplified": round(float(np.mean(n_amp)), 2) if n_amp else None,
                     "h13": ref}
        parts.append(f"{zh} 上场 {len(with_e) / len(fb):.0%}（万三 {ref:.0%}）" +
                     (f"，这些局场上{eff}随从平均 {np.mean(n_amp):.1f} 个" if n_amp else ""))
    print("  引擎卡：" + "；".join(parts))
    # 所有引擎（rl/comp.ENGINE_RULES，训练的 engine_bonus 同一张表）：上场比例、上场时配了几张燃料（最终场上）
    from rl.comp import ENGINE_FUEL_FULL, engine_tables
    ids = sorted({c for x in fb for c in x["board"]})
    tab = engine_tables(ids)
    row = {c: i for i, c in enumerate(ids)}
    eng_rows, all_eng = [], {}
    for j, (eid, zh) in enumerate(zip(tab["ids"], tab["names"])):
        with_e = [x for x in fb if eid in x["board"]]
        if not with_e and not h13.get(eid):
            continue
        nf = [int(tab["fuel"][[row[c] for c in x["board"]], j].sum()) for x in with_e]
        full = sum(v >= ENGINE_FUEL_FULL for v in nf)
        all_eng[zh] = {"share": round(len(with_e) / len(fb), 3), "fuel": round(float(np.mean(nf)), 2) if nf else None,
                       "full": round(full / len(fb), 3), "h13": h13.get(eid, 0)}
        eng_rows.append(f"{zh} {len(with_e) / len(fb):.0%}（万三 {h13.get(eid, 0):.0%}）" + (f" 燃料 {np.mean(nf):.1f}" if nf else ""))
    anyfull = sum(any(int(tab["fuel"][[row[c] for c in x["board"]], j].sum()) >= ENGINE_FUEL_FULL
                      for j, eid in enumerate(tab["ids"]) if eid in x["board"]) for x in fb)
    print(f"  引擎联动（场上有引擎、配齐 {ENGINE_FUEL_FULL} 张它放大的随从的局 {anyfull / len(fb):.0%}）：" + "、".join(eng_rows))
    out["all"] = all_eng
    out["full_share"] = round(anyfull / len(fb), 3)
    return out


def show_simcards(picks) -> dict | None:
    """引擎实测的关键卡（rl/sim_value.json，bg.py simcards）：模型买它们的倍率。"""
    sp = ROOT / "rl" / "sim_value.json"
    if not sp.exists() or picks is None or not picks.n:
        return None
    sv = json.loads(sp.read_text(encoding="utf-8"))
    mt = picks.table(10)
    rows = [(k, sv["cards"][k]["name"], sv["cards"][k]["contrib"], mt[k]["lift"]) for k in sv.get("key", []) if k in mt]
    if not rows:
        return None
    low = sorted((r for r in rows if r[3] < 1.0), key=lambda r: -r[2])[:12]
    high = sorted((r for r in rows if r[3] >= 1.5), key=lambda r: -r[3])[:8]
    print("  引擎实测关键卡（拿掉它同期胜率掉多少）里模型不爱买的：" +
          "、".join(f"{n} +{c:.2f}（倍率 {m:.1f}）" for _, n, c, m in low))
    if high:
        print("  引擎实测关键卡里模型爱买的：" + "、".join(f"{n} +{c:.2f}（倍率 {m:.1f}）" for _, n, c, m in high))
    return {"low": [[n, c, m] for _, n, c, m in low], "high": [[n, c, m] for _, n, c, m in high]}


def advice(model: dict, pro: dict | None) -> list[str]:
    """差距最大的几条，用一句话说。"""
    p = (pro or {}).get("turns") or {}
    out = []
    for t in ("2", "4", "5", "6", "7"):
        m, h = model.get(t), p.get(t)
        if not m or not h:
            continue
        if h["up"] - m["up"] >= 0.25:
            spent = sorted(((m.get(k) or 0, k) for k in COLS if k != "upgrade"), reverse=True)[:2]
            out.append(f"第 {t} 回合升本率 {m['up']:.0%}（高手 {h['up']:.0%}）；钱主要花在 "
                       + "、".join(f"{ZH[k]} {v:.1f}" for v, k in spent if v > 0))
        if (m.get("refresh") or 0) - (h.get("refresh") or 0) >= 1:
            out.append(f"第 {t} 回合刷新花 {m['refresh']:.1f}（高手 {h['refresh']:.1f}）")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(RUNS / "v3" / "latest.pt"))
    ap.add_argument("--games", type=int, default=64)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="")
    ap.add_argument("--sample", action="store_true", help="按概率抽动作（默认和竞技场一样取最大概率）")
    ap.add_argument("--pro", default="", help="高分回放目录：算高手那一列写进 rl/pro_profile.json（gold_flow）")
    ap.add_argument("--turns", type=int, default=9)
    ap.add_argument("--boosted", action="store_true", help="对手换成高分对局的真实阵容（和训练考试 eval_boost_place 一样）")
    ap.add_argument("--late-ckpt", default="", help="第 --late-from 回合起换这个模型打（例如 runs\\teacher.pt：老师后期会不会养场面），"
                                                    "结果文件名后面加 _late")
    ap.add_argument("--late-from", type=int, default=10)
    ap.add_argument("--lookahead-from", type=int, default=0, help="例如 9：第 9 回合起每个决策先在副本上推演几步、选结束时场面最强的"
                                                               "（comptest_lookahead.bat），结果文件名后面加 _look")
    ap.add_argument("--lookahead-k", type=int, default=6)
    ap.add_argument("--lookahead-depth", type=int, default=8)
    ap.add_argument("--seed-comps", default="", help="例如 8,9,10：每局在其中随机一个回合把模型的场面换成高手已经成型的 S / A 阵容"
                                                     "（comptest_seeded.bat），结果写进 comptest_seeded.json")
    a = ap.parse_args()
    prof = json.loads(PROFILE.read_text(encoding="utf-8")) if PROFILE.exists() else {}
    if a.pro:
        prof["gold_flow"] = pro_table(Path(a.pro))
        PROFILE.write_text(json.dumps(prof, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"高手金币流向：{prof['gold_flow']['games']} 局 → {PROFILE}")
        show({}, prof["gold_flow"], a.turns)
        return 0
    if not Path(a.ckpt).exists():
        print(f"没有检查点 {a.ckpt}")
        return 1
    dev = a.device
    if not dev:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    late = None
    if a.late_ckpt:
        if not Path(a.late_ckpt).exists():
            print(f"没有检查点 {a.late_ckpt}")
            return 1
        print(f"第 {a.late_from} 回合起换 {a.late_ckpt} 打 …", flush=True)
        late = model_policy(a.late_ckpt, dev, not a.sample)
    games, info = play_model(a.ckpt, a.games, a.seed, a.threads, not a.sample, dev, a.boosted, seed_comps=a.seed_comps,
                             late=late, late_from=a.late_from, look_from=a.lookahead_from, look_k=a.lookahead_k,
                             look_depth=a.lookahead_depth)
    if late is not None:
        print(f"（第 1–{a.late_from - 1} 回合 {Path(a.ckpt).name}，第 {a.late_from} 回合起 {Path(a.late_ckpt).name}）", flush=True)
    model = summarize(games)
    pro = prof.get("gold_flow")
    show(model, pro, a.turns)
    tips = advice(model, pro)
    if tips:
        print("\n差距最大的地方：")
        for x in tips:
            print("  ·", x)
    if not pro:
        print("\n（rl/pro_profile.json 里还没有高手的金币流向：bg.py goldflow --pro <高分回放目录>）")
    comps = show_comps(info.get("finals") or [])
    pk = show_picks(info.get("picks"), info.get("sells") or {})
    bs = buy_style(info.get("buy_rows") or [])
    if bs:
        cvp = ROOT / "rl" / "card_value.json"
        ps = (json.loads(cvp.read_text(encoding="utf-8")).get("style") if cvp.exists() else None) or \
            {"tribe": 0.36, "syn": 0.07, "key": 0.59, "eco": 0.11}
        zh = {"tribe": "围绕种族", "syn": "和场上常见搭配", "key": "关键卡", "eco": "经济卡",
              "simkey": "引擎实测关键卡", "simsyn": "和场上是引擎实测好搭配",
              "engine": "被场上引擎卡放大（有布莱恩买战吼 / 有提图斯买亡语）", "engcard": "买引擎卡本身",
              "triple": "凑三连（场上 / 手里已有同名）", "tgtS": "S 级阵容核心卡", "tgtA": "A 级核心卡", "tgtB": "B 级核心卡",
              "tgtfit": "核心卡和场上 / 手里的核心卡同属一套", "tgtsup": "正在做 S / A 阵容时买它的配套卡",
              "engfuel": "引擎联动（买场上引擎的燃料 / 有燃料时买引擎）"}
        tcp = ROOT / "rl" / "target_comps.json"
        pb = (json.loads(tcp.read_text(encoding="utf-8")).get("playbook") if tcp.exists() else None) or {}
        if "pro_support_share" in pb and "tgtsup" not in ps:
            ps = {**ps, "tgtsup": pb["pro_support_share"]}
        print("  买法：" + "、".join(f"{zh[w]} {bs[w]:.0%}" + (f"（高手 {ps[w]:.0%}）" if w in ps else "")
                                  for w in zh if w in bs))
        if pk is not None:
            pk["style"] = bs
        ks = keep_sold(info.get("sell_rows") or [])
        if ks is not None:
            pref = (pb or {}).get("pro_keep_sold_share")
            print(f"  卖牌：卖掉正在做的 S / A 阵容里该留的牌（高手做这套时留着不卖的核心 / 配套卡，没有第二张） {ks['share']:.1%}"
                  + (f"（高手 {pref:.1%}）" if pref is not None else "") +
                  ("；卖得最多：" + "、".join(f"{k} {v}" for k, v in ks["top"]) if ks["top"] else ""))
            if pk is not None:
                pk["keep_sold"] = ks
    tg = show_targets(info.get("finals") or [])
    if comps is not None and tg:
        comps["targets"] = tg
    eng = show_engines(info.get("finals") or [])
    if comps is not None and eng:
        comps["engines"] = eng
    sv = show_simcards(info.get("picks"))
    if pk is not None and sv:
        pk["sim"] = sv
    strength = show_strength(info)
    out = Path(a.ckpt).resolve().parent / (("comptest_seeded.json" if a.seed_comps else "comptest.json") if a.boosted
                                           else "goldflow.json")
    if late is not None:
        out = out.with_name(out.stem + "_late.json")
    if a.lookahead_from > 0:
        out = out.with_name(out.stem + "_look.json")
    out.write_text(json.dumps({"ckpt": str(a.ckpt), "iter": info.get("iter"), "games": a.games, "seed": a.seed,
                               "greedy": not a.sample, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                               "boosted": a.boosted, "turns": model, "final": comps, "picks": pk,
                               "strength": strength,
                               "places": [p for p in (info.get("places") or {}).values() if p > 0],
                               "seeded": info.get("seeded"),
                               "late": {"ckpt": a.late_ckpt, "from": a.late_from} if late is not None else None,
                               "look": info.get("look")},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ {out}")
    if a.lookahead_from > 0:
        compare_look(out.with_name(out.name.replace("_look.json", ".json")), out)
    return 0


def compare_look(base: Path, look: Path):
    """推演那一次和同一个模型、同一批对局不推演的那一次（comptest_lookahead.bat 先后跑的两次）并排比。"""
    try:
        b, k = (json.loads(p.read_text(encoding="utf-8")) for p in (base, look))
    except (OSError, ValueError):
        return
    if any(b.get(x) != k.get(x) for x in ("ckpt", "iter", "games", "seed")):
        print(f"（{base.name} 不是同一个模型 / 同一批对局，不并排比）")
        return

    def rate(ps, f):
        return f"{np.mean([f(p) for p in ps]):.0%}" if ps else "-"
    pb, pk = b.get("places") or [], k.get("places") or []
    print(f"\n和不推演的同一批对局比（{base.name}，第 {k.get('iter')} 轮，{len(pk)} 局）：")
    print(f"  平均名次 {np.mean(pb):.2f} → {np.mean(pk):.2f}，吃鸡 {rate(pb, lambda p: p == 1)} → {rate(pk, lambda p: p == 1)}，"
          f"前四 {rate(pb, lambda p: p <= 4)} → {rate(pk, lambda p: p <= 4)}")
    sb, sk = (x.get("strength") or {} for x in (b, k))
    sb = sb.get("rows", sb) if isinstance(sb, dict) else {}
    sk = sk.get("rows", sk) if isinstance(sk, dict) else {}
    for key, name in (("model", "场面强度"), ("grow", "养了多少")):
        cells = []
        for t in range(9, 15):
            x, y = (s_.get(str(t), s_.get(t, {})) or {} for s_ in (sb, sk))
            if x.get(key) is not None and y.get(key) is not None:
                cells.append(f"第 {t} 回合 {x[key]:.2f}→{y[key]:.2f}")
        if cells:
            print(f"  {name}：" + "，".join(cells))


if __name__ == "__main__":
    sys.exit(main())
