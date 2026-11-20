"""奖励塑形：在名次奖励之外，每一步当场加 / 扣的小奖励（都由 Config 里的系数控制，0 = 关）。

  tempo_bonus / tempo_deficit     升本节奏：按目标曲线（真人或高手的每回合平均等级）升本加分、结束回合时落后扣分
  upgrade_starve_penalty 等       金币流向：落后时把升本的钱花掉、钱够升本却结束回合扣分；剩钱扣分
  refresh_waste_penalty           无效刷新：刷新后什么都没买就又刷新 / 结束回合
  econ_bonus                      经济：结束回合时“存下的资源”（下回合额外金币 + 2 × 永久金币上限加成 + 手牌价值）比这局上次
                                  多了多少就加多少，一局结束时扣回最后的存款（纯势能差，整局和为 0；资源变成场面由 combat_bonus 接着算）
  combat_bonus                    阵容强度：结束回合时让引擎拿当前场面打一组同期高分阵容（rl/boardsim.py），
                                  胜率比上次量的时候涨了多少就加多少（势能差：整局加起来 = 最后的强度 - 第一次的强度）
  grow_bonus                      养场面：后期每回合开始、结束各量一次同一把尺子的强度，加 结束 - 开始（这回合自己养了多少，不扣回）
  keep_penalty                    卖掉正在做的阵容里该留的牌
（第 33328 轮删掉 comp_bonus 主种族纯度、target_bonus 目标阵容、engine_bonus 引擎联动：C3 消融里关掉以后持平或变好，
  见 docs/recipe_history.md。）

Trainer（rl/ppo.py）继承 ShapingMixin。这里用到 Trainer 的：cfg、iter、rollout（ep / last / r：改已经存下的奖励）、
mix（日志计数）。每个座位的状态（upg_ep、upg_n、ref_idx、gs）在这里建；tempo_t0 存在检查点里。
_init_shaping() 在 Trainer.__init__ 里调用一次。
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np

from tg.actions import A_BUY, A_DARK_GIFT, A_END, A_REFRESH, A_SELL, A_UPGRADE
from tg.obs import C_HAND, F_GOLD, F_TIER, F_UPCOST, N_HAND, gold_next, max_gold_bonus

from .league import load_real_profile


class ShapingMixin:
    def _init_shaping(self):
        cfg = self.cfg
        prof = load_real_profile().get("turns") or {}
        pro_prof = Path(__file__).resolve().parent / "pro_profile.json"
        if cfg.tempo_curve == "pro" and pro_prof.exists():
            prof = json.loads(pro_prof.read_text(encoding="utf-8")).get("turns") or {}
        self.tier_curve = {int(t): float(v.get("tier") or 0) for t, v in prof.items() if (v.get("n") or 0) >= 5}
        self.tempo_t0 = None                           # 升本奖励从哪一轮开始减（存在检查点里）
        self.upg_ep = np.full(self.n_slots, -1, dtype=np.int64)    # 每个槽位：这一局升了几次本
        self.upg_n = np.zeros(self.n_slots, dtype=np.int64)
        self.ref_idx = np.full(self.n_slots, -1, dtype=np.int64)   # 每个槽位：上一次还没买东西的刷新（本轮的步编号）
        self.judge = None                              # 阵容强度（combat_bonus）：每个座位上次量到的强度
        self.cs_ep = np.full(self.n_slots, -1, dtype=np.int64)
        self.cs_prev = np.zeros(self.n_slots, dtype=np.float64)
        self.cs_rng = random.Random(cfg.seed + 11)
        self.ec_ep = np.full(self.n_slots, -1, dtype=np.int64)  # 经济（econ_bonus）：每个座位上次的存款
        self.ec_prev = np.zeros(self.n_slots, dtype=np.float64)
        self.fc_ep = np.full(self.n_slots, -1, dtype=np.int64)  # 体系感知辅助头的标签：每个座位最近一次结束回合时成型的阵容
        self.fc_idx = np.full(self.n_slots, -1, dtype=np.int64)  # （rl/target_comps.json 里的序号；-1 = 没成型）
        if cfg.combat_bonus > 0:
            self.judge = self._make_judge()
        self.grow_judge = None                         # 养场面（grow_bonus）：每个座位这回合开始时的强度
        self.gr_ep = np.full(self.n_slots, -1, dtype=np.int64)
        self.gr_turn = np.full(self.n_slots, -1, dtype=np.int64)
        self.gr_start = np.full(self.n_slots, np.nan, dtype=np.float64)
        self.gr_rng = random.Random(cfg.seed + 13)
        if getattr(cfg, "grow_bonus", 0) > 0:
            self.grow_judge = self._make_judge(fights=cfg.grow_fights, what="养场面")

    def _make_judge(self, fights: int | None = None, what: str = ""):
        from .boardsim import BoardJudge
        lib = getattr(self, "_pro_lib", None)
        if not lib:
            from paths import RUNS
            from .curricula import merge_pro_boards
            paths = [Path(p) for p in str(self.cfg.pro_boards).split(";") if p.strip()] or [RUNS / "pro_boards.json"]
            lib = merge_pro_boards(p for p in paths if p.exists())
        if not lib:
            print("没有高分阵容库（runs/pro_boards.json、runs/h13_boards.json），阵容强度奖励这次不用", flush=True)
            return None
        cfg = self.cfg
        if what:
            print(f"{what}奖励：第 {cfg.grow_from_turn} 回合起（抽 {cfg.grow_frac:.0%} 的回合）每回合开始、结束各打一次 "
                  f"{cfg.combat_boards} 个同期高分阵容 × {fights} 场，加 {cfg.grow_bonus} ×（结束 - 开始）", flush=True)
            return BoardJudge(lib, k=cfg.combat_boards, n=int(fights or cfg.combat_fights), seed=cfg.seed)
        print(f"阵容强度奖励：每次结束回合（第 {cfg.combat_from_turn} 回合起，抽 {cfg.combat_frac:.0%}）打 "
              f"{cfg.combat_boards} 个同期高分阵容 × {cfg.combat_fights} 场，系数 {cfg.combat_bonus}", flush=True)
        return BoardJudge(lib, k=cfg.combat_boards, n=cfg.combat_fights, seed=cfg.seed)

    def _combat_shape(self, env, off, idx, f, ak, r0):
        """阵容强度奖励（combat_bonus）：学习者结束回合时（局面还是招募阶段的最后样子），拿这局的副本
        让引擎打一组同期高分阵容，Φ = 平均胜率；奖励 = combat_bonus × (Φ - 这一局上次量到的 Φ)。
        整局加起来是 combat_bonus × (最后一次 - 第一次)：为了升本暂时变弱不吃亏，之后变强补回来；
        真正奖励的是“场面越打越强”——卡牌配合、关键卡、金色、卖掉弱的换强的，引擎都替我们算了。
        bot 环境 40 局实测（脚本机器人）：第 7–14 回合的 Φ 和最终名次的相关系数 -0.43 到 -0.57。"""
        cfg = self.cfg
        if self.judge is None or cfg.combat_bonus <= 0:
            return r0
        ends = np.flatnonzero(ak == A_END)
        if not len(ends):
            return r0
        t0 = time.time()
        turns = np.rint(f[ends, 0] * 20).astype(int)
        for j, t in zip(ends, turns):
            if t < cfg.combat_from_turn or (cfg.combat_frac < 1 and self.cs_rng.random() >= cfg.combat_frac):
                continue
            i = int(idx[j])
            g, seat = env.slot(i)
            phi = self.judge.judge(env.game(int(g)), int(seat), int(t))
            if phi is None:
                continue
            raw = phi
            if t >= getattr(cfg, "combat_late_from", 10):
                phi = phi * getattr(cfg, "combat_late_mult", 1.0)
            s = off + i
            ep = int(self.rollout.ep[s])
            if self.cs_ep[s] == ep:
                r0[j] += cfg.combat_bonus * (phi - self.cs_prev[s])
            self.cs_ep[s], self.cs_prev[s] = ep, phi
            self.mix["cs_sum"] += raw
            self.mix["cs_n"] += 1
            if t >= 9:
                self.mix["cs_late_sum"] += raw
                self.mix["cs_late_n"] += 1
        self.mix["t_combat"] += time.time() - t0
        return r0

    def _grow_shape(self, env, off, idx, f, ak, r0):
        """养场面奖励（grow_bonus）：第 grow_from_turn 回合起，学习者这回合第一次决策前（招募阶段刚开始）拿这局的副本
        打一组同期高分阵容量 Φ_开始，结束回合时（局面还是招募阶段的最后样子）同一组再量 Φ_结束，结束回合那一步加
        grow_bonus × (Φ_结束 - Φ_开始)。同一组考官、同一局的战斗随机数（combat_preview 按对局种子），两次一减噪声小。

        为什么不用 combat_bonus：它和上一次结束回合比，对手每回合变强那一截也扣在模型头上，整局加起来是最后的强度，
        局末扣回以后后期那几步几乎没有功劳；不扣回又等于奖励“早点死、死的时候场面像样”。这一项只看这回合自己养了多少：
        引擎（布莱恩、提图斯、热气球测绘员……）的价值是以后每回合都养得更多，留着它就一直有分；卖掉它当场看不出亏，
        之后每回合少拿。死得早就少几回合，不会多拿。"""
        cfg = self.cfg
        if self.grow_judge is None or cfg.grow_bonus <= 0:
            return r0
        t0 = time.time()
        turns = np.rint(f[:, 0] * 20).astype(int)
        for j in range(len(idx)):
            t = int(turns[j])
            if t < cfg.grow_from_turn:
                continue
            i = int(idx[j])
            s = off + i
            ep = int(self.rollout.ep[s])
            if self.gr_ep[s] != ep or self.gr_turn[s] != t:      # 这回合第一次决策：决定量不量，量的话先量开始
                self.gr_ep[s], self.gr_turn[s] = ep, t
                self.gr_start[s] = np.nan
                if cfg.grow_frac >= 1 or self.gr_rng.random() < cfg.grow_frac:
                    g, seat = env.slot(i)
                    v = self.grow_judge.judge(env.game(int(g)), int(seat), t)
                    if v is not None:
                        self.gr_start[s] = v
            if ak[j] == A_END and not np.isnan(self.gr_start[s]):
                g, seat = env.slot(i)
                v = self.grow_judge.judge(env.game(int(g)), int(seat), t)
                if v is not None:
                    d = float(v - self.gr_start[s])
                    r0[j] += cfg.grow_bonus * d
                    self.mix["grow_sum"] += d
                    self.mix["grow_n"] += 1
                self.gr_start[s] = np.nan
        self.mix["t_grow"] += time.time() - t0
        return r0

    def tempo_now(self) -> float:
        cfg = self.cfg
        if cfg.tempo_bonus <= 0:
            return 0.0
        if self.tempo_t0 is None:
            self.tempo_t0 = self.iter
        x = (self.iter - self.tempo_t0) / max(1, cfg.tempo_anneal_iters)
        return cfg.tempo_bonus * max(0.0, 1.0 - x)

    def _tier_target(self, turn: int) -> float:
        if not self.tier_curve:
            return 6.0
        t = min(max(turn, min(self.tier_curve)), max(self.tier_curve))
        while t not in self.tier_curve and t > min(self.tier_curve):
            t -= 1
        return self.tier_curve.get(t, 6.0)

    def _shape(self, slots, f, ak, r0, c=None):
        """升本节奏奖励（当场加在升本这一步上）；记下刷新，供 _after_add 判断是不是无效刷新。"""
        tb = self.tempo_now()
        turns = np.rint(f[:, 0] * 20).astype(int)
        dc = self.cfg.tempo_deficit
        au = getattr(self, "audit", None)                  # 塑形审计（rl/audit.py）：每项给了多少分，按项记
        b0 = r0.copy() if au is not None else None

        def mark(term):
            nonlocal b0
            if au is not None:
                au.add(term, r0 - b0, ak, turns)
                b0 = r0.copy()
        for j in np.flatnonzero(ak == A_END):              # 结束回合：落后于目标曲线就扣分
            s = int(slots[j])
            ep = int(self.rollout.ep[s])
            tier = 1 + (int(self.upg_n[s]) if self.upg_ep[s] == ep else 0)
            gap = max(0.0, self._tier_target(int(turns[j])) - 0.5 - tier)
            self.mix["gap_sum"] += gap
            self.mix["gap_n"] += 1
            if dc > 0 and gap > 0:
                r0[j] -= dc * gap
        mark("tempo_gap")
        for j in np.flatnonzero(ak == A_UPGRADE):
            s = int(slots[j])
            ep = int(self.rollout.ep[s])
            if self.upg_ep[s] != ep:
                self.upg_ep[s], self.upg_n[s] = ep, 0
            if tb > 0 and 1 + self.upg_n[s] < self._tier_target(int(turns[j])) + 0.5:
                r0[j] += tb
                self.mix["tempo_paid"] += 1
            self.upg_n[s] += 1
        mark("tempo_up")
        if f.shape[1] > F_UPCOST:
            self._gold_shape(slots, f, ak, r0)
            mark("gold")
        if getattr(self.cfg, "econ_bonus", 0) > 0 and f.shape[1] > 9:
            self._econ_shape(slots, f, ak, r0, c)
            mark("econ")
        if getattr(self.cfg, "keep_penalty", 0) > 0 and c is not None:
            ShapingMixin._keep_shape(self, f, c, ak, r0)
            mark("keep")
        return r0

    def _engine_tab(self):
        if getattr(self, "_eng_tab", None) is None:
            from tg.catalog import base_of_index, n_cards
            from .comp import engine_tables
            self._eng_tab = engine_tables([base_of_index(i) if i else "" for i in range(n_cards())])
        return self._eng_tab

    def _keep_shape(self, f, c, ak, r0):
        """别卖正在做的阵容里该留的牌（keep_penalty）：卖掉的是高手做这套时留着不卖的核心 / 配套卡
        （bg.py playbook 的 keep：热气球测绘员、布莱恩·铜须、提图斯、断手被遗忘者……），而且没有第二张同名的 → 扣分。
        尖利的鲨鱼、新锐植物学家这类买来就要卖的不在 keep 里，不扣。高手的卖牌里这种占 5.2%，第 15180 轮模型卖得最多的牌里
        有布莱恩·铜须 48 次、热气球测绘员 47 次。"""
        if getattr(self, "_keep_tab", None) is None:
            from tg.catalog import base_of_index, n_cards
            from .comp import CompTables, load_target_comps
            self._keep_tab = CompTables(load_target_comps(), [base_of_index(i) if i else "" for i in range(n_cards())],
                                        tiers="SA")
        bad = self._keep_tab.keep_sold(c, f, ak)
        if bad.any():
            r0[bad] -= self.cfg.keep_penalty
            self.mix["keep_sold"] += int(bad.sum())
        self.mix["sells"] += int(((ak >= A_SELL) & (ak < A_SELL + 7)).sum())
        return r0

    def _hand_values(self):
        """手牌每张值几块钱（按卡牌编号查表）：随从 3（金色 6）；酒馆法术按它的价钱；
        不要钱的法术：三连奖励 4、其它（鲜血宝石之类）0.5。"""
        if getattr(self, "_hv", None) is None:
            from tg.catalog import base_id, card_id, catalog, is_golden_id, n_cards
            det = catalog()
            hv = np.zeros(n_cards(), dtype=np.float64)
            for i in range(1, n_cards()):
                cid = card_id(i)
                x = det.get(cid) or det.get(base_id(cid)) or {}
                if x.get("kind") == "minion":
                    hv[i] = 6.0 if is_golden_id(cid) else 3.0
                elif x.get("kind") in ("spell", "tavern_spell"):
                    cost = float(x.get("cost") or 0)
                    hv[i] = cost if cost > 0 else 4.0 if "Triples" in cid else 0.5
            self._hv = hv
        return self._hv

    def _econ_value(self, f, c=None):
        """存款 E（金币当量）：下回合额外金币 + 2 × 永久金币上限加成 + 手牌的价值（c 给了才算）。"""
        e = (gold_next(f) + 2 * max_gold_bonus(f)).astype(np.float64)
        if c is not None:
            hv = ShapingMixin._hand_values(self)
            h = np.asarray(c)[:, C_HAND:C_HAND + N_HAND].astype(np.int64)
            e = e + np.where(h > 0, hv[np.clip(h, 0, len(hv) - 1)], 0.0).sum(1)
        return e

    def _econ_shape(self, slots, f, ak, r0, c=None):
        """经济奖励（econ_bonus）：结束回合时存款 E = 下回合额外金币 + 2 × 永久金币上限加成 + 手牌的价值，
        奖励 = econ_bonus × (E - 这局上次结束回合时的 E)；这一局结束时把最后的 E 扣回来（_econ_close），
        所以整局加起来正好是 0，只是把功劳记在产出资源的那一步——囤牌、囤钱到最后都不白拿。
        例：场上有布莱恩·铜须时打出南海卖艺者，下回合多 2 块（没有布莱恩 1 块）；打出钢铁场地管理员拿 4 张强固
        （没有布莱恩 2 张）、竞技表演者发现 2 张酒馆法术（没有布莱恩 1 张）→ 当回合拿到翻倍的功劳。
        下回合钱到手 / 牌打出去，E 减少、扣回来——换成了更强的场面由阵容强度（combat_bonus）补上。"""
        ends = np.flatnonzero(ak == A_END)
        if not len(ends):
            return r0
        e = ShapingMixin._econ_value(self, f[ends], None if c is None else np.asarray(c)[ends])
        for j, ej in zip(ends, e):
            s = int(slots[j])
            ep = int(self.rollout.ep[s])
            prev = self.ec_prev[s] if self.ec_ep[s] == ep else 0.0
            r0[j] += self.cfg.econ_bonus * (float(ej) - prev)
            self.ec_ep[s], self.ec_prev[s] = ep, float(ej)
            self.mix["econ_sum"] += float(ej)
            self.mix["econ_n"] += 1
        return r0

    def _econ_close(self, s: int) -> float:
        """座位 s 这一局结束：返回要加在最后一步上的奖励（- econ_bonus × 最后一次的存款），之后清零。"""
        bonus = getattr(self.cfg, "econ_bonus", 0)
        if bonus <= 0 or self.ec_ep[s] != int(self.rollout.ep[s]):
            return 0.0
        v = -bonus * self.ec_prev[s]
        self.ec_ep[s], self.ec_prev[s] = -1, 0.0
        return v

    def _repay_now(self) -> float:
        """这一轮局末扣回势能的比例（shape_repay）：第一次用起 shape_repay_ramp 轮从 0 线性加上去。"""
        k = float(getattr(self.cfg, "shape_repay", 0) or 0)
        if k <= 0:
            return 0.0
        if getattr(self, "repay_t0", None) is None:
            self.repay_t0 = int(self.iter)
        ramp = max(1, int(getattr(self.cfg, "shape_repay_ramp", 1) or 1))
        return k * min(1.0, max(0.0, (self.iter - self.repay_t0 + 1) / ramp))

    def _shape_close(self, s: int) -> float:
        """座位 s 这一局结束：把阵容强度（combat）这局最后的 Φ 扣回 shape_repay 那么多（见 Config.shape_repay）。
        扣完以后整局加起来 ≈ -第一次的 Φ，和动作无关：养场面那一步还是当场拿到功劳，最后由名次决定值不值。
        （第 33328 轮以前还有目标阵容 target、引擎联动 engine 两项，C3 消融后删了。）"""
        k = ShapingMixin._repay_now(self)
        ep = int(self.rollout.ep[s])
        cfg = self.cfg
        v = rep = 0.0
        b = getattr(cfg, "combat_bonus", 0)
        if b > 0 and self.cs_ep[s] == ep:
            v = rep = b * float(self.cs_prev[s])
        self.mix["fin_sum"] += v                        # 这局局末还挂着多少（不扣时就是白拿的）
        self.mix["fin_n"] += 1
        return -k * rep

    def _gold_shape(self, slots, f, ak, r0):
        """金币流向（见 upgrade_starve_penalty）。观察里第 7 / 10 / 11 个数是 金币/10、等级/6、升本费/10。
        同时统计：回合开始时钱够升本的回合里升了本的比例（第 2–8 回合，up_afford；高手约 0.68）和第 2 回合升本率（up_t2）。"""
        pen = self.cfg.upgrade_starve_penalty
        if not hasattr(self, "gs"):
            n = len(self.rollout.last)
            self.gs = {k: np.full(n, -1, dtype=np.int64) for k in ("ep", "turn", "gold", "up", "tier", "act", "ts_turn",
                                                                  "ts_tier", "ts_aff")}
            self.gs["behind"] = np.zeros(n, dtype=bool)
        g = self.gs
        turns = np.rint(f[:, 0] * 20).astype(int)
        golds = np.rint(f[:, F_GOLD] * 10).astype(int)
        ups = np.rint(f[:, F_UPCOST] * 10).astype(int)
        tiers = np.rint(f[:, F_TIER] * 6).astype(int)
        for j in range(len(ak)):
            s, a, t = int(slots[j]), int(ak[j]), int(turns[j])
            ep = int(self.rollout.ep[s])
            gold, up, tier = int(golds[j]), int(ups[j]), int(tiers[j])
            same = g["ep"][s] == ep and g["turn"][s] == t
            # 上一步把金币从“够升本”花到“不够”（上一步不是升本、卖），而且当时落后
            if same and g["behind"][s] and g["act"][s] != A_UPGRADE and g["up"][s] > 0 \
                    and g["gold"][s] >= g["up"][s] > gold:
                self.mix["starve"] += 1
                prev = int(self.rollout.last[s])
                if pen > 0 and prev >= 0:
                    buy = A_BUY <= g["act"][s] < A_SELL and t >= self.cfg.starve_buy_from_turn
                    w = self.cfg.starve_buy_weight if buy else self.cfg.starve_gift_weight if g["act"][s] == A_DARK_GIFT else 1.0
                    self.rollout.r[prev] -= pen * w
                    if getattr(self, "audit", None) is not None:    # 扣在上一步（花钱那一步）上
                        self.audit.add_one("gold", -pen * w, g["act"][s], t)
            if not (g["ts_turn"][s] == t and g["ep"][s] == ep):         # 新回合的第一步
                g["ts_turn"][s], g["ts_tier"][s], g["ts_aff"][s] = t, tier, int(up > 0 and gold >= up)
            behind = tier < self._tier_target(t) - 0.5
            if a == A_END:
                if 2 <= t <= 8 and g["ts_aff"][s] == 1:
                    self.mix["aff_n"] += 1
                    self.mix["aff_up"] += int(tier > g["ts_tier"][s])
                if t == 2:
                    self.mix["t2_n"] += 1
                    self.mix["t2_up"] += int(tier > g["ts_tier"][s])
                if t >= 2 and gold > 0:
                    self.mix["unspent"] += gold
                    if self.cfg.unspent_gold_penalty > 0:
                        r0[j] -= self.cfg.unspent_gold_penalty * gold
                if behind and up > 0 and gold >= up:                    # 钱够升本、落后，却结束回合
                    self.mix["end_rich"] += 1
                    if pen > 0:
                        r0[j] -= pen
            g["ep"][s], g["turn"][s], g["gold"][s], g["up"][s], g["tier"][s], g["act"][s] = ep, t, gold, up, tier, a
            g["behind"][s] = behind
        return r0

    def _after_add(self, slots, ak, turns=None):
        """无效刷新：上一次刷新之后没买任何东西就又刷新 / 结束回合 → 给那次刷新扣分（改已经存下的奖励）。
        turns 给了时只在第 refresh_waste_until 回合以前扣（后期找三连、找关键卡本来就要多刷，高手第 8–9 回合
        刷新花 1.7–2.6 块，很多次刷完不买）。"""
        pen = self.cfg.refresh_waste_penalty
        until = getattr(self.cfg, "refresh_waste_until", 99)
        for j, a in enumerate(ak):
            s = int(slots[j])
            a = int(a)
            if a == A_REFRESH or a == A_END:
                prev = self.ref_idx[s]
                if prev >= 0:
                    self.mix["refresh_waste"] += 1
                    if pen > 0 and (turns is None or int(turns[j]) <= until):
                        self.rollout.r[prev] -= pen
                        if getattr(self, "audit", None) is not None:
                            self.audit.add_one("refresh_waste", -pen, A_REFRESH, None if turns is None else int(turns[j]))
                self.ref_idx[s] = self.rollout.last[s] if a == A_REFRESH else -1
            elif A_BUY <= a < A_SELL:
                self.ref_idx[s] = -1
            kind = ("end" if a == A_END else "refresh" if a == A_REFRESH else "upgrade" if a == A_UPGRADE else
                    "buy" if A_BUY <= a < A_SELL else "sell" if A_SELL <= a < A_SELL + 7 else None)
            if kind:
                self.mix[kind] += 1
