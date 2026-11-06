"""奖励塑形：在名次奖励之外，每一步当场加 / 扣的小奖励（都由 Config 里的系数控制，0 = 关）。

  tempo_bonus / tempo_deficit     升本节奏：按目标曲线（真人或高手的每回合平均等级）升本加分、结束回合时落后扣分
  upgrade_starve_penalty 等       金币流向：落后时把升本的钱花掉、钱够升本却结束回合扣分；剩钱扣分
  refresh_waste_penalty           无效刷新：刷新后什么都没买就又刷新 / 结束回合
  comp_bonus                      阵容成型：第 comp_from_turn 回合起，结束回合按场面配合程度加分（rl/comp.py 的参照）
  target_bonus                    目标阵容：结束回合时场面离 S / A / B 级目标阵容（rl/target_comps.json）多近，
                                  Φ = 等级权重（S 1 / A 0.8 / B 0.6）× 核心卡占比，比这局上次涨多少加多少（整局 = 最后的 Φ）
  econ_bonus                      经济：结束回合时“存下的资源”（下回合额外金币 + 2 × 永久金币上限加成 + 手牌价值）比这局上次
                                  多了多少就加多少，一局结束时扣回最后的存款（纯势能差，整局和为 0；资源变成场面由 combat_bonus 接着算）
  combat_bonus                    阵容强度：结束回合时让引擎拿当前场面打一组同期高分阵容（rl/boardsim.py），
                                  胜率比上次量的时候涨了多少就加多少（势能差：整局加起来 = 最后的强度 - 第一次的强度）

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
from tg.obs import C_BOARD, C_HAND, F_GOLD, F_TIER, F_UPCOST, N_HAND, gold_next, max_gold_bonus

from .comp import load_comp_refs
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
        self.tg_ep = np.full(self.n_slots, -1, dtype=np.int64)  # 目标阵容（target_bonus）：每个座位上次的 Φ
        self.tg_prev = np.zeros(self.n_slots, dtype=np.float64)
        self.ec_ep = np.full(self.n_slots, -1, dtype=np.int64)  # 经济（econ_bonus）：每个座位上次的存款
        self.ec_prev = np.zeros(self.n_slots, dtype=np.float64)
        self.eg_ep = np.full(self.n_slots, -1, dtype=np.int64)  # 引擎联动（engine_bonus）：每个座位上次的 Φ
        self.eg_prev = np.zeros(self.n_slots, dtype=np.float64)
        self.fc_ep = np.full(self.n_slots, -1, dtype=np.int64)  # 体系感知辅助头的标签：每个座位最近一次结束回合时成型的阵容
        self.fc_idx = np.full(self.n_slots, -1, dtype=np.int64)  # （rl/target_comps.json 里的序号；-1 = 没成型）
        if cfg.combat_bonus > 0:
            self.judge = self._make_judge()

    def _make_judge(self):
        from .boardsim import BoardJudge
        lib = getattr(self, "_pro_lib", None)
        if not lib:
            from paths import RUNS
            from .calibrate import merge_pro_boards
            paths = [Path(p) for p in str(self.cfg.pro_boards).split(";") if p.strip()] or [RUNS / "pro_boards.json"]
            lib = merge_pro_boards(p for p in paths if p.exists())
        if not lib:
            print("没有高分阵容库（runs/pro_boards.json、runs/h13_boards.json），阵容强度奖励这次不用", flush=True)
            return None
        cfg = self.cfg
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
        for j in np.flatnonzero(ak == A_END):              # 结束回合：落后于目标曲线就扣分
            s = int(slots[j])
            ep = int(self.rollout.ep[s])
            tier = 1 + (int(self.upg_n[s]) if self.upg_ep[s] == ep else 0)
            gap = max(0.0, self._tier_target(int(turns[j])) - 0.5 - tier)
            self.mix["gap_sum"] += gap
            self.mix["gap_n"] += 1
            if dc > 0 and gap > 0:
                r0[j] -= dc * gap
        for j in np.flatnonzero(ak == A_UPGRADE):
            s = int(slots[j])
            ep = int(self.rollout.ep[s])
            if self.upg_ep[s] != ep:
                self.upg_ep[s], self.upg_n[s] = ep, 0
            if tb > 0 and 1 + self.upg_n[s] < self._tier_target(int(turns[j])) + 0.5:
                r0[j] += tb
                self.mix["tempo_paid"] += 1
            self.upg_n[s] += 1
        if f.shape[1] > F_UPCOST:
            self._gold_shape(slots, f, ak, r0)
        if c is not None and self.cfg.comp_bonus > 0:
            self._comp_shape(f, c, ak, r0)
        if getattr(self.cfg, "target_bonus", 0) > 0 and c is not None:
            ShapingMixin._target_shape(self, slots, f, c, ak, r0)
        if getattr(self.cfg, "econ_bonus", 0) > 0 and f.shape[1] > 9:
            self._econ_shape(slots, f, ak, r0, c)
        if getattr(self.cfg, "keep_penalty", 0) > 0 and c is not None:
            ShapingMixin._keep_shape(self, f, c, ak, r0)
        if getattr(self.cfg, "engine_bonus", 0) > 0 and c is not None:
            ShapingMixin._engine_shape(self, slots, f, c, ak, r0)
        return r0

    def _engine_shape(self, slots, f, c, ak, r0):
        """引擎联动奖励（engine_bonus，rl/comp.ENGINE_RULES）：第 engine_from_turn 回合起结束回合时，
        Φ = 场上每个引擎卡（布莱恩·铜须、提图斯、巴琳达、热气球测绘员、达卡莱附魔师、小瞎眼、逐亡陆行鸟、邪能野猪人、
        活化艾泽里特、火花破坏机）配上的燃料（场上 + 手里被它放大的随从：战吼 / 亡语 / 鲜血宝石 / 回合结束 / 进击 / 法术 / 机械）
        min(1, 张数 / 3) 加起来（最多 2），奖励 = engine_bonus × (Φ - 这局上次的 Φ)。有引擎没燃料、有燃料没引擎都不算。
        第 16420 轮阵容测试：布莱恩·铜须 / 提图斯上场 0% / 0%（万三 17% / 16%），被引擎放大的买卡 0%（高手 8%）。"""
        from .comp import engine_score
        ends = np.flatnonzero(ak == A_END)
        if not len(ends):
            return r0
        tab = ShapingMixin._engine_tab(self)
        turns = np.rint(f[ends, 0] * 20).astype(int)
        for j, t in zip(ends, turns):
            if t < getattr(self.cfg, "engine_from_turn", 5):
                continue
            row = np.asarray(c[j])
            phi, _ = engine_score(row[C_BOARD:C_BOARD + 7], np.concatenate([row[C_BOARD:C_BOARD + 7], row[C_HAND:C_HAND + 10]]), tab)
            s = int(slots[j])
            ep = int(self.rollout.ep[s])
            prev = self.eg_prev[s] if self.eg_ep[s] == ep else 0.0
            rb = getattr(self, "eg_rebase", None)
            if rb and s in rb:                         # 成型阵容课程刚换过场面：从新场面起算
                rb.discard(s)
                prev = phi
            r0[j] += self.cfg.engine_bonus * (phi - prev)
            self.eg_ep[s], self.eg_prev[s] = ep, phi
            self.mix["eng_sum"] += phi
            self.mix["eng_n"] += 1
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

    def _target_tables(self):
        if getattr(self, "_tgt", None) is None:
            from tg.catalog import base_of_index, n_cards
            from .comp import load_target_comps
            self._tgt = ([base_of_index(i) if i else "" for i in range(n_cards())],
                         load_target_comps(weights=getattr(self.cfg, "target_weights", "")))
        return self._tgt

    def _target_shape(self, slots, f, c, ak, r0):
        """目标阵容奖励（target_bonus）：第 target_from_turn 回合起结束回合时，Φ = max（等级权重 × 场上核心卡占比），
        奖励 = target_bonus × (Φ - 这局上次的 Φ)。例：场上放下布莱恩·铜须 → 喝茶流 1/3、野猪人·邪能野猪人 1/3 …… Φ = 0.33；
        再凑齐巴琳达 → 0.67。整局加起来 = target_bonus × 最后一次的 Φ：最后成了 S 级阵容拿得最多。
        只算本局凑得成的那几套（核心卡的种族本局开了；tg/obs.lobby）：没开野猪人的局里不追邪能野猪人那套。"""
        from tg.obs import lobby
        from .comp import nearest_comp, target_score
        ends = np.flatnonzero(ak == A_END)
        if not len(ends):
            return r0
        base, comps = ShapingMixin._target_tables(self)
        if not comps:
            return r0
        turns = np.rint(f[ends, 0] * 20).astype(int)
        for j, t in zip(ends, turns):
            if t < getattr(self.cfg, "target_from_turn", 5):
                continue
            ids = [base[int(x)] for x in np.asarray(c[j])[C_BOARD:C_BOARD + 7] if 0 < int(x) < len(base)]
            lob = lobby(f[j])
            phi = target_score(ids, comps, lob)[0]
            s = int(slots[j])
            ep = int(self.rollout.ep[s])
            if hasattr(self, "fc_ep"):                 # 局末成了哪一套（体系感知辅助头的标签，ppo.collect 在对局结束时取）
                fc, _ = nearest_comp(ids, comps, lob=lob, strict=True)
                self.fc_ep[s], self.fc_idx[s] = ep, (comps.index(fc) if fc is not None else -1)
            prev = self.tg_prev[s] if self.tg_ep[s] == ep else 0.0
            rb = getattr(self, "tg_rebase", None)
            if rb and s in rb:                         # 成型阵容课程刚换过场面：从新场面起算，不算学习者的功劳
                rb.discard(s)
                prev = phi
            r0[j] += self.cfg.target_bonus * (phi - prev)
            self.tg_ep[s], self.tg_prev[s] = ep, phi
            self.mix["tgt_sum"] += phi
            self.mix["tgt_n"] += 1
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
        """座位 s 这一局结束：把 shape_repay_keys 里那几项（目标阵容 target / 引擎联动 engine / 阵容强度 combat）
        这局最后的 Φ 扣回 shape_repay 那么多（见 Config.shape_repay）。扣完以后那几项整局加起来 ≈ -第一次的 Φ，和动作无关：
        做阵容那一步还是当场拿到功劳，最后由名次决定值不值。不扣的项局末的 Φ 照常算进回报。"""
        k = ShapingMixin._repay_now(self)
        ep = int(self.rollout.ep[s])
        cfg = self.cfg
        keys = {x.strip() for x in str(getattr(cfg, "shape_repay_keys", "target,engine,combat")).split(",")}
        v = rep = 0.0
        for name, bonus, e_, prev in (("target", "target_bonus", self.tg_ep, self.tg_prev),
                                      ("engine", "engine_bonus", self.eg_ep, self.eg_prev),
                                      ("combat", "combat_bonus", self.cs_ep, self.cs_prev)):
            b = getattr(cfg, bonus, 0)
            if b > 0 and e_[s] == ep:
                x = b * float(prev[s])
                v += x
                if name in keys:
                    rep += x
        self.mix["fin_sum"] += v                        # 这局三项局末一共还挂着多少（不扣时就是白拿的）
        self.mix["fin_n"] += 1
        return -k * rep

    def _comp_tables(self):
        """阵容打分用的查表（按卡牌编号）：种族位掩码（全种族 = 所有位）、是不是随从、关键卡、高手常见搭配矩阵。"""
        if getattr(self, "_ct", None) is not None:
            return self._ct
        from tg.catalog import base_of_index, catalog, n_cards
        pairs, key = load_comp_refs()
        det = catalog()
        n = n_cards()
        tribes = sorted({r for x in det.values() for r in x.get("races", []) if r != "ALL"})
        mask = np.zeros(n, dtype=np.int64)
        minion = np.zeros(n, dtype=bool)
        keyv = np.zeros(n, dtype=bool)
        base = [""] * n
        for i in range(1, n):
            b = base_of_index(i)
            base[i] = b
            x = det.get(b) or {}
            minion[i] = x.get("kind") == "minion"
            rs = x.get("races", [])
            mask[i] = (1 << len(tribes)) - 1 if "ALL" in rs else sum(1 << tribes.index(r) for r in rs if r in tribes)
            keyv[i] = b in key
        idx: dict[str, list[int]] = {}
        for i in range(1, n):
            idx.setdefault(base[i], []).append(i)
        pair = np.zeros((n, n), dtype=bool)
        for x, y in pairs:
            for i in idx.get(x, []):
                for j in idx.get(y, []):
                    pair[i, j] = pair[j, i] = True
        self._ct = {"mask": mask, "minion": minion, "key": keyv, "pair": pair, "ntribe": len(tribes)}
        return self._ct

    def _comp_shape(self, f, c, ak, r0):
        """阵容成型奖励（comp_bonus）：第 comp_from_turn 回合起，每次结束回合按当前场面的配合程度加分：
          q = 0.6 × 纯度（主种族占比从 0.3 到 0.9 线性给 0→1）+ 0.2 × 高手常见搭配对数（最多 3 对）/ 3
              + 0.2 × 关键卡张数（最多 5 张）/ 5，奖励 = comp_bonus × q（每回合最多 comp_bonus）。
        第 8661–9279 轮（旧算法：纯度从 0.5 起算、搭配占 0.3，bonus 0.03）：q 0.115 → 0.124、纯度 0.51 → 0.52，几乎没动——
        模型的纯度在 0.5 附近，旧算法在那里几乎不给分，推不动；它玩的海盗 / 龙 / 机械没有万三常见搭配（那 40 对是野猪人、
        亡灵、野兽……），搭配那 0.3 拿不到。阵容测试里它自己前四名的局纯度 0.55、后四名 0.47，纯度确实和名次有关。
        第 9280–10064 轮（新算法，bonus 0.05）：纯度 0.52 → 0.56–0.58 后停住，加强考试 4.2 → 4.07，阵容测试 3.80
        （前四名纯度 0.57、后四名 0.45）→ train.bat 提到 0.08。
        第 7950–8664 轮只靠模仿加权（tribe:2、syn:3、key:2）：纯度 0.50、搭配 0.01、关键卡 0.53，一点没动——
        模拟器里堆增益法术的混合阵容照样能赢，模仿的力度（0.25 × 0.1）推不动。这一项直接要求“成型”。"""
        ends = np.flatnonzero(ak == A_END)
        if not len(ends):
            return r0
        turns = np.rint(f[ends, 0] * 20).astype(int)
        ends = ends[turns >= self.cfg.comp_from_turn]
        if not len(ends):
            return r0
        t = self._comp_tables()
        b = np.asarray(c[ends][:, C_BOARD:C_BOARD + 7], dtype=np.int64)
        valid = t["minion"][b]
        nmin = valid.sum(1)
        masks = np.where(valid, t["mask"][b], 0)
        counts = np.stack([((masks >> k) & 1).sum(1) for k in range(t["ntribe"])], 1)
        coh = counts.max(1) / np.maximum(1, nmin)
        syn = np.zeros(len(ends))
        for i in range(7):
            for j in range(i + 1, 7):
                syn += t["pair"][b[:, i], b[:, j]] & valid[:, i] & valid[:, j]
        keyn = (t["key"][b] & valid).sum(1)
        wc, ws, wk = (float(x) for x in str(getattr(self.cfg, "comp_weights", "0.6,0.2,0.2")).split(","))
        q = wc * np.clip((coh - 0.3) / 0.6, 0, 1) + ws * np.minimum(syn, 3) / 3 + wk * np.minimum(keyn, 5) / 5
        q = np.where(nmin >= 4, q, 0.0)
        r0[ends] += self.cfg.comp_bonus * q
        self.mix["comp_q_sum"] += float(q.sum())
        self.mix["comp_q_n"] += len(q)
        return r0

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
                self.ref_idx[s] = self.rollout.last[s] if a == A_REFRESH else -1
            elif A_BUY <= a < A_SELL:
                self.ref_idx[s] = -1
            kind = ("end" if a == A_END else "refresh" if a == A_REFRESH else "upgrade" if a == A_UPGRADE else
                    "buy" if A_BUY <= a < A_SELL else "sell" if A_SELL <= a < A_SELL + 7 else None)
            if kind:
                self.mix[kind] += 1
