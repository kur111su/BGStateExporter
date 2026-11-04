"""最终阵容的统计和打分：主种族、常见搭配、关键卡。

  comp_stats      一批最终阵容的汇总（主种族分布、常用卡），写进 rl/h13_profile.json 的 final
  synergy         经常一起上场的两张卡（提升度）
  board_scores    一个阵容的配合程度：主种族纯度 coh、常见搭配对数 syn、关键卡张数 key
  load_comp_refs  训练和统计用的参照：rl/h13_profile.json 的搭配 + rl/card_value.json 的关键卡

训练里的阵容成型奖励（rl/shaping.py _comp_shape）、goldflow / comptest 的阵容对比、
native_demos 生成万三画像都用这一份。卡牌 ID 都是去掉金色 _G 之后的。
"""
from __future__ import annotations

import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

from tg.catalog import catalog

REFS = Path(__file__).resolve().parent          # rl/：h13_profile.json、card_value.json 在这里
ZH_RACE = {"QUILBOAR": "野猪人", "UNDEAD": "亡灵", "BEAST": "野兽", "MURLOC": "鱼人", "DEMON": "恶魔", "MECHANICAL": "机械",
           "ELEMENTAL": "元素", "PIRATE": "海盗", "DRAGON": "龙", "NAGA": "纳迦", "ABERRATION": "畸变", "混合": "混合"}


def comp_stats(boards: list[list[str]], goldens: list[int], tiers: list[int], det: dict | None = None) -> dict:
    """最终阵容汇总：boards = 每局最后的场面（卡牌 ID，金色已去掉 _G）。主种族：同一种族（含全种族）≥ 4 个。"""
    det = det or catalog()
    main, cards = Counter(), Counter()
    for ids in boards:
        for x in set(ids):
            cards[x] += 1
        rc = Counter(r for x in ids for r in (det.get(x) or {}).get("races", []) if r != "ALL")
        allr = sum(1 for x in ids if "ALL" in (det.get(x) or {}).get("races", []))
        top = rc.most_common(1)
        main[top[0][0] if top and top[0][1] + allr >= 4 else "混合"] += 1
    n = max(1, len(boards))
    return {"n": len(boards), "tier": round(float(np.mean(tiers)), 2) if tiers else None,
            "golden": round(float(np.mean(goldens)), 2) if goldens else None,
            "main": [[k, round(v / n, 3)] for k, v in main.most_common()],
            "cards": [[x, (det.get(x) or {}).get("name_zh", x), round(v / n, 3)] for x, v in cards.most_common(40)]}


def synergy(boards: list[list[str]], det: dict, min_games: int = 12, top: int = 40) -> list:
    """经常一起上场的两张卡（提升度 = 一起出现的局数 / 各自独立时的期望）。"""
    one, two = Counter(), Counter()
    for ids in boards:
        u = sorted(set(ids))
        one.update(u)
        two.update(combinations(u, 2))
    n = len(boards)
    out = [[round(k * n / (one[x] * one[y]), 2), k, (det.get(x) or {}).get("name_zh", x), (det.get(y) or {}).get("name_zh", y), x, y]
           for (x, y), k in two.items() if k >= min_games]
    return sorted(out, reverse=True)[:top]


def board_scores(ids: list[str], det: dict, pairs: set, key: set) -> dict:
    """一个最终阵容的配合程度：
      coh   主种族纯度 = 最多的那个种族（全种族随从都算）的随从数 / 场上随从数
      syn   场上两两组合里，属于高手常见搭配（synergy 里提升度 ≥ 3）的有几对
      key   场上关键卡（rl/card_value.json 的 key）有几张"""
    races = Counter()
    for x in ids:
        rs = (det.get(x) or {}).get("races", [])
        for r in ({r for r in rs if r != "ALL"} if "ALL" not in rs else set(ZH_RACE) - {"混合"}):
            races[r] += 1
    u = sorted(set(ids))
    return {"coh": round(max(races.values()) / len(ids), 3) if races and ids else 0.0,
            "syn": sum(1 for x, y in combinations(u, 2) if (x, y) in pairs),
            "key": sum(1 for x in ids if x in key)}


def load_sim_refs(root: Path = REFS) -> tuple[set, set]:
    """(引擎实测的好搭配 {(id, id)}，引擎实测的关键卡 {id})：rl/sim_value.json（bg.py simcards 生成）；没有时都是空的。
    好搭配 = 两张一起在场比各自单独多赢 ≥ 2 个百分点。"""
    p = Path(root) / "sim_value.json"
    if not p.exists():
        return set(), set()
    v = json.loads(p.read_text(encoding="utf-8"))
    return {tuple(sorted((r[4], r[5]))) for r in v.get("pairs", []) if r[0] >= 0.02}, set(v.get("key", []))


def load_comp_refs(root: Path = REFS) -> tuple[set, set]:
    """(高手常见搭配 {(id, id)}, 关键卡 {id})：rl/h13_profile.json 和 rl/card_value.json。"""
    pairs, key = set(), set()
    hp, cv = Path(root) / "h13_profile.json", Path(root) / "card_value.json"
    if hp.exists():
        for row in json.loads(hp.read_text(encoding="utf-8")).get("synergy", []):
            if len(row) >= 6 and row[0] >= 3:
                pairs.add(tuple(sorted((row[4], row[5]))))
    if cv.exists():
        key = set(json.loads(cv.read_text(encoding="utf-8")).get("key", []))
    return pairs, key


# ---------------- 目标阵容（rl/target_comps.json：bgbuddy.gg 的 S / A / B 级终局阵容和核心卡） ----------------

def parse_tier_weights(spec) -> dict:
    """"S:1,A:0.85,B:0.4" → {"S": 1.0, …}；空 = 用 target_comps.json 里的。"""
    out = {}
    for part in str(spec or "").replace(" ", "").split(","):
        if ":" in part:
            k, v = part.split(":", 1)
            out[k] = float(v)
    return out


def load_target_comps(root: Path = REFS, weights=None) -> list[dict]:
    """[{name, tier, w（默认 S 1.0 / A 0.8 / B 0.6，weights 可以改）, core: {卡牌 ID}, need: [本局要开的种族]}]；没有文件时空列表。
    need：每张核心卡一项，这张卡的种族（中立 / 全种族的不算）——这些种族本局都没开，这张卡就不会出现，这套凑不成。"""
    p = Path(root) / "target_comps.json"
    if not p.exists():
        return []
    v = json.loads(p.read_text(encoding="utf-8"))
    tw = {**(v.get("tiers") or {"S": 1.0, "A": 0.8, "B": 0.6}), **parse_tier_weights(weights)}
    det = catalog()
    keep = set((v.get("playbook") or {}).get("keep") or [])
    out = []
    for c in v.get("comps", []):
        need = []
        for cid in c["core"]:
            rs = set((det.get(cid) or {}).get("races") or [])
            if rs and "ALL" not in rs:
                need.append(frozenset(rs))
        tribes = frozenset().union(*need) if need else frozenset()
        out.append({"name": c["name"], "tier": c["tier"], "w": float(tw.get(c["tier"], 0.5)), "core": set(c["core"]),
                    "avg_place": c.get("avg_place"), "need": need, "tribes": tribes, "support": set(c.get("support") or []),
                    "keep": (set(c["core"]) | set(c.get("support") or [])) & keep,
                    "keep_support": set(c.get("support") or []) & keep})
    return out


TRIBE_FULL = 4       # 种族阵容：场上这么多个本族（或全种族）随从才算撑起来了


def _tribe_count(ids, tribes) -> int:
    det = catalog()
    n = 0
    for cid in ids:
        rs = set((det.get(cid) or {}).get("races") or [])
        n += bool(rs & tribes or "ALL" in rs)
    return n


def comp_progress(ids, comp: dict) -> tuple[float, int, float]:
    """(核心卡占比, 核心卡张数, 本族随从撑起来的程度 0–1)。中立阵容（喝茶流这种，核心卡没有种族）撑起来的程度算 1。"""
    have = set(ids)
    k = len(comp["core"] & have)
    tribes = comp.get("tribes") or frozenset()
    sup = min(1.0, _tribe_count(ids, tribes) / TRIBE_FULL) if tribes else 1.0
    return k / len(comp["core"]), k, sup


SUPPORT_FULL = 2     # 场上有这么多张这套“留着用”的配套卡（bg.py playbook 的 keep ∩ support）算配齐


def support_factor(ids, comp: dict) -> float:
    """配套卡配齐的程度 → 0.7–1：成型以后场上还有高手留着用的配套卡（热气球测绘员配宝石、断手被遗忘者配宰割……）才是满分。
    这套没有这种配套卡时算 1。"""
    ks = comp.get("keep_support") or set()
    if not ks:
        return 1.0
    return 0.7 + 0.3 * min(1.0, len(ks & set(ids)) / min(SUPPORT_FULL, len(ks)))


def formed(ids, comp: dict) -> bool:
    """算成型：核心卡至少 2 张、至少 2/3；种族阵容场上还要有 3 个以上本族随从。
    第 14240 轮阵容测试按“核心卡一半”算，S 级 43%——其实是场上放一张提图斯（畸变亡语 2 张核心卡里的 1 张）、
    一张守门融合怪就算数，这些局平均 4.57 名，没成的 4.65 名，几乎没区别。"""
    fr, k, sup = comp_progress(ids, comp)
    return k >= 2 and fr >= 2 / 3 - 1e-9 and sup >= 0.75 - 1e-9


def reachable(comp: dict, lob) -> bool:
    """本局开放的种族 lob（集合；None = 不知道，算能凑成）下，这套阵容的核心卡都可能出现。"""
    return lob is None or all(n & lob for n in comp.get("need") or [])


def target_score(ids, comps: list[dict], lob=None) -> tuple[float, dict | None, float]:
    """场面（卡牌 ID，金色已去掉 _G）离目标阵容多近：(最高的 等级权重 × 核心卡占比² × 本族随从撑起来的程度 × 配套卡配齐的程度,
    那一套, 核心卡占比)。
    lob：本局开放的种族——只算本局凑得成的那几套。
    核心卡这一项 = (占比 + 占比²) / 2：开头就有分（4 张核心卡：1 张 0.16、2 张 0.38、3 张 0.66、4 张 1），凑齐最后几张值得更多；
    第 15300–16420 轮只用占比²（1 张才 0.06），第 8 回合有主选的局只有 32%，开头太不值钱。本族程度：场上本族随从 / 4。
    第 14140–14240 轮用的是“等级权重 × 占比”：放一张提图斯就是畸变亡语的 0.5，模型学会了只放一张核心卡。"""
    best, comp, frac = 0.0, None, 0.0
    for c in comps:
        if not reachable(c, lob):
            continue
        fr, _, sup = comp_progress(ids, c)
        q = c["w"] * (0.5 * fr + 0.5 * fr * fr) * sup * support_factor(ids, c)
        if q > best:
            best, comp, frac = q, c, fr
    return best, comp, frac


def nearest_comp(ids, comps: list[dict], min_frac: float = 0.5, lob=None, strict: bool = False) -> tuple[dict | None, float]:
    """最终阵容算哪一套：核心卡占比最高的那套（一样高时等级高的优先），占比不到 min_frac 算“不像任何一套”。
    strict：按 formed() 算（核心卡至少 2 张、2/3，种族阵容还要 3 个本族随从）。"""
    have = set(ids)
    order = {"S": 0, "A": 1, "B": 2}
    best = None
    for c in comps:
        if not reachable(c, lob):
            continue
        fr = len(c["core"] & have) / len(c["core"])
        key = (fr, -order.get(c["tier"], 3), -(c.get("avg_place") or 9))
        ok = formed(ids, c) if strict else fr >= min_frac
        if ok and (best is None or key > best[0]):
            best = (key, c, fr)
    return (best[1], best[2]) if best else (None, 0.0)


class CompTables:
    """目标阵容的整批查表（训练每一步、模仿示范挑选、阵容测试都用）：卡牌编号 × 阵容 的核心卡 / 配套卡 / 本族表。
    配套卡（support）：高手已经在做这套（场上 + 手里核心卡过半、本族随从 ≥ 2）时，比平时明显更爱买的卡
    （bg.py playbook 从高分示范里算，存在 rl/target_comps.json）——成型以后拿来提升实力、倒转经济、卡牌联动的那些。"""

    def __init__(self, comps: list[dict], ids: list[str], tiers: str = "SAB"):
        self.comps = [c for c in comps if c["tier"] in tiers]
        det = catalog()
        n, k = len(ids), len(self.comps)
        self.core = np.zeros((n, k), dtype=bool)
        self.supp = np.zeros((n, k), dtype=bool)
        self.trib = np.zeros((n, k), dtype=bool)
        self.keep = np.zeros((n, k), dtype=bool)          # 高手做这套时留着不卖的核心 / 配套卡
        self.ksup = np.zeros((n, k), dtype=bool)          # 其中的配套卡（目标阵容得分的“配套卡配齐的程度”）
        for i, cid in enumerate(ids):
            if not cid:
                continue
            rs = set((det.get(cid) or {}).get("races") or [])
            for j, c in enumerate(self.comps):
                self.core[i, j] = cid in c["core"]
                self.supp[i, j] = cid in c.get("support", ())
                self.keep[i, j] = cid in c.get("keep", ())
                self.ksup[i, j] = cid in c.get("keep_support", ())
                self.trib[i, j] = bool(c.get("tribes")) and bool(rs & c["tribes"] or "ALL" in rs)
        self.ncore = np.array([len(c["core"]) for c in self.comps], dtype=np.float32)
        self.w = np.array([c["w"] for c in self.comps], dtype=np.float32)
        self.nks = np.array([len(c.get("keep_support") or ()) for c in self.comps], dtype=np.float32)
        keys = {b: k for k, b in enumerate(sorted(set(ids)))}
        self.base_key = np.array([keys[b] for b in ids], dtype=np.int64)     # 卡牌编号 → 同名（金色并到普通）的编号
        self.tribal = np.array([bool(c.get("tribes")) for c in self.comps])
        self._reach: dict[int, np.ndarray] = {}

    def reach(self, bits) -> np.ndarray:
        """本局种族位图（tg/obs.lobby_bits）→ [行, 阵容] 凑得成。"""
        from tg.obs import F_LOBBY
        bits = np.asarray(bits)
        out = np.zeros((len(bits), len(self.comps)), dtype=bool)
        names = list(F_LOBBY)
        for b in np.unique(bits):
            r = self._reach.get(int(b))
            if r is None:
                lob = frozenset(x for k, x in enumerate(names) if int(b) >> k & 1)
                r = self._reach[int(b)] = np.array([reachable(c, lob) for c in self.comps], dtype=bool)
            out[bits == b] = r
        return out

    def progress(self, c) -> tuple[np.ndarray, np.ndarray]:
        """([行, 阵容] 场上 + 手里核心卡占比（同名只算一张）, [行, 阵容] 场上本族随从个数)。"""
        from tg.obs import C_BOARD, C_HAND, N_BOARD, N_HAND
        c = np.asarray(c)
        own = np.concatenate([c[:, C_BOARD:C_BOARD + N_BOARD], c[:, C_HAND:C_HAND + N_HAND]], 1)
        own = np.clip(own, 0, len(self.core) - 1)
        pres = np.zeros((len(c), len(self.core)), dtype=bool)
        pres[np.arange(len(c))[:, None], own] = True
        pres[:, 0] = False
        frac = (pres.astype(np.float32) @ self.core.astype(np.float32)) / self.ncore
        board = np.clip(c[:, C_BOARD:C_BOARD + N_BOARD], 0, len(self.core) - 1)
        return frac, self.trib[board].sum(1)

    def building(self, c, bits=None, min_frac: float = 0.5, best: bool = False) -> np.ndarray:
        """[行, 阵容]：正在做这套——场上 + 手里的核心卡 ≥ min_frac，种族阵容场上本族随从 ≥ 2，本局凑得成。
        best：每行只留最像的那一套（等级权重 × 占比² × 本族程度最高；喝茶流和元素都有布莱恩、巴琳达时分得开）。"""
        frac, tribe = self.progress(c)
        ok = (frac >= min_frac - 1e-6) & (~self.tribal | (tribe >= 2))
        if bits is not None:
            ok &= self.reach(bits)
        if best:
            w = np.array([c_["w"] for c_ in self.comps], dtype=np.float32)
            sup = np.where(self.tribal, np.minimum(1.0, tribe / TRIBE_FULL), 1.0)
            sc = np.where(ok, w * frac * frac * sup + 1e-3 * w, -1.0)
            top = sc.argmax(1)
            ok = ok & (np.arange(len(self.comps))[None, :] == top[:, None])
        return ok

    def scores(self, c, bits=None) -> np.ndarray:
        """[行, 阵容] 目标阵容得分（和 target_score 同一个算法：等级权重 × 占比² × 本族程度 × 配套卡程度；本局凑不成的 0）。"""
        from tg.obs import C_BOARD, N_BOARD
        frac, tribe = self.progress(c)
        sup = np.where(self.tribal, np.minimum(1.0, tribe / TRIBE_FULL), 1.0)
        board = np.clip(np.asarray(c)[:, C_BOARD:C_BOARD + N_BOARD], 0, len(self.core) - 1)
        ks = np.zeros((len(board), len(self.comps)), dtype=np.float32)
        for r in range(len(board)):                     # 场上“留着用”的配套卡（同名算一张）
            seen = {}
            for x in board[r]:
                if x > 0:
                    seen[int(self.base_key[x])] = int(x)
            if seen:
                ks[r] = self.ksup[list(seen.values())].sum(0)
        supf = np.where(self.nks > 0, np.minimum(1.0, ks / np.minimum(np.maximum(self.nks, 1), SUPPORT_FULL)), 1.0)
        sc = self.w * frac * frac * sup * (0.7 + 0.3 * supf)
        if bits is not None:
            sc = sc * self.reach(bits)
        return sc

    def main_backup(self, c, bits=None, n_backup: int = 2) -> tuple[np.ndarray, np.ndarray]:
        """([行, 阵容] 主选, [行, 阵容] 备选)：得分最高的那套是主选，接下来 n_backup 套（得分 > 0）是备选。"""
        sc = self.scores(c, bits)
        order = np.argsort(-sc, 1)
        rank = np.empty_like(order)
        rank[np.arange(len(sc))[:, None], order] = np.arange(sc.shape[1])[None, :]
        pos = sc > 0
        return pos & (rank == 0), pos & (rank >= 1) & (rank <= n_backup)

    def shop_hit(self, c, comps_mask, table) -> np.ndarray:
        """[行]：酒馆里有 table（core / supp）里属于 comps_mask 那几套的卡。"""
        from tg.obs import C_SHOP, N_SHOP
        shop = np.clip(np.asarray(c)[:, C_SHOP:C_SHOP + N_SHOP], 0, len(table) - 1)
        return (table[shop] & np.asarray(comps_mask)[:, None, :]).any((1, 2))

    def keep_sold(self, c, f, actions) -> np.ndarray:
        """[行]：这一步卖掉的是正在做的 S / A 阵容里高手留着不卖的牌（这套开了头时的核心卡 / 在做这套时的配套卡），
        而且手里 / 场上没有第二张同名的。"""
        from tg.actions import A_SELL
        from tg.obs import C_BOARD, C_HAND, N_BOARD, N_HAND, lobby_bits
        c = np.asarray(c)
        a = np.asarray(actions).astype(np.int64) - A_SELL
        out = np.zeros(len(c), dtype=bool)
        rows = np.flatnonzero((a >= 0) & (a < N_BOARD))
        if not len(rows):
            return out
        cr = c[rows]
        card = np.clip(cr[np.arange(len(rows)), C_BOARD + a[rows]], 0, len(self.core) - 1)
        bits = lobby_bits(np.asarray(f)[rows])
        frac, tribe = self.progress(cr)
        started = ((frac > 0) | (self.tribal & (tribe >= 2))) & self.reach(bits)
        bm = self.building(cr, bits)
        rel = (self.keep[card] & ((self.core[card] & started) | (self.supp[card] & bm))).any(1)
        own = np.concatenate([cr[:, C_BOARD:C_BOARD + N_BOARD], cr[:, C_HAND:C_HAND + N_HAND]], 1)
        same = (np.clip(own, 0, len(self.base_key) - 1) > 0) & (self.base_key[np.clip(own, 0, len(self.base_key) - 1)]
                                                               == self.base_key[card][:, None])
        out[rows] = rel & (same.sum(1) < 2)
        return out


K_MAX = 32          # 模型里阵容进度输入的格数（target_comps.json 现在 23 套；多出来的格子全是 0，换阵容表不用改模型形状）
COMP_F = 10         # 每套阵容的输入：凑得成、核心卡占比、占比²、本族程度、配套卡程度、得分、主选、备选、酒馆有核心卡、酒馆有配套卡


def model_tables(comps: list[dict] | None = None, ids: list[str] | None = None) -> dict:
    """给模型（rl/model.CompSense）用的查表，全是 numpy 数组：模型自己从卡牌编号和观察里算每套阵容做到几成，
    训练、影子模式、竞技场都一样（竞技场的打包里存一份 comp.npz，不用装 TavernGym）。"""
    from tg.catalog import base_of_index, n_cards
    from tg.obs import F_LOBBY, LOBBY_IDX
    comps = load_target_comps() if comps is None else comps
    ids = [base_of_index(i) if i else "" for i in range(n_cards())] if ids is None else ids
    comps = comps[:K_MAX]
    det = catalog()
    bases = sorted(set(ids))                                   # "" 排第一 → 编号 0 = 空
    bk = {b: k for k, b in enumerate(bases)}
    base_key = np.array([bk[b] for b in ids], dtype=np.int64)
    nb, n = len(bases), len(ids)
    core_b = np.zeros((nb, K_MAX), np.float32)
    ks_b = np.zeros((nb, K_MAX), np.float32)
    core_c = np.zeros((n, K_MAX), bool)
    supp_c = np.zeros((n, K_MAX), bool)
    trib_c = np.zeros((n, K_MAX), bool)
    races = list(F_LOBBY)
    nmax = max([len(c.get("need") or []) for c in comps] + [1])
    need = np.zeros((K_MAX, nmax, len(races)), bool)
    need_valid = np.zeros((K_MAX, nmax), bool)
    w = np.zeros(K_MAX, np.float32)
    ncore = np.ones(K_MAX, np.float32)
    nks = np.zeros(K_MAX, np.float32)
    tribal = np.zeros(K_MAX, bool)
    valid = np.zeros(K_MAX, bool)
    for j, c in enumerate(comps):
        valid[j], w[j], ncore[j] = True, c["w"], len(c["core"])
        tribal[j] = bool(c.get("tribes"))
        ks = c.get("keep_support") or set()
        nks[j] = len(ks)
        for b in c["core"]:
            if b in bk:
                core_b[bk[b], j] = 1
        for b in ks:
            if b in bk:
                ks_b[bk[b], j] = 1
        for q, ns in enumerate(c.get("need") or []):
            need_valid[j, q] = True
            for r in ns:
                if r in races:
                    need[j, q, races.index(r)] = True
    for i, cid in enumerate(ids):
        if not cid:
            continue
        rs = set((det.get(cid) or {}).get("races") or [])
        for j, c in enumerate(comps):
            core_c[i, j] = cid in c["core"]
            supp_c[i, j] = cid in c.get("support", ())
            trib_c[i, j] = bool(c.get("tribes")) and bool(rs & c["tribes"] or "ALL" in rs)
    return {"base_key": base_key, "core_b": core_b, "ks_b": ks_b, "core_c": core_c, "supp_c": supp_c, "trib_c": trib_c,
            "need": need, "need_valid": need_valid, "w": w, "ncore": ncore, "nks": nks, "tribal": tribal, "valid": valid,
            "lobby_idx": np.asarray(LOBBY_IDX, np.int64), "names": np.array([c["name"] for c in comps])}
