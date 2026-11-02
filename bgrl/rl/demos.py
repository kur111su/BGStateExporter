"""示范数据（模仿学习样本）的格式：读、写、动作类别。不需要 torch。

三种来源写出来的格式一样，训练（rl/ppo.py）只认这里：
  runs/demos.npz          你自己的对局（tools/make_demos.py，单个文件，观察是原始浮点）
  runs/pro_demos/         Firestone 6500+ 回放（tools/make_pro_demos.py，分片，观察是 soft_log 后的 float16）
  runs/h13_demos/         别人万三的对局（tools/native_demos.py，分片，同上）

每条样本：f 浮点观察、c 卡牌编号、m 合法动作（存的时候按位打包）、cand 示范动作（最多 MAX_CANDS 个等价的，
不足补 -1）、kind 动作类别（KINDS 的下标）、turn 回合、split（1 = 验证，不参与训练）；分片的还有 mmr。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from tg.obs import C_BOARD, C_SHOP

MAX_CANDS = 8
KINDS = ("buy", "sell", "play", "cast", "magnetize", "upgrade", "refresh", "hero_power", "move", "end",
         "freeze", "dark_gift", "activate")          # 后三个只在高分回放里有（tools/make_pro_demos.py）；只能往后加
SHARD_KEYS = ("f", "c", "m", "cand", "kind", "turn", "split", "mmr")


def soft_log_np(x: np.ndarray) -> np.ndarray:
    """和 rl/model.py 的 soft_log_np 一样（这里不导入 rl.model：转换不需要 torch；tests/test_layout.py 核对两者一致）。"""
    a = np.abs(x)
    return np.where(a <= 1, x, np.sign(x) * (1 + np.log(np.maximum(a, 1)))).astype(np.float32)


# ---------------- 单文件（你自己的对局） ----------------

def save(rows, out: Path, meta: dict):
    """rows：[(f, c, m, 候选动作, 类别名, 回合, split)]，写成一个 npz（先写临时文件再换名）。"""
    n = len(rows)
    cand = np.full((n, MAX_CANDS), -1, dtype=np.int16)
    for i, r in enumerate(rows):
        cand[i, :len(r[3])] = r[3]
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, f=np.stack([r[0] for r in rows]).astype(np.float32),
                        c=np.stack([r[1] for r in rows]).astype(np.int32),
                        m=np.packbits(np.stack([r[2] for r in rows]), axis=1), cand=cand,
                        kind=np.array([KINDS.index(r[4]) for r in rows], dtype=np.int8),
                        turn=np.array([r[5] for r in rows], dtype=np.int16),
                        split=np.array([r[6] for r in rows], dtype=np.int8),
                        n_actions=np.int32(len(rows[0][2])), meta=np.array(json.dumps(meta, ensure_ascii=False)))
    os.replace(tmp, out)


def load(path) -> dict:
    """训练用：返回各数组（m 已解包成 bool）。"""
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["m"] = np.unpackbits(d["m"], axis=1)[:, :int(d["n_actions"])].astype(bool)
    d["meta"] = json.loads(str(d["meta"]))
    return d


# ---------------- 分片目录（高分回放 / 万三对局） ----------------

def reset_dir(out_dir: Path) -> Path:
    """建目录，删掉旧分片（重新生成时不留上次多出来的片）。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("part_*.npz"):
        old.unlink()
    return out_dir


def write_shard(out_dir: Path, k: int, arrays: dict, n_actions: int, meta: dict) -> int:
    """写 part_{k:03d}.npz；arrays 里 m 已经按位打包、f 已经是 soft_log 后的 float16。返回条数。"""
    np.savez_compressed(Path(out_dir) / f"part_{k:03d}.npz", **arrays, n_actions=np.int32(n_actions),
                        compressed=np.int8(1), meta=np.array(json.dumps(meta, ensure_ascii=False)))
    return len(arrays["kind"])


def load_dir(path) -> dict | None:
    """训练用：读目录里所有分片，拼成和 load 一样的字典（多一个 compressed=True）。"""
    files = sorted(Path(path).glob("part_*.npz"))
    if not files:
        return None
    parts = [np.load(f, allow_pickle=False) for f in files]
    d = {k: np.concatenate([p[k] for p in parts]) for k in SHARD_KEYS}
    n_act = int(parts[0]["n_actions"])
    d["m"] = np.unpackbits(d["m"], axis=1)[:, :n_act].astype(bool)
    d["compressed"] = True
    meta = Path(path) / "meta.json"
    d["meta"] = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}
    return d


# ---------------- 打乱酒馆顺序（模仿时的数据增强） ----------------

def permute_shop(f, c, m, cand, rng, shop_f: int, item_f: int = 40, n_shop: int = 8):
    """每条样本把酒馆里有东西的那几格随机换个顺序：观察里每格的浮点块（从 shop_f 开始，每格 item_f 个）、
    卡牌编号（C_SHOP 起）、合法动作里的“买第几格”、示范动作里的“买第几格”一起换。空格留在后面（和引擎一样）。
    返回新数组，不改原来的。tests/test_boardsim.py 核对：和在引擎里把酒馆列表换顺序后再 observe() 完全一样。"""
    from tg.actions import A_BUY
    B = len(c)
    occ = np.asarray(c[:, C_SHOP:C_SHOP + n_shop]) > 0
    key = np.where(occ, rng.random((B, n_shop)), 2.0 + np.arange(n_shop))
    perm = np.argsort(key, 1)                       # 新第 k 格 = 原来第 perm[b, k] 格
    inv = np.argsort(perm, 1)                       # 原来第 k 格 → 新的第 inv[b, k] 格
    f = np.array(f, copy=True)
    blk = f[:, shop_f:shop_f + n_shop * item_f].reshape(B, n_shop, item_f)
    f[:, shop_f:shop_f + n_shop * item_f] = np.take_along_axis(blk, perm[:, :, None], 1).reshape(B, -1)
    c = np.array(c, copy=True)
    c[:, C_SHOP:C_SHOP + n_shop] = np.take_along_axis(c[:, C_SHOP:C_SHOP + n_shop], perm, 1)
    m = np.array(m, copy=True)
    m[:, A_BUY:A_BUY + n_shop] = np.take_along_axis(m[:, A_BUY:A_BUY + n_shop], perm, 1)
    cand = np.array(cand, copy=True)
    rows, cols = np.nonzero((cand >= A_BUY) & (cand < A_BUY + n_shop))
    cand[rows, cols] = A_BUY + inv[rows, cand[rows, cols] - A_BUY]
    return f, c, m, cand


# ---------------- 引擎卡：放大某一类效果的随从 ----------------

ENGINES = {"battlecry": ("BG_LOE_077", "TB_BaconUps_045", "TB_BaconUps_045t"),      # 布莱恩·铜须：战吼触发两次 / 三次
           "deathrattle": ("BG25_354", "BG25_354_G")}                              # 提图斯·瑞文戴尔：亡语多触发


def _effects(card: dict) -> set:
    """一张卡有哪些能被引擎卡放大的效果（按英文说明里的关键词）。"""
    t = str(card.get("text") or "")
    if card.get("kind") != "minion":
        return set()
    return {k for k, w in (("battlecry", "Battlecry"), ("deathrattle", "Deathrattle")) if w in t}


def engine_table() -> dict:
    """卡牌 ID（去掉 _G）→ 它放大哪类效果。"""
    out = {}
    for kind, ids in ENGINES.items():
        for cid in ids:
            out.setdefault(cid[:-2] if cid.endswith("_G") else cid, set()).add(kind)
    return out


# ---------------- 按买的是什么卡挑示范（--bc-weights / --pro-weights 的 key、eco、tribe、syn） ----------------

def bought_in(d, which: str, root: Path | None = None):
    """哪些示范是“买了一张 …… 的卡”（买的那格的卡牌），返回布尔数组（参照文件不在时返回 None）：
      key / eco  在 rl/card_value.json 的关键卡 / 经济卡里
      tribe      是随从，而且场上已经有 ≥ 2 个和它同种族的随从（全种族算任何种族）——围绕主种族成型
      syn        和场上某个随从是高手常见搭配（rl/h13_profile.json 里提升度 ≥ 3 的组合）——卡牌联动
      simkey     在引擎实测的关键卡里（rl/sim_value.json，bg.py simcards：拿掉它胜率掉得多）
      simsyn     和场上某个随从是引擎实测的好搭配（rl/sim_value.json 的 pairs）
      engcard    买的是引擎卡本身（布莱恩·铜须、提图斯·瑞文戴尔）
      triple     买的随从场上或手里已经有同名的（不是金色）——凑三连
      engine     买的卡被场上的“引擎卡”放大：场上有布莱恩·铜须时买战吼随从、有提图斯·瑞文戴尔时买亡语随从——
                 布莱恩让战吼触发两次（南海卖艺者下回合 2 块钱、各种加攻防的战吼翻倍），打战吼随从换经济 / 换身材"""
    if "c" not in d:
        return None
    from tg.catalog import base_of_index, catalog, n_cards
    root = Path(__file__).resolve().parent if root is None else Path(root)
    base = {i: base_of_index(i) for i in range(1, n_cards())}
    buy = np.flatnonzero(d["kind"] == KINDS.index("buy"))
    slot = d["cand"][buy, 0].astype(np.int64) - 8
    ok = (slot >= 0) & (slot < 8)
    buy, slot = buy[ok], slot[ok]
    cards = d["c"][buy, C_SHOP + slot]
    sel = np.zeros(len(d["kind"]), dtype=bool)
    if which == "engcard":
        eng = set(engine_table())
        idx = [i for i, b in base.items() if b in eng]
        sel[buy[np.isin(cards, idx)]] = True
        return sel
    if which == "triple":
        from tg.catalog import card_id, is_golden_id
        from tg.obs import C_HAND, N_HAND
        det = catalog()
        golden = {i for i in base if is_golden_id(card_id(i))}
        for row, x in zip(buy, cards):
            x = int(x)
            if x in golden or (det.get(base.get(x)) or {}).get("kind") != "minion":
                continue
            own = [int(y) for y in list(d["c"][row, C_BOARD:C_BOARD + 7]) + list(d["c"][row, C_HAND:C_HAND + N_HAND])
                   if y > 0 and int(y) not in golden]
            sel[row] = any(base.get(y) == base.get(x) for y in own)
        return sel
    if which == "engine":
        det = catalog()
        amp = engine_table()
        kinds = {i: _effects(det.get(b) or {}) for i, b in base.items()}
        for row, x in zip(buy, cards):
            want = kinds.get(int(x)) or set()
            if not want:
                continue
            have = set()
            for y in d["c"][row, C_BOARD:C_BOARD + 7]:
                have |= amp.get(base.get(int(y)), set())
            sel[row] = bool(want & have)
        return sel
    if which in ("simkey", "simsyn"):
        from .comp import load_sim_refs
        if not (root / "sim_value.json").exists():
            return None
        spairs, skey = load_sim_refs(root)
        if which == "simkey":
            idx = [i for i, b in base.items() if b in skey]
            sel[buy[np.isin(cards, idx)]] = True
            return sel
        pairs = {q for x, y in spairs for q in ((x, y), (y, x))}
        for row, x in zip(buy, cards):
            bx = base.get(int(x))
            sel[row] = any((bx, base.get(int(y))) in pairs for y in d["c"][row, C_BOARD:C_BOARD + 7] if y > 0)
        return sel
    if which in ("key", "eco"):
        cv = root / "card_value.json"
        if not cv.exists():
            return None
        want = set(json.loads(cv.read_text(encoding="utf-8")).get(which, []))
        idx = [i for i, b in base.items() if b in want]
        sel[buy[np.isin(cards, idx)]] = True
        return sel
    det = catalog()
    if which == "tribe":
        races = {i: set((det.get(b) or {}).get("races", [])) for i, b in base.items()}
        minion = {i for i, b in base.items() if (det.get(b) or {}).get("kind") == "minion"}
        for row, x in zip(buy, cards):
            x = int(x)
            if x not in minion or not races.get(x):
                continue
            mine = races[x]
            n = 0
            for y in d["c"][row, C_BOARD:C_BOARD + 7]:
                ry = races.get(int(y)) or set()
                if ry and ("ALL" in ry or "ALL" in mine or ry & mine):
                    n += 1
            sel[row] = n >= 2
        return sel
    if which == "syn":
        hp = root / "h13_profile.json"
        if not hp.exists():
            return None
        from .comp import load_comp_refs
        pairs = {q for x, y in load_comp_refs(root)[0] for q in ((x, y), (y, x))}
        for row, x in zip(buy, cards):
            bx = base.get(int(x))
            sel[row] = any((bx, base.get(int(y))) in pairs for y in d["c"][row, C_BOARD:C_BOARD + 7] if y > 0)
        return sel
    return None
