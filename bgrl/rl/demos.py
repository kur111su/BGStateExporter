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
