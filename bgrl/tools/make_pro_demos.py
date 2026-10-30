"""高分对局回放（Firestone 6500+ 分享包）→ 高分玩家的决策样本 runs/pro_demos/。

每局：回放 XML → 和插件一样的对局记录（tg/hsreplay.py，在每个决策点按 StateReader.cs 的规则读局面）
→ 切成动作（tg/logs.segment，和你自己的对局同一套）→ 每个动作前的局面在 TavernGym 里重建（tg/bridge.build_game）
→ 观察编码 + 动作编号；每回合最后加一条“结束回合”。重建后这个动作不合法的跳过（8 局测试 2977 个决策里 5 个）。

打出法术、打出随从、调站位太多（一局几百次），按比例抽一部分（--keep），升本、结束回合、买、卖全留。
每 10 局留 1 局做验证（不参与训练），日志里的 pro_top1 / pro_upgrade / pro_end 就是在这些局上算的。

观察存成 soft_log 之后的 float16（和训练采样一样，模型用 compressed=True 读），分片写成 runs/pro_demos/part_XXX.npz。
这些是别人的对局，生成的样本只在本机（runs/ 不进 Git）。

用法：bg.py prodemos <分享包文件夹：解压后的，或者就是那 5 个 zip 所在的文件夹> [--workers N] [--limit N]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paths import RUNS  # noqa: E402

def soft_log_np(x: np.ndarray) -> np.ndarray:
    """和 rl/model.py 的 soft_log_np 一样（这里不导入 rl.model：转换不需要 torch）。"""
    a = np.abs(x)
    return np.where(a <= 1, x, np.sign(x) * (1 + np.log(np.maximum(a, 1)))).astype(np.float32)


KEEP_DEFAULT = "cast:0.35,play:0.35,move:0.35,activate:0.5,refresh:0.7"


_CTX: dict = {}


def _init(mmr_by_game: dict, keep: dict):
    _CTX.update(mmr=mmr_by_game, keep=keep)


def _jobs(src: Path) -> list[tuple[str, str]]:
    """找回放：目录里的 rawxml-sanitized/*.xml.zip、*.xml.zip，或者分享包原样的 *part*.zip（不用解压）。
    返回 [(文件, 包内名字或空)]。"""
    import zipfile
    files = sorted(glob.glob(str(src / "rawxml-sanitized" / "*.xml.zip"))) or sorted(glob.glob(str(src / "*.xml.zip")))
    if files:
        return [(f, "") for f in files]
    out = []
    for part in sorted(glob.glob(str(src / "*.zip"))):
        try:
            with zipfile.ZipFile(part) as z:
                out += [(part, n) for n in sorted(z.namelist()) if n.endswith(".xml.zip")]
        except zipfile.BadZipFile:
            continue
    return out


def _xml(job: tuple[str, str]) -> bytes:
    import io
    import zipfile
    path, inner = job
    if not inner:
        from tg.hsreplay import read_xml
        return read_xml(path)
    with zipfile.ZipFile(path) as z:
        data = z.read(inner)
    with zipfile.ZipFile(io.BytesIO(data)) as z2:
        return z2.read(next(n for n in z2.namelist() if n.endswith(".xml")))


def convert_game(args):
    """一局 → (样本数组字典, 统计)。回放 → 插件格式记录（tg/hsreplay.py）→ 切动作（tg/logs.segment，和你的对局一样）
    → 每个动作前的局面在 TavernGym 里重建 → 观察 + 动作编号（不合法的跳过）。每回合最后加一条“结束回合”。"""
    job, split = args
    from tg.bridge import build_game
    from tg.hsreplay import parse
    from tg.logs import load_records, segment
    from tg.replay import translate
    from tools.make_demos import KINDS, MAX_CANDS
    from tg.actions import A_END
    gid = Path(job[1] or job[0]).name.split(".")[0]
    rng = np.random.default_rng(int(hashlib.md5(gid.encode()).hexdigest()[:8], 16))
    st = Counter()
    try:
        states = load_records(parse(_xml(job)), gid)
        acts = segment(states)
    except Exception as ex:                           # noqa: BLE001
        st[f"解析失败 {type(ex).__name__}"] += 1
        return None, st
    tiers = {}
    for s_ in states:                                 # 每回合招募结束时的酒馆等级（升本节奏曲线用，见 tools/pro_library.py）
        tiers[int(s_.get("turn") or 0)] = int((s_.get("player") or {}).get("tavernTier") or 0)
    F, Cc, M, CA, K, Tn = [], [], [], [], [], []
    for i, a in enumerate(acts):
        nxt = acts[i + 1] if i + 1 < len(acts) else None
        todo = [] if a.kind == "multi" else [(a.before, a.kind, a)]
        if nxt is None or nxt.turn != a.turn:
            todo.append(((a.afters or [a.after])[-1], "end", None))
        for state, kind, act in todo:
            st[f"决策 {kind}"] += 1
            if rng.random() >= _CTX["keep"].get(kind, 1.0):
                continue
            try:
                g, _ = build_game(state, seed=i)
            except Exception as ex:                   # noqa: BLE001
                st[f"重建失败 {type(ex).__name__}"] += 1
                continue
            m = np.frombuffer(g.legal_mask(0), dtype=np.uint8).astype(bool)
            cands = [A_END] if kind == "end" else translate(act)
            cands = [x for x in cands if 0 <= x < len(m) and m[x]][:MAX_CANDS]
            if not cands:
                st[f"不合法 {kind}"] += 1
                continue
            f, c = g.observe(0)
            F.append(soft_log_np(np.frombuffer(f, "<f4")).astype(np.float16))
            Cc.append(np.frombuffer(c, "<i4").copy())
            M.append(m)
            row = np.full(MAX_CANDS, -1, np.int16)
            row[:len(cands)] = cands
            CA.append(row)
            K.append(KINDS.index(kind))
            Tn.append(int(a.turn or 0))
            st[f"样本 {kind}"] += 1
    if not K:
        return None, st
    n = len(K)
    out = {"f": np.stack(F), "c": np.stack(Cc).astype(np.int32), "m": np.packbits(np.stack(M), axis=1),
           "cand": np.stack(CA), "kind": np.asarray(K, np.int8), "turn": np.asarray(Tn, np.int16),
           "split": np.full(n, split, np.int8), "mmr": np.full(n, _CTX["mmr"].get(gid, 0), np.int32),
           "n_actions": len(M[0]), "tiers": tiers, "gid": gid}
    return out, st


def _mmr(src: Path) -> dict:
    """可选：core/games.jsonl（或 firestone-bg-core.zip 里的）给每局的 MMR。"""
    import zipfile
    lines = []
    f = src / "core" / "games.jsonl"
    if f.exists():
        lines = open(f, encoding="utf-8")
    else:
        for z in glob.glob(str(src / "*core*.zip")):
            with zipfile.ZipFile(z) as zz:
                n = next((x for x in zz.namelist() if x.endswith("games.jsonl")), None)
                if n:
                    lines = zz.read(n).decode("utf-8").splitlines()
                    break
    out = {}
    for line in lines:
        if line.strip():
            g = json.loads(line)
            out[g.get("review_id")] = int(g.get("player_rank") or 0)
    return out


def _write(out_dir: Path, k: int, parts: list[dict], meta: dict):
    cat = {key: np.concatenate([p[key] for p in parts]) for key in ("f", "c", "m", "cand", "kind", "turn", "split", "mmr")}
    np.savez_compressed(out_dir / f"part_{k:03d}.npz", **cat, n_actions=np.int32(parts[0]["n_actions"]),
                        compressed=np.int8(1), meta=np.array(json.dumps(meta, ensure_ascii=False)))
    return len(cat["kind"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="分享包目录：解压后的（含 rawxml-sanitized/）或者放着 5 个原始 zip 的文件夹")
    ap.add_argument("--out", default=str(RUNS / "pro_demos"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--val-every", type=int, default=10)
    ap.add_argument("--keep", default=KEEP_DEFAULT, help="各类动作保留的比例")
    ap.add_argument("--shard", type=int, default=4000)
    a = ap.parse_args()
    src = Path(a.src)
    files = _jobs(src)
    if a.limit:
        files = files[:a.limit]
    if not files:
        print(f"{src} 里没有回放（rawxml-sanitized/*.xml.zip，或分享包的 *part*.zip）")
        return 1
    mmr = _mmr(src)
    keep = {k: float(v) for k, v in (p.split(":") for p in a.keep.split(",") if ":" in p)}
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("part_*.npz"):
        old.unlink()
    jobs = [(f, 1 if a.val_every and i % a.val_every == a.val_every - 1 else 0) for i, f in enumerate(files)]
    t0 = time.time()
    stats, buf, nbuf, k, total = Counter(), [], 0, 0, 0
    tiers = {}
    meta = {"source": "Firestone 6500+ 精彩对局回放（他人对局，已脱敏）", "games": len(files),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"), "keep": keep}
    if a.workers > 1:
        from multiprocessing import Pool
        pool = Pool(a.workers, initializer=_init, initargs=(mmr, keep))
        it = pool.imap(convert_game, jobs, chunksize=1)
    else:
        _init(mmr, keep)
        it = map(convert_game, jobs)
    for i, (res, st) in enumerate(it, 1):
        stats += st
        if res is not None:
            tiers[res.pop("gid")] = res.pop("tiers")
            buf.append(res)
            nbuf += len(res["kind"])
        if nbuf >= a.shard:
            total += _write(out_dir, k, buf, meta)
            k, buf, nbuf = k + 1, [], 0
        if i % 10 == 0 or i == len(jobs):
            el = time.time() - t0
            print(f"[{i}/{len(jobs)}] 样本 {total + nbuf}，{el / 60:.1f} 分钟，预计还要 {el / i * (len(jobs) - i) / 60:.0f} 分钟",
                  flush=True)
    if buf:
        total += _write(out_dir, k, buf, meta)
        k += 1
    meta.update(samples=total, shards=k, stats=dict(stats))
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "tiers.json").write_text(json.dumps(tiers), encoding="utf-8")
    dec = sum(v for kk, v in stats.items() if kk.startswith("决策"))
    print(f"完成：{len(files)} 局，{dec} 个决策 → {total} 条样本（{k} 片）→ {out_dir}；{time.time() - t0:.0f} 秒")
    print({kk: v for kk, v in sorted(stats.items())})
    return 0


def load_dir(path) -> dict | None:
    """训练用：读 runs/pro_demos/ 里所有分片，拼成和 tools/make_demos.load 一样的字典（多一个 compressed=True）。"""
    files = sorted(Path(path).glob("part_*.npz"))
    if not files:
        return None
    parts = [np.load(f, allow_pickle=False) for f in files]
    d = {k: np.concatenate([p[k] for p in parts]) for k in ("f", "c", "m", "cand", "kind", "turn", "split", "mmr")}
    n_act = int(parts[0]["n_actions"])
    d["m"] = np.unpackbits(d["m"], axis=1)[:, :n_act].astype(bool)
    d["compressed"] = True
    meta = Path(path) / "meta.json"
    d["meta"] = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}
    return d


if __name__ == "__main__":
    sys.exit(main())
