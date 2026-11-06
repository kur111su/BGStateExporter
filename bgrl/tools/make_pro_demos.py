"""高分对局回放（Firestone 6500+ 分享包）→ 高分玩家的决策样本 runs/pro_demos/。

每局：回放 XML → 和插件一样的对局记录（tg/hsreplay.py，在每个决策点按 StateReader.cs 的规则读局面）
→ 切成动作（tg/logs.segment，和你自己的对局同一套）→ 每个动作前的局面在 TavernGym 里重建（tg/bridge.build_game）
→ 观察编码 + 动作编号；每回合最后加一条“结束回合”。重建后这个动作不合法的跳过（8 局测试 2977 个决策里 5 个）。

打出法术、打出随从、调站位太多（一局几百次），按比例抽一部分（--keep），升本、结束回合、买、卖全留。
打出 / 施法 / 技能 / 发动之后要选目标或发现的，另外做一条“选择”示范（kind = choose，全留）：每个选项在引擎里
试一遍，和回放里的结果比出高手选的是哪个——宝石给谁、法术打谁、发现拿哪张（--no-choose 不做）。
每 10 局留 1 局做验证（不参与训练），日志里的 pro_top1 / pro_upgrade / pro_end 就是在这些局上算的。

观察存成 soft_log 之后的 float16（和训练采样一样，模型用 compressed=True 读），分片写成 runs/pro_demos/part_XXX.npz。
这些是别人的对局，生成的样本只在本机（runs/ 不进 Git）。

用法：bg.py prodemos <回放文件夹> [--workers N] [--limit N]
回放文件夹（默认 data\\firestone，包括子文件夹）里可以混着放：分享包原样的 part*.zip、单局的 *.xml.zip、*.xml（HSReplay 格式）。
每次都从这个文件夹全部重新生成 runs/pro_demos/，所以新回放和旧的放在一起，不要只放新的。
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
# 示范格式在 rl/demos.py；load_dir / soft_log_np 的旧名字保留（tools.make_pro_demos.load_dir）
from rl.demos import KINDS, MAX_CANDS, SHARD_KEYS, load_dir, reset_dir, soft_log_np, write_shard  # noqa: E402,F401

KEEP_DEFAULT = "cast:0.35,play:0.35,move:0.35,activate:0.5,refresh:0.7"


_CTX: dict = {}


def _init(mmr_by_game: dict, keep: dict, choose: bool = True):
    _CTX.update(mmr=mmr_by_game, keep=keep, choose=choose)


def _jobs(src: Path) -> list[tuple[str, str]]:
    """找回放（包括子文件夹）：单局的 *.xml.zip / *.xml（HSReplay 格式，Firestone 分享包解压后的 rawxml-sanitized/ 里就是这种），
    和分享包原样的 *part*.zip（里面一堆 *.xml.zip，不用解压）。几种可以混着放；同一局（文件名相同）只算一次。
    返回 [(文件, 包内名字或空)]。"""
    import zipfile
    out, seen = [], set()

    def add(path: str, inner: str = ""):
        gid = Path(inner or path).name.split(".")[0]
        if gid not in seen:
            seen.add(gid)
            out.append((path, inner))
    for f in sorted(glob.glob(str(src / "**" / "*.xml.zip"), recursive=True)) + \
            sorted(glob.glob(str(src / "**" / "*.xml"), recursive=True)):
        add(f)
    for part in sorted(glob.glob(str(src / "**" / "*.zip"), recursive=True)):
        if part.endswith(".xml.zip"):
            continue
        try:
            with zipfile.ZipFile(part) as z:
                for n in sorted(z.namelist()):
                    if n.endswith(".xml.zip") or n.endswith(".xml"):
                        add(part, n)
        except zipfile.BadZipFile:
            continue
    return sorted(out, key=lambda x: Path(x[1] or x[0]).name)


def _xml(job: tuple[str, str]) -> bytes:
    import io
    import zipfile
    path, inner = job
    if not inner:
        from tg.hsreplay import read_xml
        return read_xml(path)
    with zipfile.ZipFile(path) as z:
        data = z.read(inner)
    if not inner.endswith(".zip"):
        return data
    with zipfile.ZipFile(io.BytesIO(data)) as z2:
        return z2.read(next(n for n in z2.namelist() if n.endswith(".xml")))


CHOOSERS = ("play", "cast", "hero_power", "activate")      # 这些动作之后可能要选目标 / 发现


def _choice_sample(g, action: int, act):
    """打出 / 施法 / 技能 / 发动之后要是停在待选项上（选目标、发现），把高手选的那个读出来，做成一条“选择”示范：
    每个选项都在引擎里试一遍，和回放里这一步之后的局面比，最接近的就是他选的（几个一样接近时都算对）。
    全都一样接近（例如对谁用都一样）不要——看不出选了哪个。"""
    from tg.actions import A_CHOOSE
    from tg.replay import best_choice, visible_log
    g2 = g.copy()
    try:
        g2.step(0, int(action))
    except ValueError:
        return None
    m2 = np.frombuffer(g2.legal_mask(0), dtype=np.uint8).astype(bool)
    legal = [x for x in range(A_CHOOSE, A_CHOOSE + 16) if m2[x]]
    if len(legal) < 2:
        return None
    reals = [visible_log(x) for x in (act.afters or [act.after])]
    idx, score = best_choice(g2, reals, act.kind not in ("refresh", "upgrade"))
    cands = [A_CHOOSE + i for i in idx if m2[A_CHOOSE + i]]
    if not cands or len(cands) >= len(legal):
        return None
    return g2, m2, cands, "choose"


def convert_game(args):
    """一局 → (样本数组字典, 统计)。回放 → 插件格式记录（tg/hsreplay.py）→ 切动作（tg/logs.segment，和你的对局一样）
    → 每个动作前的局面在 TavernGym 里重建 → 观察 + 动作编号（不合法的跳过）。每回合最后加一条“结束回合”。"""
    job, split = args
    from tg.bridge import build_game
    from tg.hsreplay import parse
    from tg.logs import load_records, segment
    from tg.replay import translate
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
            keep_main = rng.random() < _CTX["keep"].get(kind, 1.0)
            want_choose = _CTX.get("choose", True) and kind in CHOOSERS
            if not keep_main and not want_choose:
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
            rows = [(g, m, cands, kind)] if keep_main else []
            if want_choose:
                ch = _choice_sample(g, cands[0], act)
                if ch is not None:
                    rows.append(ch)
                    st["决策 choose"] += 1
            for gg, mm, cc_, kk in rows:
                f, c = gg.observe(0)
                F.append(soft_log_np(np.frombuffer(f, "<f4")).astype(np.float16))
                Cc.append(np.frombuffer(c, "<i4").copy())
                M.append(mm)
                row = np.full(MAX_CANDS, -1, np.int16)
                row[:len(cc_)] = cc_[:MAX_CANDS]
                CA.append(row)
                K.append(KINDS.index(kk))
                Tn.append(int(a.turn or 0))
                st[f"样本 {kk}"] += 1
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
    cat = {key: np.concatenate([p[key] for p in parts]) for key in SHARD_KEYS}
    return write_shard(out_dir, k, cat, parts[0]["n_actions"], meta)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="分享包目录：解压后的（含 rawxml-sanitized/）或者放着 5 个原始 zip 的文件夹")
    ap.add_argument("--out", default=str(RUNS / "pro_demos"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--val-every", type=int, default=10)
    ap.add_argument("--keep", default=KEEP_DEFAULT, help="各类动作保留的比例")
    ap.add_argument("--no-choose", action="store_true", help="不做“选择”示范（选目标、发现）")
    ap.add_argument("--shard", type=int, default=4000)
    a = ap.parse_args()
    src = Path(a.src)
    files = _jobs(src)
    if a.limit:
        files = files[:a.limit]
    if not files:
        where = "这个文件夹不存在" if not src.exists() else "里面没有回放"
        print(f"{src.resolve()}：{where}。\n"
              f"把 Firestone 分享包的 zip（firestone-bg-rawxml-sanitized-part1.zip … part4.zip，不用解压）放进这个文件夹，\n"
              f"或者在 prodemos.bat 后面写 zip 所在的文件夹，例如：prodemos.bat D:\\Downloads\\firestone")
        return 1
    mmr = _mmr(src)
    keep = {k: float(v) for k, v in (p.split(":") for p in a.keep.split(",") if ":" in p)}
    out_dir = reset_dir(Path(a.out))
    jobs = [(f, 1 if a.val_every and i % a.val_every == a.val_every - 1 else 0) for i, f in enumerate(files)]
    t0 = time.time()
    stats, buf, nbuf, k, total = Counter(), [], 0, 0, 0
    tiers = {}
    meta = {"source": "Firestone 6500+ 精彩对局回放（他人对局，已脱敏）", "games": len(files),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"), "keep": keep}
    if a.workers > 1:
        from multiprocessing import Pool
        pool = Pool(a.workers, initializer=_init, initargs=(mmr, keep, not a.no_choose))
        it = pool.imap(convert_game, jobs, chunksize=1)
    else:
        _init(mmr, keep, not a.no_choose)
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


if __name__ == "__main__":
    sys.exit(main())
