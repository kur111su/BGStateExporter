"""核对竞技场契约和本地模型（在 arena\\.venv 里运行，只读竞技场的公开接口，不需要也不读 Agent Key）。

读 /api/v1/manifest 和 /api/v1/catalog，和 model_bundle/meta.json 比：
  1. 观察：竞技场 v2 是 4017 浮点 + 447 卡牌编号，前 3227 / 124 个是 v1（我们模型的输入）。
     核对 manifest 的布局里和我们一样的字段数值一致，v1 长度一致。
  2. 动作：245 个，编号表（manifest.actions）原样存下来给人看。
  3. 卡牌词表：竞技场词表哈希和训练词表一样就直接用；不一样时按卡牌 ID 对应（竞技场编号 → 训练编号），
     竞技场里有、训练词表里没有的卡列出来，有的话不通过。
通过后写 compat.json（agent.py 启动时读）和 remap.npy，最后一行打印给连接器 --catalog-sha256 用的哈希。

用法：arena\\.venv\\Scripts\\python arena\\prep.py --url https://...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import ssl
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

HERE = Path(__file__).resolve().parent
BUNDLE = HERE / "model_bundle"


def canonical(x) -> bytes:
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


class Unreachable(Exception):
    pass


def get(origin: str, path: str, tries: int = 4):
    """读公开接口；连不上（握手超时、断线）时等 5 / 15 / 30 秒再试，最后还不行抛 Unreachable。"""
    import time
    req = urllib.request.Request(origin.rstrip("/") + path, headers={"Accept": "application/json",
                                                                     "User-Agent": "TavernArena-Connector/1"})
    last = None
    for k in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=30, context=ssl.create_default_context()) as r:
                return json.loads(r.read(2 ** 23))
        except (OSError, ValueError) as ex:             # URLError / 超时 / 连接重置都是 OSError
            last = ex
            if k + 1 < tries:
                wait = (5, 15, 30)[min(k, 2)]
                print(f"连不上 {origin}{path}（{type(ex).__name__}），{wait} 秒后重试 {k + 2}/{tries} …", flush=True)
                time.sleep(wait)
    raise Unreachable(f"{origin}{path}：{last}")


def _cards(catalog) -> list[dict]:
    if isinstance(catalog, dict):
        for k in ("cards", "vocabulary", "items"):
            if isinstance(catalog.get(k), list):
                return catalog[k]
    if isinstance(catalog, list):
        return catalog
    raise ValueError("认不出 /api/v1/catalog 的格式")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    a = ap.parse_args()
    u = urlsplit(a.url)
    if u.scheme != "https" and u.hostname not in ("127.0.0.1", "localhost", "::1"):
        print("竞技场地址必须是 https")
        return 1
    if not (BUNDLE / "meta.json").exists():
        print("还没有 model_bundle（先运行 arena\\export.py）")
        return 1
    meta = json.loads((BUNDLE / "meta.json").read_text(encoding="utf-8"))
    man = get(a.url, "/api/v1/manifest")
    (HERE / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    problems, notes = [], []
    obs = man.get("observation") or {}
    lens = [obs.get(k, {}).get("length") for k in ("features", "cards", "action_mask")]
    if lens != [4017, 447, 245]:
        problems.append(f"观察长度 {lens}，连接器只支持 [4017, 447, 245]")
    lay = obs.get("layout") or {}
    diff = {k: (v, lay.get(k)) for k, v in meta["layout"].items() if k in lay and lay[k] != v}
    missing = [k for k in meta["layout"] if k not in lay]
    if diff:
        problems.append(f"观察布局和训练时不一样：{diff}")
    if missing:
        notes.append(f"manifest 布局里没有这些字段（无法核对）：{missing}")
    v1 = json.dumps(obs, ensure_ascii=False)
    for n in (meta["floats"], meta["cards"]):
        if str(n) not in v1:
            notes.append(f"manifest.observation 里没写 {n}（v1 前缀长度），按协议说明“v1 是 v2 前缀”处理")
    if meta["actions"] != 245:
        problems.append(f"模型动作数 {meta['actions']} ≠ 245")
    # 卡牌词表
    cat = get(a.url, "/api/v1/catalog")
    arena_cards = _cards(cat)
    cat_hash = hashlib.sha256(canonical(cat)).hexdigest()
    if man.get("catalog_sha256") and cat_hash != man["catalog_sha256"]:
        notes.append("/api/v1/catalog 的哈希和 manifest.catalog_sha256 对不上（可能只是格式包装不同），以 manifest 为准")
    ours = {cid: i for i, cid in meta["vocab"]}
    n_arena = max([int(c["index"]) for c in arena_cards] + [0]) + 1
    remap = np.zeros(max(n_arena, 1), dtype=np.int64)
    unknown, moved = [], 0
    for c in arena_cards:
        i, cid = int(c["index"]), c.get("id")
        if i == 0:
            continue
        j = ours.get(cid)
        if j is None:
            unknown.append(f"{cid}（{c.get('name_zh') or c.get('name')}）")
            continue
        remap[i] = j
        moved += j != i
    same_vocab = man.get("catalog_sha256") == meta["catalog_sha256"]
    if unknown:
        problems.append(f"竞技场有 {len(unknown)} 张训练词表里没有的卡：{unknown[:10]}")
    env = man.get("environment") or {}
    result = {"ok": not problems, "url": f"{u.scheme}://{u.netloc}", "ruleset_id": man.get("ruleset_id"),
              "protocol_version": man.get("protocol_version"), "arena_taverngym": env,
              "our_taverngym": meta.get("taverngym_version"), "our_engine_source_hash": meta.get("engine_source_hash"),
              "same_vocab_hash": same_vocab, "arena_cards": len(arena_cards), "remapped": int(moved),
              "catalog_sha256": man.get("catalog_sha256"), "model_version": meta["model_version"],
              "problems": problems, "notes": notes, "actions": man.get("actions"),
              "time_control": {k: man.get(k) for k in ("time_control", "turn_seconds", "limits") if k in man}}
    (HERE / "compat.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    np.save(HERE / "remap.npy", remap)
    print(f"竞技场 {result['url']}：协议 {result['protocol_version']}，规则 {result['ruleset_id']}，"
          f"引擎 {env.get('version') or env.get('taverngym_version') or env}")
    print(f"本地模型 {meta['model_version']}（训练用 TavernGym {meta.get('taverngym_version')}）")
    print(f"卡牌词表：竞技场 {len(arena_cards)} 张；哈希{'相同' if same_vocab else '不同'}；"
          f"编号不同需要换算的 {moved} 张；训练词表里没有的 {len(unknown)} 张")
    for x in notes:
        print("注意：", x)
    for x in problems:
        print("不通过：", x)
    if problems:
        return 1
    print("CATALOG_SHA256=" + str(man.get("catalog_sha256")))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Unreachable as ex:
        print(f"连不上竞技场：{ex}\n这是网络问题（网站暂时不可用，或网络 / 代理拦住了），不是模型或代码的问题；"
              f"过一会儿再运行 arena.bat check。")
        sys.exit(2)
