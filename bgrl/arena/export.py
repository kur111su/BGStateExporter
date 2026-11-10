"""把训练好的模型打包给竞技场连接器用（在训练环境 bgrl\\.venv 里运行，需要 TavernGym 和 torch）。

竞技场连接器跑在单独的 arena\\.venv 里（不装 TavernGym：本机的 TavernGym 是训练用的旧版，和竞技场打过补丁的
引擎源码指纹不同，装着它连接器会拒绝）。所以这里把模型需要从 TavernGym 读的东西一次存下来：
  arena/model_bundle/model.pt    模型参数（只在本机，不上传、不进 Git）
  arena/model_bundle/static.npy  卡牌静态特征表（card_static）
  arena/model_bundle/meta.json   观察布局、数组长度、模型大小、训练轮数、训练用的卡牌词表（编号 → 卡牌 ID）及其哈希

用法：.venv\\Scripts\\python arena\\export.py [--ckpt runs\\v3\\latest.pt] [--name capri]（竞技场上显示 capri-it<训练轮数>）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NAME = "capri"                     # 竞技场上的模型名：连接器的 --model-version 发的是 "capri-it<训练轮数>"
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))


def canonical(x) -> bytes:
    """和官方连接器 arena_client.py 的 validate 一样的规范化 JSON。"""
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def training_catalog() -> dict:
    """训练用 TavernGym 的卡牌词表，结构和连接器核对时算哈希用的一样。"""
    import taverngym
    vocab = [{"index": i, "id": c[0], "name": c[1], "name_zh": c[2], "tier": c[3]}
             for i in range(1, taverngym.N_CARDS) for c in (taverngym.card_info(i),)]
    return {"empty_index": 0, "cards": vocab, "details": json.loads(taverngym.catalog_json())}


def main() -> int:
    import taverngym
    import taverngym._core as c
    import torch
    from paths import DEFAULT_RUN
    from rl.model import load_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(DEFAULT_RUN / "latest.pt"))
    ap.add_argument("--out", default=str(HERE / "model_bundle"))
    ap.add_argument("--name", default=NAME, help="竞技场上显示的模型名（后面自动加 -it训练轮数）")
    a = ap.parse_args()
    model, info = load_model(a.ckpt, "cpu")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n, raw = c.card_static()
    static = np.frombuffer(raw, dtype="<f4").reshape(-1, n).copy()
    cat = training_catalog()
    try:
        src = taverngym.build_info().get("source_hash")
    except Exception:                                  # noqa: BLE001
        src = None
    run = Path(a.ckpt).resolve().parent.name
    it = info.get("iter")
    if it is None:                                     # 快照文件只有参数：轮数从文件名读（it12450.pt）
        import re
        m = re.search(r"it(\d+)", Path(a.ckpt).name)
        it = int(m.group(1)) if m else None
    meta = {"ckpt": str(Path(a.ckpt).resolve()), "run": run, "iter": it,
            "model_version": f"{a.name}-it{it}" if it is not None else a.name,
            "model_kwargs": info["size"], "layout": dict(c.layout()),
            "floats": int(c.OBS_FLOATS), "cards": int(c.OBS_CARDS), "actions": int(c.N_ACTIONS),
            "taverngym_version": taverngym.version(), "engine_source_hash": src,
            "catalog_sha256": hashlib.sha256(canonical(cat)).hexdigest(),
            "vocab": [[x["index"], x["id"]] for x in cat["cards"]]}
    torch.save({"model": model.state_dict(), "model_kwargs": info["size"]}, out / "model.pt")
    np.save(out / "static.npy", static)
    from rl.model import comp_tables
    tab = comp_tables()                                # 体系感知的阵容查表（竞技场那边没有 TavernGym，算不了）
    if tab is not None:
        np.savez(out / "comp.npz", **{k: v for k, v in tab.items() if k != "names"})
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"已打包 {meta['model_version']}（{a.ckpt}）→ {out}；观察 {meta['floats']} 浮点 + {meta['cards']} 卡牌编号，"
          f"{len(meta['vocab'])} 张卡，词表哈希 {meta['catalog_sha256'][:12]}…", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
