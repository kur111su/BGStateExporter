"""竞技场适配（arena/）：打包 → 不装 TavernGym 也能建模型 → act() 在真实 TavernGym 局面里只出合法动作。

用一个随机初始化的小模型；竞技场的 v2 观察用 v1 后面补零模拟（agent 只用前缀）。不连网。"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    import taverngym
    import torch
    from rl.model import build_model

    arena = ROOT / "arena"
    saved = {n: (arena / n).read_bytes() for n in ("compat.json", "remap.npy") if (arena / n).exists()}
    bundle_bak = None
    if (arena / "model_bundle").exists():
        bundle_bak = Path(tempfile.mkdtemp()) / "model_bundle"
        shutil.copytree(arena / "model_bundle", bundle_bak)
    tmp = Path(tempfile.mkdtemp())
    try:
        kw = dict(d=64, layers=2, heads=2, dc=32)
        torch.manual_seed(0)
        m = build_model(**kw)
        ck = tmp / "latest.pt"
        torch.save({"model": m.state_dict(), "model_kwargs": kw, "iter": 7}, ck)
        sys.argv = ["export.py", "--ckpt", str(ck)]
        sys.path.insert(0, str(arena))
        import export
        assert export.main() == 0
        meta = json.loads((arena / "model_bundle" / "meta.json").read_text(encoding="utf-8"))
        assert meta["floats"] == 3227 and meta["cards"] == 124 and meta["actions"] == 245, meta
        (arena / "compat.json").write_text(json.dumps({"ok": True, "model_version": meta["model_version"],
                                                       "catalog_sha256": meta["catalog_sha256"]}), encoding="utf-8")
        np.save(arena / "remap.npy", np.arange(taverngym.N_CARDS, dtype=np.int64))
        import os
        os.environ["BGRL_ARENA_TEST"] = "1"
        import agent                                    # 只用 bundle 建模型，不读 TavernGym
        # 和训练时的模型输出一致
        g = taverngym.Game(seed=3)
        f, c = g.observe(0)
        f = np.frombuffer(f, "<f4")
        c = np.frombuffer(c, "<i4")
        mask = np.frombuffer(g.legal_mask(0), dtype=np.uint8).astype(bool)
        t = lambda x, dt: torch.as_tensor(np.array(x)[None], dtype=dt)  # noqa: E731
        with torch.no_grad():
            ref = m.eval()(t(f, torch.float32), t(c, torch.long), t(mask, torch.bool))[0]
            got = agent.MODEL(t(f, torch.float32), t(c, torch.long), t(mask, torch.bool))[0]
        assert torch.allclose(ref, got, atol=1e-4), "打包后的模型输出和原模型不一样"
        # 一整局：0 号座位用 agent.act，其它用脚本机器人
        state, steps = {}, 0
        while not g.is_over() and steps < 20000:
            for p in range(g.num_players):
                if not g.needs_action(p):
                    continue
                if p == 0:
                    f, c = g.observe(0)
                    obs = {"features": np.concatenate([np.frombuffer(f, "<f4"), np.zeros(4017 - 3227, np.float32)]),
                           "cards": np.concatenate([np.frombuffer(c, "<i4"), np.zeros(447 - 124, np.int32)]),
                           "action_mask": np.frombuffer(g.legal_mask(0), dtype=np.uint8).astype(bool)}
                    ctx = {"match_id": "test", "seat": 0, "turn": g.turn, "phase": "recruit", "remaining_ms": 60000,
                           "actions_used": 0, "action_limit": 100, "player_view": {}, "state": state}
                    a = agent.act(obs, ctx)
                    assert isinstance(a, int) and obs["action_mask"][a], a
                    g.step(0, a)
                    steps += 1
                else:
                    g.step(p, g.bot_action(p))
        assert steps > 20, steps
        rec = [json.loads(x) for x in (arena / "records" / "test.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(rec) == steps and all(r["text"] for r in rec) and rec[0]["top"], rec[0]
        assert agent.describe(8, {"me": {"shop": [{"name_zh": "测试"}]}}) == "买 测试"
        print(f"arena OK：{steps} 步，全部合法；日志 {sorted(p.name for p in (arena / 'logs').glob('test*'))}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        (arena / "logs" / "test.jsonl").unlink(missing_ok=True)
        (arena / "records" / "test.jsonl").unlink(missing_ok=True)
        if (arena / "records").exists() and not any((arena / "records").iterdir()):
            (arena / "records").rmdir()
        if (arena / "logs").exists() and not any((arena / "logs").iterdir()):
            (arena / "logs").rmdir()
        for n in ("compat.json", "remap.npy"):
            (arena / n).unlink(missing_ok=True)
        for n, b in saved.items():
            (arena / n).write_bytes(b)
        shutil.rmtree(arena / "model_bundle", ignore_errors=True)
        if bundle_bak is not None:
            shutil.copytree(bundle_bak, arena / "model_bundle")
    return 0


if __name__ == "__main__":
    sys.exit(main())
