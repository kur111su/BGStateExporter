"""C1 影子模式测试（CI，CPU）：真实局面重建 → 模型给方案；中文描述；检查点读取。"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rl.advisor import Advisor, load_model  # noqa: E402
from rl.model import build_model  # noqa: E402
from tg.bridge import build_game  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures" / "recruit_states_v3.jsonl"


def main() -> int:
    kw = dict(d=64, layers=2, heads=2, dc=32)
    m = build_model(**kw).eval()
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "latest.pt"
        torch.save({"model": m.state_dict(), "iter": 7, "model_kwargs": kw}, p)
        m2, info = load_model(p)
        assert info["iter"] == 7 and info["size"] == kw
    adv = Advisor(m2)
    states = [json.loads(x) for x in FIX.read_text(encoding="utf-8").splitlines() if x.strip()]
    n_steps = 0
    worst = 0
    for s in states[::4]:
        g, _ = build_game(s)
        before = g.state_json()
        out = adv.plan(g)
        assert g.state_json() == before, "plan 改动了局面"
        assert out["steps"] and all(isinstance(x["text"], str) and "?" not in x["text"] for x in out["steps"]), out
        assert 1 <= out["expectedPlace"] <= 8 and out["alternatives"]
        n_steps += len(out["steps"])
        worst = max(worst, out["ms"])
    print(f"{len(states[::4])} 个局面都给出了方案，共 {n_steps} 步，最慢 {worst} ms；例：{out['steps'][:3]}")
    print("SHADOW TEST OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
