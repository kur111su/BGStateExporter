"""外部模型（rl/extnet.py）：安全读取、torch 版和 numpy 参考实现一致、不让移动、恶意文件被拒绝。
用随机小权重造一个同格式的文件（不用别人的权重）。需要 torch 和 TavernGym。"""
import collections
import pickle
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                     # noqa: E402
import torch                                           # noqa: E402

from rl import extnet as X                             # noqa: E402


def fake_ckpt(path: Path, d=32, heads=4, layers=2, n_cards=60, attr_f=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g) * 0.2                              # noqa: E731
    sd = collections.OrderedDict()
    sd["pos"] = r(X.N_TOK, d)
    sd["attr"] = torch.rand(n_cards, attr_f, generator=g)
    sd["zcard"] = torch.rand(2, n_cards, 3, generator=g)
    sd["card.weight"] = r(n_cards, d)
    sd["attr_proj.weight"] = r(d, attr_f)
    for n, i, o in (("f_global", X.GF, d), ("f_board", X.MF, d), ("f_item", X.IF, d), ("f_opp", X.OPPF, d),
                    ("f_oppb", X.MF, d)):
        sd[n + ".weight"], sd[n + ".bias"] = r(o, i), r(o)
    for k in range(layers):
        p = f"encoder.layers.{k}."
        sd[p + "self_attn.in_proj_weight"], sd[p + "self_attn.in_proj_bias"] = r(3 * d, d), r(3 * d)
        sd[p + "self_attn.out_proj.weight"], sd[p + "self_attn.out_proj.bias"] = r(d, d), r(d)
        sd[p + "linear1.weight"], sd[p + "linear1.bias"] = r(4 * d, d), r(4 * d)
        sd[p + "linear2.weight"], sd[p + "linear2.bias"] = r(d, 4 * d), r(d)
        for nn_ in ("norm1", "norm2"):
            sd[p + nn_ + ".weight"], sd[p + nn_ + ".bias"] = 1 + r(d), r(d)
    sd["norm.weight"], sd["norm.bias"] = 1 + r(d), r(d)
    for n, i, o in (("h_global", 2 * d, 8), ("h_buy", d, 1), ("h_sell", d, 1), ("h_activate", d, 1), ("h_choose", d, 1),
                    ("h_play", d, 8), ("q_mag", d, d), ("h_move", d, 7), ("value.0", 2 * d, d), ("value.2", d, 1)):
        sd[n + ".weight"], sd[n + ".bias"] = r(o, i), r(o)
    sd["z_global.weight"], sd["z_tok.weight"] = r(d, 2), r(d, 3)
    torch.save({"format": X.FORMAT, "net_config": {"d": d, "layers": layers, "heads": heads, "n_cards": n_cards,
                                                   "z_vocab": ["a", "none"]},
                "state_dict": sd, "meta": {"iter": 7}}, path)


def sample_obs(n=12):
    import json
    import taverngym._core as c
    F, C, M = [], [], []
    g = c.Game(3, json.dumps({}))
    while len(F) < n and not g.is_over():
        for p in range(8):
            if g.needs_action(p):
                fb, cb = g.observe(p)
                F.append(np.frombuffer(fb, dtype="<f4").copy())
                C.append(np.frombuffer(cb, dtype="<i4").copy())
                M.append(np.frombuffer(g.legal_mask(p), dtype=np.uint8).astype(bool))
                g.bot_play(p)
    return np.stack(F[:n]), np.stack(C[:n]), np.stack(M[:n])


def main():
    tmp = Path(tempfile.mkdtemp())
    p = tmp / "ext.pt"
    fake_ckpt(p)
    assert X.is_external(p)
    ck = X.load_checkpoint(p)
    assert ck["meta"]["iter"] == 7 and ck["state_dict"]["pos"].shape == (X.N_TOK, 32)

    f, c, m = sample_obs()
    f = f.copy()
    c = c % 60                                          # 假权重只有 60 张卡
    lg_np, v_np = X.forward_np(ck, f, c)
    from rl.model import load_model
    model, info = load_model(p, "cpu")
    assert info.get("external") and getattr(model, "external", False)
    mm = torch.ones(len(f), 245, dtype=torch.bool)
    model.allow_moves = True
    with torch.no_grad():
        lg, v, pl = model(torch.from_numpy(f), torch.from_numpy(c), mm)
    d_lg = float(np.abs(lg.numpy() - lg_np).max())
    d_v = float(np.abs(v.numpy() - v_np).max())
    print(f"torch vs numpy: logits {d_lg:.2e}, value {d_v:.2e}")
    assert d_lg < 2e-3 and d_v < 2e-3, (d_lg, d_v)
    # 压缩过的输入（训练数据）结果一样
    from rl.model import soft_log
    with torch.no_grad():
        lg2 = model(soft_log(torch.from_numpy(f)), torch.from_numpy(c), mm, compressed=True)[0]
    assert float((lg2 - lg).abs().max()) < 1e-4
    # 默认不让移动（还有别的动作时）；非法动作 -1e9
    model.allow_moves = False
    mt = torch.from_numpy(m)
    with torch.no_grad():
        lg3 = model(torch.from_numpy(f), torch.from_numpy(c), mt)[0]
    other = mt[:, :X.A_MOVE].any(-1)
    assert bool((lg3[other][:, X.A_MOVE:] <= -1e8).all())
    assert bool((lg3[~mt] <= -1e8).all())
    assert not any(q.requires_grad for q in model.parameters())
    model.train()
    assert not model.training

    # 恶意文件：pickle 里引用别的全局对象 → 拒绝，不执行
    bad = tmp / "bad.pt"
    payload = pickle.dumps({"format": X.FORMAT, "x": print}, protocol=2)        # 引用 builtins.print
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("bad/data.pkl", payload)
        z.writestr("bad/byteorder", "little")
    assert X.is_external(bad)
    try:
        X.load_checkpoint(bad)
        raise AssertionError("恶意文件没被拒绝")
    except pickle.UnpicklingError:
        pass
    # 普通的我们自己的检查点不是外部模型
    own = tmp / "own.pt"
    torch.save({"model": {"w": torch.zeros(2)}}, own)
    assert not X.is_external(own)
    print("extnet ok")


if __name__ == "__main__":
    main()
