"""MLX 值不值得换（python bg.py bench-mlx）：同样大小的网络、同样的一份样本，比 MLX 和 PyTorch（MPS）训练一步各要多久。

只是测速，不动训练：网络是和我们的模型一样大的 Transformer（d=320、6 层、8 头、前馈 1280、48 个词、每份 512 个样本），
每种跑“前向 + 反向 + Adam 一步”若干次（前几次预热不算）。结果写进 reports/mlx_bench.json。
  PyTorch：MPS float32、MPS bfloat16（现在训练用的）
  MLX：float32、bfloat16
判断：MLX 最快的比 PyTorch 最快的快 1.5 倍以上才值得把整个项目换成 MLX；不到 1.3 倍就不换。
MLX 没装时自己装（pip install mlx，只有 Apple 芯片的 Mac 能装）。约 1–2 分钟。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "mlx_bench.json"
D, LAYERS, HEADS, TOK, BATCH, ACTIONS = 320, 6, 8, 48, 512, 245


def bench_torch(dtype: str, steps: int, warm: int) -> dict:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    if not torch.backends.mps.is_available():
        return {"ok": False, "error": "MPS 用不了"}
    dev = torch.device("mps")
    torch.manual_seed(0)
    layer = nn.TransformerEncoderLayer(D, HEADS, 4 * D, dropout=0.0, batch_first=True, norm_first=True, activation="gelu")
    net = nn.ModuleDict({"inp": nn.Linear(64, D), "enc": nn.TransformerEncoder(layer, LAYERS, enable_nested_tensor=False),
                         "norm": nn.LayerNorm(D), "pi": nn.Linear(D, ACTIONS), "v": nn.Linear(D, 1)}).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    x = torch.randn(BATCH, TOK, 64, device=dev)
    a = torch.randint(0, ACTIONS, (BATCH,), device=dev)
    ret = torch.randn(BATCH, device=dev)
    amp = dtype == "bfloat16"

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("mps", dtype=torch.bfloat16, enabled=amp):
            h = net["norm"](net["enc"](net["inp"](x)))
            g = h.mean(1)
            lg, v = net["pi"](g), net["v"](g).squeeze(-1)
        loss = F.cross_entropy(lg.float(), a) + (v.float() - ret).pow(2).mean()
        loss.backward()
        opt.step()
        return loss

    for _ in range(warm):
        step()
    torch.mps.synchronize()
    t0 = time.time()
    for _ in range(steps):
        loss = step()
    torch.mps.synchronize()
    dt = (time.time() - t0) / steps
    return {"ok": bool(torch.isfinite(loss)), "ms": round(dt * 1000, 1), "sps": round(BATCH / dt)}


def bench_mlx(dtype: str, steps: int, warm: int) -> dict:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.n1, self.n2 = nn.LayerNorm(D), nn.LayerNorm(D)
            self.attn = nn.MultiHeadAttention(D, HEADS, bias=True)
            self.l1, self.l2 = nn.Linear(D, 4 * D), nn.Linear(4 * D, D)

        def __call__(self, x):
            y = self.n1(x)
            x = x + self.attn(y, y, y)
            return x + self.l2(nn.gelu(self.l1(self.n2(x))))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(64, D)
            self.blocks = [Block() for _ in range(LAYERS)]
            self.norm = nn.LayerNorm(D)
            self.pi, self.v = nn.Linear(D, ACTIONS), nn.Linear(D, 1)

        def __call__(self, x):
            h = self.inp(x)
            for b in self.blocks:
                h = b(h)
            g = self.norm(h).mean(axis=1)
            return self.pi(g), self.v(g).squeeze(-1)

    mx.random.seed(0)
    net = Net()
    dt_ = mx.bfloat16 if dtype == "bfloat16" else mx.float32
    if dtype == "bfloat16":
        net.set_dtype(mx.bfloat16)
    x = mx.random.normal((BATCH, TOK, 64)).astype(dt_)
    a = mx.random.randint(0, ACTIONS, (BATCH,))
    ret = mx.random.normal((BATCH,))
    opt = optim.Adam(learning_rate=1e-4)

    def loss_fn(model, x, a, ret):
        lg, v = model(x)
        return nn.losses.cross_entropy(lg.astype(mx.float32), a, reduction="mean") + ((v.astype(mx.float32) - ret) ** 2).mean()

    lg_fn = nn.value_and_grad(net, loss_fn)

    def step():
        loss, grads = lg_fn(net, x, a, ret)
        opt.update(net, grads)
        mx.eval(net.parameters(), opt.state, loss)
        return loss

    for _ in range(warm):
        step()
    t0 = time.time()
    for _ in range(steps):
        loss = step()
    dt = (time.time() - t0) / steps
    lv = float(loss.item())
    return {"ok": lv == lv and abs(lv) < 1e6, "ms": round(dt * 1000, 1), "sps": round(BATCH / dt)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warm", type=int, default=5)
    a = ap.parse_args(argv)
    if sys.platform != "darwin":
        print("MLX 只能在 Apple 芯片的 Mac 上跑，这台不是。", flush=True)
        return 0
    try:
        import mlx.core  # noqa: F401
    except ImportError:
        print("装 MLX（pip install mlx）……", flush=True)
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "mlx"], capture_output=True, text=True)
        if r.returncode != 0:
            print("MLX 装不上：" + (r.stderr or r.stdout).strip().splitlines()[-1], flush=True)
            return 1
    print(f"网络：d={D}、{LAYERS} 层、{HEADS} 头、{TOK} 个词，每份 {BATCH} 个样本；每种 {a.steps} 步（先预热 {a.warm} 步）", flush=True)
    res = {}
    for name, fn, dtype in (("PyTorch MPS float32", bench_torch, "float32"), ("PyTorch MPS bfloat16", bench_torch, "bfloat16"),
                            ("MLX float32", bench_mlx, "float32"), ("MLX bfloat16", bench_mlx, "bfloat16")):
        try:
            r = fn(dtype, a.steps, a.warm)
        except Exception as ex:                          # noqa: BLE001
            r = {"ok": False, "error": f"{type(ex).__name__}: {str(ex)[:160]}"}
        res[name] = r
        print(f"  {name:22s} " + (f"{r['ms']:.0f} ms / 步，{r['sps']} 样本/秒" if r.get("ok") else "失败：" + str(r.get("error", "结果不对"))),
              flush=True)
    best = lambda k: max((r["sps"] for n, r in res.items() if n.startswith(k) and r.get("ok")), default=0)   # noqa: E731
    bt, bm = best("PyTorch"), best("MLX")
    ratio = round(bm / bt, 2) if bt and bm else None
    if ratio is None:
        verdict = "有一边跑不起来，没法比"
    elif ratio >= 1.5:
        verdict = f"MLX 快 {ratio} 倍：值得把项目换成 MLX"
    elif ratio >= 1.3:
        verdict = f"MLX 快 {ratio} 倍：有提升但不大，换不换两可"
    else:
        verdict = f"MLX 是 PyTorch 的 {ratio} 倍：不值得换"
    print("结论：" + verdict, flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "results": res, "ratio": ratio, "verdict": verdict},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
