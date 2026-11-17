"""模型测速（python bg.py bench-model）：训练更新为什么慢——把模型的几个部件逐个关掉，各测“前向 + 反向”要多久。

分段计时（--profile-update）显示每轮更新约 42 秒里，反向 21 秒、前向 11 秒、老师 9 秒；
每个样本前向 + 反向约 0.94 毫秒，同宽同层的标准 Transformer 在这台 Mac 上约 0.14 毫秒。这个脚本找多出来的时间在哪：

  1. 用 runs/v3/latest.pt 和机器人打一会儿，攒一批真实局面（各个回合都有）。
  2. 每种设置跑若干次“前向 + 损失 + 反向”（和训练同样的输入格式、同样的混合精度），前几次预热不算：
       完整（现在训练用的）、不打包空词（pack 关）、不要体系 / 联动感知、不要 311 维卡牌属性、全精度（不用 bfloat16）、
       只算前向、不过 Transformer（只剩输入和输出头）、只算 Transformer（带 / 不带屏蔽），还有 micro 256 / 1024 两种批大小。
  3. 老师（data/external/model.pt，有的话）只算前向。
结果按“每个样本几微秒”和“换算成一轮 3.4 万样本要几秒”打印，写进 reports/bench_model.json。
只读模型，不改训练、不写 runs/。约 2–4 分钟。训练正在跑时别测（抢同一块显卡，数不准）。
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.8")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.6")

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "reports" / "bench_model.json"


def collect(model, dev, n_rows: int, games: int, threads: int, seed: int):
    """学习者（每局 0 号座位）由模型按概率出动作、其余座位脚本机器人，攒 n_rows 个局面（各个回合都有）。"""
    import numpy as np
    import torch
    from taverngym import BatchEnv
    env = BatchEnv(games, seed=seed, controlled=(0,), threads=threads)
    obs = env.observe()
    F_, C_, M_, A_ = [], [], [], []
    rng = np.random.default_rng(seed)
    keep_p = 0.35                                      # 每步只留一部分，局面在回合上分散一点
    for _ in range(4000):
        ready = np.flatnonzero(obs["ready"])
        actions = np.zeros(env.n_slots, dtype=np.int64)
        if len(ready):
            f, c, m = obs["features"][ready], obs["cards"][ready], obs["action_mask"][ready]
            with torch.inference_mode():
                lg = model(torch.as_tensor(f).to(dev), torch.as_tensor(c, dtype=torch.long).to(dev),
                           torch.as_tensor(m).to(dev))[0].float()
                a = torch.distributions.Categorical(logits=lg).sample().cpu().numpy()
            actions[ready] = a
            pick = rng.random(len(ready)) < keep_p
            F_.append(f[pick])
            C_.append(c[pick])
            M_.append(m[pick])
            A_.append(a[pick])
        env.step(actions)
        obs = env.observe()
        if sum(len(x) for x in F_) >= n_rows * 3:
            break
    f, c, m, a = (np.concatenate(x) for x in (F_, C_, M_, A_))
    sel = rng.permutation(len(f))[:n_rows]
    turns = np.rint(f[sel, 0] * 20).astype(int)
    return f[sel], c[sel], m[sel], a[sel], turns


def make_variant(model, name: str):
    """在同一个模型上临时改一处，返回还原函数。"""
    import torch
    undo = []
    if name == "nopack":
        old = model.pack
        model.pack = False
        undo.append(lambda: setattr(model, "pack", old))
    elif name == "nosense":
        L, d = model.L, model.d
        ne = L.NB + L.NH + L.NS + L.NC

        def zero_sense(f, c):
            B = c.shape[0]
            z = torch.zeros(B, 1, d, device=c.device)
            return z, torch.zeros(B, ne, d, device=c.device)
        model.sense_input = zero_sense
        undo.append(lambda: delattr(model, "sense_input"))
    elif name == "noenc":                               # 不过 Transformer：只剩输入（嵌入、感知、对手随从）和输出头
        enc = model.enc

        class Skip(torch.nn.Module):
            def forward(self, x, src_key_padding_mask=None):
                return x
        model.enc = Skip()
        undo.append(lambda: setattr(model, "enc", enc))
    elif name == "noattr":
        tab, w = model.card_attr, model.attr_proj
        model.set_card_attr(None, k=0)

        def back():
            model.register_buffer("card_attr", tab, persistent=False)
            model.attr_proj = w
        undo.append(back)
    return lambda: [u() for u in reversed(undo)]


def time_steps(model, dev, data, micro: int, amp: bool, backward: bool, reps: int, warm: int) -> dict:
    import torch
    import torch.nn.functional as F

    from rl.model import soft_log
    f, c, m, a = data
    n = len(f)
    sync = torch.mps.synchronize if dev.type == "mps" else torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
    model.train(backward)
    times = []
    for r in range(warm + reps):
        s = (r * micro) % max(1, n - micro)
        j = slice(s, s + micro)
        ft = soft_log(torch.as_tensor(f[j]).to(dev).float())
        ct = torch.as_tensor(c[j], dtype=torch.long).to(dev)
        mt = torch.as_tensor(m[j]).to(dev)
        at = torch.as_tensor(a[j], dtype=torch.long).to(dev)
        sync()
        t0 = time.perf_counter()
        if backward:
            model.zero_grad(set_to_none=True)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
                lg, v, pl = model(ft, ct, mt, compressed=True)
            lp = F.log_softmax(lg.float(), -1)
            loss = -lp.gather(1, at.unsqueeze(1)).mean() + v.float().pow(2).mean() + \
                F.cross_entropy(pl.float(), torch.zeros(len(at), dtype=torch.long, device=dev))
            if model.last_comp_logits is not None:
                loss = loss + F.cross_entropy(model.last_comp_logits, torch.zeros(len(at), dtype=torch.long, device=dev))
            loss.backward()
        else:
            with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
                model(ft, ct, mt, compressed=True)
        sync()
        if r >= warm:
            times.append(time.perf_counter() - t0)
    times.sort()
    med = times[len(times) // 2]
    return {"ms_per_micro": round(med * 1000, 1), "us_per_sample": round(med / micro * 1e6, 1)}


def time_encoder(model, dev, data, micro: int, amp: bool, masked: bool, reps: int, warm: int) -> tuple[dict, int]:
    """只算 Transformer 那几层（前向 + 反向）：输入是这批真实局面打包以后的词（同样的词数、同样的屏蔽）；
    masked=False：同样的词数但不屏蔽（看屏蔽本身贵不贵）。"""
    import torch

    from rl.model import soft_log
    f, c, m, _ = data
    sync = torch.mps.synchronize if dev.type == "mps" else torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
    model.train(True)
    times, tms = [], []
    for r in range(warm + reps):
        s = (r * micro) % max(1, len(f) - micro)
        j = slice(s, s + micro)
        with torch.no_grad():
            ft = soft_log(torch.as_tensor(f[j]).to(dev).float())
            x, present = model._tokens(ft, torch.as_tensor(c[j], dtype=torch.long).to(dev))
            order = torch.argsort((~present).to(torch.int32), dim=1, stable=True)
            tm = int(present.sum(1).max())
            tm = min(x.shape[1], -(-tm // 8) * 8)
            idx = order[:, :tm]
            xp = torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1])).detach()
            pp = torch.gather(present, 1, idx)
        xp.requires_grad_(True)
        tms.append(tm)
        sync()
        t0 = time.perf_counter()
        model.zero_grad(set_to_none=True)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
            h = model.enc(xp, src_key_padding_mask=~pp if masked else None)
        h.float().pow(2).mean().backward()
        sync()
        if r >= warm:
            times.append(time.perf_counter() - t0)
    times.sort()
    med = times[len(times) // 2]
    tms.sort()
    return {"ms_per_micro": round(med * 1000, 1), "us_per_sample": round(med / micro * 1e6, 1)}, tms[len(tms) // 2]


def time_pass(model, dev, data, micro: int, amp: bool, sort: bool, reps: int) -> dict:
    """整批（这里是全部局面）按 micro 切开，前向 + 反向走一遍的总时间 / 样本数；sort：先按词数排序（训练的 bucket_tokens）。"""
    import numpy as np
    import torch
    import torch.nn.functional as F

    from rl.model import soft_log
    f, c, m, a = data
    order = np.argsort(model.token_counts(c), kind="stable") if sort else np.arange(len(f))
    sync = torch.mps.synchronize if dev.type == "mps" else torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
    model.train(True)
    best = None
    for _ in range(reps + 1):
        sync()
        t0 = time.perf_counter()
        for u in range(0, len(f), micro):
            j = order[u:u + micro]
            ft = soft_log(torch.as_tensor(f[j]).to(dev).float())
            ct = torch.as_tensor(c[j], dtype=torch.long).to(dev)
            mt = torch.as_tensor(m[j]).to(dev)
            at = torch.as_tensor(a[j], dtype=torch.long).to(dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
                lg, v, pl = model(ft, ct, mt, compressed=True)
            lp = F.log_softmax(lg.float(), -1)
            loss = -lp.gather(1, at.unsqueeze(1)).mean() + v.float().pow(2).mean()
            loss.backward()
        sync()
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
        model.zero_grad(set_to_none=True)
    return {"us_per_sample": round(best / len(f) * 1e6, 1)}


def time_teacher(teacher, dev, data, micro: int, amp: bool, reps: int, warm: int) -> dict:
    import torch
    f, c, m, _ = data
    sync = torch.mps.synchronize if dev.type == "mps" else torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
    times = []
    for r in range(warm + reps):
        s = (r * micro) % max(1, len(f) - micro)
        j = slice(s, s + micro)
        ft = torch.as_tensor(f[j]).to(dev).float()
        ct = torch.as_tensor(c[j], dtype=torch.long).to(dev)
        mt = torch.as_tensor(m[j]).to(dev)
        sync()
        t0 = time.perf_counter()
        with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
            teacher(ft, ct, mt)
        sync()
        if r >= warm:
            times.append(time.perf_counter() - t0)
    times.sort()
    med = times[len(times) // 2]
    return {"ms_per_micro": round(med * 1000, 1), "us_per_sample": round(med / micro * 1e6, 1)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="runs/v3/latest.pt")
    ap.add_argument("--teacher", default="data/external/model.pt")
    ap.add_argument("--device", default="")
    ap.add_argument("--rows", type=int, default=4096, help="攒多少个局面")
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--per-iter", type=int, default=34000, help="换算成一轮多少个样本")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="CI：随机小模型、少量局面，只检查能跑通")
    a = ap.parse_args(argv)
    import numpy as np
    import torch

    from rl.device import default_threads, resolve
    from rl.model import build_model, load_model
    dev = torch.device(a.device or resolve("auto", "auto")[0])
    threads = a.threads or default_threads()
    if a.smoke:
        model = build_model(64, 2, 2, 32).to(dev).eval()
        a.rows, a.reps, a.warm = 256, 2, 1
        info = {"iter": None}
    else:
        ck = ROOT / a.ckpt
        if not ck.exists():
            print(f"没有 {ck}", flush=True)
            return 1
        model, info = load_model(ck, str(dev))
    amp = dev.type == "mps"
    n_par = sum(p.numel() for p in model.parameters())
    print(f"设备 {dev}，模型 {n_par / 1e6:.2f}M 参数（{info.get('iter')} 轮），攒 {a.rows} 个局面……", flush=True)
    t0 = time.time()
    data = collect(model, dev, a.rows, games=64 if not a.smoke else 8, threads=threads, seed=7)
    turns = data[4]
    data = data[:4]
    print(f"  {len(data[0])} 个局面，回合 {int(turns.min())}–{int(turns.max())}（中位 {int(np.median(turns))}），"
          f"{time.time() - t0:.0f} 秒", flush=True)
    micro = 512 if not a.smoke else 64
    rows = [
        ("完整（现在训练用的）", "full", micro, amp, True),
        ("只算前向", "full", micro, amp, False),
        ("不打包空词", "nopack", micro, amp, True),
        ("不要体系 / 联动感知", "nosense", micro, amp, True),
        ("不要 311 维卡牌属性", "noattr", micro, amp, True),
        ("全精度（不用 bfloat16）", "full", micro, False, True),
        ("不过 Transformer（只剩输入和输出头）", "noenc", micro, amp, True),
        (f"micro {micro // 2}", "full", micro // 2, amp, True),
        (f"micro {micro * 2}", "full", micro * 2, amp, True),
    ]
    res = {}
    base = None
    print(f"{'设置':<22} {'毫秒/份':>8} {'微秒/样本':>9} {'一轮(秒)':>8} {'比完整':>7}", flush=True)
    for label, var, mic, amp_, bwd in rows:
        undo = make_variant(model, var)
        try:
            r = time_steps(model, dev, data, mic, amp_, bwd, a.reps, a.warm)
        except Exception as ex:                          # noqa: BLE001
            r = {"error": f"{type(ex).__name__}: {str(ex)[:160]}"}
        finally:
            undo()
            if dev.type == "mps":
                torch.mps.empty_cache()
        res[label] = r
        if "error" in r:
            print(f"{label:<22} 失败：{r['error']}", flush=True)
            continue
        per_iter = r["us_per_sample"] * a.per_iter / 1e6
        r["sec_per_iter"] = round(per_iter, 1)
        if base is None:
            base = r["us_per_sample"]
        r["vs_full"] = round(r["us_per_sample"] / base, 2)
        print(f"{label:<22} {r['ms_per_micro']:>8} {r['us_per_sample']:>9} {per_iter:>8.1f} {r['vs_full']:>6}×", flush=True)
    for masked in (True, False):                       # 只算 Transformer 那几层
        label = "只算 Transformer" + ("（带屏蔽）" if masked else "（不屏蔽）")
        try:
            r, tm = time_encoder(model, dev, data, micro, amp, masked, a.reps, a.warm)
            r["tokens"] = tm
            r["sec_per_iter"] = round(r["us_per_sample"] * a.per_iter / 1e6, 1)
            r["vs_full"] = round(r["us_per_sample"] / base, 2) if base else None
            res[label] = r
            print(f"{label:<22} {r['ms_per_micro']:>8} {r['us_per_sample']:>9} {r['sec_per_iter']:>8.1f} {r['vs_full']:>6}×"
                  f"（每行 {tm} 个词）", flush=True)
        except Exception as ex:                          # noqa: BLE001
            print(f"{label:<22} 失败：{type(ex).__name__}: {str(ex)[:160]}", flush=True)
        finally:
            if dev.type == "mps":
                torch.mps.empty_cache()
    pa = {}                                            # 整批走一遍：不排序（原来）/ 按词数排序（bucket_tokens）
    for sort in (False, True):
        label = "整批一遍（" + ("按词数排序" if sort else "不排序") + "）"
        try:
            r = time_pass(model, dev, data, micro, amp, sort, max(1, a.reps // 4))
            r["sec_per_iter"] = round(r["us_per_sample"] * a.per_iter / 1e6, 1)
            res[label] = pa[sort] = r
            print(f"{label:<22} {'':>8} {r['us_per_sample']:>9} {r['sec_per_iter']:>8.1f}", flush=True)
        except Exception as ex:                          # noqa: BLE001
            print(f"{label:<22} 失败：{type(ex).__name__}: {str(ex)[:160]}", flush=True)
    if len(pa) == 2:
        tc = model.token_counts(data[1])
        print(f"  按词数排序：快 {1 - pa[True]['us_per_sample'] / pa[False]['us_per_sample']:.0%}"
              f"（每行词数 {int(tc.min())}–{int(tc.max())}，中位 {int(np.median(tc))}）", flush=True)
    tp = ROOT / a.teacher
    if not a.smoke and tp.exists():
        try:
            teacher, _ = load_model(tp, str(dev))
            for amp_ in (amp, False) if amp else (False,):
                r = time_teacher(teacher, dev, data, micro, amp_, a.reps, a.warm)
                r["sec_per_iter_half"] = round(r["us_per_sample"] * a.per_iter / 2 / 1e6, 1)
                label = "老师只算前向" + ("（bfloat16）" if amp_ else "（全精度）")
                res[label] = r
                print(f"{label:<22} {r['ms_per_micro']:>8} {r['us_per_sample']:>9} {r['sec_per_iter_half']:>8.1f}"
                      "（一半局面）", flush=True)
        except Exception as ex:                          # noqa: BLE001
            print(f"老师测不了：{type(ex).__name__}: {str(ex)[:160]}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "device": str(dev), "params_m": round(n_par / 1e6, 2),
                               "iter": info.get("iter"), "rows": len(data[0]), "results": res}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"→ {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
