"""老师模型：只拿高分示范（runs/pro_demos 6500+ 回放 + runs/h13_demos 万三对局）训练一个模仿模型，
按留出局的交叉熵挑最好的那一版，存成 runs/teacher.pt。训练时（configs/v3.json 的 teacher / teacher_coef）
在模型自己打出来的局面上加 KL(老师 ‖ 模型)：告诉它“高手在这个局面会怎么做”。

  bg.py teacher                          从头学（默认），约 15–25 分钟（显卡）
  bg.py teacher --init runs/v3/latest.pt 从训练中的模型开始学（不推荐：见下）
  bg.py teacher --steps 3000 --batch 256 学多少步、每步多少条

为什么要老师（rl/ppo.py 的 teacher 注释里有数）：训练里直接模仿高手（pro_coef）只能在回放里的那些局面上学，
8.4M 参数背 10 万条，第 4450 轮以后训练示范命中率涨到 0.7–0.8、留出局一直 0.37–0.42——背题；
老师先在留出局最好的时候停下来，再到模型自己的局面上去教，背不了题。
老师只用在训练里，不进 Git、不上传（runs/ 不在 Git 里）。

第一次（2026-11-03）从 runs/v3/latest.pt（第 14140 轮）开始学：一开始留出局 0.405 / 交叉熵 2.45 就是最好，
再学只会更差——它已经把训练示范背下来了（命中率 0.70），接着学只是背得更熟。所以默认从头学；
最好的一版如果就是开始时那一版（什么都没学到），不存。
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paths import ROOT, RUNS  # noqa: E402
from rl.imitation import ImitationMixin  # noqa: E402

DEFAULT_RECIPE = ROOT / "configs" / "v3.json"


class Teach(ImitationMixin):
    """只用 ImitationMixin 的读示范、加权、损失、命中率（没有环境）。"""

    def __init__(self, cfg, model, dev):
        self.cfg, self.model, self.dev = cfg, model, dev
        self.amp = dev.type == "cuda"
        self.bc_rng = np.random.default_rng(cfg.seed + 11)

    def _demo_format_ok(self, d) -> bool:
        return d["m"].shape[1] == self.model.L.actions


def _path(p: str) -> Path:
    q = Path(str(p).replace("\\", "/"))
    return q if q.exists() or q.is_absolute() else ROOT / q


def build(init: str, size: str, dev):
    from rl.model import load_model
    from rl.model import SIZES, build_model
    if init and init.lower() != "none":
        p = _path(init)
        if p.exists():
            m, info = load_model(p, str(dev))
            print(f"从 {p}（第 {info.get('iter')} 轮）的参数开始", flush=True)
            return m.train(), info["size"]
        print(f"没有 {p}，从头学", flush=True)
    kw = dict(SIZES[size])
    return build_model(**kw).to(dev).train(), kw


def main(argv=None) -> int:
    from tools.train import load_recipe
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", default=str(DEFAULT_RECIPE), help="读 pro_data、pro_weights、模型大小（默认 configs/v3.json）")
    ap.add_argument("--init", default="none", help="从这个检查点开始（默认 none = 从头学）")
    ap.add_argument("--out", default=str(RUNS / "teacher.pt"))
    ap.add_argument("--pro-data", default="", help="默认用配方里的 pro_data")
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--patience", type=int, default=6, help="留出局交叉熵连续几次不再变好就停")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    rc = load_recipe(a.recipe).get("config") or {}
    data = a.pro_data or rc.get("pro_data") or str(RUNS / "pro_demos")
    data = ";".join(str(_path(p)) for p in data.split(";") if p.strip())
    cfg = SimpleNamespace(pro_data=data, pro_coef=1.0, pro_weights=rc.get("pro_weights", ""), seed=a.seed,
                          bc_smooth=rc.get("bc_smooth", 0.1), demo_shop_shuffle=rc.get("demo_shop_shuffle", True),
                          micro_batch=512)
    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    model, kw = build(a.init, rc.get("size", "medium"), dev)
    t = Teach(cfg, model, dev)
    d = t._load_pro()
    if d is None:
        print("没有高分决策样本（先运行 prodemos.bat）", flush=True)
        return 1
    w = t._bc_weights(d, cfg.pro_weights)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, eps=1e-5)

    def show(step, pe, mark=""):
        kinds = "、".join(f"{k} {pe[f'pro_{k}']:.2f}" for k in ("upgrade", "end", "buy", "refresh", "sell", "choose")
                         if f"pro_{k}" in pe)
        print(f"  第 {step} 步：留出局命中率 {pe.get('pro_top1')}（训练示范 {pe.get('pro_top1_train')}），"
              f"交叉熵 {pe.get('pro_nll')}；{kinds}{mark}", flush=True)

    pe = t.demo_eval(d, "pro")
    show(0, pe, "（开始）")
    best = (pe["pro_nll"], 0, copy.deepcopy(model.state_dict()), pe)
    bad, t0, losses = 0, time.time(), []
    for step in range(1, a.steps + 1):
        loss = t.bc_loss(d, w, a.batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if step % a.eval_every and step != a.steps:
            continue
        pe = t.demo_eval(d, "pro")
        better = pe["pro_nll"] < best[0] - 1e-3
        el = time.time() - t0
        show(step, pe, f"；训练损失 {np.mean(losses):.3f}，{el / 60:.1f} 分钟" + ("，目前最好" if better else ""))
        losses = []
        if better:
            best, bad = (pe["pro_nll"], step, copy.deepcopy(model.state_dict()), pe), 0
        else:
            bad += 1
            if bad >= a.patience:
                print(f"  连续 {bad} 次没变好，停", flush=True)
                break
    nll, step, sd, pe = best
    if step == 0:
        print(f"最好的一版就是开始时那一版（留出局交叉熵 {nll}），等于什么都没学到，不存老师模型。"
              f"{'从头学试试：teacher.bat --init none' if str(a.init).lower() != 'none' else '把这段输出发给 Claude'}", flush=True)
        return 1
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": sd, "model_kwargs": kw, "iter": None,
                "teacher": {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "init": a.init, "data": data, "step": step,
                            "val": {k: v for k, v in pe.items() if k.startswith("pro_")}}}, out)
    print(f"老师模型 → {out}：第 {step} 步那一版，留出局命中率 {pe.get('pro_top1')}、交叉熵 {nll}"
          f"（选择 {pe.get('pro_choose')}、买 {pe.get('pro_buy')}、升本 {pe.get('pro_upgrade')}）", flush=True)
    print("接着照常运行 train.bat：配方里 teacher_coef > 0 时会自动用它（启动时会打印“老师模型：…”）", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
