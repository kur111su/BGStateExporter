"""模仿学习：你的对局示范（bc_*）和高分玩家示范（pro_*）的读取、加权、损失和命中率。

Trainer（rl/ppo.py）继承 ImitationMixin。这里用到 Trainer 的：cfg、model、dev、amp、obs、envs（检查示范格式）、
bc_rng、bc_scale / pro_scale（存在检查点里）。_init_imitation() 在 Trainer.__init__ 里调用一次。
数据格式和类别见 rl/demos.py。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .demos import KINDS, SHARD_KEYS, bought_in, load as load_demos, load_dir, permute_shop


class ImitationMixin:
    def _init_imitation(self):
        cfg = self.cfg
        self.demos = self._load_demos()
        self.bc_rng = np.random.default_rng(cfg.seed + 7)
        self.bc_w = self._bc_weights() if self.demos is not None else None
        self.pro = self._load_pro()
        self.pro_w = self._bc_weights(self.pro, cfg.pro_weights) if self.pro is not None else None
        self.bc_scale = 1.0                            # 模仿强度的倍数（见 bc_train_cap；存在检查点里）
        self.pro_scale = 1.0                           # 模仿高手强度的倍数（见 pro_gap_cap；存在检查点里）

    def _load_demos(self):
        from paths import RUNS
        path = Path(self.cfg.bc_data) if self.cfg.bc_data else RUNS / "demos.npz"
        if not path.exists():
            if self.cfg.bc_coef > 0:
                print(f"没有示范数据 {path}（先运行 bg.py demos），这次不模仿", flush=True)
            return None
        try:
            d = load_demos(path)
        except Exception as ex:                        # noqa: BLE001
            print(f"示范数据 {path} 读不了（{type(ex).__name__}: {ex}），这次不模仿", flush=True)
            return None
        if d["m"].shape[1] != self.model.L.actions or d["f"].shape[1:] != self.obs[self.envs[0][0]]["features"].shape[1:]:
            print(f"示范数据 {path} 和当前环境的格式不一样（TavernGym 换过版本？），重新运行 bg.py demos", flush=True)
            return None
        d["train_idx"] = np.flatnonzero(d["split"] == 0)
        d["val_idx"] = np.flatnonzero(d["split"] == 1)
        print(f"示范数据：{d['meta'].get('games')} 局，训练 {len(d['train_idx'])} 条、验证 {len(d['val_idx'])} 条"
              f"（{d['meta'].get('created')}）；模仿系数 {self.cfg.bc_coef}", flush=True)
        return d

    def _load_pro(self):
        from paths import RUNS
        # 可以写几个目录，用分号隔开（例如 "runs\\pro_demos;runs\\h13_demos"），合在一起用
        paths = [Path(p) for p in str(self.cfg.pro_data).split(";") if p.strip()] or [RUNS / "pro_demos"]
        parts = []
        for path in paths:
            if not path.exists():
                if self.cfg.pro_coef > 0:
                    print(f"没有高分决策样本 {path}（先运行 bg.py prodemos / nativedemos），跳过", flush=True)
                continue
            try:
                x = load_dir(path)
            except Exception as ex:                    # noqa: BLE001
                print(f"高分决策样本 {path} 读不了（{type(ex).__name__}: {ex}）", flush=True)
                continue
            if x is not None:
                print(f"  {path}：{len(x['kind'])} 条（{x['meta'].get('source', '')}）", flush=True)
                parts.append(x)
        if not parts:
            return None
        d = parts[0]
        if len(parts) > 1:
            d = {k: np.concatenate([p[k] for p in parts]) for k in SHARD_KEYS}
            d["compressed"] = True
            d["meta"] = {"games": sum(int(p["meta"].get("games") or 0) for p in parts)}
        path = ";".join(str(p) for p in paths)
        if not self._demo_format_ok(d):
            print(f"高分决策样本 {path} 和当前环境的格式不一样，重新运行 bg.py prodemos", flush=True)
            return None
        d["train_idx"] = np.flatnonzero(d["split"] == 0)
        d["val_idx"] = np.flatnonzero(d["split"] == 1)
        print(f"高分决策样本：{d['meta'].get('games')} 局，训练 {len(d['train_idx'])} 条、验证 {len(d['val_idx'])} 条；"
              f"模仿系数 {self.cfg.pro_coef}", flush=True)
        return d

    def _demo_format_ok(self, d) -> bool:
        """示范的动作数、观察长度和当前环境一样（老师训练 tools/teacher.py 没有环境，只比动作数）。"""
        return d["m"].shape[1] == self.model.L.actions and d["f"].shape[1:] == self.obs[self.envs[0][0]]["features"].shape[1:]

    def _demo_tensors(self, j, d=None, shuffle: bool = False):
        """示范第 j 条（数组）→ 张量。shuffle：酒馆顺序随机打乱（demo_shop_shuffle，只在训练时用，考试不打乱）。"""
        d = self.demos if d is None else d
        f, c, m, cand = d["f"][j], d["c"][j], d["m"][j], d["cand"][j]
        if shuffle:
            f, c, m, cand = permute_shop(f, c, m, cand, self.bc_rng, self.model.L.fslices()["shop"][0], self.model.L.IF,
                                         self.model.L.NS, rows=d["kind"][j] != KINDS.index("choose"))
        t = lambda x, dt: torch.as_tensor(x, dtype=dt).to(self.dev, non_blocking=True)  # noqa: E731
        return (t(f, torch.float32), t(c, torch.long), t(m, torch.bool), t(cand.astype(np.int64), torch.long))

    def _demo_logp(self, f, c, m, compressed=False):
        with torch.autocast(self.dev.type, dtype=torch.bfloat16, enabled=self.amp):
            logits = self.model(f, c, m, compressed=compressed)[0]
        return F.log_softmax(logits.float(), -1)

    def _bc_weights(self, d=None, spec=None):
        d = self.demos if d is None else d
        spec = self.cfg.bc_weights if spec is None else spec
        w, tw = {}, []
        for part in str(spec or "").replace(" ", "").split(","):
            if ":" in part:
                k, v = part.split(":", 1)
                if k in KINDS:
                    w[KINDS.index(k)] = float(v)
                elif k in ("key", "eco", "tribe", "syn", "simkey", "simsyn", "engine", "engcard", "triple", "tgtS", "tgtA", "tgtB", "tgtfit", "tgtsup", "spelltgt"):     # 买关键卡 / 经济卡 / 同种族 / 常见搭配的示范再乘（见 _bought_in）
                    tw.append((k, None, float(v)))
                elif k[:1] == "t" and "turn" in d:              # 按回合再乘：t2:3 第 2 回合 ×3，t3-6:2 第 3–6 回合 ×2
                    lo, _, hi = k[1:].partition("-")
                    tw.append((int(lo), int(hi or lo), float(v)))
        out = np.ones(len(d["kind"]), dtype=np.float32)
        for k, v in w.items():
            out[d["kind"] == k] = v
        for lo, hi, v in tw:
            if lo in ("key", "eco", "tribe", "syn", "simkey", "simsyn", "engine", "engcard", "triple", "tgtS", "tgtA", "tgtB", "tgtfit", "tgtsup", "spelltgt"):
                sel = bought_in(d, lo)
                if sel is not None:
                    out[sel] *= v
                continue
            out[(d["turn"] >= lo) & (d["turn"] <= hi)] *= v
        return out

    _bought_in = staticmethod(bought_in)

    def bc_loss(self, d=None, wts=None, batch=None):
        d = self.demos if d is None else d
        wts = self.bc_w if wts is None else wts
        batch = batch or self.cfg.bc_batch
        j = self.bc_rng.choice(d["train_idx"], size=min(batch, len(d["train_idx"])))
        f, c, m, cand = self._demo_tensors(j, d, shuffle=self.cfg.demo_shop_shuffle)
        lp_all = self._demo_logp(f, c, m, bool(d.get("compressed")))
        lp = lp_all.gather(1, cand.clamp(min=0)).masked_fill(cand < 0, float("-inf"))
        nll = -torch.logsumexp(lp, 1)                  # 候选有几个（磁力吸附目标不确定）时，任一个都算对
        eps = self.cfg.bc_smooth
        if eps > 0:                                    # 标签平滑：合法动作上的平均对数概率
            uni = -(lp_all.masked_fill(~m, 0).sum(-1) / m.sum(-1).clamp(min=1))
            nll = (1 - eps) * nll + eps * uni
        w = torch.as_tensor(wts[j], device=nll.device)
        return (nll * w).sum() / w.sum()

    def adjust_bc(self, he: dict) -> None:
        """按训练示范命中率调模仿强度（考试时调用）。"""
        cap = self.cfg.bc_train_cap
        tr = he.get("human_top1_train")
        if cap <= 0 or tr is None:
            return
        if tr > cap:
            self.bc_scale = max(self.cfg.bc_scale_min, self.bc_scale * 0.5)
        elif tr < cap - 0.1:
            self.bc_scale = min(1.0, self.bc_scale * 1.5)

    def adjust_pro(self, pe: dict) -> None:
        """按训练示范和留出局命中率的差距调模仿高手的强度（考试时调用）。"""
        cap = self.cfg.pro_gap_cap
        tr, va = pe.get("pro_top1_train"), pe.get("pro_top1")
        if cap <= 0 or tr is None or va is None:
            return
        if tr - va > cap:
            self.pro_scale = max(self.cfg.pro_scale_min, self.pro_scale * 0.7)
        elif tr - va < cap - 0.04:
            self.pro_scale = min(1.0, self.pro_scale * 1.25)

    @torch.no_grad()
    def human_eval(self, d=None, prefix: str = "human") -> dict:
        """留出的验证局上：模型最可能的动作是不是示范里的那个（{prefix}_top1），按类别的命中率，平均交叉熵。"""
        d = self.demos if d is None else d
        self.model.eval()
        out = {}
        val = d["val_idx"]
        if len(val) > 6000:                            # 高分样本的验证集很大：固定抽 6000 条（每次同一批）
            val = val[np.random.default_rng(0).permutation(len(val))[:6000]]
        # 训练示范也随机抽一批固定的（不能取前 2000 条：几个来源拼在一起时前面全是同一个来源，
        # 和验证集的组成不一样——第 5100–5750 轮万三数据加进来后，这个差距一直 > 0.1，模仿强度被一路减到 0.01）
        tr_idx = d["train_idx"]
        if len(tr_idx) > len(val):
            tr_idx = tr_idx[np.random.default_rng(1).permutation(len(tr_idx))[:len(val)]]
        for tag, idx in (("", val), ("_train", tr_idx)):
            if not len(idx):
                continue
            hit, nll = [], []
            for s in range(0, len(idx), self.cfg.micro_batch):
                j = idx[s:s + self.cfg.micro_batch]
                f, c, m, cand = self._demo_tensors(j, d)
                lp = self._demo_logp(f, c, m, bool(d.get("compressed")))
                top = lp.argmax(1, keepdim=True)
                hit.append(((cand == top) & (cand >= 0)).any(1).cpu().numpy())
                nll.append((-torch.logsumexp(lp.gather(1, cand.clamp(min=0)).masked_fill(cand < 0, float("-inf")), 1))
                           .clamp(max=20).cpu().numpy())
            hit, nll = np.concatenate(hit), np.concatenate(nll)
            out[f"{prefix}_top1" + tag] = round(float(hit.mean()), 3)
            if tag == "":
                out[f"{prefix}_nll"] = round(float(nll.mean()), 3)
                kind = d["kind"][idx]
                for k in ("upgrade", "end", "buy", "refresh", "sell", "choose"):
                    sel = kind == KINDS.index(k)
                    if sel.any():
                        out[f"{prefix}_{k}"] = round(float(hit[sel].mean()), 3)
                mm = d.get("mmr")
                if mm is not None:                     # 几个来源合在一起时（--pro-data "a;b"）：万分以上的那部分单独看
                    hi = mm[idx] >= 10000
                    if hi.any() and not hi.all():
                        out[f"{prefix}_top1_hi"] = round(float(hit[hi].mean()), 3)
                        sel = hi & (kind == KINDS.index("upgrade"))
                        if sel.any():
                            out[f"{prefix}_upgrade_hi"] = round(float(hit[sel].mean()), 3)
        self.model.train()
        return out
