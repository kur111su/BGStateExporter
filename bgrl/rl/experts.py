"""流派专家当老师（第二阶段，见 docs/comp_expert.md）：主模型训练时，在“专家擅长的局面”上向专家学。

专家 = 用 configs/exp/<流派>.json 单独练出来的检查点（例如 runs/beast/best.pt：龙虾流野兽）。
擅长的局面 = 三个条件都满足：
  1. 本局酒馆开了专家的种族（must），没开它训练时排除的种族（exclude）——和专家练的时候一样的世界；
  2. 回合 ≥ from_turn；
  3. 场上 / 手里 / 酒馆里有这套阵容的牌（engage，卡牌编号）——已经在做这套，或者酒馆里正好出了，
     要不要拿也问专家。engage 空 = 条件 3 不要求。
学法和老师的 hard 模式一样：专家最想做的动作概率 ≥ conf 的局面，加 coef × −log π(专家的动作)。
每个专家一行配置（Config.experts，JSON 列表）：
  [{"ckpt": "runs/beast/best.pt", "name": "龙虾流野兽", "must": "BEAST", "exclude": "QUILBOAR,DEMON,ELEMENTAL",
    "engage": "BG36_202,BG36_210,…", "from_turn": 3, "coef": 0.05, "conf": 0.4}]
外部模型（data/external/，CC BY-NC）不能当专家：只用自己练的检查点。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def parse_experts(spec) -> list[dict]:
    """Config.experts（JSON 字符串、JSON 文件路径或已经是列表）→ [专家配置]。"""
    if not spec:
        return []
    if isinstance(spec, list):
        items = spec
    else:
        s = str(spec).strip()
        if not s.startswith("["):
            s = Path(s).read_text(encoding="utf-8")
        items = json.loads(s)
    out = []
    for e in items:
        if not e.get("ckpt"):
            raise ValueError(f"专家缺 ckpt：{e}")
        out.append({"ckpt": str(e["ckpt"]), "name": str(e.get("name") or Path(e["ckpt"]).parent.name),
                    "must": str(e.get("must", "")), "exclude": str(e.get("exclude", "")),
                    "engage": str(e.get("engage", "")), "from_turn": int(e.get("from_turn", 3)),
                    "coef": float(e.get("coef", 0.05)), "conf": float(e.get("conf", 0.4))})
    return out


def _lobby_ok(f, must: str, exclude: str) -> np.ndarray:
    """[行]：酒馆开了 must 里所有种族、一个 exclude 都没开。"""
    from tg.obs import F_LOBBY, lobby_bits
    bits = lobby_bits(f)
    ok = np.ones(len(bits), dtype=bool)
    for t in [x.strip().upper() for x in must.split(",") if x.strip()]:
        ok &= (bits >> F_LOBBY.index(t)) & 1 == 1
    for t in [x.strip().upper() for x in exclude.split(",") if x.strip()]:
        ok &= (bits >> F_LOBBY.index(t)) & 1 == 0
    return ok


class ExpertMixin:
    def _load_experts(self):
        """读进专家（冻住）。读不了的跳过并说明，训练照常。"""
        from .model import load_model
        out = []
        for e in parse_experts(getattr(self.cfg, "experts", "")):
            p = Path(e["ckpt"].replace("\\", "/"))
            if not p.exists():
                print(f"专家 {e['name']}：没有 {p}，这次不用", flush=True)
                continue
            try:
                net, info = load_model(p, str(self.dev))
            except Exception as ex:                    # noqa: BLE001
                print(f"专家 {e['name']}：{p} 读不了（{type(ex).__name__}: {ex}），这次不用", flush=True)
                continue
            if info.get("external"):
                print(f"专家 {e['name']}：{p} 是外部模型（不能当专家，只用自己练的检查点），这次不用", flush=True)
                continue
            if net.L.actions != self.model.L.actions:
                print(f"专家 {e['name']}：{p} 的动作格式和现在的不一样，这次不用", flush=True)
                continue
            for q in net.parameters():
                q.requires_grad_(False)
            e = dict(e, net=net.eval(), engage_mask=None)
            if e["engage"]:
                from .collector import _ids_mask
                e["engage_mask"] = _ids_mask(e["engage"])
            print(f"专家 {e['name']}：{p}；酒馆必须有 {e['must'] or '-'}、不能有 {e['exclude'] or '-'}；"
                  f"{'场上 / 手里 / 酒馆有这套的牌、' if e['engage'] else ''}第 {e['from_turn']} 回合起；"
                  f"系数 {e['coef']}，有把握（≥ {e['conf']:.0%}）的局面才学", flush=True)
            out.append(e)
        return out

    def _expert_rows(self, e: dict, f, c) -> np.ndarray:
        """[行]：这个专家擅长的局面。"""
        from tg.obs import C_BOARD, C_HAND, C_SHOP, N_BOARD, N_HAND, N_SHOP
        f = np.asarray(f)
        ok = _lobby_ok(f, e["must"], e["exclude"])
        ok &= np.rint(f[:, 0].astype(np.float32) * 20) >= e["from_turn"]
        em = e["engage_mask"]
        if em is not None:
            c = np.clip(np.asarray(c).astype(np.int64), 0, len(em) - 1)
            cols = np.r_[C_BOARD:C_BOARD + N_BOARD, C_HAND:C_HAND + N_HAND, C_SHOP:C_SHOP + N_SHOP]
            ok &= em[c[:, cols]].any(1)
        return ok

    @torch.no_grad()
    def _expert_targets(self, b, n_actions: int):
        """每轮一次：每个专家在它擅长的局面上最想做的动作和概率 → [(专家, 动作 [n]（不学 = -1）, 概率 [n])]。"""
        out = []
        n = len(b["a"])
        ch = max(1, int(getattr(self.cfg, "teacher_chunk", 512)))
        for e in self.experts:
            ta = np.full(n, -1, dtype=np.int64)
            tp = np.zeros(n, dtype=np.float32)
            rows = np.flatnonzero(self._expert_rows(e, b["f"], b["c"]))
            for q in range(0, len(rows), ch):
                j = rows[q:q + ch]
                k = len(j)
                if k < ch:
                    j = np.concatenate([j, np.repeat(j[-1:], ch - k)])
                fx = torch.as_tensor(b["f"][j], dtype=torch.float32).to(self.dev)
                cx = torch.as_tensor(b["c"][j], dtype=torch.long).to(self.dev)
                mx = torch.as_tensor(np.unpackbits(b["m"][j], axis=1)[:, :n_actions].astype(bool)).to(self.dev)
                with torch.autocast(self.dev.type, dtype=torch.bfloat16, enabled=self.amp):
                    lg = e["net"](fx, cx, mx, compressed=True)[0].float()
                p, a = F.softmax(lg, -1).max(-1)
                ta[j[:k]] = a.cpu().numpy()[:k]
                tp[j[:k]] = p.cpu().numpy()[:k]
            out.append((e, ta, tp))
        if self.dev.type == "mps" and out:
            torch.mps.empty_cache()
        return out
