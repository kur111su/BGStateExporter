"""老师模型（训练时让学习者向它靠）：读老师、每轮先算好它的答案、系数怎么加上来。Trainer 的一部分（TeacherMixin）。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .shaping import ShapingMixin

TEACHER_FAMILIES = {"end": (0, 1), "refresh": (1, 2), "freeze": (2, 3), "upgrade": (3, 4), "buy": (8, 16), "sell": (16, 23),
                    "play": (46, 126)}



def teacher_action_mask(spec: str, n_actions: int):
    """"sell,buy" → [动作数] 的 0/1（这几类动作是 1）。空 → None（全学）。不认识的名字报错。"""
    names = [x.strip() for x in str(spec or "").split(",") if x.strip()]
    if not names:
        return None
    out = np.zeros(int(n_actions), dtype=np.float32)
    for nm in names:
        if nm not in TEACHER_FAMILIES:
            raise ValueError(f"teacher_actions 里不认识 {nm!r}，可选 {sorted(TEACHER_FAMILIES)}")
        a, b = TEACHER_FAMILIES[nm]
        out[a:b] = 1.0
    return out



class TeacherMixin:
    @torch.no_grad()
    def _teacher_targets(self, b, n_actions: int):
        """老师在这一轮每个局面上最想做的动作和它的概率（hard 模式用）：→ (动作 [n]，不学的局面是 -1；概率 [n])。
        只算要学的局面（teacher_from_turn 起），每块大小固定（最后一块补齐）——形状不变，MPS 不会越攒越多。
        teacher_actions 限定了动作类别的，老师最想做的不在里面就记 -1。"""
        cfg = self.cfg
        n = len(b["a"])
        ta = np.full(n, -1, dtype=np.int64)
        tp = np.zeros(n, dtype=np.float32)
        rows = np.arange(n)
        if getattr(cfg, "teacher_from_turn", 0) > 0:     # 第 0 个特征 = 回合 / 20（≤ 1，压缩不改它）
            turn = np.rint(np.asarray(b["f"][:, 0], dtype=np.float32) * 20)
            rows = rows[turn >= cfg.teacher_from_turn]
        ch = max(1, int(getattr(cfg, "teacher_chunk", 512)))
        want = teacher_action_mask(getattr(cfg, "teacher_actions", ""), n_actions)
        for q in range(0, len(rows), ch):
            j = rows[q:q + ch]
            k = len(j)
            if k < ch:
                j = np.concatenate([j, np.repeat(j[-1:], ch - k)])
            f = torch.as_tensor(b["f"][j], dtype=torch.float32).to(self.dev)
            c = torch.as_tensor(b["c"][j], dtype=torch.long).to(self.dev)
            m = torch.as_tensor(np.unpackbits(b["m"][j], axis=1)[:, :n_actions].astype(bool)).to(self.dev)
            with torch.autocast(self.dev.type, dtype=torch.bfloat16, enabled=self.amp):
                lg = self.teacher(f, c, m, compressed=True)[0].float()
            p, a = F.softmax(lg, -1).max(-1)
            ta[j[:k]] = a.cpu().numpy()[:k]
            tp[j[:k]] = p.cpu().numpy()[:k]
        if want is not None:
            ok = ta >= 0
            ta[ok & (want[np.clip(ta, 0, None)] <= 0)] = -1
        if self.dev.type == "mps":
            torch.mps.empty_cache()
        return ta, tp

    def _load_teacher(self):
        """老师模型（teacher / teacher_coef）：读进来、冻住。没有文件或格式不对就不用，训练照常。"""
        cfg = self.cfg
        if getattr(cfg, "teacher_coef", 0) <= 0 or not getattr(cfg, "teacher", ""):
            return None
        p = Path(str(cfg.teacher).replace("\\", "/"))
        if not p.exists():
            print(f"没有老师模型 {cfg.teacher}（先在面板“训练”里跑“老师模型”，约 15–25 分钟），这次不用", flush=True)
            return None
        from .model import load_model
        try:
            t, info = load_model(p, str(self.dev))
        except Exception as ex:                        # noqa: BLE001
            print(f"老师模型 {p} 读不了（{type(ex).__name__}: {ex}），这次不用", flush=True)
            return None
        if t.L.actions != self.model.L.actions:
            print(f"老师模型 {p} 和当前环境的格式不一样（TavernGym 换过版本？），重新跑一次面板里的“老师模型”", flush=True)
            return None
        for q in t.parameters():
            q.requires_grad_(False)
        if info.get("external"):                       # 外部模型（rl/extnet.py）：不 torch.load 它
            self.teacher_id = f"external:{p.name}:{int(info['mtime'])}"
            print(f"老师模型：外部模型 {p}（{info.get('license')}；只在本机用，不进 Git、不打包进竞技场）；"
                  f"系数 {cfg.teacher_coef}，前 {cfg.teacher_ramp} 轮慢慢加上，teacher_mode {cfg.teacher_mode}", flush=True)
            return t.eval()
        meta = torch.load(p, map_location="cpu", weights_only=False).get("teacher") or {}
        self.teacher_id = meta.get("created")
        val = meta.get("val") or {}
        print(f"老师模型：{p}（{meta.get('created', '?')}，留出局命中率 {val.get('pro_top1', '?')}、交叉熵 {val.get('pro_nll', '?')}）；"
              f"KL 系数 {cfg.teacher_coef}，前 {cfg.teacher_ramp} 轮慢慢加上", flush=True)
        return t.eval()

    def _teacher_rows(self, f, c):
        """每个局面的老师权重（体系选手）：先认出主选体系（S / A 里目标阵容得分最高、本局凑得成的那套）和 2 套备选；
        酒馆里有主选的核心卡 / 配套卡 → teacher_core_mult，有备选的 → (1 + teacher_core_mult) / 2，否则 1。
        还没有任何一套开头（前几回合）时不加权。不加权时返回 None。"""
        mult = float(getattr(self.cfg, "teacher_core_mult", 1.0))
        if mult == 1.0:
            return None
        if getattr(self, "_comp_tab", None) is None:
            from tg.catalog import base_of_index, n_cards

            from .comp import CompTables, load_target_comps
            self._comp_tab = CompTables(load_target_comps(), [base_of_index(i) if i else "" for i in range(n_cards())],
                                        tiers="SA")
        from tg.obs import lobby_bits
        tab = self._comp_tab
        main, backup = tab.main_backup(c, lobby_bits(f))
        hit_m = tab.shop_hit(c, main, tab.core) | tab.shop_hit(c, main, tab.supp)
        hit_b = tab.shop_hit(c, backup, tab.core) | tab.shop_hit(c, backup, tab.supp)
        hit_m |= TeacherMixin._engine_hits(self, c)                  # 引擎联动：酒馆里有引擎（手里 / 场上已有 2 张燃料）或者场上有引擎、酒馆里有燃料
        return np.where(hit_m, mult, np.where(hit_b, (1.0 + mult) / 2, 1.0)).astype(np.float32)

    def _engine_hits(self, c) -> np.ndarray:
        """[行]：酒馆里有引擎卡、而且场上 + 手里已经有 ≥ 2 张它的燃料；或者场上有引擎、酒馆里有它的燃料（rl/comp.ENGINE_RULES）。"""
        from tg.obs import C_BOARD, C_HAND, C_SHOP, N_BOARD, N_HAND, N_SHOP
        tab = ShapingMixin._engine_tab(self)
        eng, fuel = tab["engine"], tab["fuel"]
        c = np.clip(np.asarray(c), 0, len(eng) - 1)
        board = c[:, C_BOARD:C_BOARD + N_BOARD]
        own = np.concatenate([board, c[:, C_HAND:C_HAND + N_HAND]], 1)
        shop = c[:, C_SHOP:C_SHOP + N_SHOP]
        nf = (fuel[own] & (own > 0)[:, :, None]).sum(1)
        on = (eng[board] & (board > 0)[:, :, None]).any(1)
        return ((eng[shop].any(1) & (nf >= 2)) | (on & fuel[shop].any(1))).any(1)

    def _teacher_coef(self) -> float:
        """这一轮的老师 KL 系数：每次启动后从 0 线性加到 teacher_coef（teacher_ramp 轮）。"""
        if getattr(self, "teacher", None) is None:
            return 0.0
        if self.teacher_t0 is None:
            self.teacher_t0 = self.iter
        ramp = max(1, int(self.cfg.teacher_ramp))
        return float(self.cfg.teacher_coef) * min(1.0, max(0.0, (self.iter - self.teacher_t0 + 1) / ramp))
