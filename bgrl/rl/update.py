"""PPO 更新（rl/ppo.Trainer 的一部分，作为 mixin 继承进去）：PPO + 价值 + 名次 / 体系辅助头 + 老师蒸馏 + 高手模仿，
按词数排序切 micro、梯度累积、按 KL 提前结束；MPS 上第一次更新前探测 bfloat16 能不能用。
"""
from __future__ import annotations

import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F


class UpdateMixin:
    def _probe_mps_amp(self, b) -> None:
        """Mac：实测 bfloat16 混合精度在这台机器上对不对、快不快（第一次更新前跑一次，几秒）。
        同一份样本算两遍：结果要和 float32 对得上（动作一样的 ≥ 98%、没有 NaN），而且快 15% 以上才启用。"""
        cfg = self.cfg
        A_N = self.model.L.actions
        j = np.arange(min(len(b["a"]), cfg.micro_batch))
        f = torch.as_tensor(b["f"][j], dtype=torch.float32).to(self.dev)
        c = torch.as_tensor(b["c"][j], dtype=torch.long).to(self.dev)
        m = torch.as_tensor(np.unpackbits(b["m"][j], axis=1)[:, :A_N].astype(bool)).to(self.dev)

        def run(amp: bool, reps: int):
            t0 = time.time()
            for _ in range(reps):
                self.opt.zero_grad(set_to_none=True)
                with torch.autocast("mps", dtype=torch.bfloat16, enabled=amp):
                    lg, v, _ = self.model(f, c, m, compressed=True)
                lp = F.log_softmax(lg.float(), -1)
                (-(lp.exp() * lp.masked_fill(~m, 0)).sum(-1).mean() + v.float().pow(2).mean()).backward()
                torch.mps.synchronize()
            self.opt.zero_grad(set_to_none=True)
            return (time.time() - t0) / reps, lg.detach().float()

        self.model.eval()
        try:
            run(False, 1)
            t32, l32 = run(False, 3)
            run(True, 1)
            t16, l16 = run(True, 3)
            same = float((l32.argmax(-1) == l16.argmax(-1)).float().mean())
            ok = bool(torch.isfinite(l16[m]).all()) and same >= 0.98
            use = ok and t16 < 0.85 * t32
            print(f"  混合精度实测：float32 {t32 * 1000:.0f} ms / 份，bfloat16 {t16 * 1000:.0f} ms / 份，动作一致 {same:.1%} → "
                  f"{'启用' if use else '不用'}", flush=True)
        except Exception as ex:                          # noqa: BLE001 —— 这台机器 / 这个 torch 不支持：不用
            use = False
            print(f"  混合精度实测：不支持（{type(ex).__name__}: {str(ex)[:80]}）→ 不用", flush=True)
        self.opt.zero_grad(set_to_none=True)
        torch.mps.empty_cache()
        if use:
            self.amp = True
            cfg.micro_batch = max(cfg.micro_batch, min(512, getattr(self, "_micro_recipe", cfg.micro_batch)))
            cfg.teacher_chunk = max(cfg.teacher_chunk, cfg.micro_batch)
            print(f"  半精度内存减半：每份放回 {cfg.micro_batch} 个样本", flush=True)
            self.learner.amp = True
            for pol in self.snap_cache.values():
                pol.amp = True

    def update(self, b):
        cfg = self.cfg
        if self.dev.type == "mps" and not self.amp and str(cfg.amp_mps).lower() == "auto" and not getattr(self, "_amp_probed", False):
            self._amp_probed = True
            self._probe_mps_amp(b)
        self.model.train()
        n = len(b["a"])
        adv = (b["adv"] - b["adv"].mean()) / (b["adv"].std() + 1e-8)
        idx = np.random.permutation(n)
        stats = defaultdict(list)
        use_anchor = self.anchor is not None and self.iter < self.anchor_until
        tcoef = self._teacher_coef()
        if tcoef > 0:
            stats["teacher_coef"].append(tcoef)
        stop = False
        A_N = self.model.L.actions
        hard = getattr(cfg, "teacher_mode", "hard") != "kl"
        t_ta = t_tp = None
        if tcoef > 0 and hard:                           # 老师的答案先算好（每轮一次，不是每个 epoch 每一份都算）
            t0_t = time.time()
            t_ta, t_tp = self._teacher_targets(b, A_N)
            stats["t_teacher"].append(time.time() - t0_t)
            stats["teacher_rows"].append(float((t_ta >= 0).mean()))
        ex_t = []
        ex_ramp = 0.0
        if getattr(self, "experts", None):                 # 流派专家：在它擅长的局面上的答案，也是每轮先算好
            if self.expert_t0 is None:
                self.expert_t0 = self.iter
            ex_ramp = min(1.0, (self.iter - self.expert_t0 + 1) / max(1, int(getattr(cfg, "expert_ramp", 100))))
            t0_e = time.time()
            ex_t = self._expert_targets(b, A_N)
            stats["t_expert"].append(time.time() - t0_e)
            for e_, ta_, _ in ex_t:
                stats["expert_rows"].append(float((ta_ >= 0).mean()))
        if self.dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        # 统计量留在显卡上（张量），每个小批结束时一起取回一次：每取一个数（.item() / float() / .any()）MPS 都要停下来
        # 等队列清空，原来每份 micro 十几次。prof：分段计时（cfg.profile_update；分段点要同步，会稍慢，只在测速时开）
        prof = bool(getattr(cfg, "profile_update", False))
        sync = (torch.mps.synchronize if self.dev.type == "mps" else torch.cuda.synchronize
                if self.dev.type == "cuda" else (lambda: None))
        tseg = defaultdict(float)

        def mark(key, t0):
            if prof:
                sync()
                now = time.perf_counter()
                tseg[key] += now - t0
                return now
            return t0
        zero = torch.zeros((), device=self.dev)
        tok = None
        if getattr(cfg, "bucket_tokens", False) and hasattr(self.model, "token_counts"):
            tok = self.model.token_counts(b["c"])
        for ep_i in range(cfg.epochs):
            if ep_i:
                idx = np.random.permutation(n)                 # 每遍重新打乱，小批的组合不重复
            for s in range(0, n, cfg.minibatch):
                mb = idx[s:s + cfg.minibatch]
                if tok is not None:                        # 词数相近的排在一起（梯度不变，打包更省）
                    mb = mb[np.argsort(tok[mb], kind="stable")]
                self.opt.zero_grad(set_to_none=True)
                acc: dict[str, torch.Tensor] = {}          # 名字 → 这个小批里各份 micro 的和（张量）
                cnt: dict[str, int] = defaultdict(int)     # 名字 → 加了几份（求平均用）
                ratio_pairs: dict[str, tuple] = {}         # 名字 → (分子, 分母)：按行数加权的比例

                def add(k_, x_):
                    acc[k_] = acc[k_] + x_.detach().float() if k_ in acc else x_.detach().float()
                    cnt[k_] += 1

                def add_ratio(k_, num, den):
                    a_, b_ = ratio_pairs.get(k_, (zero, zero))
                    ratio_pairs[k_] = (a_ + num.detach().float(), b_ + den.detach().float())
                kl_mb = zero
                tp = time.perf_counter()
                for u in range(0, len(mb), cfg.micro_batch):     # 分份算梯度再累加（省显存）
                    j = mb[u:u + cfg.micro_batch]
                    w = len(j) / len(mb)
                    t = lambda x, dt: torch.as_tensor(x, dtype=dt).to(self.dev, non_blocking=True)  # noqa: E731
                    f = t(b["f"][j], torch.float32)
                    c = t(b["c"][j], torch.long)
                    m = t(np.unpackbits(b["m"][j], axis=1)[:, :A_N].astype(bool), torch.bool)
                    a = t(b["a"][j], torch.long)
                    old = t(b["logp"][j], torch.float32)
                    A = t(adv[j], torch.float32)
                    R = t(b["ret"][j], torch.float32)
                    P = t(b["place"][j], torch.long)
                    tp = mark("t_data", tp)
                    with torch.autocast(self.dev.type, dtype=torch.bfloat16, enabled=self.amp):
                        logits, v, pl = self.model(f, c, m, compressed=True)
                    lb = getattr(self.model, "last_bad", 0)
                    add("bad_rows", lb if torch.is_tensor(lb) else torch.tensor(float(lb), device=self.dev))
                    logp_all = F.log_softmax(logits.float(), -1)
                    logp = logp_all.gather(1, a.unsqueeze(1)).squeeze(1)
                    ratio = torch.exp(logp - old)
                    pg = -torch.min(ratio * A, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * A).mean()
                    ent = -(logp_all.exp() * logp_all.masked_fill(~m, 0)).sum(-1).mean()
                    vf = F.mse_loss(v.float(), R)
                    # 名次头：只算打完的局（P > 0）。不分支（.any() 要同步）：按行加权，没有就是 0，和原来的平均一样
                    has_p = (P > 0).float()
                    ce_p = F.cross_entropy(pl.float(), (P - 1).clamp(min=0), reduction="none")
                    aux = (ce_p * has_p).sum() / has_p.sum().clamp(min=1)
                    loss = pg + cfg.vf_coef * vf - cfg.ent_coef * ent + cfg.aux_coef * aux
                    if getattr(cfg, "aux_comp_coef", 0) > 0 and "comp" in b and self.model.last_comp_logits is not None:
                        K = t(b["comp"][j], torch.long)
                        has_k = (K >= 0).float()
                        if (b["comp"][j] >= 0).any():          # 在 CPU 上判断（不用同步）
                            ce_k = F.cross_entropy(self.model.last_comp_logits, K.clamp(min=0), reduction="none")
                            aux_c = (ce_k * has_k).sum() / has_k.sum().clamp(min=1)
                            loss = loss + cfg.aux_comp_coef * aux_c
                            add("aux_comp", aux_c)
                            add_ratio("comp_acc", ((self.model.last_comp_logits.argmax(-1) == K).float() * has_k).sum(),
                                      has_k.sum())
                    if use_anchor:
                        with torch.no_grad(), torch.autocast(self.dev.type, dtype=torch.bfloat16, enabled=self.amp):
                            ref = F.log_softmax(self.anchor(f, c, m, compressed=True)[0].float(), -1)
                        kl_anchor = (ref.exp() * (ref - logp_all)).masked_fill(~m, 0).sum(-1).mean()
                        loss = loss + cfg.anchor_coef * kl_anchor
                        add("kl_anchor", kl_anchor)
                    if tcoef > 0:
                        tw = self._teacher_rows(b["f"][j], b["c"][j])
                        if tw is not None:
                            stats["teacher_core"].append(float((tw > 1).mean()))
                        tw_t = t(tw, torch.float32) if tw is not None else torch.ones(len(j), device=self.dev)
                        if not hard:                                     # kl：整个分布向老师靠（要现算老师的分布）
                            with torch.no_grad(), torch.autocast(self.dev.type, dtype=torch.bfloat16, enabled=self.amp):
                                ch = int(getattr(cfg, "teacher_chunk", 512)) or f.shape[0]
                                tl = torch.cat([F.log_softmax(self.teacher(f[q:q + ch], c[q:q + ch], m[q:q + ch],
                                                                           compressed=True)[0].float(), -1)
                                                for q in range(0, f.shape[0], ch)])
                            kl_rows = (tl.exp() * (tl - logp_all)).masked_fill(~m, 0).sum(-1)
                            loss = loss + tcoef * (kl_rows * tw_t).mean()
                            add("kl_teacher", kl_rows.mean())
                        else:                                            # 只在老师有把握的局面，学老师最想做的那个动作
                            ta = t(t_ta[j], torch.long)
                            sure = ((t(t_tp[j], torch.float32) >= cfg.teacher_conf) & (ta >= 0)).float()
                            ta = ta.clamp(min=0)
                            nll_t = -logp_all.gather(1, ta.unsqueeze(1)).squeeze(1).clamp(max=20)
                            boost = 1.0 / min(1.0, max(0.05, float(getattr(cfg, "teacher_frac", 1.0))))   # 只算了一部分局面：按比例放大
                            loss = loss + tcoef * boost * (nll_t * sure * tw_t).mean()
                            add("teacher_sure", sure.mean())
                            add_ratio("teacher_agree", ((logp_all.argmax(-1) == ta).float() * sure).sum(), sure.sum())
                    for e_, ta_all, tp_all in ex_t:                       # 流派专家：有把握的局面学它最想做的动作
                        ta = t(ta_all[j], torch.long)
                        sure = ((t(tp_all[j], torch.float32) >= e_["conf"]) & (ta >= 0)).float()
                        ta = ta.clamp(min=0)
                        nll_e = -logp_all.gather(1, ta.unsqueeze(1)).squeeze(1).clamp(max=20)
                        loss = loss + e_["coef"] * ex_ramp * (nll_e * sure).mean()
                        add("expert_sure", sure.mean())
                        add_ratio("expert_agree", ((logp_all.argmax(-1) == ta).float() * sure).sum(), sure.sum())
                    tp = mark("t_fwd", tp)
                    self.scaler.scale(loss * w).backward()
                    tp = mark("t_bwd", tp)
                    with torch.no_grad():
                        kl = ((ratio - 1) - torch.log(ratio)).mean()
                        kl_mb = kl_mb + kl * w
                        for k_, v_ in (("loss", loss), ("pg", pg), ("vf", vf), ("ent", ent), ("aux", aux), ("kl", kl),
                                       ("clipfrac", ((ratio - 1).abs() > cfg.clip).float().mean())):
                            add(k_, v_)
                if self.pro is not None and cfg.pro_coef > 0 and len(self.pro["train_idx"]):
                    pb = self.bc_loss(self.pro, self.pro_w, cfg.pro_batch)
                    self.scaler.scale(cfg.pro_coef * self.pro_scale * pb).backward()
                    add("pro_bc", pb)
                    tp = mark("t_bc", tp)
                gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                # 这个小批唯一一次取回：梯度范数、KL（早停用）、所有统计
                keys = list(acc)
                rkeys = list(ratio_pairs)
                vals = torch.stack([gn.float(), kl_mb.float()] + [acc[k_] for k_ in keys]
                                   + [x for k_ in rkeys for x in ratio_pairs[k_]]).tolist()
                gn_v, kl_v = vals[0], vals[1]
                for i_, k_ in enumerate(keys):
                    x_ = vals[2 + i_]
                    if k_ == "bad_rows":
                        stats[k_].append(x_)
                    else:
                        stats[k_].append(x_ / max(1, cnt[k_]))
                off_r = 2 + len(keys)
                for i_, k_ in enumerate(rkeys):
                    num, den = vals[off_r + 2 * i_], vals[off_r + 2 * i_ + 1]
                    if den > 0:
                        stats[k_].append(num / den)
                if not np.isfinite(gn_v):                  # 梯度坏了：这一步不更新，记下来
                    stats["skipped"].append(1.0)
                    self.opt.zero_grad(set_to_none=True)
                    continue
                stats["skipped"].append(0.0)
                stats["grad_norm"].append(float(gn_v))
                self.opt.step()
                self.model._ver = getattr(self.model, "_ver", 0) + 1          # 采样用的副本下次出动作前同步
                tp = mark("t_opt", tp)
                if kl_v > cfg.target_kl:
                    stop = True
                    break
            if stop:
                break
        for k_, v_ in tseg.items():
            stats[k_].append(v_)
        # 价值头解释了多少回报的方差（1 = 全解释，0 = 和猜平均一样，< 0 = 更差）；优势、回报的分布
        ret_np, v_np = np.asarray(b["ret"], dtype=np.float64), np.asarray(b["v"], dtype=np.float64)
        var_r = float(ret_np.var())
        stats["ev"].append(1.0 - float((ret_np - v_np).var()) / var_r if var_r > 1e-12 else 0.0)
        stats["adv_std"].append(float(np.std(b["adv"])))
        stats["ret_mean"].append(float(ret_np.mean()))
        stats["ret_std"].append(float(ret_np.std()))
        out = {k: float(np.mean(v)) for k, v in stats.items()}
        out["bad_rows"] = int(sum(stats.get("bad_rows", [])))
        out["skipped"] = int(sum(stats.get("skipped", [])))
        out["early_stop"] = stop
        if self.dev.type == "cuda":
            out["gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        elif self.dev.type == "mps":                   # MPS 会把用过的显存一直留着（和系统内存是同一块）：每轮还回去
            out["gpu_mem_gb"] = round(torch.mps.driver_allocated_memory() / 2**30, 2)
            torch.mps.empty_cache()
        return out
