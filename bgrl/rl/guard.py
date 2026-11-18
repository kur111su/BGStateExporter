"""训练的保险：发现训练坏了（health / _drift）就退回上一个好的状态（rollback）、考试守护（最好成绩存 best.pt，变差就退回并把惹事的系数减半）、
学习率预热、熵系数自动调、--restore。Trainer 的一部分（GuardMixin）。"""
from __future__ import annotations

import json
import re
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch


class GuardMixin:
    def health(self, upd: dict) -> str:
        """这一轮有没有出问题（返回问题描述，没有返回空串）。第一次训练第 310 轮起更新失控、
        第 455 轮起参数变成 NaN，之前的保险把 NaN 悄悄换成了随机动作，又白跑了 800 轮——现在直接回退。"""
        if any(not torch.isfinite(p).all() for p in self.model.parameters()):
            return "模型参数出现 NaN / inf"
        if self.learner.bad_rows or upd.get("bad_rows"):
            return f"网络输出出现 NaN（采样 {self.learner.bad_rows} 个、更新 {upd.get('bad_rows', 0)} 个局面）"
        if upd.get("skipped", 0) >= 2:
            return f"有 {upd['skipped']} 个小批的梯度是 NaN / inf"
        if upd.get("aux", 0) > 4.0:
            return f"名次头损失发散（{upd['aux']:.2f}）"
        return ""

    def _drift(self, ent, noop) -> str:
        """策略慢慢变乱的检查（见 drift_ent / drift_noop）。返回问题描述，没有返回空串。"""
        cfg = self.cfg
        if not hasattr(self, "ent_hist"):
            self.ent_hist, self.drift_n = deque(maxlen=300), 0
        if ent is None or (cfg.drift_ent <= 0 and cfg.drift_noop <= 0 and getattr(cfg, "drift_ent_max", 0) <= 0):
            return ""
        base = float(np.median(self.ent_hist)) if len(self.ent_hist) >= 100 else None
        bad = []
        if cfg.drift_ent > 0 and base is not None and ent > base + cfg.drift_ent:
            bad.append(f"熵 {ent:.2f}（最近中位数 {base:.2f}）")
        if getattr(cfg, "drift_ent_max", 0) > 0 and ent > cfg.drift_ent_max:
            bad.append(f"熵 {ent:.2f}（上限 {cfg.drift_ent_max}）")
        if cfg.drift_noop > 0 and len(self.ent_hist) >= 100 and noop > cfg.drift_noop:
            bad.append(f"无效操作 {noop:.0%}")
        if not bad:
            self.drift_n = 0
            self.ent_hist.append(float(ent))
            return ""
        self.drift_n += 1
        self.mix["drift"] += 1
        if self.drift_n >= cfg.drift_iters:
            self.drift_n = 0
            return f"策略连续 {cfg.drift_iters} 轮变乱：" + "、".join(bad)
        return ""

    def _guard_keys(self) -> list[str]:
        return [k.strip() for k in str(getattr(self.cfg, "guard_keys", "") or "").split(",")
                if k.strip() and hasattr(self.cfg, k.strip())]

    @staticmethod
    def _guard_new() -> dict:
        return {"evals": [], "best": None, "best_iter": None, "bad": 0, "n": 0, "scale": 1.0, "metric": None}

    @classmethod
    def _guard_restart(cls, old: dict, ck: dict) -> dict:
        """--restore 以后的考试守护：最近几次考试从头记，但“最好成绩”留着，和硬盘上的 best.pt 对得上。
        退回的检查点本身带考试成绩（best.pt 存的 guard.eval_avg）就用它；否则（退回某个快照）沿用原来的最好成绩。
        以前这里整个清零：第 28579 轮退回 best.pt（第 27800 轮，4.375）以后，第 28850 轮第一次凑满的平均 4.57
        直接覆盖了 best.pt。"""
        g = {**cls._guard_new(), "metric": (old or {}).get("metric")}
        gi = ck.get("guard") if isinstance(ck, dict) else None
        if isinstance(gi, dict) and gi.get("eval_avg") is not None:
            g["best"], g["best_iter"] = float(gi["eval_avg"]), (int(ck["iter"]) if ck.get("iter") is not None else None)
        elif (old or {}).get("best") is not None:
            g["best"], g["best_iter"] = old["best"], old.get("best_iter")
        return g

    def _guard_metric(self) -> str:
        """守护比的是什么：平均名次（"place"），或者按名次奖励表算的等效名次（"score:<表>"）。"""
        spec = str(getattr(self.cfg, "place_reward", "") or "").replace(" ", "")
        return f"score:{spec}" if spec else "place"

    def _guard_check_metric(self):
        """换了名次奖励表：之前的最好成绩是按别的标准算的，不能比。best.pt 改名留着（best_<轮>.pt），守护从头记（缩放保留）。"""
        g, m = self.guard_st, self._guard_metric()
        if g.get("metric") in (None, m):
            g["metric"] = m
            return
        p = self.run / "best.pt"
        if p.exists():
            q = self.run / f"best_it{g.get('best_iter') or 0}.pt"
            p.replace(q)
            print(f"考试守护换了标准（{g.get('metric')} → {m}）：原来的 best.pt 改名为 {q.name}，从头记", flush=True)
        self.guard_st = {**self._guard_new(), "scale": g.get("scale", 1.0), "n": g.get("n", 0), "metric": m}

    def _guard_apply(self):
        """把守护的缩放（scale）乘到模仿 / 老师系数上（基准是启动时配方里的值）。"""
        sc = float(self.guard_st.get("scale", 1.0))
        for k, v in getattr(self, "guard_base", {}).items():
            setattr(self.cfg, k, v * sc)

    def _guard(self, x) -> dict:
        """每次加强局考试后调用。返回要写进日志的字段。"""
        cfg = self.cfg
        if not cfg.guard or x is None:
            return {}
        self._guard_check_metric()
        g = self.guard_st
        w = max(1, int(cfg.guard_window))
        g["evals"] = (g["evals"] + [float(x)])[-w:]
        out = {"guard_scale": round(float(g["scale"]), 3)}
        if len(g["evals"]) < w:
            return out
        m = float(np.mean(g["evals"]))
        out["guard_avg"] = round(m, 3)
        if g["best"] is None or m < g["best"]:
            g["best"], g["best_iter"], g["bad"] = m, int(self.iter), 0
            self._save_best(m)
            out["guard_best"] = round(m, 3)
            return out
        out["guard_ref"] = round(float(g["best"]), 3)
        if m < g["best"] + cfg.guard_margin:
            g["bad"] = 0
            return out
        g["bad"] += 1
        out["guard_bad"] = g["bad"]
        if g["bad"] < cfg.guard_patience:
            return out
        if g["n"] >= cfg.guard_max:
            out["guard_capped"] = True
            return out
        out["guard_fix"] = self._guard_fix(m)
        return out

    def _save_best(self, m: float):
        torch.save({"model": self.model.state_dict(), "model_kwargs": self.mkw, "iter": self.iter,
                    "guard": {"eval_avg": round(m, 3), "window": int(self.cfg.guard_window),
                              "time": time.strftime("%Y-%m-%d %H:%M:%S")}}, self.run / "best.pt")

    def _guard_fix(self, m: float) -> str:
        g, cfg = self.guard_st, self.cfg
        p = self.run / "best.pt"
        if not p.exists():
            g["bad"] = 0
            return "没有 best.pt，只记不改"
        sd = torch.load(p, map_location=self.dev, weights_only=False)["model"]
        self.model.load_state_dict(sd)
        self.model._ver = getattr(self.model, "_ver", 0) + 1          # 采样用的副本下次出动作前同步
        lr = self.opt.param_groups[0]["lr"]
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)
        self.snap_cache.clear()
        g["n"] += 1
        g["scale"] = float(g["scale"]) * float(cfg.guard_decay)
        g["evals"], g["bad"] = [], 0
        self._guard_apply()
        self._start_warmup()
        self._reset_drift()
        self.mark_good()
        msg = (f"第 {self.iter} 轮考试最近 {cfg.guard_window} 次平均 {m:.2f}，比最好的 {g['best']:.2f}（第 {g['best_iter']} 轮）"
               f"差 {cfg.guard_margin} 以上连续 {cfg.guard_patience} 次：换回 best.pt，"
               + "、".join(f"{k} → {getattr(cfg, k):.4g}" for k in self.guard_base) + f"（第 {g['n']} / {cfg.guard_max} 次）")
        print(msg, flush=True)
        return msg

    def mark_good(self):
        import copy
        self.good = (copy.deepcopy(self.model.state_dict()), copy.deepcopy(self.opt.state_dict()), self.iter)
        self.rollbacks = 0

    def _reset_drift(self):
        """换了一套参数（--restore、考试守护换回 best.pt）：变乱检查的“最近熵中位数”从头记。
        不清的话拿新参数的熵去比旧参数的中位数：第 17000 轮换回 best.pt（第 15450 轮那版熵 1.6）后，
        和之前的中位数 1.10 比一直算“变乱”，回退点又就是换回来的那版，连退 5 次训练停了。"""
        self.ent_hist, self.drift_n = deque(maxlen=300), 0

    def _start_warmup(self):
        """优化器刚重新开始：学习率从 10% 爬到 cfg.lr（warmup_iters 轮）。"""
        if getattr(self.cfg, "warmup_iters", 0) > 0:
            self.warm_t0 = int(self.iter)
            self._apply_warmup()

    def _apply_warmup(self):
        t0, n = getattr(self, "warm_t0", None), int(getattr(self.cfg, "warmup_iters", 0) or 0)
        if t0 is None or n <= 0:
            return
        k = (self.iter - t0) / n
        if k >= 1 or k < 0:
            self.warm_t0 = None
            k = 1.0
        for g in self.opt.param_groups:
            g["lr"] = self.cfg.lr * (0.1 + 0.9 * k)

    def _adapt_ent(self, ent) -> None:
        """熵对着 ent_target 调 ent_coef（见 Config.ent_target）。"""
        cfg = self.cfg
        if not getattr(cfg, "ent_target", 0) or ent is None:
            return
        f = (1 - cfg.ent_adapt) if ent > cfg.ent_target else (1 + cfg.ent_adapt)
        cfg.ent_coef = float(min(cfg.ent_coef_max, max(cfg.ent_coef_min, cfg.ent_coef * f)))

    def rollback(self, problem: str):
        self.rollbacks += 1
        self.warm_t0 = None
        print(f"第 {self.iter} 轮出问题：{problem}。回退到第 {self.good[2] if self.good else '?'} 轮，学习率减半。", flush=True)
        if self.rollbacks > self.cfg.max_rollbacks or self.good is None:
            self.log_f.write(json.dumps({"iter": self.iter, "stopped": problem}, ensure_ascii=False) + "\n")
            self.log_f.flush()
            raise RuntimeError(f"连续回退 {self.rollbacks} 次仍然出问题（{problem}），训练停止。把 log.jsonl 发给 Claude。")
        sd, od, it = self.good
        self.model.load_state_dict(sd)
        self.model._ver = getattr(self.model, "_ver", 0) + 1          # 采样用的副本下次出动作前同步
        self.opt.load_state_dict(od)
        for g in self.opt.param_groups:
            g["lr"] *= 0.5
        self.iter = it
        self.snap_cache.clear()
        if self.rollbacks >= 2 and "最近中位数" in problem and "上限" not in problem:
            # 连着第二次因为“熵比最近中位数高”回退：回退点自己的熵就在这个水平（比如换了名次奖励表以后，前 100 轮
            # 熵中位数还没攒够、检查没开，熵涨上去的那段也被当成了回退点），再退还是一样，第 5 次训练就停了。
            # 熵基准从头记（这 100 轮只看 drift_ent_max 上限），交给熵目标（ent_target）往回拉
            self._reset_drift()
            print("回退点本身的熵就在这个水平：变乱检查的熵基准从头记（只看上限），由熵目标往回拉", flush=True)

    def restore(self, path):
        """接着训练目录练（对手库、加强强度、模仿强度都保留），但模型参数换成某个快照的、优化器重新开始，
        轮数退回快照那一轮，对手库里比它新的快照删掉（那是变坏以后的自己）。"""
        ck = self.init_from(path) or {}
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr, eps=1e-5)
        m = re.search(r"it(\d+)", Path(path).name)
        if m:
            it = int(m.group(1))
            drop = [x for x in self.league.members if (re.search(r"it(\d+)", x.name or "") and
                                                      int(re.search(r"it(\d+)", x.name).group(1)) > it)]
            self.league.members = [x for x in self.league.members if x not in drop]
            self.iter = it
            print(f"退回第 {it} 轮的模型；对手库删掉 {len(drop)} 个更新的快照", flush=True)
        self.snap_cache.clear()
        self.active_snaps = []
        self.good = None
        # 老师强度、局末扣回势能（shape_repay）的爬坡：退回的模型本来就是在全强度下练出来的，接着用全强度。
        # 以前这里清零、从 0 重新爬 300 轮：第 30060 轮退回以后审计看到局末扣回只有 3–31%，每局约 1.0 的
        # 目标阵容 / 引擎联动 / 阵容强度势能白给（名次奖励每局平均才 0.24），老师也只剩一部分——每次退回都有 300 轮在练另一个目标。
        # 退回快照时轮数会往回退，t0 要跟着挪到“已经爬满”，不然 iter < t0 会一直是 0。
        cfg = self.cfg
        if self.teacher_t0 is not None:
            self.teacher_t0 = int(self.iter) - int(getattr(cfg, "teacher_ramp", 0) or 0)
        if getattr(self, "repay_t0", None) is not None:
            self.repay_t0 = int(self.iter) - int(getattr(cfg, "shape_repay_ramp", 1) or 1)
        self.guard_st = self._guard_restart(self.guard_st, ck)
        self._guard_apply()
        self._start_warmup()
        self._reset_drift()
        self.log_f.write(json.dumps({"iter": self.iter, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                                     "restored_from": str(path)}, ensure_ascii=False) + "\n")
        self.log_f.flush()
