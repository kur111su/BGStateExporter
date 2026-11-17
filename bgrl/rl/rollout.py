"""采样用的东西：Policy（一个能出动作的模型，可以在另一个设备上放副本）、Rollout（一轮的经验，算优势和回报）、名次奖励表。"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from .model import soft_log


class Policy:
    """一个可以出动作的策略：学习者或对手库里的快照。
    device 和模型所在的设备不同时（采样用 CPU、更新用 MPS / CUDA），在 device 上放一份副本，
    模型的版本号（_ver，训练器每次改参数时加 1）变了就同步一次。"""

    def __init__(self, model, device, amp=False, copy=False, groups: int = 1, min_group: int = 64):
        self.src, self.device = model, device
        # 按词数分组前向（groups > 1）：同一步里各局处在不同回合，前期局面约 20 个词、后期约 60 个，打包时一批截到最长那行；
        # 按词数排序切成 groups 份分别算，少算空词。每份至少 min_group 行（太小了多一次调用的固定开销不划算）
        self.groups, self.min_group = max(1, int(groups)), max(1, int(min_group))
        self.amp = amp and device.type in ("cuda", "mps")
        self.bad_rows = 0                      # 输出出现 NaN 的局面数（应该一直是 0）
        if next(model.parameters()).device.type == device.type and not copy:
            self.model, self.ver = model, None
        else:
            import copy
            self.model = copy.deepcopy(model).to(device).eval()
            for q in self.model.parameters():
                q.requires_grad_(False)
            self.ver = getattr(model, "_ver", 0)

    def sync(self) -> None:
        if self.ver is not None and self.ver != getattr(self.src, "_ver", 0):
            with torch.no_grad():
                for a, b in zip(self.model.state_dict().values(), self.src.state_dict().values()):
                    a.copy_(b.detach().to(a.device))
            self.ver = getattr(self.src, "_ver", 0)

    @torch.inference_mode()
    def __call__(self, f, c, m, greedy=False, want_obs=False):
        """want_obs=True 时另外返回压缩后的观察（float16，存训练数据用：压缩后最大十几，不会溢出）。"""
        self.sync()
        n = len(f)
        k = min(self.groups, n // self.min_group)
        if k > 1 and hasattr(self.model, "token_counts"):
            order = np.argsort(self.model.token_counts(c), kind="stable")
            parts = [self._run(f[j], c[j], m[j], greedy, want_obs) for j in np.array_split(order, k)]
            inv = np.empty(n, dtype=np.int64)
            inv[order] = np.arange(n)
            return tuple(np.concatenate([p[i] for p in parts])[inv] for i in range(len(parts[0])))
        return self._run(f, c, m, greedy, want_obs)

    def _run(self, f, c, m, greedy, want_obs):
        n = len(f)
        if self.device.type == "mps" and n % 64:        # MPS：局面数凑成 64 的倍数（重复最后一行，算完丢掉），原因同上
            pad = 64 - n % 64
            rep = lambda x: np.concatenate([x, np.repeat(np.asarray(x)[-1:], pad, 0)])   # noqa: E731
            f, c, m = rep(f), rep(c), rep(m)
        f = torch.as_tensor(np.ascontiguousarray(f), dtype=torch.float32).to(self.device, non_blocking=True)
        c = torch.as_tensor(np.ascontiguousarray(c), dtype=torch.long).to(self.device, non_blocking=True)
        m = torch.as_tensor(np.ascontiguousarray(m), dtype=torch.bool).to(self.device, non_blocking=True)
        fz = soft_log(f)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.amp):
            logits, v, _ = self.model(fz, c, m, compressed=True)
        if greedy:
            a = logits.argmax(-1)
        else:
            a = torch.distributions.Categorical(logits=logits).sample()
        logp = torch.log_softmax(logits, -1).gather(1, a.unsqueeze(1)).squeeze(1)
        out = (a.cpu().numpy()[:n], logp.float().cpu().numpy()[:n], v.float().cpu().numpy()[:n])
        self.bad_rows += int(getattr(self.model, "last_bad", 0))       # 动作已经取回了，这时候数不会多等
        if want_obs:
            out = out + (fz.half().cpu().numpy()[:n],)
        return out



def parse_place_reward(spec: str):
    """"1:1,2:0.55,…,8:-1" → 长度 9 的数组（下标 = 名次，0 不用）；空 / 不全 → None（用引擎的线性奖励）。"""
    if not str(spec or "").strip():
        return None
    tab = np.full(9, np.nan)
    for part in str(spec).split(","):
        if ":" in part:
            k, v = part.split(":", 1)
            tab[int(k)] = float(v)
    if np.isnan(tab[1:]).any():
        raise ValueError(f"place_reward 要写全 1–8 名：{spec}")
    tab[0] = 0.0
    return tab



def linear_place_reward(p: int) -> float:
    """引擎自带的名次奖励：第 1 名 1，第 8 名 -1，线性。"""
    return 1.0 - 2.0 * (int(p) - 1) / 7.0



def seed_row(v) -> list:
    """成型阵容课程一套阵容的结果 [(名次, 加强局?)] → [局数, 平均名次, 吃鸡率, 其中加强局的局数, 平均名次, 吃鸡率]。"""
    ps = [int(p) for p, _ in v]
    bs = [int(p) for p, b in v if b]
    row = [len(ps), round(float(np.mean(ps)), 2), round(float(np.mean([p == 1 for p in ps])), 3), len(bs)]
    if bs:
        row += [round(float(np.mean(bs)), 2), round(float(np.mean([p == 1 for p in bs])), 3)]
    else:
        row += [0.0, 0.0]
    return row



def place_score_equiv(places, tab) -> float:
    """一组名次按奖励表算平均分，再换成“线性奖励下的等效名次”（4.5 - 3.5 × 平均分；线性表就是平均名次本身），
    考试守护拿它比，越小越好。"""
    if tab is None:
        return float(np.mean(places))
    return float(4.5 - 3.5 * np.mean([tab[int(p)] for p in places]))



def grow_opt_state(opt, od: dict) -> dict:
    """旧检查点的优化器状态比现在的模型少几个参数（第 15300 轮加了体系感知 comp_proj / comp_head，排在参数表最后）：
    把新参数的编号接到参数组后面（它们没有 Adam 状态，第一次更新时从 0 开始），旧参数的状态原样保留。
    数量一样时原样返回；新模型参数更少、或者有好几个参数组时也原样返回（让 load_state_dict 照常报错）。"""
    import copy
    groups = od.get("param_groups") or []
    if len(groups) != 1 or len(opt.param_groups) != 1:
        return od
    have, want = len(groups[0]["params"]), len(opt.param_groups[0]["params"])
    if have >= want:
        return od
    od = copy.deepcopy(od)
    ids = list(od["param_groups"][0]["params"])
    nxt = max(ids) + 1 if ids else 0
    od["param_groups"][0]["params"] = ids + list(range(nxt, nxt + want - have))
    print(f"优化器状态：检查点里 {have} 个参数，现在 {want} 个（新加的体系感知参数从 0 开始）", flush=True)
    return od



class Rollout:
    """学习者的决策按步成批存（numpy），名次奖励在对局结束时补到该座位最后一个决策上。
    槽位 = 全局座位编号（各环境的槽位依次排开）；ep = 这个槽位的第几局。"""

    def __init__(self, n_slots: int):
        self.n_slots = n_slots
        self.ep = np.zeros(n_slots, dtype=np.int64)
        self.place: dict[tuple, int] = {}
        self.comp: dict[tuple, int] = {}               # 局末成了哪一套（体系感知辅助头的标签）
        self.reset()

    def cut(self, slot: int):
        """成型阵容课程换场面之前调用：在这个座位最后一个决策后面切断 GAE。
        不切的话，换场面带来的价值跳变（课程局平均好将近 1 名）全算在换之前最后一步（上一回合的结束回合）头上——
        等于每回合早点结束回合、钱不花完也有奖励。切断后那一步用自己的价值自举（不换场面时下一步的价值≈这一步的），
        换之前那几步的阵容标签也不要（局末的阵容是换来的，不是它们做出来的）。"""
        i = int(self.last[slot])
        if i >= 0:
            self.cuts.add(i)
            self.seed_cut[(int(slot), int(self.ep[slot]))] = i

    def reset(self):
        self.parts = defaultdict(list)
        self.r: list[float] = []
        self.done: list[bool] = []
        self.n = 0
        self.last = np.full(self.n_slots, -1, dtype=np.int64)
        self.cuts: set[int] = set()
        self.seed_cut: dict[tuple, int] = {}

    def add(self, slots, f, c, m, a, logp, v, r0=None):
        k = len(slots)
        if not k:
            return
        slots = np.asarray(slots, dtype=np.int64)
        for name, x in (("slot", slots), ("ep", self.ep[slots]), ("f", np.asarray(f, dtype=np.float16)), ("c", c),
                        ("m", np.packbits(m, axis=1)), ("a", a), ("logp", logp), ("v", v)):
            self.parts[name].append(np.asarray(x))
        self.last[slots] = np.arange(self.n, self.n + k)
        self.r.extend([0.0] * k if r0 is None else [float(x) for x in r0])
        self.done.extend([False] * k)
        self.n += k

    def reward(self, slot, r, done, placement, comp=None):
        i = self.last[slot]
        if i >= 0:
            self.r[i] += float(r)
            if done:
                self.done[i] = True
        if done:
            self.place[(int(slot), int(self.ep[slot]))] = int(placement)
            if comp is not None:
                self.comp[(int(slot), int(self.ep[slot]))] = int(comp)
            self.ep[slot] += 1
            self.last[slot] = -1

    def collect(self, gamma, lam, last_values: np.ndarray | None):
        """GAE（按槽位、按局分段）。没打完的用当前局面的价值自举。返回扁平化的数组。"""
        if self.n == 0:
            return None
        b = {k: np.concatenate(v) for k, v in self.parts.items()}
        r = np.asarray(self.r, dtype=np.float32)
        done = np.asarray(self.done, dtype=bool)
        v = b["v"].astype(np.float32)
        slot, ep = b["slot"], b["ep"]
        order = np.lexsort((np.arange(self.n), slot))           # 按槽位，槽位内按时间
        adv = np.zeros(self.n, dtype=np.float32)
        nxt_i = -1
        last = 0.0
        for pos in range(self.n - 1, -1, -1):
            i = order[pos]
            same = nxt_i >= 0 and slot[nxt_i] == slot[i] and ep[nxt_i] == ep[i]
            if done[i]:
                nv, last = 0.0, 0.0
            elif i in self.cuts:                               # 换场面之前的最后一步：切断
                nv, last = float(v[i]), 0.0
            elif same:
                nv = v[nxt_i]
            else:                                              # 本轮这一段的最后一个决策
                last = 0.0
                live = last_values is not None and self.last[slot[i]] == i
                nv = float(last_values[slot[i]]) if live else 0.0
            delta = r[i] + gamma * nv - v[i]
            last = delta + gamma * lam * last
            adv[i] = last
            nxt_i = i
        b["adv"] = adv
        b["ret"] = adv + v
        b["place"] = np.asarray([self.place.get((int(s), int(e)), 0) for s, e in zip(slot, ep)], dtype=np.int64)
        b["comp"] = np.asarray([self.comp.get((int(s), int(e)), -1) for s, e in zip(slot, ep)], dtype=np.int64)
        for (s_, e_), i_ in self.seed_cut.items():          # 换场面之前那几步不要阵容标签
            b["comp"][(slot == s_) & (ep == e_) & (np.arange(self.n) <= i_)] = -1
        self.place.clear()                 # 之后的决策只会属于还没结束的局，旧的名次用不到了
        self.comp.clear()
        self.reset()
        return b
