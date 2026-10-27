"""B3 PPO 训练（TavernGym BatchEnv + 实体 Transformer + 对手库）。

超参按论文：γ=1、λ=0.995、clip 0.2、目标 KL 0.05（超过就提前结束本轮更新）、熵 0.002、价值 0.5、
名次辅助 0.1、梯度裁剪 1.0、每轮 1 遍；新阶段开始时对上一阶段策略加 KL 锚定（系数 0.1，前 100 轮）。
规模按一张 3070（8GB）缩小：默认每轮约 32k 个决策、小批 4096。

对局安排（每局重新抽）：
  - 一部分对局：0 号座位是学习者，其余 7 个是 TavernGym 脚本机器人（controlled=(0,) 的环境）；
  - 其余对局：8 个座位都由我们控制；0 号是学习者，其它座位按对手库权重抽（库里还没有快照时就是学习者
    自己 = 自我对弈）。学习者控制的座位都产生训练数据。
奖励：TavernGym 的名次奖励（第 1 名 1 … 第 8 名 -1），对局结束时给出。
"""
from __future__ import annotations

import json
import math
import random
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .league import League, load_real_profile
from .model import build_model, n_params


@dataclass
class Config:
    run_dir: str = "runs/default"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0
    threads: int = 8
    games_bot: int = 64            # 和机器人打的并行对局数
    games_league: int = 32         # 8 座位都由我们控制的并行对局数
    steps_per_iter: int = 256      # 每轮每个环境走多少步
    iters: int = 1000
    # 模型
    d: int = 192
    layers: int = 4
    heads: int = 4
    # PPO
    gamma: float = 1.0
    lam: float = 0.995
    clip: float = 0.2
    target_kl: float = 0.05
    ent_coef: float = 0.002
    vf_coef: float = 0.5
    aux_coef: float = 0.1
    max_grad_norm: float = 1.0
    lr: float = 3e-4
    minibatch: int = 4096
    epochs: int = 1
    anchor_coef: float = 0.1
    anchor_iters: int = 100
    # 对手库 / 检查点
    snapshot_every: int = 25       # 每多少轮存一个快照进对手库
    profile_games: int = 16        # 测快照画像用多少局
    save_every: int = 10
    eval_every: int = 25
    eval_games: int = 64
    amp: bool = True


class Policy:
    """一个可以出动作的策略：学习者或对手库里的快照。"""

    def __init__(self, model, device):
        self.model, self.device = model, device

    @torch.no_grad()
    def __call__(self, f, c, m, greedy=False):
        t = lambda x, dt: torch.as_tensor(x, device=self.device, dtype=dt)     # noqa: E731
        a, logp, v = self.model.act(t(f, torch.float32), t(c, torch.long), t(m, torch.bool), greedy)
        return a.cpu().numpy(), logp.cpu().numpy(), v.float().cpu().numpy()


class Rollout:
    """按座位（槽位）记录学习者的决策；名次奖励在对局结束时补到该槽位最后一个决策上。"""

    def __init__(self):
        self.seq = defaultdict(list)       # 槽位键 -> [dict]
        self.done_seqs = []

    def add(self, key, **tr):
        self.seq[key].append(tr)

    def reward(self, key, r, done, placement):
        s = self.seq.get(key)
        if s:
            s[-1]["r"] = s[-1].get("r", 0.0) + float(r)
            if done:
                s[-1]["done"] = True
                for tr in s:
                    if "place" not in tr:
                        tr["place"] = int(placement)
                self.done_seqs.append(s)
                self.seq[key] = []

    def collect(self, gamma, lam, last_values: dict):
        """GAE。没打完的序列用当前局面的价值做自举。返回扁平化的数组。"""
        seqs = [(s, 0.0) for s in self.done_seqs] + [(s, last_values.get(k, 0.0)) for k, s in self.seq.items() if s]
        out = defaultdict(list)
        for s, boot in seqs:
            n = len(s)
            adv = np.zeros(n, dtype=np.float32)
            last = 0.0
            for i in reversed(range(n)):
                nxt_v = boot if i == n - 1 and not s[i].get("done") else (0.0 if s[i].get("done") else s[i + 1]["v"])
                delta = s[i].get("r", 0.0) + gamma * nxt_v - s[i]["v"]
                last = delta + gamma * lam * (0.0 if s[i].get("done") else last)
                adv[i] = last
            for i, tr in enumerate(s):
                out["f"].append(tr["f"])
                out["c"].append(tr["c"])
                out["m"].append(tr["m"])
                out["a"].append(tr["a"])
                out["logp"].append(tr["logp"])
                out["v"].append(tr["v"])
                out["adv"].append(adv[i])
                out["ret"].append(adv[i] + tr["v"])
                out["place"].append(tr.get("place", 0))
        self.done_seqs = []
        self.seq = defaultdict(list, {k: [] for k in self.seq})
        if not out:
            return None
        return {k: np.stack(v) if k in ("f", "c", "m") else np.asarray(v) for k, v in out.items()}


class Trainer:
    def __init__(self, cfg: Config):
        import taverngym
        from taverngym import BatchEnv
        self.cfg = cfg
        self.dev = torch.device(cfg.device)
        torch.manual_seed(cfg.seed)
        self.rng = random.Random(cfg.seed)
        self.run = Path(cfg.run_dir)
        (self.run / "snapshots").mkdir(parents=True, exist_ok=True)
        self.model = build_model(cfg.d, cfg.layers, cfg.heads).to(self.dev)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, eps=1e-5)
        self.scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and self.dev.type == "cuda")
        self.learner = Policy(self.model, self.dev)
        self.envs = []
        if cfg.games_bot:
            self.envs.append(("bot", BatchEnv(cfg.games_bot, seed=cfg.seed, controlled=(0,), threads=cfg.threads)))
        if cfg.games_league:
            self.envs.append(("league", BatchEnv(cfg.games_league, seed=cfg.seed + 10**6, controlled=range(8),
                                                 threads=cfg.threads)))
        self.obs = {name: env.observe() for name, env in self.envs}
        self.episode = {name: [env.episode_index(i) for i in range(env.n_games)] for name, env in self.envs}
        self.roles: dict[tuple, str] = {}             # (环境, 对局, 座位) -> "learner" 或快照名
        self.snap_cache: dict[str, Policy] = {}
        self.league = League(load_real_profile())
        self.league.add("bot")
        self.rollout = Rollout()
        self.iter = 0
        self.anchor = None
        self.anchor_until = 0
        self.stats = deque(maxlen=2000)               # 学习者名次（对机器人 / 对手库）
        self.log_f = open(self.run / "log.jsonl", "a", encoding="utf-8")
        (self.run / "config.json").write_text(json.dumps(asdict(cfg), indent=1), encoding="utf-8")
        print(f"模型参数 {n_params(self.model) / 1e6:.2f}M，设备 {self.dev}，taverngym {taverngym.version()}")

    # ---------------- 对手 ----------------

    def _assign(self, name, env, g):
        for s in range(8):
            self.roles.pop((name, g, s), None)
        if name == "bot":
            self.roles[(name, g, 0)] = "learner"
            return
        self.roles[(name, g, 0)] = "learner"
        snaps = [m for m in self.league.training_weights() if m[0].name != "bot"]
        for s in range(1, 8):
            if snaps and self.rng.random() < 0.75:
                m = self.rng.choices([x for x, _ in snaps], weights=[w for _, w in snaps])[0]
                self.roles[(name, g, s)] = m.name
            else:
                self.roles[(name, g, s)] = "learner"

    def _policy(self, role):
        if role == "learner":
            return self.learner
        if role not in self.snap_cache:
            m = next(x for x in self.league.members if x.name == role)
            model = build_model(self.cfg.d, self.cfg.layers, self.cfg.heads).to(self.dev)
            model.load_state_dict(torch.load(m.path, map_location=self.dev))
            model.eval()
            self.snap_cache[role] = Policy(model, self.dev)
            if len(self.snap_cache) > 12:
                self.snap_cache.pop(next(iter(self.snap_cache)))
        return self.snap_cache[role]

    # ---------------- 采样 ----------------

    def collect(self):
        cfg = self.cfg
        self.model.eval()
        n_dec = 0
        t0 = time.time()
        for name, env in self.envs:
            for g in range(env.n_games):
                if (name, g, 0) not in self.roles:
                    self._assign(name, env, g)
        for _ in range(cfg.steps_per_iter):
            for name, env in self.envs:
                obs = self.obs[name]
                k = len(env.controlled)
                ready = np.flatnonzero(obs["ready"])
                actions = np.zeros(env.n_slots, dtype=np.int64)
                groups = defaultdict(list)
                for i in ready:
                    g, seat = divmod(int(i), k)
                    seat = env.controlled[seat]
                    groups[self.roles.get((name, g, seat), "learner")].append(i)
                for role, idx in groups.items():
                    idx = np.asarray(idx)
                    a, logp, v = self._policy(role)(obs["features"][idx], obs["cards"][idx], obs["action_mask"][idx])
                    actions[idx] = a
                    if role == "learner":
                        for j, i in enumerate(idx):
                            self.rollout.add((name, int(i)), f=obs["features"][i].astype(np.float16),
                                             c=obs["cards"][i], m=np.packbits(obs["action_mask"][i]),
                                             a=int(a[j]), logp=float(logp[j]), v=float(v[j]))
                        n_dec += len(idx)
                rewards, dones, info = env.step(actions)
                for i in np.flatnonzero((rewards != 0) | dones):
                    g, seat = divmod(int(i), k)
                    seat = env.controlled[seat]
                    role = self.roles.get((name, g, seat), "learner")
                    if role == "learner":
                        self.rollout.reward((name, int(i)), rewards[i], bool(dones[i]), info["placement"][i])
                        if dones[i] and seat == 0:
                            self.stats.append((name, int(info["placement"][i])))
                self.obs[name] = env.observe()
                for g in range(env.n_games):                 # 新开的对局：重新抽对手
                    e = env.episode_index(g)
                    if e != self.episode[name][g]:
                        self.episode[name][g] = e
                        self._assign(name, env, g)
        # 没打完的序列：用当前局面的价值自举
        last_values = {}
        for name, env in self.envs:
            obs = self.obs[name]
            keys = [(name, i) for i in range(env.n_slots) if self.rollout.seq.get((name, i))]
            if keys:
                idx = np.asarray([i for _, i in keys])
                _, _, v = self.learner(obs["features"][idx], obs["cards"][idx], obs["action_mask"][idx] |
                                       (~obs["action_mask"][idx].any(1, keepdims=True)))
                last_values.update({k: float(x) for k, x in zip(keys, v)})
        batch = self.rollout.collect(cfg.gamma, cfg.lam, last_values)
        return batch, n_dec, time.time() - t0

    # ---------------- 更新 ----------------

    def update(self, b):
        cfg = self.cfg
        self.model.train()
        n = len(b["a"])
        adv = (b["adv"] - b["adv"].mean()) / (b["adv"].std() + 1e-8)
        idx = np.random.permutation(n)
        stats = defaultdict(list)
        use_anchor = self.anchor is not None and self.iter < self.anchor_until
        stop = False
        for _ in range(cfg.epochs):
            for s in range(0, n, cfg.minibatch):
                j = idx[s:s + cfg.minibatch]
                f = torch.as_tensor(b["f"][j], device=self.dev, dtype=torch.float32)
                c = torch.as_tensor(b["c"][j], device=self.dev, dtype=torch.long)
                m = torch.as_tensor(np.unpackbits(b["m"][j], axis=1)[:, :self.model.L.actions].astype(bool),
                                    device=self.dev)
                a = torch.as_tensor(b["a"][j], device=self.dev)
                old = torch.as_tensor(b["logp"][j], device=self.dev, dtype=torch.float32)
                A = torch.as_tensor(adv[j], device=self.dev, dtype=torch.float32)
                R = torch.as_tensor(b["ret"][j], device=self.dev, dtype=torch.float32)
                P = torch.as_tensor(b["place"][j], device=self.dev, dtype=torch.long)
                with torch.autocast(self.dev.type, enabled=self.scaler.is_enabled()):
                    logits, v, pl = self.model(f, c, m)
                logits, v, pl = logits.float(), v.float(), pl.float()
                logp_all = F.log_softmax(logits, -1)
                logp = logp_all.gather(1, a.unsqueeze(1)).squeeze(1)
                ratio = torch.exp(logp - old)
                pg = -torch.min(ratio * A, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * A).mean()
                probs = logp_all.exp()
                ent = -(probs * logp_all.masked_fill(~m, 0)).sum(-1).mean()
                vf = F.mse_loss(v, R)
                has_p = P > 0
                aux = F.cross_entropy(pl[has_p], P[has_p] - 1) if has_p.any() else torch.zeros((), device=self.dev)
                loss = pg + cfg.vf_coef * vf - cfg.ent_coef * ent + cfg.aux_coef * aux
                if use_anchor:
                    with torch.no_grad():
                        ref = F.log_softmax(self.anchor(f, c, m)[0].float(), -1)
                    kl_anchor = (ref.exp() * (ref - logp_all)).masked_fill(~m, 0).sum(-1).mean()
                    loss = loss + cfg.anchor_coef * kl_anchor
                    stats["kl_anchor"].append(kl_anchor.item())
                self.opt.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.scaler.step(self.opt)
                self.scaler.update()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                for k_, v_ in (("loss", loss.item()), ("pg", pg.item()), ("vf", vf.item()), ("ent", ent.item()),
                               ("aux", aux.item()), ("kl", approx_kl),
                               ("clipfrac", ((ratio - 1).abs() > cfg.clip).float().mean().item())):
                    stats[k_].append(v_)
                if approx_kl > cfg.target_kl:
                    stop = True
                    break
            if stop:
                break
        out = {k: float(np.mean(v)) for k, v in stats.items()}
        out["early_stop"] = stop
        return out

    # ---------------- 对手库快照、评估 ----------------

    def snapshot(self):
        from .league import sim_profile
        path = self.run / "snapshots" / f"it{self.iter:05d}.pt"
        torch.save(self.model.state_dict(), path)
        pol = self.learner

        def act(g, p):
            f, c = g.observe(p)
            m = np.frombuffer(g.legal_mask(p), dtype=np.uint8).astype(bool)[None]
            a, _, _ = pol(np.frombuffer(f, "<f4")[None], np.frombuffer(c, "<i4")[None], m)
            return int(a[0])
        self.model.eval()
        prof = sim_profile(act, games=self.cfg.profile_games, seed=10**7 + self.iter)
        mem = self.league.add(path.stem, str(path), prof)
        (self.run / "league.json").write_text(json.dumps(self.league.to_json(), ensure_ascii=False, indent=1),
                                              encoding="utf-8")
        return mem

    def evaluate(self, games=None, seed=999):
        """考试：学习者（0 号座位）对 7 个脚本机器人，平均名次。"""
        from taverngym import BatchEnv
        games = games or self.cfg.eval_games
        env = BatchEnv(games, seed=seed, controlled=(0,), threads=self.cfg.threads)
        obs = env.observe()
        places = []
        seen = set()
        self.model.eval()
        while len(places) < games:
            ready = np.flatnonzero(obs["ready"])
            actions = np.zeros(env.n_slots, dtype=np.int64)
            if len(ready):
                a, _, _ = self.learner(obs["features"][ready], obs["cards"][ready], obs["action_mask"][ready])
                actions[ready] = a
            _, dones, info = env.step(actions)
            for i in np.flatnonzero(dones):
                key = (i, env.episode_index(i))
                if key not in seen:
                    seen.add(key)
                    places.append(int(info["placement"][i]))
            obs = env.observe()
        places = [p for p in places if p > 0][:games]
        return float(np.mean(places)) if places else float("nan"), float(np.mean([p == 1 for p in places]))

    # ---------------- 主循环 ----------------

    def save(self):
        torch.save({"model": self.model.state_dict(), "opt": self.opt.state_dict(), "iter": self.iter,
                    "league": self.league.to_json()}, self.run / "latest.pt")

    def load(self, path):
        ck = torch.load(path, map_location=self.dev)
        self.model.load_state_dict(ck["model"])
        self.opt.load_state_dict(ck["opt"])
        self.iter = ck["iter"]
        self.league = League.from_json(ck["league"], load_real_profile())

    def start_stage(self):
        """新阶段：以当前策略为锚，前 anchor_iters 轮加 KL 约束。"""
        import copy
        self.anchor = copy.deepcopy(self.model).eval()
        for p in self.anchor.parameters():
            p.requires_grad_(False)
        self.anchor_until = self.iter + self.cfg.anchor_iters

    def train(self, iters=None, on_log=None):
        cfg = self.cfg
        end = self.iter + (iters or cfg.iters)
        while self.iter < end:
            batch, n_dec, t_col = self.collect()
            t1 = time.time()
            upd = self.update(batch) if batch is not None else {}
            self.iter += 1
            rec = {"iter": self.iter, "decisions": n_dec, "dps": round(n_dec / max(t_col, 1e-6)),
                   "t_collect": round(t_col, 1), "t_update": round(time.time() - t1, 1), **upd}
            for name in ("bot", "league"):
                ps = [p for e, p in self.stats if e == name and p > 0]
                if ps:
                    rec[f"place_{name}"] = round(float(np.mean(ps)), 3)
                    rec[f"top1_{name}"] = round(float(np.mean([p == 1 for p in ps])), 3)
            if cfg.snapshot_every and self.iter % cfg.snapshot_every == 0:
                m = self.snapshot()
                rec["snapshot"] = m.name
                rec["snapshot_dist"] = round(m.dist, 3)
            if cfg.eval_every and self.iter % cfg.eval_every == 0:
                rec["eval_place"], rec["eval_top1"] = self.evaluate()
            if cfg.save_every and self.iter % cfg.save_every == 0:
                self.save()
            self.log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.log_f.flush()
            if on_log:
                on_log(rec)
        self.save()
