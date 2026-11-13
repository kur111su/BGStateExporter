"""B2 实体 Transformer（策略 + 价值 + 名次辅助头）。

输入只用 TavernGym 自己的观察编码 observe()（3227 个浮点 + 124 个卡牌编号），切片位置在运行时从
taverngym._core.layout() 读，避免两套编码器漂移（论文教训）。

词（token）：
  全局 1（全局特征 + 任务进度 + 待选类型）、英雄 1、英雄技能 2、神祇 + 待选来源 2、任务 3、奥秘 1（合并）、
  饰品 8、场面 7、手牌 10、酒馆 8、待选项 16、对手 7（公开信息 + 英雄 + 上次看到的阵容汇总）
  空位用注意力屏蔽。
卡 = 编号嵌入 + 静态特征（card_static：身材、等级、种族、关键词等 66 维）投影。
动作：指针式打分，245 个动作各自来自对应的词（买 i ← 酒馆词 i；卖 / 发动 i ← 场面词 i；
  打出 (手牌 h, 位置 p) ← 手牌词 h；磁力 (h, b) ← 手牌词 h 和场面词 b 的双线性；移动 (from, to) ← 场面词 from；
  选择 i ← 待选项词 i；英雄技能 i ← 技能词 i；结束 / 刷新 / 冻结 / 升本 / 黑暗之赐 / 取消 ← 全局词），
  再按合法动作屏蔽。
头：价值（名次奖励 1…-1 的期望）、名次分布（8 类，辅助损失）。
大小见 SIZES：small 约 2.3M、medium 约 8M（默认）、large 约 15M；论文最终版是 d=384、8 层、18M。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Layout:
    MF: int
    IF: int
    GF: int
    NQ: int
    QF: int
    NPK: int
    OPPF: int
    NOPP: int
    NB: int
    NH: int
    NS: int
    NC: int
    NPOS: int
    NPW: int
    floats: int
    cards: int
    actions: int

    @classmethod
    def from_taverngym(cls) -> "Layout":
        import taverngym._core as c
        return cls.from_dict(dict(c.layout()), c.OBS_FLOATS, c.OBS_CARDS, c.N_ACTIONS)

    @classmethod
    def from_dict(cls, L: dict, floats: int, cards: int, actions: int) -> "Layout":
        """不装 TavernGym 时用（竞技场连接器的环境里，见 arena/agent.py）：L = taverngym.layout() 存下来的字典。"""
        return cls(L["MINION_F"], L["ITEM_F"], L["GLOBAL_F"], L["N_QUESTS"], L["QUEST_F"], L["PENDING_KINDS"],
                   L["OPP_F"], L["N_OPP"], L["N_BOARD"], L["N_HAND"], L["N_SHOP"], L["N_CHOICE"], L["N_POS"],
                   L["N_POWERS"], int(floats), int(cards), int(actions))

    # 浮点切片
    def fslices(self):
        o = 0
        out = {}
        for name, n in (("global", self.GF + self.NQ * self.QF + self.NPK), ("board", self.NB * self.MF),
                        ("hand", self.NH * self.IF), ("shop", self.NS * self.IF), ("choice", self.NC * self.IF),
                        ("opp", self.NOPP * self.OPPF), ("oppboard", self.NOPP * self.NB * self.MF)):
            out[name] = (o, o + n)
            o += n
        assert o == self.floats, (o, self.floats)
        return out

    # 卡牌编号切片
    def cslices(self):
        o = 0
        out = {}
        for name, n in (("hero", 1), ("powers", self.NPW), ("deity", 1), ("source", 1), ("quests", 3 * self.NQ),
                        ("secrets", 5), ("board", self.NB), ("hand", self.NH), ("shop", self.NS),
                        ("choice", self.NC), ("trinkets", 8), ("opphero", self.NOPP),
                        ("oppboard", self.NOPP * self.NB)):
            out[name] = (o, o + n)
            o += n
        assert o == self.cards, (o, self.cards)
        return out


def soft_log_np(x: np.ndarray) -> np.ndarray:
    a = np.abs(x)
    return np.where(a <= 1, x, np.sign(x) * (1 + np.log(np.maximum(a, 1)))).astype(np.float32)


def soft_log(x: torch.Tensor) -> torch.Tensor:
    """输入压缩：|x| ≤ 1 不变，更大的按对数压（1 + ln|x|）。后期随从攻防上千（观察里是攻防/10），
    不压的话半精度下注意力会溢出成 NaN（训练第 200 轮测强度时崩溃就是这个）。"""
    a = x.abs()
    return torch.where(a <= 1, x, torch.sign(x) * (1 + torch.log(a.clamp(min=1))))


class CardEmbed(nn.Module):
    def __init__(self, static: np.ndarray, dc: int):
        super().__init__()
        n, k = static.shape
        table = np.zeros((n + 1, k), dtype=np.float32)
        table[1:] = static
        self.register_buffer("static", torch.from_numpy(table), persistent=False)
        self.ident = nn.Embedding(n + 1, dc, padding_idx=0)
        self.proj = nn.Linear(k, dc)
        nn.init.normal_(self.ident.weight, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        idx = idx.clamp(min=0, max=self.static.shape[0] - 1)
        return self.ident(idx) + self.proj(self.static[idx])


K_MAX, COMP_F = 32, 10          # 和 rl/comp.py 一样（这里不 import rl/comp：竞技场按文件路径单独导入本文件）
E_MAX, SYN_F = 16, 10           # 引擎格数（rl/comp.E_MAX）；每张卡的联动标记个数（SynergySense）


class CompSense(nn.Module):
    """阵容进度（体系感知）：从卡牌编号和观察里算每套目标阵容（rl/target_comps.json，最多 K_MAX 套）做到几成，
    给模型当输入——“我在做哪套、做到几成、还有哪几套备选、酒馆里有没有它的核心卡 / 配套卡”。
    查表（rl/comp.model_tables）不是参数，不进检查点；没有查表时全是 0（输入层是 0 起步，模型行为不变）。"""

    def __init__(self, tables: dict | None, cs: dict):
        super().__init__()
        self.cs = cs
        self.on = tables is not None
        if not self.on:
            return
        t = lambda k, dt: torch.as_tensor(np.asarray(tables[k]), dtype=dt)          # noqa: E731
        for k in ("core_b", "ks_b", "w", "ncore", "nks"):
            self.register_buffer(k, t(k, torch.float32), persistent=False)
        for k in ("core_c", "supp_c", "trib_c", "need", "need_valid", "tribal", "valid"):
            self.register_buffer(k, t(k, torch.bool), persistent=False)
        for k in ("base_key", "lobby_idx"):
            self.register_buffer(k, t(k, torch.long), persistent=False)
        n = self.core_c.shape[0]
        for k, shape in (("eng_c", (n, E_MAX)), ("fuel_c", (n, E_MAX)), ("econ_c", (n,))):   # 旧的 comp.npz 没有：全是 0
            v = t(k, torch.bool) if k in tables else torch.zeros(shape, dtype=torch.bool)
            self.register_buffer(k, v, persistent=False)

    def forward(self, x_in: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """x_in：soft_log 过的浮点；c：卡牌编号。返回 [B, K_MAX, COMP_F]。"""
        return self.sense(x_in, c)[0]

    def sense(self, x_in: torch.Tensor, c: torch.Tensor):
        """返回 (每套阵容的进度 [B, K_MAX, COMP_F], 场上 / 手里 / 酒馆 / 发现里每张卡的联动标记 [B, NE, SYN_F])。

        联动标记（SynergySense，第 15313 轮起）——每张卡和我现在的体系是什么关系：
          0 主选的核心卡  1 主选的配套卡  2 备选的核心卡  3 备选的配套卡  4 主选种族阵容的本族随从
          5 主选 / 备选里“留着用”的卡（卖了亏）  6 我手上（场上 + 手里）某个引擎的燃料
          7 引擎卡：它的燃料我已经有几成（场上 + 手里燃料张数 / 3，最多 1）
          8 同名卡我已经有几张（三连进度：0 / 0.5 / 1）  9 倒转经济的卡（发金币 / 铸币、便宜刷新、卖得贵）"""
        B = c.shape[0]
        cs = self.cs
        ne = cs["choice"][1] - cs["board"][0]
        if not self.on:
            return (torch.zeros(B, K_MAX, COMP_F, device=c.device), torch.zeros(B, ne, SYN_F, device=c.device))
        cs = self.cs
        nc = self.core_c.shape[0]
        cl = c.clamp(0, nc - 1)
        board = cl[:, cs["board"][0]:cs["board"][1]]
        own = torch.cat([board, cl[:, cs["hand"][0]:cs["hand"][1]]], 1)
        shop = cl[:, cs["shop"][0]:cs["shop"][1]]
        nb = self.core_b.shape[0]
        pres = torch.zeros(B, nb, device=c.device)
        pres.scatter_(1, self.base_key[own], 1.0)
        pres[:, 0] = 0.0
        presb = torch.zeros(B, nb, device=c.device)
        presb.scatter_(1, self.base_key[board], 1.0)
        presb[:, 0] = 0.0
        frac = (pres @ self.core_b) / self.ncore                            # 场上 + 手里核心卡占比（同名算一张）
        tribe = self.trib_c[board].float().sum(1)                         # 场上本族随从个数
        sup = torch.where(self.tribal, (tribe / 4).clamp(max=1.0), torch.ones_like(tribe))
        ksn = presb @ self.ks_b                                            # 场上“留着用”的配套卡张数
        supf = torch.where(self.nks > 0, (ksn / self.nks.clamp(min=1).clamp(max=2)).clamp(max=1.0), torch.ones_like(ksn))
        lob = x_in[:, self.lobby_idx] > 0.3                               # 本局开放的种族
        hit = (self.need.unsqueeze(0) & lob[:, None, None, :]).any(-1)    # [B,K,需要几组]
        reach = ((hit | ~self.need_valid.unsqueeze(0)).all(-1) & self.valid).float()
        score = self.w * frac * frac * sup * (0.7 + 0.3 * supf) * reach
        top = torch.topk(score, 3, dim=1)
        rank = torch.zeros_like(score).scatter(1, top.indices, torch.tensor([1.0, 2.0, 3.0], device=c.device).expand(B, 3))
        rank = rank * (score > 0)
        main = (rank == 1).float()
        backup = ((rank == 2) | (rank == 3)).float()
        shop_core = self.core_c[shop].any(1).float()
        shop_supp = self.supp_c[shop].any(1).float()
        feats = torch.stack([reach, frac, frac * frac, sup, supf, score, main, backup, shop_core, shop_supp], -1) * \
            self.valid.float()[None, :, None]
        # 每张卡的联动标记
        ent = cl[:, cs["board"][0]:cs["choice"][1]]                         # [B, NE]：场上、手里、酒馆、发现
        mb, bb = main.bool()[:, None, :], backup.bool()[:, None, :]
        core, supp = self.core_c[ent], self.supp_c[ent]                    # [B, NE, K]
        bk_e, bk_o = self.base_key[ent], self.base_key[own]
        keep = self.ks_b[bk_e] > 0
        eng_own = self.eng_c[own].any(1)                                   # [B, E]：我手上有的引擎
        nf = self.fuel_c[own].float().sum(1)                               # [B, E]：每个引擎的燃料张数
        same = ((bk_e[:, :, None] == bk_o[:, None, :]) & (bk_o[:, None, :] > 0)).float().sum(-1)
        n_own = own.shape[1]
        same = same - (torch.arange(ne, device=c.device) < n_own).float()[None, :] * (ent > 0).float()   # 自己不算
        syn = torch.stack([
            (core & mb).any(-1), (supp & mb).any(-1), (core & bb).any(-1), (supp & bb).any(-1),
            (self.trib_c[ent] & mb & self.tribal[None, None, :]).any(-1),
            (keep & (mb | bb)).any(-1) | (core & (mb | bb)).any(-1),
            (self.fuel_c[ent] & eng_own[:, None, :]).any(-1),
        ], -1).float()
        eng_lv = (self.eng_c[ent].float() * (nf / 3.0).clamp(max=1.0)[:, None, :]).amax(-1)
        syn = torch.cat([syn, eng_lv.unsqueeze(-1), (same / 2).clamp(0, 1).unsqueeze(-1),
                         self.econ_c[ent].float().unsqueeze(-1)], -1) * (ent > 0).float().unsqueeze(-1)
        return feats, syn


class EntityTransformer(nn.Module):
    N_TYPES = 12

    def __init__(self, layout: Layout, static: np.ndarray, d: int = 192, layers: int = 4, heads: int = 4,
                 dc: int = 64, comp_tables: dict | None = None):
        super().__init__()
        self.L = layout
        self.fs = layout.fslices()
        self.cs = layout.cslices()
        L = layout
        self.d = d
        self.card = CardEmbed(static, dc)
        self.card_to_d = nn.Linear(dc, d)
        self.type_emb = nn.Embedding(self.N_TYPES, d)
        self.slot_emb = nn.Embedding(16, d)                  # 槽位序号（场面站位、手牌顺序等）
        g_in = self.fs["global"][1] - self.fs["global"][0]
        self.f_global = nn.Linear(g_in, d)
        self.f_minion = nn.Linear(L.MF, d)
        self.f_item = nn.Linear(L.IF, d)
        self.f_opp = nn.Linear(L.OPPF, d)
        self.opp_minion = nn.Sequential(nn.Linear(L.MF + dc, d), nn.ReLU(), nn.Linear(d, d))
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True,
                                         activation="gelu")
        self.enc = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        # 动作头
        self.h_global = nn.Linear(d, 6)          # 结束 刷新 冻结 升本 黑暗之赐 取消
        self.h_power = nn.Linear(d, 1)
        self.h_buy = nn.Linear(d, 1)
        self.h_sell = nn.Linear(d, 1)
        self.h_activate = nn.Linear(d, 1)
        self.h_choose = nn.Linear(d, 1)
        self.h_play = nn.Linear(d, L.NPOS)
        self.h_move = nn.Linear(d, L.NB)
        self.mag_q = nn.Linear(d, d)
        self.mag_k = nn.Linear(d, d)
        # 价值 / 名次
        self.v_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.p_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 8))
        # 体系感知（第 15300 轮起）：每套目标阵容的进度 → 加到全局词上（0 起步：旧检查点接着用，行为不变）；
        # 辅助头：预测这局最后成了哪一套（K_MAX 类）或没成（第 K_MAX 类），训练时用局末结果当标签
        self.comp = CompSense(comp_tables, self.cs)
        self.comp_proj = nn.Linear(K_MAX * COMP_F, d)
        nn.init.zeros_(self.comp_proj.weight)
        nn.init.zeros_(self.comp_proj.bias)
        self.comp_head = nn.Linear(d, K_MAX + 1)
        # 联动感知（第 15313 轮起）：每张卡和体系 / 引擎 / 三连 / 经济的关系 → 加到场上、手里、酒馆、发现的词上（0 起步）
        self.syn_proj = nn.Linear(SYN_F, d)
        nn.init.zeros_(self.syn_proj.weight)
        nn.init.zeros_(self.syn_proj.bias)
        self.last_comp_logits = None
        self._register_load_state_dict_pre_hook(self._fill_new_keys, with_module=False)
        # 每个词的类型和槽位序号（不是参数，不进检查点）
        groups = [(0, 1), (1, 1), (2, L.NPW), (3, 2), (4, L.NQ), (5, 1), (6, 8), (7, L.NB), (8, L.NH), (9, L.NS),
                  (10, L.NC), (11, L.NOPP)]
        self.register_buffer("type_ids", torch.tensor([t for t, n in groups for _ in range(n)]), persistent=False)
        self.register_buffer("slot_ids", torch.tensor([i for t, n in groups for i in range(n)]), persistent=False)
        self.pack = True          # 只把有内容的词送进 Transformer（结果不变，省约一半计算）

    def _fill_new_keys(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """旧检查点（第 15300 轮以前没有体系感知、第 15313 轮以前没有联动感知）：用现在的初始值补上（输入层是 0，行为和原来一样）。"""
        for name in ("comp_proj.weight", "comp_proj.bias", "comp_head.weight", "comp_head.bias", "syn_proj.weight",
                     "syn_proj.bias"):
            if prefix + name not in state_dict:
                mod, attr = name.split(".")
                state_dict[prefix + name] = getattr(getattr(self, mod), attr).detach().clone()

    def comp_input(self, x_in: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.comp_proj(self.comp(x_in, c).reshape(c.shape[0], -1)).unsqueeze(1)

    def sense_input(self, x_in: torch.Tensor, c: torch.Tensor):
        """(加到全局词上的体系感知 [B,1,d], 加到场上 / 手里 / 酒馆 / 发现词上的联动感知 [B,NE,d])。"""
        feats, syn = self.comp.sense(x_in, c)
        return self.comp_proj(feats.reshape(c.shape[0], -1)).unsqueeze(1), self.syn_proj(syn)

    # ---------------- 编码 ----------------

    def _tokens(self, f: torch.Tensor, c: torch.Tensor):
        """把观察变成 66 个词。和 _tokens_slow 数学上完全一样（同一套参数），只是把十几次小的嵌入 / 线性层
        合并成几次大的，显卡上快很多。"""
        B = f.shape[0]
        L, fs, cs, d = self.L, self.fs, self.cs, self.d
        E = self.card(c)                                       # [B,124,dc] 所有卡一次算
        D = self.card_to_d(E)                                  # [B,124,d]
        bias = self.card_to_d.bias
        sl = lambda T, a, b=None: T[:, cs[a][0]:cs[b or a][1]]   # noqa: E731
        gi, si = self.sense_input(f, c)
        g = self.f_global(f[:, fs["global"][0]:fs["global"][1]]).unsqueeze(1) + gi.to(D.dtype)
        si = si.to(D.dtype)
        nq = cs["quests"][1] - cs["quests"][0]
        ns = cs["secrets"][1] - cs["secrets"][0]
        # 任务、奥秘：原来是先把卡嵌入加起来再过线性层，线性层的偏置只算一次
        q = sl(D, "quests").view(B, L.NQ, nq // L.NQ, d).sum(2) - (nq // L.NQ - 1) * bias
        sec = sl(D, "secrets").sum(1, keepdim=True) - (ns - 1) * bias
        bf = f[:, fs["board"][0]:fs["board"][1]].view(B, L.NB, L.MF)
        board = self.f_minion(bf) + sl(D, "board") + si[:, :L.NB]
        n_it = L.NH + L.NS + L.NC
        items_f = f[:, fs["hand"][0]:fs["choice"][1]].view(B, n_it, L.IF)
        items = self.f_item(items_f) + sl(D, "hand", "choice") + si[:, L.NB:]
        of = f[:, fs["opp"][0]:fs["opp"][1]].view(B, L.NOPP, L.OPPF)
        obf = f[:, fs["oppboard"][0]:fs["oppboard"][1]].view(B, L.NOPP, L.NB, L.MF)
        obc = sl(E, "oppboard").view(B, L.NOPP, L.NB, -1)
        om = self.opp_minion(torch.cat([obf, obc.to(obf.dtype)], -1))
        opres = (obf[..., 0] > 0).to(om.dtype).unsqueeze(-1)
        pooled = (om * opres).sum(2) / opres.sum(2).clamp(min=1.0)
        ox = self.f_opp(of) + sl(D, "opphero") + pooled
        x = torch.cat([g, sl(D, "hero"), sl(D, "powers"), sl(D, "deity", "source"), q, sec, sl(D, "trinkets"),
                       board, items, ox], 1)
        x = x + (self.type_emb(self.type_ids) + self.slot_emb(self.slot_ids)).to(x.dtype)
        cc = c
        qc = sl(cc, "quests").view(B, L.NQ, -1)
        present = torch.cat([
            torch.ones(B, 1, dtype=torch.bool, device=f.device),
            sl(cc, "hero") > 0, sl(cc, "powers") > 0, sl(cc, "deity", "source") > 0, (qc > 0).any(2),
            (sl(cc, "secrets") > 0).any(1, keepdim=True), sl(cc, "trinkets") > 0, bf[:, :, 0] > 0,
            (items_f[:, :, 0] > 0) | (sl(cc, "hand", "choice") > 0), of[:, :, 0] > 0], 1)
        return x, present

    def _tokens_slow(self, f: torch.Tensor, c: torch.Tensor):
        B = f.shape[0]
        L, fs, cs, d = self.L, self.fs, self.cs, self.d
        dev = f.device
        cardv = lambda name: self.card(c[:, cs[name][0]:cs[name][1]])           # noqa: E731
        toks, pres, types = [], [], []

        def add(x, p, t):
            n = x.shape[1]
            toks.append(x + self.type_emb.weight[t] + self.slot_emb.weight[:n])
            pres.append(p)
            types.append(t)

        gi, si = self.sense_input(f, c)
        g = self.f_global(f[:, fs["global"][0]:fs["global"][1]]).unsqueeze(1) + gi
        add(g, torch.ones(B, 1, dtype=torch.bool, device=dev), 0)
        hero = self.card_to_d(cardv("hero"))
        add(hero, c[:, cs["hero"][0]:cs["hero"][1]] > 0, 1)
        pw = self.card_to_d(cardv("powers"))
        add(pw, c[:, cs["powers"][0]:cs["powers"][1]] > 0, 2)
        ds = self.card_to_d(torch.cat([cardv("deity"), cardv("source")], 1))
        add(ds, torch.cat([c[:, cs["deity"][0]:cs["deity"][1]], c[:, cs["source"][0]:cs["source"][1]]], 1) > 0, 3)
        q = cardv("quests").view(B, L.NQ, 3, -1).sum(2)
        qc = c[:, cs["quests"][0]:cs["quests"][1]].view(B, L.NQ, 3)
        add(self.card_to_d(q), (qc > 0).any(2), 4)
        sec = cardv("secrets").sum(1, keepdim=True)
        add(self.card_to_d(sec), (c[:, cs["secrets"][0]:cs["secrets"][1]] > 0).any(1, keepdim=True), 5)
        add(self.card_to_d(cardv("trinkets")), c[:, cs["trinkets"][0]:cs["trinkets"][1]] > 0, 6)
        bf = f[:, fs["board"][0]:fs["board"][1]].view(B, L.NB, L.MF)
        add(self.f_minion(bf) + self.card_to_d(cardv("board")) + si[:, :L.NB], bf[:, :, 0] > 0, 7)
        o = L.NB
        for name, t, n in (("hand", 8, L.NH), ("shop", 9, L.NS), ("choice", 10, L.NC)):
            it = f[:, fs[name][0]:fs[name][1]].view(B, n, L.IF)
            add(self.f_item(it) + self.card_to_d(cardv(name)) + si[:, o:o + n],
                (it[:, :, 0] > 0) | (c[:, cs[name][0]:cs[name][1]] > 0), t)
            o += n
        of = f[:, fs["opp"][0]:fs["opp"][1]].view(B, L.NOPP, L.OPPF)
        obf = f[:, fs["oppboard"][0]:fs["oppboard"][1]].view(B, L.NOPP, L.NB, L.MF)
        obc = self.card(c[:, cs["oppboard"][0]:cs["oppboard"][1]]).view(B, L.NOPP, L.NB, -1)
        om = self.opp_minion(torch.cat([obf, obc], -1))
        opres = (obf[..., 0] > 0).float().unsqueeze(-1)
        pooled = (om * opres).sum(2) / opres.sum(2).clamp(min=1.0)
        ox = self.f_opp(of) + self.card_to_d(cardv("opphero")) + pooled
        add(ox, of[:, :, 0] > 0, 11)
        x = torch.cat(toks, 1)
        present = torch.cat(pres, 1)
        return x, present

    def encode(self, x: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        """Transformer。pack=True 时每行只取有内容的词（挪到前面、截到本批最多的个数）再算，算完放回原位。
        没有位置编码、空词被注意力屏蔽，所以和整行一起算结果相同。"""
        if not self.pack:
            return self.norm(self.enc(x, src_key_padding_mask=~present))
        B, T, d = x.shape
        order = torch.argsort((~present).to(torch.int32), dim=1, stable=True)   # 有内容的在前
        tm = int(present.sum(1).max())
        idx = order[:, :tm]
        xp = torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, d))
        pp = torch.gather(present, 1, idx)
        hp = self.norm(self.enc(xp, src_key_padding_mask=~pp))
        h = torch.zeros(B, T, d, dtype=hp.dtype, device=x.device)
        return h.scatter(1, idx.unsqueeze(-1).expand(-1, -1, d), hp)

    def forward(self, f: torch.Tensor, c: torch.Tensor, mask: torch.Tensor, compressed: bool = False,
                slow: bool = False):
        """返回 (动作 logits [B,245]（非法 = -1e9），价值 [B]，名次 logits [B,8])。
        compressed=True：f 已经做过 soft_log（训练数据按压缩后的半精度存，避免溢出）。"""
        x_in = f.float() if compressed else soft_log(f.float())
        x, present = (self._tokens_slow if slow else self._tokens)(x_in, c.long())
        h = self.encode(x, present)
        h = h * present.unsqueeze(-1).to(h.dtype)
        idx = {}
        o = 0
        for name, n in (("global", 1), ("hero", 1), ("powers", self.L.NPW), ("ds", 2), ("quests", self.L.NQ),
                        ("secrets", 1), ("trinkets", 8), ("board", self.L.NB), ("hand", self.L.NH),
                        ("shop", self.L.NS), ("choice", self.L.NC), ("opp", self.L.NOPP)):
            idx[name] = (o, o + n)
            o += n
        tok = lambda name: h[:, idx[name][0]:idx[name][1]]                       # noqa: E731
        g = tok("global")[:, 0]
        board, hand = tok("board"), tok("hand")
        hg = self.h_global(g)
        B = f.shape[0]
        mag = torch.einsum("bhd,bkd->bhk", self.mag_q(hand), self.mag_k(board)) / math.sqrt(self.d)
        parts = [hg[:, 0:4],                                   # 0 结束 1 刷新 2 冻结 3 升本
                 self.h_power(tok("powers")).squeeze(-1),       # 4..5 英雄技能
                 hg[:, 4:6],                                   # 6 黑暗之赐 7 取消
                 self.h_buy(tok("shop")).squeeze(-1),           # 8..15
                 self.h_sell(board).squeeze(-1),                # 16..22
                 self.h_activate(board).squeeze(-1),            # 23..29
                 self.h_choose(tok("choice")).squeeze(-1),      # 30..45
                 self.h_play(hand).reshape(B, -1),              # 46..125
                 mag.reshape(B, -1),                            # 126..195
                 self.h_move(board).reshape(B, -1)]             # 196..244
        # 混合精度（float16）下 -1e9 会溢出：屏蔽和后面的概率计算都在 float32 里做
        logits = torch.cat(parts, 1).float()
        assert logits.shape[1] == self.L.actions, logits.shape
        logits = logits.masked_fill(~mask, -1e9)
        # 保险：某一行出现 NaN / inf（数值溢出）时，这一行退回到合法动作里均匀选，不让整个训练崩掉
        bad = ~torch.isfinite(logits).all(-1)
        self.last_bad = int(bad.sum())             # 训练程序据此判断出没出问题（不能悄悄掩盖）
        if self.last_bad:
            logits = torch.where(bad.unsqueeze(-1), torch.zeros_like(logits).masked_fill(~mask, -1e9), logits)
        pooled = h.sum(1) / present.sum(1, keepdim=True).clamp(min=1)
        z = g + pooled
        v = torch.nan_to_num(torch.tanh(self.v_head(z).float()).squeeze(-1), nan=0.0)
        self.last_comp_logits = self.comp_head(z).float()          # 训练的辅助损失用（ppo.update 读）
        return logits, v, torch.nan_to_num(self.p_head(z).float(), nan=0.0)

    @torch.no_grad()
    def act(self, f, c, mask, greedy: bool = False):
        logits, v, _ = self(f, c, mask)
        if greedy:
            a = logits.argmax(-1)
        else:
            a = torch.distributions.Categorical(logits=logits).sample()
        logp = F.log_softmax(logits, -1).gather(1, a.unsqueeze(1)).squeeze(1)
        return a, logp, v


# 模型大小预设（论文同训练量下：3.5M / 8M / 18M → 平均名次 3.84 / 3.77 / 3.67）
SIZES = {
    "small": dict(d=192, layers=4, heads=4, dc=64),     # 约 2.3M
    "medium": dict(d=320, layers=6, heads=8, dc=96),    # 约 8M（默认）
    "large": dict(d=384, layers=8, heads=8, dc=128),    # 约 15M（接近论文最终版）
}


_COMP_TABLES: list = []


def comp_tables():
    """rl/target_comps.json 的阵容查表（rl/comp.model_tables，进程里只算一次）；读不了时 None（体系感知输入全是 0）。"""
    if _COMP_TABLES:
        return _COMP_TABLES[0]
    try:
        from .comp import model_tables
        _COMP_TABLES.append(model_tables())
        return _COMP_TABLES[0]
    except Exception as ex:                             # noqa: BLE001
        print(f"阵容查表读不了（{type(ex).__name__}: {ex}），体系感知输入全是 0", flush=True)
        return None


def build_model(d: int = 192, layers: int = 4, heads: int = 4, dc: int = 64, tables="auto") -> EntityTransformer:
    import taverngym._core as c
    n, raw = c.card_static()
    static = np.frombuffer(raw, dtype="<f4").reshape(-1, n).copy()
    tables = comp_tables() if isinstance(tables, str) else tables
    return EntityTransformer(Layout.from_taverngym(), static, d=d, layers=layers, heads=heads, dc=dc, comp_tables=tables)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def load_model(path: str | Path, device: str = "cpu"):
    """读 latest.pt（含模型大小）或对手库快照（只有参数，大小从同一训练目录的 config.json 读）。
    外部模型（rl/extnet.py，data\\external\\model.pt）也能读：不用 torch.load，返回接口一样的模型。"""
    path = Path(path)
    from .extnet import is_external, load_external
    if is_external(path):
        return load_external(path, device)
    ck = torch.load(path, map_location=device, weights_only=False)
    info = {"path": str(path), "mtime": path.stat().st_mtime}
    if isinstance(ck, dict) and "model" in ck:
        state, kw = ck["model"], ck.get("model_kwargs")
        info["iter"] = ck.get("iter")
    else:
        state, kw = ck, None
    if not kw:
        cfg = None
        for d in (path.parent, path.parent.parent):
            if (d / "config.json").exists():
                cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
                break
        kw = dict(SIZES[(cfg or {}).get("size", "medium")])
        for k in ("d", "layers", "heads", "dc"):
            if cfg and cfg.get(k):
                kw[k] = cfg[k]
    if any(torch.is_tensor(v) and v.is_floating_point() and not torch.isfinite(v).all() for v in state.values()):
        raise ValueError(f"{path.name} 的参数有 NaN / inf（训练出过问题），不能用")
    model = build_model(**kw).to(device).eval()
    model.load_state_dict(state)
    info["size"] = kw
    return model, info
