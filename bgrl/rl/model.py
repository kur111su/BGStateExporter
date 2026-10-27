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

import math
from dataclasses import dataclass

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
        L = dict(c.layout())
        return cls(L["MINION_F"], L["ITEM_F"], L["GLOBAL_F"], L["N_QUESTS"], L["QUEST_F"], L["PENDING_KINDS"],
                   L["OPP_F"], L["N_OPP"], L["N_BOARD"], L["N_HAND"], L["N_SHOP"], L["N_CHOICE"], L["N_POS"],
                   L["N_POWERS"], c.OBS_FLOATS, c.OBS_CARDS, c.N_ACTIONS)

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


class EntityTransformer(nn.Module):
    N_TYPES = 12

    def __init__(self, layout: Layout, static: np.ndarray, d: int = 192, layers: int = 4, heads: int = 4,
                 dc: int = 64):
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

    # ---------------- 编码 ----------------

    def _tokens(self, f: torch.Tensor, c: torch.Tensor):
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

        g = self.f_global(f[:, fs["global"][0]:fs["global"][1]]).unsqueeze(1)
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
        add(self.f_minion(bf) + self.card_to_d(cardv("board")), bf[:, :, 0] > 0, 7)
        for name, t, n in (("hand", 8, L.NH), ("shop", 9, L.NS), ("choice", 10, L.NC)):
            it = f[:, fs[name][0]:fs[name][1]].view(B, n, L.IF)
            add(self.f_item(it) + self.card_to_d(cardv(name)), (it[:, :, 0] > 0) | (c[:, cs[name][0]:cs[name][1]] > 0), t)
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

    def forward(self, f: torch.Tensor, c: torch.Tensor, mask: torch.Tensor):
        """返回 (动作 logits [B,245]（非法 = -1e9），价值 [B]，名次 logits [B,8])。"""
        x, present = self._tokens(f, c.long())
        h = self.norm(self.enc(x, src_key_padding_mask=~present))
        h = h * present.unsqueeze(-1)
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
        pooled = h.sum(1) / present.sum(1, keepdim=True).clamp(min=1)
        z = g + pooled
        return logits, torch.tanh(self.v_head(z).float()).squeeze(-1), self.p_head(z).float()

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


def build_model(d: int = 192, layers: int = 4, heads: int = 4, dc: int = 64) -> EntityTransformer:
    import taverngym._core as c
    n, raw = c.card_static()
    static = np.frombuffer(raw, dtype="<f4").reshape(-1, n).copy()
    return EntityTransformer(Layout.from_taverngym(), static, d=d, layers=layers, heads=heads, dc=dc)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
