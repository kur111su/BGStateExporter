"""外部模型：jimmypretend/hearthstone-bg-ppo（作者 PrentendYan，CC BY-NC 4.0，只能非商业使用、要署名）。

权重不在仓库里：自己下载 model.pt 放到 bgrl/data/external/（data/ 不进 Git），这里只有读它、跑它的代码。
网络结构照作者 2026-11-13 公开的代码（github.com/PrentendYan/hearthstone-bg-ppo，hsbg_net/net_cz46.py）：

  词（125 个）：全局 1 | 场面 7 | 手牌 10 | 酒馆 8 | 待选 16 | 对手 7 | 对手场面 49 | 英雄、技能、神祇、来源、任务、奥秘 19 | 饰品 8
  卡 = card[编号] + attr_proj(attr[编号]) + z_tok(zcard[z, 编号])；有浮点特征的词再加 f_board / f_item / f_opp / f_oppb；
  全局词 = f_global(全局特征) + z_global[:, z]；每个词加位置嵌入 pos。z 默认 45 = none（zcard 那一行全 0）
  输入是引擎的原始特征（不压缩）；4 层先归一化的 Transformer（前馈层 ReLU），注意力不屏蔽空词；最后 LayerNorm
  头：h_global([全局, 125 个词的平均]) → 动作 0..7；买 / 卖 / 激活 / 选 / 打出 / 磁力（q_mag(手牌)·场面，不缩放）/ 移动
  价值：value.2(gelu(value.0([全局, 平均])))

（作者公开代码之前，这里是从权重反推的结构，有四处不对：输入多做了一次压缩、前馈层用了 GELU、平均只算有内容的词、
磁力多除了 √d。2026-11-14 改成和作者的一样。）

我们的训练数据存的是压缩过的特征（soft_log，float16）：compressed=True 时先还原再喂给它。
allow_moves=False（默认）：还有别的动作可选时不让它挪随从。
读文件不用 torch.load：只认 OrderedDict、_rebuild_tensor_v2、FloatStorage 三个对象，张量直接从 zip 里的原始字节读。
"""
from __future__ import annotations

import collections
import io
import math
import pickle
import zipfile
from pathlib import Path

import numpy as np

FORMAT = "tg-ckpt-v1"
# 观察布局（taverngym.layout()）
GF, MF, IF, OPPF = 68, 31, 40, 9
NB, NH, NS, NC, NOPP = 7, 10, 8, 16, 7
F_BOARD = (GF, GF + NB * MF)
F_ITEMS = (F_BOARD[1], F_BOARD[1] + (NH + NS + NC) * IF)
F_OPP = (F_ITEMS[1], F_ITEMS[1] + NOPP * OPPF)
F_OPPB = (F_OPP[1], F_OPP[1] + NOPP * NB * MF)
N_FLOATS = F_OPPB[1]
# (名字, 卡牌编号位置, 浮点特征范围, 每个词的特征数, 投影)；按词的顺序
GROUPS = (
    ("board", list(range(19, 26)), F_BOARD, MF, "f_board"),
    ("hand", list(range(26, 36)), (F_ITEMS[0], F_ITEMS[0] + NH * IF), IF, "f_item"),
    ("shop", list(range(36, 44)), (F_ITEMS[0] + NH * IF, F_ITEMS[0] + (NH + NS) * IF), IF, "f_item"),
    ("choice", list(range(44, 60)), (F_ITEMS[0] + (NH + NS) * IF, F_ITEMS[1]), IF, "f_item"),
    ("opp", list(range(68, 75)), F_OPP, OPPF, "f_opp"),
    ("oppb", list(range(75, 124)), F_OPPB, MF, "f_oppb"),
    ("misc", list(range(0, 19)) + list(range(60, 68)), None, 0, None),
)
C_ORDER = np.array([i for g in GROUPS for i in g[1]], dtype=np.int64)
OFF = {}
_o = 1
for _g in GROUPS:
    OFF[_g[0]] = (_o, _o + len(_g[1]))
    _o += len(_g[1])
N_TOK = _o
A_MOVE = 196


# ---------------------------------------------------------------- 读文件（不执行任意代码）
class _Storage:
    def __init__(self, key, n):
        self.key, self.n = key, n


def _rebuild_tensor_v2(storage, offset, size, stride, requires_grad=False, backward_hooks=None, metadata=None):
    return ("tensor", storage, int(offset), tuple(int(x) for x in size), tuple(int(x) for x in stride))


class _FloatStorage:
    pass


_ALLOWED = {("collections", "OrderedDict"): collections.OrderedDict,
            ("torch._utils", "_rebuild_tensor_v2"): _rebuild_tensor_v2,
            ("torch", "FloatStorage"): _FloatStorage}


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in _ALLOWED:
            raise pickle.UnpicklingError(f"外部模型文件里有不允许的对象 {module}.{name}，不读")
        return _ALLOWED[(module, name)]

    def persistent_load(self, pid):
        if not (isinstance(pid, tuple) and len(pid) == 5 and pid[0] == "storage" and pid[1] is _FloatStorage):
            raise pickle.UnpicklingError(f"外部模型文件里有不认识的存储 {pid!r:.80}")
        return _Storage(str(pid[2]), int(pid[4]))


def is_external(path) -> bool:
    """是不是外部模型（tg-ckpt-v1）的文件：zip 里有 */data.pkl，且 pickle 里写着 format = tg-ckpt-v1。"""
    try:
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.endswith("/data.pkl")]
            return bool(names) and FORMAT.encode() in z.read(names[0])[:4096]
    except (OSError, zipfile.BadZipFile, KeyError):
        return False


def load_checkpoint(path) -> dict:
    """读成 {format, net_config, state_dict（numpy float32）, meta}。"""
    with zipfile.ZipFile(path) as z:
        root = next(n for n in z.namelist() if n.endswith("/data.pkl"))[:-len("data.pkl")]
        if z.read(root + "byteorder").strip() != b"little":
            raise ValueError("外部模型不是小端字节序")
        obj = _SafeUnpickler(io.BytesIO(z.read(root + "data.pkl"))).load()
        cache: dict = {}

        def conv(x):
            if isinstance(x, tuple) and len(x) == 5 and x[0] == "tensor":
                _, st, off, size, stride = x
                if st.key not in cache:
                    cache[st.key] = np.frombuffer(z.read(root + "data/" + st.key), dtype="<f4")
                n = int(np.prod(size)) if size else 1
                if stride != tuple(int(np.prod(size[i + 1:])) for i in range(len(size))):
                    raise ValueError("外部模型的张量不是连续存储")
                return cache[st.key][off:off + n].reshape(size).astype(np.float32)
            if isinstance(x, dict):
                return type(x)((k, conv(v)) for k, v in x.items()) if isinstance(x, collections.OrderedDict) \
                    else {k: conv(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return type(x)(conv(v) for v in x)
            return x
        ck = conv(obj)
    if not isinstance(ck, dict) or ck.get("format") != FORMAT or "state_dict" not in ck:
        raise ValueError(f"不是 {FORMAT} 格式的外部模型")
    return ck


def soft_log(x):
    a = np.abs(x)
    return np.where(a <= 1, x, np.sign(x) * (1 + np.log(np.maximum(a, 1)))).astype(np.float32)


# ---------------------------------------------------------------- numpy 参考实现（没有 torch 也能跑；测试拿来对 torch 版）
def soft_exp(y):
    """soft_log 的逆：|y| ≤ 1 不变，否则 sign × exp(|y| − 1)。"""
    a = np.abs(y)
    return np.where(a <= 1, y, np.sign(y) * np.exp(np.minimum(a, 40.0) - 1.0)).astype(np.float32)


def _erf(x):
    s, a = np.sign(x), np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * a)
    return s * (1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592)
                * t * np.exp(-a * a))


def _ln(x, w, b):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + 1e-5) * w + b


def forward_np(ck: dict, f, c, compressed: bool = False):
    """→ (logits [B,245] 未屏蔽, 价值 [B])。"""
    s = ck["state_dict"]
    cfg = ck["net_config"]
    d, H = int(cfg["d"]), int(cfg["heads"])
    z_none = list(cfg["z_vocab"]).index("none")
    f = np.asarray(f, dtype=np.float32)
    c = np.asarray(c, dtype=np.int64)
    B = f.shape[0]
    x = soft_exp(f) if compressed else f              # 作者的网络吃原始特征
    lin = lambda n, t: t @ s[n + ".weight"].T + s[n + ".bias"]                    # noqa: E731
    emb = s["card.weight"] + s["attr"] @ s["attr_proj.weight"].T
    cc = np.clip(c[:, C_ORDER], 0, emb.shape[0] - 1)
    tok = np.empty((B, N_TOK, d), dtype=np.float32)
    tok[:, 0] = lin("f_global", x[:, :GF]) + s["z_global.weight"][:, z_none]
    tok[:, 1:] = emb[cc]
    present = np.ones((B, N_TOK), dtype=bool)
    present[:, 1:] = cc > 0
    for name, _, fs, nf, proj in GROUPS:
        if fs is None:
            continue
        a0, a1 = OFF[name]
        tok[:, a0:a1] += lin(proj, x[:, fs[0]:fs[1]].reshape(B, a1 - a0, nf))
        present[:, a0:a1] |= x[:, fs[0]:fs[1]].reshape(B, a1 - a0, nf)[:, :, 0] > 0
    h = tok + s["pos"][None]
    hd = d // H
    for i in range(int(cfg["layers"])):
        p = f"encoder.layers.{i}."
        y = _ln(h, s[p + "norm1.weight"], s[p + "norm1.bias"])
        qkv = y @ s[p + "self_attn.in_proj_weight"].T + s[p + "self_attn.in_proj_bias"]
        q = qkv[..., :d].reshape(B, N_TOK, H, hd).transpose(0, 2, 1, 3) * hd ** -0.5
        k = qkv[..., d:2 * d].reshape(B, N_TOK, H, hd).transpose(0, 2, 3, 1)
        v = qkv[..., 2 * d:].reshape(B, N_TOK, H, hd).transpose(0, 2, 1, 3)
        att = q @ k
        att = np.exp(att - att.max(-1, keepdims=True))
        att /= att.sum(-1, keepdims=True)
        o = (att @ v).transpose(0, 2, 1, 3).reshape(B, N_TOK, d)
        h = h + o @ s[p + "self_attn.out_proj.weight"].T + s[p + "self_attn.out_proj.bias"]
        y = _ln(h, s[p + "norm2.weight"], s[p + "norm2.bias"])
        u = lin(p + "linear1", y)
        h = h + lin(p + "linear2", np.maximum(u, 0.0))
    h = _ln(h, s["norm.weight"], s["norm.bias"])
    gp = np.concatenate([h[:, 0], h.mean(1)], -1)
    sl = lambda n: h[:, OFF[n][0]:OFF[n][1]]                                      # noqa: E731
    board, hand = sl("board"), sl("hand")
    mag = np.einsum("bhd,bkd->bhk", lin("q_mag", hand), board)
    logits = np.concatenate([lin("h_global", gp), lin("h_buy", sl("shop"))[..., 0], lin("h_sell", board)[..., 0],
                             lin("h_activate", board)[..., 0], lin("h_choose", sl("choice"))[..., 0],
                             lin("h_play", hand).reshape(B, -1), mag.reshape(B, -1), lin("h_move", board).reshape(B, -1)], 1)
    u = lin("value.0", gp)
    v = lin("value.2", 0.5 * u * (1.0 + _erf(u / math.sqrt(2.0))))[:, 0]
    return logits.astype(np.float32), v.astype(np.float32)


# ---------------------------------------------------------------- torch 版（训练当老师、测试工具用）
def build_torch(ck: dict, allow_moves: bool = False):
    """→ nn.Module，接口和 rl.model 的模型一样：forward(f, c, mask, compressed=False) → (logits（非法 = -1e9）, 价值, 名次 logits)。
    名次 logits 没有（全 0）；价值是作者的效用尺度。参数全冻住。"""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from .model import Layout

    s = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in ck["state_dict"].items()}
    cfg = ck["net_config"]
    d, H, L = int(cfg["d"]), int(cfg["heads"]), int(cfg["layers"])
    z_none = list(cfg["z_vocab"]).index("none")

    class ExtModel(nn.Module):
        external = True

        def __init__(self):
            super().__init__()
            self.L = Layout.from_taverngym()
            if self.L.floats != N_FLOATS or self.L.cards != 124 or self.L.actions != 245:
                raise ValueError("外部模型的输入格式和现在的 TavernGym 不一样（换过版本？），不能用")
            self.d = d
            self.allow_moves = bool(allow_moves)
            self.register_buffer("emb", s["card.weight"] + s["attr"] @ s["attr_proj.weight"].T)
            self.register_buffer("pos", s["pos"].clone())
            self.register_buffer("g0", s["z_global.weight"][:, z_none].clone())
            self.register_buffer("c_order", torch.from_numpy(C_ORDER))
            mk = lambda n, i, o: self._lin(n, i, o)                               # noqa: E731
            self.f_global, self.f_board, self.f_item = mk("f_global", GF, d), mk("f_board", MF, d), mk("f_item", IF, d)
            self.f_opp, self.f_oppb = mk("f_opp", OPPF, d), mk("f_oppb", MF, d)
            layer = nn.TransformerEncoderLayer(d, H, 4 * d, dropout=0.0, activation="relu", batch_first=True,
                                               norm_first=True)
            self.enc = nn.TransformerEncoder(layer, L, enable_nested_tensor=False)
            self.enc.load_state_dict({k[len("encoder."):]: v for k, v in s.items() if k.startswith("encoder.")})
            self.norm = nn.LayerNorm(d)
            self.norm.load_state_dict({"weight": s["norm.weight"], "bias": s["norm.bias"]})
            self.h_global, self.h_buy, self.h_sell = mk("h_global", 2 * d, 8), mk("h_buy", d, 1), mk("h_sell", d, 1)
            self.h_activate, self.h_choose = mk("h_activate", d, 1), mk("h_choose", d, 1)
            self.h_play, self.q_mag, self.h_move = mk("h_play", d, 8), mk("q_mag", d, d), mk("h_move", d, 7)
            self.v0, self.v2 = mk("value.0", 2 * d, d), mk("value.2", d, 1)
            for q in self.parameters():
                q.requires_grad_(False)
            self.eval()

        @staticmethod
        def _lin(name, i, o):
            m = nn.Linear(i, o)
            m.load_state_dict({"weight": s[name + ".weight"], "bias": s[name + ".bias"]})
            return m

        def train(self, mode: bool = True):                                      # 一直是推理模式
            return super().train(False)

        def forward(self, f, c, mask, compressed: bool = False, slow: bool = False):
            x = f.float()
            if compressed:                              # 训练数据是压缩过的：还原成原始特征
                ax = x.abs()
                x = torch.where(ax <= 1, x, torch.sign(x) * torch.exp(ax.clamp(max=40.0) - 1.0))
            B = x.shape[0]
            cc = c.long()[:, self.c_order].clamp(0, self.emb.shape[0] - 1)
            tok = [self.f_global(x[:, :GF]) + self.g0]
            parts, pres = [self.emb[cc]], [cc > 0]
            feat = []
            for name, idx, fs, nf, proj in GROUPS:
                if fs is None:
                    continue
                n = len(idx)
                xs = x[:, fs[0]:fs[1]].reshape(B, n, nf)
                feat.append(getattr(self, proj)(xs))
                pres.append(xs[:, :, 0] > 0)
            E = parts[0]
            nfeat = OFF["misc"][0] - 1
            E = torch.cat([E[:, :nfeat] + torch.cat(feat, 1), E[:, nfeat:]], 1)
            h = torch.cat([tok[0].unsqueeze(1), E], 1) + self.pos
            present = torch.ones(B, N_TOK, dtype=torch.bool, device=x.device)
            present[:, 1:] = pres[0]
            present[:, 1:OFF["misc"][0]] |= torch.cat(pres[1:], 1)
            h = self.norm(self.enc(h))
            gp = torch.cat([h[:, 0], h.mean(1)], -1)
            sl = lambda n: h[:, OFF[n][0]:OFF[n][1]]                              # noqa: E731
            board, hand = sl("board"), sl("hand")
            mag = torch.einsum("bhd,bkd->bhk", self.q_mag(hand), board)
            lg = torch.cat([self.h_global(gp), self.h_buy(sl("shop")).squeeze(-1), self.h_sell(board).squeeze(-1),
                            self.h_activate(board).squeeze(-1), self.h_choose(sl("choice")).squeeze(-1),
                            self.h_play(hand).reshape(B, -1), mag.reshape(B, -1), self.h_move(board).reshape(B, -1)], 1)
            m = mask.bool()
            if not self.allow_moves:
                other = m[:, :A_MOVE].any(-1, keepdim=True)
                m = m & ~(other & (torch.arange(m.shape[1], device=m.device) >= A_MOVE))
            lg = lg.float().masked_fill(~m, -1e9)
            v = self.v2(F.gelu(self.v0(gp))).squeeze(-1).float()
            return lg, v, torch.zeros(B, 8, device=x.device)

    return ExtModel()


def load_external(path, device: str = "cpu", allow_moves: bool = False):
    """外部模型文件 → (模型, 信息)。和 rl.model.load_model 返回的一样用。"""
    path = Path(path)
    ck = load_checkpoint(path)
    meta = ck.get("meta") or {}
    model = build_torch(ck, allow_moves=allow_moves).to(device)
    info = {"path": str(path), "mtime": path.stat().st_mtime, "iter": meta.get("iter"), "external": True,
            "size": {k: ck["net_config"].get(k) for k in ("d", "layers", "heads")},
            "license": "CC BY-NC 4.0（jimmypretend/hearthstone-bg-ppo，非商业、要署名）"}
    return model, info
