"""C1 大脑：插件局面 → TavernGym 局面 → 模型给出本回合方案（影子模式，只显示、不操作）。

方案 = 在重建的 TavernGym 局面里让模型一步一步选概率最高的动作，直到结束回合；遇到刷新就停
（刷出什么是随机的，之后的步骤要等真的刷了再算）。另外给出第一步的几个备选：概率、做完之后价值头估计的名次。

注意：TavernGym 里的随机数和真实对局无关（重建时随便设的种子），所以不存在“复制局面泄露未来随机数”的问题；
但凡结果随机的步骤（刷新、发现的选项）都只能等真实结果出来再看下一步。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .model import SIZES, build_model

from tg.actions import (A_ACTIVATE, A_BUY, A_CANCEL, A_CHOOSE, A_DARK_GIFT, A_END, A_FREEZE, A_MAGNETIZE, A_MOVE,
                        A_PLAY, A_REFRESH, A_SELL, A_UPGRADE)
from tg.actions import A_HERO_POWER as A_HP


def _name(x) -> str:
    if not isinstance(x, dict):
        return "?"
    for k in ("name_zh", "name"):
        if x.get(k):
            return x[k] + ("（金）" if x.get("golden") else "")
    for k in ("minion", "card", "item", "reward"):
        if isinstance(x.get(k), dict):
            return _name(x[k])
    return x.get("id") or "?"


def describe(me: dict, a: int) -> str:
    """把动作编号翻译成中文（用动作前的局面）。"""
    board, hand, shop = me.get("board") or [], me.get("hand") or [], me.get("shop") or []
    at = lambda lst, i: lst[i] if 0 <= i < len(lst) else {}         # noqa: E731
    if a == A_END:
        return "结束回合"
    if a == A_REFRESH:
        return "刷新酒馆"
    if a == A_FREEZE:
        return "冻结酒馆"
    if a == A_UPGRADE:
        return "升级酒馆"
    if A_HP <= a < A_DARK_GIFT:
        return f"英雄技能：{_name(at(me.get('hero_powers') or [], a - A_HP))}"
    if a == A_DARK_GIFT:
        return "黑暗之赐"
    if a == A_CANCEL:
        return "取消"
    if A_BUY <= a < A_SELL:
        return f"买 {_name(at(shop, a - A_BUY))}"
    if A_SELL <= a < A_ACTIVATE:
        return f"卖 {_name(at(board, a - A_SELL))}（第 {a - A_SELL + 1} 位）"
    if A_ACTIVATE <= a < A_CHOOSE:
        return f"发动 {_name(at(board, a - A_ACTIVATE))}（第 {a - A_ACTIVATE + 1} 位）"
    if A_CHOOSE <= a < A_PLAY:
        i = a - A_CHOOSE
        pend = me.get("pending") or {}
        if pend.get("targets") is not None:
            eid = at(pend["targets"], i) if isinstance(at(pend["targets"], i), int) else None
            for lst, where in ((board, "场上"), (shop, "酒馆"), (hand, "手牌")):
                for k, m in enumerate(lst):
                    if m.get("eid") == eid:
                        return f"目标：{where}{_name(m)}" + (f"（第 {k + 1} 位）" if where == "场上" else "")
            return f"目标 {i + 1}"
        kind = {"discover": "发现", "hero": "选英雄", "choose_one": "抉择", "trinket": "选饰品"}.get(pend.get("kind"),
                                                                                              "选择")
        return f"{kind}：{_name(at(pend.get('options') or [], i))}"
    if A_PLAY <= a < A_MAGNETIZE:
        h, pos = divmod(a - A_PLAY, 8)
        it = at(hand, h)
        if it.get("type") == "spell" or it.get("kind") in ("spell", "tavern_spell"):
            return f"使用 {_name(it)}"
        return f"打出 {_name(it)}，放在第 {pos + 1} 位"
    if A_MAGNETIZE <= a < A_MOVE:
        h, b = divmod(a - A_MAGNETIZE, 7)
        return f"把 {_name(at(hand, h))} 磁力吸附到 {_name(at(board, b))}（第 {b + 1} 位）"
    f, t = divmod(a - A_MOVE, 7)
    return f"把 {_name(at(board, f))} 从第 {f + 1} 位移到第 {t + 1} 位"


def op_type(me: dict, a: int) -> str:
    """动作类别（插件面板上色用）。"""
    if a == A_END:
        return "end"
    if a == A_REFRESH:
        return "refresh"
    if a == A_FREEZE:
        return "freeze"
    if a == A_UPGRADE:
        return "upgrade"
    if A_HP <= a < A_DARK_GIFT:
        return "hero"
    if A_BUY <= a < A_SELL:
        return "buy"
    if A_SELL <= a < A_ACTIVATE:
        return "sell"
    if A_ACTIVATE <= a < A_CHOOSE:
        return "activate"
    if A_CHOOSE <= a < A_PLAY:
        return "choose"
    if A_PLAY <= a < A_MAGNETIZE:
        hand = me.get("hand") or []
        h = (a - A_PLAY) // 8
        it = hand[h] if h < len(hand) else {}
        return "spell" if it.get("type") == "spell" or it.get("kind") in ("spell", "tavern_spell") else "play"
    if A_MAGNETIZE <= a < A_MOVE:
        return "play"
    if a >= A_MOVE:
        return "move"
    return "other"


def value_to_place(v: float) -> float:
    """名次奖励（第 1 名 1 … 第 8 名 -1，线性）→ 名次。"""
    return 1.0 + (1.0 - float(v)) * 3.5


def load_model(path: str | Path, device: str = "cpu"):
    """读 latest.pt（含模型大小）或对手库快照（只有参数，大小从同一训练目录的 config.json 读）。"""
    path = Path(path)
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


class Advisor:
    def __init__(self, model, device="cpu", max_steps: int = 25, alternatives: int = 5):
        self.model, self.dev = model, torch.device(device)
        self.max_steps, self.n_alt = max_steps, alternatives

    @torch.no_grad()
    def _eval(self, g):
        f, c = g.observe(0)
        m = np.frombuffer(g.legal_mask(0), dtype=np.uint8).astype(bool)
        t = lambda x, dt: torch.as_tensor(np.array(x)[None], dtype=dt, device=self.dev)   # noqa: E731
        logits, v, pl = self.model(t(np.frombuffer(f, "<f4"), torch.float32), t(np.frombuffer(c, "<i4"), torch.long),
                                   t(m, torch.bool))
        probs = F.softmax(logits[0], -1).cpu().numpy()
        pdist = F.softmax(pl[0], -1).cpu().numpy()
        return probs, float(v[0]), pdist, m

    @staticmethod
    def _sig(g) -> str:
        """局面签名（不含站位顺序和冻结状态）：用来发现“做了等于没做”的动作和来回绕圈。"""
        me = json.loads(g.view(0))["me"]
        key = lambda lst: sorted((x.get("id"), x.get("atk"), x.get("hp"), bool(x.get("golden"))) for x in lst or [])  # noqa: E731
        return json.dumps([me.get("gold"), me.get("tier"), key(me.get("board")), key(me.get("hand")), key(me.get("shop")),
                           bool(me.get("pending"))], sort_keys=True, default=str)

    def plan(self, g) -> dict:
        """g = 已重建的 TavernGym 局面（0 号座位 = 你）。不会改动 g。
        方案只列“有实际效果”的步骤：调站位只在最后给一次最终站位，冻结只在结束前提一次；
        做完局面没变化（或回到之前出现过的局面）的动作跳过，改选下一个概率最高的。"""
        t0 = time.time()
        probs, v, pdist, mask = self._eval(g)
        me = json.loads(g.view(0))["me"]
        out = {"value": round(v, 3), "valuePlace": round(value_to_place(v), 2),
               "expectedPlace": round(value_to_place(v), 2),          # 用价值头（名次头训练标签有偏差，前期偏悲观）
               "headPlace": round(float((pdist * np.arange(1, 9)).sum()), 2),
               "placeDist": [round(float(x), 3) for x in pdist], "steps": [], "alternatives": [], "note": ""}
        # 第一步的备选（不含调站位、冻结）
        legal = [a for a in np.flatnonzero(mask) if a < A_MOVE and a != A_FREEZE]
        for a in sorted(legal, key=lambda x: -probs[x])[:self.n_alt]:
            g2 = g.copy()
            try:
                g2.step(0, int(a))
                v2 = self._eval(g2)[1] if g2.needs_action(0) else v
            except ValueError:
                continue
            out["alternatives"].append({"action": int(a), "text": describe(me, int(a)), "type": op_type(me, int(a)),
                                        "prob": round(float(probs[a]), 3), "placeAfter": round(value_to_place(v2), 2)})
        # 贪心走完本回合（跳过没效果的动作）
        g2 = g.copy()
        seen = {self._sig(g2)}
        want_freeze = False
        moved = False
        for _ in range(self.max_steps):
            if not g2.needs_action(0):
                break
            p, _, _, m = self._eval(g2)
            me2 = json.loads(g2.view(0))["me"]
            order = [int(x) for x in np.argsort(-np.where(m, p, -1)) if m[x]]
            chosen = None
            for a in order[:12]:
                if a == A_FREEZE:
                    want_freeze = want_freeze or p[a] > 0.2
                    continue
                if a >= A_MOVE:
                    moved = True
                    continue
                if a == A_END:
                    chosen = a
                    break
                g3 = g2.copy()
                try:
                    g3.step(0, a)
                except ValueError:
                    continue
                sig = self._sig(g3)
                if sig in seen:
                    continue
                chosen = a
                break
            if chosen is None:
                chosen = A_END
            if chosen == A_END:
                if want_freeze:
                    out["steps"].append({"action": A_FREEZE, "text": "冻结酒馆（结束前）", "type": "freeze",
                                         "prob": round(float(p[A_FREEZE]), 3)})
                out["steps"].append({"action": A_END, "text": "结束回合", "type": "end", "prob": round(float(p[A_END]), 3)})
                break
            out["steps"].append({"action": chosen, "text": describe(me2, chosen), "type": op_type(me2, chosen),
                                 "prob": round(float(p[chosen]), 3)})
            g2.step(0, chosen)
            seen.add(self._sig(g2))
            if chosen == A_REFRESH:
                out["note"] = "刷新之后的步骤要看刷出了什么，刷新后会重新给方案"
                break
        else:
            out["note"] = f"超过 {self.max_steps} 步，后面省略"
        if moved:
            out["note"] = (out["note"] + "；" if out["note"] else "") + "模型还想调整站位（站位以战斗前的模拟为准）"
        out["ms"] = round((time.time() - t0) * 1000)
        return out

