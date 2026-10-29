"""竞技场策略：官方连接器 arena_client.py 用 --policy agent:act 调用。模型只在本机推理，不上传任何东西。

输入：竞技场的 observe2（4017 浮点 + 447 卡牌编号）的前 3227 / 124 个就是我们训练用的 observe() v1；
卡牌编号按 prep.py 核对出的对应表换成训练词表的编号（词表一样时不变）。
动作：模型最可能的合法动作（不随机）。另外三条保护（和影子模式一样的考虑）：
  - 同一回合调站位 / 冻结超过 2 次后不再选它们（模型偶尔会来回调站位）；
  - 本回合操作次数只剩 2 次、或回合时间只剩 1.5 秒时，能结束就结束；
  - 每局在 arena/logs/<对局>.jsonl 记每回合各类操作的次数（不记观测、不记密钥）。
模型在 import 时加载（连接器建立连接之前），第一步不会因为加载耗时超时。
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent


def _import_model():
    """按文件路径导入 rl/model.py，不把 bgrl 目录加进 sys.path：bgrl 下有个 taverngym 文件夹（只放版本号），
    加进去之后连接器会把它当成装了 TavernGym 去核对，然后出错。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("bgrl_rl_model", HERE.parent / "rl" / "model.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod                   # dataclass 需要能按名字找到模块
    spec.loader.exec_module(mod)
    return mod


_model_mod = _import_model()
EntityTransformer, Layout = _model_mod.EntityTransformer, _model_mod.Layout

A_END, A_REFRESH, A_FREEZE, A_CANCEL, A_CHOOSE, A_PLAY, A_MAGNETIZE, A_MOVE = 0, 1, 2, 7, 30, 46, 126, 196
N_POS = 8
KIND = [(0, "end"), (1, "refresh"), (2, "freeze"), (3, "upgrade"), (4, "hero_power"), (6, "dark_gift"), (7, "cancel"),
        (8, "buy"), (16, "sell"), (23, "activate"), (30, "choose"), (46, "play"), (126, "magnetize"), (196, "move")]


def kind_of(a: int) -> str:
    k = "?"
    for start, name in KIND:
        if a >= start:
            k = name
    return k


def _load():
    bundle = HERE / "model_bundle"
    compat_path = HERE / "compat.json"
    if not compat_path.exists():
        raise SystemExit("还没有核对竞技场契约（先运行 arena.bat check）")
    compat = json.loads(compat_path.read_text(encoding="utf-8"))
    if not compat.get("ok"):
        raise SystemExit(f"竞技场契约核对没通过：{compat.get('problems')}")
    meta = json.loads((bundle / "meta.json").read_text(encoding="utf-8"))
    if compat.get("model_version") != meta["model_version"]:
        raise SystemExit("模型换过了，重新运行 arena.bat check")
    L = Layout.from_dict(meta["layout"], meta["floats"], meta["cards"], meta["actions"])
    ck = torch.load(bundle / "model.pt", map_location="cpu", weights_only=True)
    model = EntityTransformer(L, np.load(bundle / "static.npy"), **ck["model_kwargs"]).eval()
    model.load_state_dict(ck["model"])
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    remap = np.load(HERE / "remap.npy")
    return model, meta, remap


def _no_taverngym():
    """连接器发现能 import 的 taverngym 就会拿它和竞技场引擎比，所以这个环境里不能有（哪怕是个同名空文件夹）。"""
    import importlib.util
    import os
    if os.environ.get("BGRL_ARENA_TEST") == "1":           # 只给 tests/test_arena.py 用（训练环境里本来就装着 TavernGym）
        return
    spec = importlib.util.find_spec("taverngym")
    if spec is not None:
        where = spec.origin or list(spec.submodule_search_locations or [])
        raise SystemExit(f"连接器的环境里找得到一个 taverngym：{where}。它来自 PYTHONPATH 或当前目录；"
                         f"arena.bat 用 -E -s 运行时不该出现，请把这行发给我")


_no_taverngym()
MODEL, META, REMAP = _load()
NF, NC = META["floats"], META["cards"]
LOGS = HERE / "logs"
print(f"模型 {META['model_version']} 已加载（CPU）", file=sys.stderr, flush=True)


def _log(st: dict, match_id: str):
    if not st.get("counts"):
        return
    LOGS.mkdir(exist_ok=True)
    with open(LOGS / f"{match_id}.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"turn": st["turn"], "phase": st.get("phase"), "actions": dict(st["counts"]),
                             "guards": dict(st["guards"]), "model": META["model_version"],
                             "time": time.strftime("%H:%M:%S")}, ensure_ascii=False) + "\n")


@torch.no_grad()
def act(observation, context):
    st = context["state"]
    key = (context["match_id"], context["turn"], context.get("phase"))
    if st.get("key") != key:                       # 新回合：写上一回合的统计
        if st.get("key"):
            _log(st, st["key"][0])
        st.update(key=key, turn=context["turn"], phase=context.get("phase"), counts=Counter(), guards=Counter(), noops=0,
                  banned=set(), cancels=0, trigger=None, used=Counter())
    mask = np.asarray(observation["action_mask"], dtype=bool)
    left = int(context.get("action_limit", 100)) - int(context.get("actions_used", 0))
    if mask[A_END] and (left <= 2 or int(context.get("remaining_ms", 10 ** 9)) < 1500):
        st["guards"]["end_early"] += 1
        st["counts"]["end"] += 1
        return A_END
    m = mask.copy()
    if st["noops"] >= 2:
        m[A_FREEZE] = False
        m[A_MOVE:] = False
        if not m.any():
            m = mask.copy()
        elif (m != mask).any():
            st["guards"]["no_more_moves"] += 1
    # 打出 → 取消 → 再打出……的死循环（第一局练习第 9–12 回合每回合 49 次打出 + 49 次取消，直到 100 次上限）：
    # 取消过的那张手牌本回合不再打出；本回合取消 2 次后不再取消（只能选目标，局面会往前走）；
    # 除结束、刷新、选择以外，同一个动作编号本回合最多 6 次
    for b in st["banned"]:
        m[b] = False
    if st["cancels"] >= 2:
        m[A_CANCEL] = False
    for b, n in st["used"].items():
        if n >= 6 and b not in (A_END, A_REFRESH) and not (A_CHOOSE <= b < A_PLAY):
            m[b] = False
    if not m.any():
        m = mask.copy()
    elif (m != mask).any() and (st["banned"] or st["cancels"] >= 2):
        st["guards"]["loop_block"] += 1
    f = np.array(observation["features"][:NF], dtype=np.float32)        # 复制：连接器给的数组是只读的
    c = np.asarray(observation["cards"][:NC], dtype=np.int64)
    c = REMAP[np.clip(c, 0, len(REMAP) - 1)]
    logits, _, _ = MODEL(torch.from_numpy(f)[None], torch.from_numpy(c)[None], torch.from_numpy(m)[None])
    lg = logits[0].float().masked_fill(~torch.from_numpy(m), float("-inf"))
    a = int(torch.argmax(F.log_softmax(lg, -1)))
    if not mask[a]:                                # 不会发生；防御一下
        a = int(np.flatnonzero(mask)[0])
    if a == A_FREEZE or a >= A_MOVE:
        st["noops"] += 1
    st["used"][a] += 1
    if a == A_CANCEL:
        st["cancels"] += 1
        t = st["trigger"]
        if t is not None and A_PLAY <= t < A_MAGNETIZE:        # 取消的是打出：这张手牌的所有位置都禁掉
            h = (t - A_PLAY) // N_POS
            st["banned"].update(range(A_PLAY + h * N_POS, A_PLAY + (h + 1) * N_POS))
        elif t is not None:
            st["banned"].add(t)
    elif not (A_CHOOSE <= a < A_PLAY):
        st["trigger"] = a
    st["counts"][kind_of(a)] += 1
    if a == A_END:
        _log(st, context["match_id"])
        st["counts"], st["guards"] = Counter(), Counter()
    return a
