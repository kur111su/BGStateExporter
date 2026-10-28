"""用 TavernBench（AutoTavern 论文的评测集，368 题）测我们的模型，和论文公布的 TavernAgent、大语言模型、Jev 比。

用法：python tavernbench.py [--ckpt 检查点.pt] [--name 名字] [--device cpu] [--smoke]
  不给 --ckpt：用当前训练目录（paths.DEFAULT_RUN）的 latest.pt，没有就用最近更新的训练目录的。
  结果：bgrl/reports/tavernbench/<名字>/answers.jsonl（每题的回答），report.txt / report.json（成绩表）。

我们的模型是一个下棋用的策略网络（选动作 + 估名次），不会读文字题，所以按题型这样答：
  选项题（E3 买哪几个、O 出牌顺序、T 选目标、D3 发现、D5 先打哪张）：每个选项在模拟器里照做，
      用价值头估计做完后的名次，选最好的。
  站位题（P）：把场面排成每个选项的顺序，用价值头比。
  选英雄（D1）：每个英雄的开局局面用价值头比。
  选饰品、发现、回合计划（D2、D4、N1、N2）：直接用策略头，选概率最高的那个动作。
  战斗预测（R2、B1–B6，问胜率、伤害、谁先攻）和规则计算题（R1、R3–R7、E1、E2、E4、E5）：
      模型不输出这些量，不作答（按评测规则算错）。另外单独算一个“能答的 23 个场景”的成绩。
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
TB = ROOT / "third_party" / "tavernbench"
sys.path.insert(0, str(TB))
from paths import DEFAULT_RUN, REPORTS  # noqa: E402

SET = TB / "data" / "eval"
OPTION_SCEN = {"E3", "O1", "O2", "O3", "O4", "O5", "T1", "T2", "T3", "T4", "T5", "D3", "D5"}
PERM_SCEN = {"P1", "P2", "P3", "P4", "P5"}
POLICY_SCEN = {"D2", "D4", "N1", "N2"}
STATES_SCEN = {"D1"}
ANSWERABLE = OPTION_SCEN | PERM_SCEN | POLICY_SCEN | STATES_SCEN
COMPARE = ["tavernagent", "claude-opus-5.5", "gpt-6-astra", "gemini-3.8-flash", "jev"]


class Net:
    """模型的价值头和策略头（单个局面）。"""

    def __init__(self, model, device="cpu"):
        import torch
        self.torch, self.model, self.dev = torch, model.eval(), torch.device(device)

    def __call__(self, g, seat=0):
        import torch.nn.functional as F
        torch = self.torch
        f, c = g.observe(seat)
        m = np.frombuffer(g.legal_mask(seat), dtype=np.uint8).astype(bool)
        mm = m.copy()
        if not mm.any():
            mm[:] = True
        t = lambda x, dt: torch.as_tensor(np.array(x)[None], dtype=dt, device=self.dev)   # noqa: E731
        with torch.inference_mode():
            logits, v, _ = self.model(t(np.frombuffer(f, "<f4"), torch.float32), t(np.frombuffer(c, "<i4"), torch.long),
                                      t(mm, torch.bool))
        return F.softmax(logits[0].float(), -1).cpu().numpy(), float(v[0]), m


def arrange(g, order: list[int]) -> bool:
    """把我方场面排成 order（原来的位置编号，从左到右）。"""
    from tavernbench.engine import A_MOVE, N_BOARD, legal, me
    eids = [m["eid"] for m in me(g)["board"]]
    want = [eids[i] for i in order]
    for k, e in enumerate(want):
        cur = [m["eid"] for m in me(g)["board"]].index(e)
        if cur != k:
            a = A_MOVE + cur * N_BOARD + k
            if a not in legal(g):
                return False
            g.step(0, a)
    return [m["eid"] for m in me(g)["board"]][:len(want)] == want


def answer(task: dict, st: dict, net: Net):
    """返回 (回答 JSON 对象, 说明)；不作答返回 (None, 原因)。"""
    from tavernbench.engine import apply_ops, from_state
    sc = task["scenario"]
    if sc in OPTION_SCEN:
        best = None
        for opt in st["options"]:
            g = from_state(st["state"])
            if not apply_ops(g, opt["ops"]):
                continue
            v = net(g)[1]
            if best is None or v > best[0]:
                best = (v, opt["answer"])
        if best is None:
            return {"answer": st["options"][0]["answer"]}, "所有选项都执行不了，答第一个"
        return {"answer": best[1]}, f"价值 {best[0]:.3f}"
    if sc in PERM_SCEN:
        best = None
        for perm, letter in zip(st["perms"], st["answers"]):
            g = from_state(st["state"])
            if not arrange(g, perm):
                continue
            v = net(g)[1]
            if best is None or v > best[0]:
                best = (v, letter)
        return ({"answer": best[1]}, f"价值 {best[0]:.3f}") if best else ({"answer": st["answers"][0]}, "排不了")
    if sc in STATES_SCEN:
        vals = [net(from_state(s))[1] for s in st["states"]]
        k = int(np.argmax(vals))
        return {"answer": st["answers"][k]}, "价值 " + " / ".join(f"{v:.3f}" for v in vals)
    if sc in POLICY_SCEN:
        g = from_state(st["state"])
        seat = int(st.get("seat") or 0)
        probs, _, m = net(g, seat)
        mp = {int(k): v for k, v in st["map"].items()}
        cand = [a for a in mp if 0 <= a < len(probs)]
        a = max(cand, key=lambda x: probs[x])
        return {"answer": mp[a]}, "策略 " + " / ".join(f"{mp[x]}:{probs[x]:.2f}" for x in cand)
    return None, "模型不输出这类量（战斗预测 / 规则计算）"


def load_net(ckpt: str, device: str, smoke: bool):
    if smoke:
        import torch
        from rl.model import build_model
        torch.manual_seed(0)
        return Net(build_model(d=64, layers=2, heads=2, dc=32), device), {"iter": 0, "path": "随机小模型（冒烟测试）"}
    from rl.advisor import load_model
    p = Path(ckpt) if ckpt else DEFAULT_RUN / "latest.pt"
    if not p.exists():
        runs = sorted(DEFAULT_RUN.parent.glob("*/latest.pt"), key=lambda x: x.stat().st_mtime)
        if not runs:
            raise SystemExit(f"找不到模型：{p}")
        p = runs[-1]
    model, info = load_model(p, device)
    return Net(model, device), info


def _kind(a: int) -> str:
    return {0: "结束", 1: "刷新", 3: "升本"}.get(a, "买" if 8 <= a < 16 else "其它")


def plan_summary(answers_path: Path) -> str:
    """回合计划题（N1、N2）：正确答案是哪类第一步、模型选了哪类，以及选错平均多掉几个名次（标签里 128 局实测）。"""
    if not answers_path.exists():
        return ""
    got = {}
    for x in answers_path.read_text(encoding="utf-8").splitlines():
        if x.strip():
            r = json.loads(x)
            got[r["id"]] = r
    labels = {d["id"]: d for d in map(json.loads, (SET / "labels.jsonl").read_text(encoding="utf-8").splitlines())}
    maps = {}
    with gzip.open(SET / "states.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            if d["id"][:2] in ("N1", "N2"):
                maps[d["id"]] = {v: int(k) for k, v in d["map"].items()}
    best, mine = Counter(), Counter()
    loss = []
    for iid, inv in maps.items():
        r = got.get(iid)
        if not r or not r.get("response"):
            continue
        lab = labels[iid]
        my = json.loads(r["response"])["answer"]
        best[_kind(inv[lab["answer"]])] += 1
        mine[_kind(inv[my])] += 1
        acts, mp = lab["meta"]["actions"], lab["meta"]["mean_place"]
        loss.append(mp[acts.index(inv[my])] - min(mp))
    if not loss:
        return ""
    fmt = lambda c: "、".join(f"{k} {v}" for k, v in c.most_common())   # noqa: E731
    return (f"回合计划题（N1、N2，共 {len(loss)} 题）：正确的第一步 {fmt(best)}；模型选的 {fmt(mine)}；"
            f"平均每题比最好的计划多掉 {sum(loss) / len(loss):.2f} 名")


def report(name: str, out_dir: Path) -> str:
    from tavernbench.chance import expected
    from tavernbench.report import evaluate, load, summarize
    labels, prompts = load(SET)
    rows = {}
    ch = defaultdict(list)
    for iid, p in prompts.items():
        ch[p["scenario"]].append(expected(labels[iid], p))
    rows["随机作答（chance）"] = {s: sum(v) / len(v) for s, v in ch.items()}
    rows[f"我们的模型 {name}"], _ = evaluate(labels, prompts, out_dir / "answers.jsonl")
    for s in COMPARE:
        path = TB / "results" / "eval" / s / "answers.jsonl"
        if path.exists():
            rows[s], _ = evaluate(labels, prompts, path)
    cols = ["combat", "positioning", "ordering", "targets", "selection", "strategy", "economy"]
    zh = {"combat": "战斗", "positioning": "站位", "ordering": "顺序", "targets": "目标", "selection": "选择",
          "strategy": "计划", "economy": "经济"}
    lines = [f"TavernBench 评测集 368 题（30 个对局场景 + 10 个规则场景）；模型 {name}",
             "“对局总分” = 30 个对局场景的平均正确率（论文的 Play 分，没答的算错）；",
             "“能答的 23 个” = 我们的模型能作答的 23 个场景的平均（去掉战斗预测 R2、B1–B6）。", "",
             f"{'系统':24s} 对局总分 能答的23个 " + " ".join(f"{zh[c]:>4s}" for c in cols)]
    out = {}
    for sysname, acc in rows.items():
        s = summarize(acc)
        sub = [v for k, v in acc.items() if k in ANSWERABLE]
        s["answerable23"] = sum(sub) / max(1, len(sub))
        out[sysname] = {**s, "by_scenario": acc}
        lines.append(f"{sysname:24s} {s['play']:8.2f} {s['answerable23']:10.2f} "
                     + " ".join(f"{s.get(c, float('nan')):6.2f}" for c in cols))
    lines += ["", "分场景（我们的模型 / TavernAgent / 随机作答）："]
    me_, ta, chs = out[f"我们的模型 {name}"]["by_scenario"], out.get("tavernagent", {}).get("by_scenario", {}), \
        out["随机作答（chance）"]["by_scenario"]
    for sc in sorted(me_):
        if sc in ANSWERABLE:
            lines.append(f"  {sc:4s} {me_[sc]:.2f} / {ta.get(sc, float('nan')):.2f} / {chs.get(sc, float('nan')):.2f}")
    lines += ["", plan_summary(out_dir / "answers.jsonl")]
    (out_dir / "report.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    text = "\n".join(lines)
    (out_dir / "report.txt").write_text(text + "\n", encoding="utf-8")
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--name", default="")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--smoke", action="store_true", help="随机小模型、每个场景 2 题（CI 用）")
    a = ap.parse_args()
    net, info = load_net(a.ckpt, a.device, a.smoke)
    src = Path(str(info.get("path", "")))
    name = a.name or ("smoke" if a.smoke else f"{src.parent.name}-it{info.get('iter')}")
    print(f"模型：{info.get('path')}（第 {info.get('iter')} 轮）", flush=True)
    prompts = [json.loads(x) for x in (SET / "prompts.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    states = {}
    with gzip.open(SET / "states.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            states[d["id"]] = d
    out_dir = REPORTS / "tavernbench" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    per = defaultdict(int)
    n_ans = 0
    t_all = time.time()
    with open(out_dir / "answers.jsonl", "w", encoding="utf-8") as fh:
        for p in prompts:
            sc = p["scenario"]
            if a.smoke and per[sc] >= 2:
                continue
            per[sc] += 1
            t0 = time.time()
            st = states.get(p["id"])
            obj, why = (None, "没有局面") if st is None else answer(p, st, net)
            row = {"id": p["id"], "model": name, "response": None if obj is None else json.dumps(obj, ensure_ascii=False),
                   "error": None if obj is not None else why, "note": why, "latency_s": round(time.time() - t0, 4)}
            if obj is None:
                continue                          # 不作答：不写这一行（评测按没答算错）
            n_ans += 1
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"作答 {n_ans} 题，用时 {time.time() - t_all:.0f} 秒", flush=True)
    print(report(name, out_dir), flush=True)
    print("详细：", out_dir, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
