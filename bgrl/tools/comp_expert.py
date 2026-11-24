"""流派专家模板：按 rl/target_comps.json 里的一套阵容生成专家配方和蒸馏配方，打印整套流程的命令（docs/comp_expert.md）。

  bg.py comp-expert --comp 龙虾流野兽                     生成 configs/exp/<run>.json（专家）、打印流程命令
  bg.py comp-expert --comp 龙虾流野兽 --distill           再生成 configs/exp/distill_<run>.json（第二阶段：主模型向专家学）
  bg.py comp-expert --stats                               列出玩家 stats 里有哪些计数（找专属奖励用的 counter_key）

生成的专家配方（第一阶段，见 configs/exp/beast.json 的完整历史）：
  · 固定酒馆：必开这套阵容的种族，其余从弱种族里抽（lobby_must，考试也一样）；外部老师关掉
  · 起步牌：一半的机器人局第 3–9 回合往手里送这套的核心 / 配套卡（gift_comps）
  · 成型阵容课程：只换成这一套（seed_comp_names）
  · 高手示范里买这套的牌 ×5（pro_weights comp-<名字>）
  · 定向局 10%：第 3 回合起只能买这套 / 本族 / 法术；核心卡必买、必打、不能卖；升本节奏；
    5 本 / 6 本还缺高本核心卡就刷新找（commit_*）
  · 专属奖励：--counter <玩家计数> 时按计数增长给分（lobster_coef / counter_key）——要自己找，默认不开
  · 连招奖励（trick_bonus）只有进击流（--trick rally）能用，默认不开
生成以后先看一遍、按这套阵容的打法改（站位、连招、哪些牌必买），再开练。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

STRONG = ("QUILBOAR", "DEMON", "ELEMENTAL")
TIER_PLAN = "3:5,4:7,5:8,6:10"


def _base_pro_weights() -> str:
    v = json.loads((ROOT / "configs" / "v3.json").read_text(encoding="utf-8"))
    return str((v.get("config") or {}).get("pro_weights") or "")


def _comp(name: str) -> dict:
    v = json.loads((ROOT / "rl" / "target_comps.json").read_text(encoding="utf-8"))
    for c in v.get("comps", []):
        if c["name"] == name:
            return c
    names = "、".join(c["name"] for c in v.get("comps", []))
    raise SystemExit(f"rl/target_comps.json 里没有 {name!r}；有：{names}")


def _card_info(ids) -> dict:
    """卡牌编号 → (酒馆等级, 种族)；用 card_value.json（不用开模拟器）。"""
    v = json.loads((ROOT / "rl" / "card_value.json").read_text(encoding="utf-8"))
    cards = v.get("cards") or {}
    return {i: (int((cards.get(i) or {}).get("tier") or 0), (cards.get(i) or {}).get("name", i)) for i in ids}


def _tribe_of(c: dict) -> str:
    """这套阵容的种族：核心卡里出现最多的种族（要开模拟器的卡牌表）。"""
    from tg.catalog import catalog
    det = catalog()
    cnt: dict = {}
    for cid in c["core"]:
        for r in (det.get(cid) or {}).get("races") or []:
            if r != "ALL":
                cnt[r] = cnt.get(r, 0) + 1
    if not cnt:
        raise SystemExit(f"{c['name']} 的核心卡没有种族（中立阵容），用 --tribe 指定要固定的种族")
    return max(cnt, key=cnt.get)


def _hunt(core: list, info: dict) -> str:
    """5 本以后要找的核心卡（≥ 5 本的），按等级累加：5:A;6:A,B。"""
    hi = sorted({info[x][0] for x in core if info[x][0] >= 5})
    out = []
    for t in hi:
        ids = [x for x in core if 5 <= info[x][0] <= t]
        out.append(f"{t}:{','.join(ids)}")
    return ";".join(out)


def stats_keys() -> int:
    """跑一局到第 8 回合，列出玩家 stats 里的计数（专属奖励 counter_key 从这里挑）。"""
    import numpy as np
    import taverngym
    env = taverngym.BatchEnv(n_games=1, seed=1, controlled=(0,), threads=1)
    rng = np.random.default_rng(0)
    for _ in range(3000):
        o = env.observe()
        if int(np.rint(o["features"][0, 0] * 20)) >= 8:
            break
        legal = np.flatnonzero(o["action_mask"][0])
        env.step(np.array([int(rng.choice(legal))]))
    st = json.loads(env.game(0).state_json())["players"][0].get("stats") or {}
    print("玩家 stats 里的计数（第 8 回合左右，随机打法）：")
    for k in sorted(st):
        print(f"  {k} = {st[k]}")
    print("挑和这套阵容的成长有关的（例如 lobster_bonus = 美味龙虾的亡语触发次数），配方里设 counter_key 和 lobster_coef")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--comp", default="", help="rl/target_comps.json 里的阵容名字，例如 龙虾流野兽")
    ap.add_argument("--tribe", default="", help="固定开的种族（默认按核心卡的种族猜）")
    ap.add_argument("--exclude", default=",".join(STRONG), help="专家的酒馆里不开的种族（默认野猪人、恶魔、元素）")
    ap.add_argument("--run", default="", help="专家的训练目录名（runs/<run>），默认按种族起名")
    ap.add_argument("--counter", default="", help="专属奖励用的玩家计数（--stats 列出来的），例如 lobster_bonus")
    ap.add_argument("--trick", default="", choices=["", "rally"], help="招募阶段连招奖励（只有进击流用 rally）")
    ap.add_argument("--force", default="", help="定向局必买的牌（卡牌编号，逗号隔开）；默认核心卡")
    ap.add_argument("--distill", action="store_true", help="再生成第二阶段的蒸馏配方")
    ap.add_argument("--init", default="runs/v3/best.pt", help="专家从哪个模型开始练")
    ap.add_argument("--stats", action="store_true", help="只列出玩家 stats 里有哪些计数")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    if a.stats:
        return stats_keys()
    if not a.comp:
        ap.error("要 --comp（或 --stats）")
    c = _comp(a.comp)
    tribe = (a.tribe or _tribe_of(c)).upper()
    exclude = ",".join(t for t in a.exclude.split(",") if t and t.upper() != tribe)
    run = a.run or re.sub(r"[^a-z0-9_]+", "_", tribe.lower())
    core, supp = list(c["core"]), list(c.get("support") or [])
    info = _card_info(core + supp)
    force = a.force or ",".join(core)
    cfg = {
        "lobby_must": tribe, "lobby_frac": 1.0, "lobby_exam": True, "teacher_coef": 0.0,
        "pro_weights": _base_pro_weights() + f",lobnew:3,tgtnew:3,comp-{c['name']}:5",
        "gift_frac": 0.5, "gift_turns": "3,4,5,6,7,8,9", "gift_p": 0.8, "gift_comps": c["name"],
        "seed_comp_frac": 0.3, "seed_comp_turns": "5,6,7,8,9,10", "seed_comp_formed_from": 8, "seed_comp_names": c["name"],
        "commit_comp": c["name"], "commit_frac": 0.1, "commit_from_turn": 3, "commit_force": force,
        "commit_tiers": TIER_PLAN, "commit_hunt": _hunt(core, info),
    }
    if a.counter:
        cfg.update({"counter_key": a.counter, "lobster_coef": 0.03})
    if a.trick:
        cfg.update({"trick_bonus": 0.02, "trick_engine": a.trick})
    recipe = {"base": "v3", "run": f"runs/{run}",
              "description": f"{c['name']} 专家（bg.py comp-expert 生成）。核心：{'、'.join(c.get('core_names') or core)}；"
                             f"打法：{c.get('play', '')}。生成后按打法改：必买的牌、专属奖励、连招。",
              "config": cfg}
    out = ROOT / "configs" / "exp" / f"{run}.json"
    if out.exists() and not a.overwrite:
        print(f"{out.relative_to(ROOT)} 已经有了（要覆盖加 --overwrite）；下面的命令照样能用")
    else:
        out.write_text(json.dumps(recipe, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"写好专家配方 {out.relative_to(ROOT)}")
    engage = ",".join(core + supp)
    ex = {t.upper() for t in exclude.split(",") if t}
    pool = ",".join(t for t in ("BEAST", "DRAGON", "MECHANICAL", "MURLOC", "PIRATE", "UNDEAD", "ABERRATION") + STRONG
                    if t not in ex and t != tribe)
    if a.distill:
        dist = {"base": "v3", "run": "runs/v3",
                "description": f"第二阶段：主模型在 {c['name']} 的局面上向专家 runs/{run}/best.pt 学"
                               f"（15% 的局强制开 {tribe}、不开 {exclude}；其它局照常）。",
                "config": {"lobby_must": tribe, "lobby_pool": pool, "lobby_frac": 0.15, "lobby_exam": False,
                           "experts": json.dumps([{"ckpt": f"runs/{run}/best.pt", "name": c["name"], "must": tribe,
                                                   "exclude": exclude, "engage": engage, "from_turn": 3,
                                                   "coef": 0.05, "conf": 0.4}], ensure_ascii=False)}}
        dout = ROOT / "configs" / "exp" / f"distill_{run}.json"
        if dout.exists() and not a.overwrite:
            print(f"{dout.relative_to(ROOT)} 已经有了（要覆盖加 --overwrite）")
        else:
            dout.write_text(json.dumps(dist, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            print(f"写好蒸馏配方 {dout.relative_to(ROOT)}")
    py = ".venv/bin/python -X utf8 bg.py"
    cf = f'--commit {c["name"]} --commit-force {force} --commit-tiers {TIER_PLAN}' + \
         (f' --commit-hunt "{cfg["commit_hunt"]}"' if cfg["commit_hunt"] else "")
    print(f"""
流程（docs/comp_expert.md）：
 0. 这套阵容在模拟器里强不强（同回合对高手场面的胜率，0.5 = 五五开）：
    {py} compstrength --comps {c['name']}
 1. 起点：主模型在这种酒馆里怎么打
    {py} goldflow --boosted --games 256 --lobby {tribe} --ckpt {a.init}
 2. 练专家（中断后把 --init … 换成 --resume）：
    {py} train --recipe configs/exp/{run}.json --init {a.init}
 3. 每几百轮测一次：
    会不会打（定向）：{py} goldflow --boosted --games 128 --lobby {tribe} {cf} --ckpt runs/{run}/snapshots/itNNNNN.pt
    会不会自己选（自由）：{py} goldflow --boosted --games 256 --lobby {tribe} --ckpt runs/{run}/snapshots/itNNNNN.pt
    看：这套成型率、成型的局名次、走这套的局和别的局的名次差、核心卡的购买倍率
 4. 专家合格（自由测试里走这套的局不比别的差、成型的局明显更好）以后，第二阶段：
    {py} comp-expert --comp {c['name']} --distill
    {py} train --recipe configs/exp/distill_{run}.json --resume
    对比：主模型 best.pt 蒸馏前后，在这种酒馆和随机酒馆的 goldflow 名次""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
