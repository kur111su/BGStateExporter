"""高分对局（Firestone 6500+ 分享包的 core/games.jsonl）→ 训练用的两个文件：

  runs/pro_boards.json   对手棋盘库：按回合存 6500+ 对局里战斗开始时的整板（卡牌 ID + 攻 / 血）和估计的酒馆等级。
                         第二阶段加强机器人用（rl/calibrate.py 的 ProBoardBooster）：把机器人场面直接换成同一回合的
                         真人高分阵容，机器人带上真的高本随从和效果，靠买增益法术堆身材的打法就不灵了。
                         只在本机（runs/ 不进 Git）：里面是别人对局的阵容，数据说明要求不要再公开分发。
  rl/pro_profile.json    汇总曲线（只有统计数字，可以进仓库）：每回合平均酒馆等级、对手场面总攻防四分位、
                         平均买 / 卖 / 刷新次数。升本节奏奖励用它当目标曲线（--tempo-curve pro）。

只收 TavernGym 认识的卡组成的整板（3 个含未实现随从的板子跳过）；按 MMR 过滤用 --min-mmr。

用法：bg.py prolib <解压后的分享包目录或 games.jsonl> [--min-mmr 6500]
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paths import ROOT, RUNS  # noqa: E402


def _games(path: Path):
    if path.is_dir():
        for cand in (path / "core" / "games.jsonl", path / "games.jsonl"):
            if cand.exists():
                path = cand
                break
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _q(v: list, q: float) -> float:
    v = sorted(v)
    if not v:
        return 0.0
    i = (len(v) - 1) * q
    lo, hi = int(i), min(int(i) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (i - lo)


def build(src: Path, min_mmr: int = 0):
    import taverngym
    cat = {c["id"]: c for c in json.loads(taverngym.catalog_json())}

    def known(cid: str):
        base = cid[:-2] if cid.endswith("_G") else cid
        c = cat.get(base)
        return c if c and c.get("kind") == "minion" and c.get("implemented", True) else None

    boards = defaultdict(list)
    tier_at, opp_tot, behave = defaultdict(list), defaultdict(list), defaultdict(lambda: defaultdict(list))
    n_games = skipped = 0
    for g in _games(src):
        mmr = int(g.get("player_rank") or 0)
        if mmr < min_mmr:
            continue
        n_games += 1
        tt = sorted((d for d in g.get("tavern_timings") or [] if d.get("tavernTier")), key=lambda d: d["turn"])
        last = max([b["turn"] for b in g.get("board_history") or []] or [0])

        def tier_of(t: int) -> int:
            tier = 1
            for d in tt:
                if d["turn"] <= t:
                    tier = int(d["tavernTier"])
            return tier
        for t in range(1, last + 1):
            tier_at[t].append(tier_of(t))
        for name in ("minions_bought", "minions_sold", "rerolls", "coins_wasted"):
            for d in g.get(name) or []:
                if d.get("value") is not None and 1 <= d["turn"] <= last:
                    behave[name][d["turn"]].append(d["value"])
        for key in ("opponent_combat_boards", "own_combat_boards"):
            for cb in g.get(key) or []:
                t = int(cb.get("turn") or 0)
                brd = cb.get("board") or []
                if t < 1 or not brd:
                    continue
                cards = [known(m["card_id"]) for m in brd]
                if not all(cards):
                    skipped += 1
                    continue
                tot = sum((m.get("attack") or 0) + (m.get("health") or 0) for m in brd)
                if key == "opponent_combat_boards":
                    opp_tot[t].append(tot)
                # 对手的酒馆等级不在数据里：取“场上最高等级随从”和“同回合高分玩家平均等级”中较大的
                est = max(max(int(c.get("tier") or 1) for c in cards), round(st.mean(tier_at[t])) if tier_at[t] else 1) \
                    if key == "opponent_combat_boards" else tier_of(t)
                boards[t].append({"tier": min(6, est), "mmr": mmr,
                                  "board": [[m["card_id"], int(m.get("attack") or 0),
                                             max(1, int(m.get("health") or 0) - int(m.get("damage") or 0))]
                                            for m in sorted(brd, key=lambda m: m.get("position") or 0)][:7]})
    turns = sorted(set(tier_at) | set(opp_tot))
    # 等级曲线优先用回放里读出来的（runs/pro_demos/tiers.json，tools/make_pro_demos.py 顺带写的）：
    # Firestone 的 tavern_timings 有时自相矛盾（例如第 4 回合 5 本、第 7 回合 4 本），会把前期曲线抬高
    rt = RUNS / "pro_demos" / "tiers.json"
    tier_src = "Firestone tavern_timings"
    if rt.exists():
        per = json.loads(rt.read_text(encoding="utf-8"))
        tier_at = defaultdict(list)
        for g in per.values():
            for t, v in g.items():
                if int(t) >= 1 and v:
                    tier_at[int(t)].append(int(v))
        tier_src = f"回放（{len(per)} 局，{rt}）"
    profile = {"source": "Firestone 6500+ 精彩对局（他人对局，策展池，不是无偏抽样）", "tier_source": tier_src, "games": n_games,
               "min_mmr": min_mmr, "created": time.strftime("%Y-%m-%d"), "turns": {}}
    for t in turns:
        row = {"n": len(tier_at[t])}
        if tier_at[t]:
            row["tier"] = round(st.mean(tier_at[t]), 2)
            row["tier_dist"] = {str(k): round(tier_at[t].count(k) / len(tier_at[t]), 3) for k in range(1, 7)}
        if opp_tot[t]:
            row.update(boardQ1=round(_q(opp_tot[t], .25)), boardMedian=round(_q(opp_tot[t], .5)),
                       boardQ3=round(_q(opp_tot[t], .75)), n_boards=len(opp_tot[t]))
        for name, d in behave.items():
            if d.get(t):
                row[name] = round(st.mean(d[t]), 2)
        profile["turns"][str(t)] = row
    lib = {"source": profile["source"], "games": n_games, "min_mmr": min_mmr, "created": profile["created"],
           "skipped_unknown": skipped, "turns": {str(t): v for t, v in sorted(boards.items())}}
    return profile, lib


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="解压后的分享包目录（含 core/games.jsonl）或 games.jsonl")
    ap.add_argument("--min-mmr", type=int, default=6500)
    ap.add_argument("--boards-out", default=str(RUNS / "pro_boards.json"))
    ap.add_argument("--profile-out", default=str(ROOT / "rl" / "pro_profile.json"))
    a = ap.parse_args()
    profile, lib = build(Path(a.src), a.min_mmr)
    Path(a.boards_out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.boards_out).write_text(json.dumps(lib, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    old = Path(a.profile_out)
    if old.exists():                                   # bg.py goldflow --pro 写的高手金币流向留着
        gf = json.loads(old.read_text(encoding="utf-8")).get("gold_flow")
        if gf:
            profile["gold_flow"] = gf
    Path(a.profile_out).write_text(json.dumps(profile, ensure_ascii=False, indent=1), encoding="utf-8")
    n = sum(len(v) for v in lib["turns"].values())
    print(f"{profile['games']} 局（MMR ≥ {a.min_mmr}）→ {n} 个整板（跳过含未知卡的 {lib['skipped_unknown']} 个）→ {a.boards_out}；"
          f"汇总曲线 → {a.profile_out}")
    for t in (3, 5, 7, 9, 11, 13):
        r = profile["turns"].get(str(t)) or {}
        print(f"  第 {t} 回合：平均等级 {r.get('tier')}，对手场面中位 {r.get('boardMedian')}，"
              f"刷新 {r.get('rerolls')}，买 {r.get('minions_bought')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
