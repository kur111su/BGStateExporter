"""高手库里各套阵容的场面在 TavernGym 里有多强：同一回合，拿某套阵容的场面打 k 个随机的同回合高分场面，算胜率。

为什么要它（第 1900 轮）：强制模型走龙虾流练了约 1400 轮，定向测试还是 5.6 名；写死规则的龙虾流机器人更差。
先确认“高手搭好的龙虾流场面，在这个模拟器的战斗里本身强不强”——强，问题在模型怎么搭；不强，就是模拟器 /
对手库里这套阵容本来就吃亏（比如高手库的场面只有卡牌和身材，没有玩家的龙虾加成这类计数）。

  bg.py compstrength                       第 8–14 回合，所有 S / A 阵容 + “其它”
  bg.py compstrength --comps 龙虾流野兽 --fights 6 --k 24

输出：回合 × 阵容的平均胜率（0.5 = 和同回合随便一个高分场面五五开）、场面数。
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paths import RUNS  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comps", default="", help="只看这几套（目标阵容名字，逗号隔开）；默认所有 S / A")
    ap.add_argument("--turns", default="8,9,10,11,12,13,14")
    ap.add_argument("--k", type=int, default=16, help="每个场面打几个随机的同回合高分场面")
    ap.add_argument("--fights", type=int, default=4, help="每对打几场")
    ap.add_argument("--max", type=int, default=40, help="每回合每套阵容最多测几个场面")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lib", default="", help="阵容库文件（分号隔开）；默认 runs/pro_boards.json;runs/h13_boards.json")
    a = ap.parse_args(argv)

    from rl.boardsim import lab_game, put_board, strength
    from rl.comp import formed, load_target_comps
    from rl.curricula.boost import merge_pro_boards
    from tg.catalog import base_id

    paths = [Path(x) for x in a.lib.split(";") if x] if a.lib else [RUNS / "pro_boards.json", RUNS / "h13_boards.json"]
    paths = [p for p in paths if p.exists()]
    if not paths:
        print("没有高分阵容库（runs/pro_boards.json / runs/h13_boards.json）")
        return 1
    lib = {int(t): v for t, v in merge_pro_boards(paths).items() if v}
    want = [x.strip() for x in a.comps.split(",") if x.strip()]
    comps = [c for c in load_target_comps() if c["tier"] in "SA" and (not want or c["name"] in want)]
    rng = random.Random(a.seed)
    turns = [int(x) for x in a.turns.split(",") if x.strip()]
    res: dict = {}
    for t in turns:
        boards = lib.get(t) or []
        if not boards:
            continue
        groups: dict = {}
        for e in boards:
            ids = [base_id(str(x[0])) for x in e["board"][:7]]
            hit = [c["name"] for c in comps if formed(ids, c)]
            groups.setdefault(hit[0] if hit else "其它", []).append(e)
        g = lab_game(seed=a.seed, turn=t)
        for name, es in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            if want and name not in want and name != "其它":
                continue
            es = rng.sample(es, min(a.max, len(es)))
            wr = []
            for e in es:
                put_board(g, 0, e["board"], e.get("tier"))
                for o in rng.sample(boards, min(a.k, len(boards))):
                    if put_board(g, 1, o["board"], o.get("tier")):
                        try:
                            wr.append(strength(g, 0, 1, a.fights))
                        except BaseException:          # noqa: BLE001 —— 引擎内部出错：这场不算
                            pass
            if wr:
                res.setdefault(name, {})[t] = (sum(wr) / len(wr), len(es))
        print(f"第 {t} 回合测完（{len(boards)} 个场面）", flush=True)
    names = sorted(res, key=lambda n: -sum(v[1] for v in res[n].values()))
    print("\n阵容 | " + " | ".join(f"第{t}回合" for t in turns) + "    （同回合胜率；括号里是测了几个场面）")
    for n in names:
        row = [f"{res[n][t][0]:.2f}({res[n][t][1]})" if t in res[n] else "   -    " for t in turns]
        print(f"{n} | " + " | ".join(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
