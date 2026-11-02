"""训练配方（configs/*.json，tools/train.py --recipe）：

- 配方里的每一项都是 Config 的字段；
- train.bat 现在的命令（配方 + 命令行）得到的设置，和改用配方之前 train.bat 那一长串参数完全一样；
- 命令行参数优先于配方。
"""
from __future__ import annotations

import dataclasses
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 改用配方之前 train.bat 的参数（去掉了 INIT，处理器个数换成 8）
OLD = ('--run runs\\v3 --resume --iters 100000 --threads 8 --boost-frac 0.75 --boost-kind pro --boost-from-turn 5 '
       '--tempo-curve pro --bc-coef 0.05 --bc-weights "upgrade:3,sell:3,buy:2" --bc-train-cap 0.6 --bc-smooth 0.1 '
       '--pro-coef 0.1 --pro-batch 64 --pro-weights "upgrade:3,end:2,buy:1.5,sell:1.5,key:2,eco:2,tribe:2,syn:3,t2-5:2" '
       '--tempo-bonus 0.03 --tempo-deficit 0.06 --refresh-waste-penalty 0.015 --upgrade-starve-penalty 0.08 '
       '--starve-gift-weight 2 --comp-bonus 0.08 --starve-buy-weight 0 --unspent-gold-penalty 0.005 --pro-gap-cap 0.12 '
       '--pro-data "runs\\pro_demos;runs\\h13_demos" --pro-boards "runs\\pro_boards.json;runs\\h13_boards.json"')


def _split(s: str) -> list[str]:
    return shlex.split(s.replace("\\", "/"))       # Windows 路径的反斜杠统一成 /（shlex 会把反斜杠当转义）


def _bat_args() -> list[str]:
    line = next(x for x in (ROOT / "train.bat").read_text(encoding="utf-8").splitlines() if "bg.py train" in x)
    rest = line.split("bg.py train", 1)[1]
    for k, v in (("%INIT%", ""), ("%*", ""), ("%NUMBER_OF_PROCESSORS%", "8")):
        rest = rest.replace(k, v)
    return _split(rest)


def _norm(cfg) -> dict:
    return {k: v.replace("\\", "/") if isinstance(v, str) else v for k, v in dataclasses.asdict(cfg).items()}


def main() -> int:
    from rl.ppo import Config
    from tools.train import load_recipe, parse
    names = {f.name for f in dataclasses.fields(Config)}
    for p in sorted((ROOT / "configs").glob("*.json")):
        r = load_recipe(str(p))
        assert set(r["config"]) <= names, (p.name, set(r["config"]) - names)

    c_old, a_old = parse(_split(OLD))
    c_new, a_new = parse(_bat_args())
    d_old, d_new = _norm(c_old), _norm(c_new)
    diff = {k: (d_old[k], d_new[k]) for k in d_old if d_old[k] != d_new[k]}
    assert not diff, diff
    assert a_old.run.replace("\\", "/") == a_new.run.replace("\\", "/") and a_old.resume == a_new.resume

    c, a = parse(_bat_args() + ["--comp-bonus", "0.02", "--run", "runs/x"])
    assert c.comp_bonus == 0.02 and a.run == "runs/x"
    print(f"recipe OK：train.bat 的 {len(d_new)} 项设置和原来一样")
    return 0


if __name__ == "__main__":
    sys.exit(main())
