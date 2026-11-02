"""训练配方（configs/*.json，tools/train.py --recipe）：

- 配方里的每一项都是 Config 的字段，值的类型能按默认值转换；
- 配方里的值和把同样的值写成命令行参数得到的 Config 完全一样（配方和命令行是同一套参数）；
- train.bat 用的是 configs/v3.json，而且命令行参数优先于配方。
（改用配方时，train.bat 原来那串参数和配方的等价性在提交 cf7568a 里核对过。）
"""
from __future__ import annotations

import dataclasses
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _bat_args() -> list[str]:
    line = next(x for x in (ROOT / "train.bat").read_text(encoding="utf-8").splitlines() if "bg.py train" in x)
    rest = line.split("bg.py train", 1)[1]
    for k, v in (("%INIT%", ""), ("%*", ""), ("%NUMBER_OF_PROCESSORS%", "8")):
        rest = rest.replace(k, v)
    return shlex.split(rest.replace("\\", "/"))       # Windows 路径的反斜杠统一成 /（shlex 会把反斜杠当转义）


def _norm(cfg) -> dict:
    return {k: v.replace("\\", "/") if isinstance(v, str) else v for k, v in dataclasses.asdict(cfg).items()}


def main() -> int:
    from rl.ppo import Config
    from tools.train import load_recipe, parse
    names = {f.name for f in dataclasses.fields(Config)}
    for p in sorted((ROOT / "configs").glob("*.json")):
        r = load_recipe(str(p))
        assert set(r["config"]) <= names, (p.name, set(r["config"]) - names)
        flags = []
        for k, v in r["config"].items():
            flags += ["--" + k.replace("_", "-"), str(v).replace("\\", "/")]
        c_cli, _ = parse(flags)
        c_rec, _ = parse(["--recipe", str(p)])
        d_cli, d_rec = _norm(c_cli), _norm(c_rec)
        diff = {k: (d_cli[k], d_rec[k]) for k in d_cli if d_cli[k] != d_rec[k]}
        assert not diff, (p.name, diff)

    args = _bat_args()
    assert args[:2] == ["--recipe", "configs/v3.json"] and "--resume" in args, args
    c, a = parse(args)
    rec = load_recipe("configs/v3.json")
    assert a.run.replace("\\", "/") == rec["run"].replace("\\", "/") and a.resume and c.threads == 8
    for k, v in rec["config"].items():
        got = getattr(c, k)
        assert (got.replace("\\", "/") if isinstance(got, str) else got) == (v.replace("\\", "/") if isinstance(v, str) else v), k
    c2, a2 = parse(args + ["--comp-bonus", "0.02", "--run", "runs/x"])
    assert c2.comp_bonus == 0.02 and a2.run == "runs/x"
    print(f"recipe OK：train.bat → configs/v3.json，{len(rec['config'])} 项设置")
    return 0


if __name__ == "__main__":
    sys.exit(main())
