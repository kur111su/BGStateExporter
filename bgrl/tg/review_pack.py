"""HDT “AI 复盘包”（manifest.json 的 schema = hdt-replay-3layer-v1）→ 和插件一样的对局记录（schema 2）。

复盘包里每一步操作（买、卖、打出、施法、刷新、升本、冻结、发现、换位）前后都有一份结构化局面
（data/timeline.jsonl：第一条整份 snapshot，之后是 JSON Patch），实体带 HDT 的标签（名字或数字）。
这里把每份局面装进 tg/hsreplay.Replay 的实体表，用同一套 StateReader.cs 规则（Replay._state）出记录，
之后和 Firestone 回放走同一条路：tg.logs 切成动作 → tools/make_pro_demos.py 生成示范。

  在哪些时刻记一条：每次操作开始 / 结束、发现弹出 / 选完、换位结束，最后在进战斗那一刻再记一条（= 回合结束时的局面）。

和 HSReplay 回放不同的地方（包里没有的东西）：
  - 没有排行榜、对手英雄、对手阵容（战斗快照只有自己的场面）：观察里这几项是空的；
  - 没有英雄技能实体：英雄技能那一步不做示范；
  - 没有按钮实体：买牌按 3 金、刷新按 1 金；升本费用用这一回合真的升本那一步前后的金币差，没升本的回合交给引擎按等级算；
  - 没有附魔实体：随从的攻 / 血已经带着加成，单独的计数（例如“本局已施法 N 次”的附魔）读不到。
manifest 的 outcome（“战绩：第 N 名，原分数 X”）里的分数记成样本的 mmr。

用法：tools/make_pro_demos.py 的 prodemos 自动认出复盘包（外层 zip 里套着的 *复盘包.zip，或者单个复盘包 zip）。
包里的 tools/*.py 不会被运行，只读 manifest.json 和 data/timeline.jsonl。
"""
from __future__ import annotations

import copy
import io
import json
import re
import zipfile
from pathlib import Path

from tg.hsreplay import (CT_HERO, T_ARMOR, T_CARDTYPE, T_CONTROLLER, T_DAMAGE, T_HEALTH, T_HERO_ENTITY, T_TECH, T_TURN,
                         T_ZONE, Z_PLAY, Ent, Replay, _enums)

SCHEMA = "hdt-replay-3layer-v1"
EMIT = {"operation_start", "operation_end", "choice_offered", "choice_selected", "move_end"}


def is_pack(z: zipfile.ZipFile) -> bool:
    n = set(z.namelist())
    return "manifest.json" in n and "data/timeline.jsonl" in n


def _enum_maps():
    d = json.loads((Path(__file__).resolve().parent / "gametags.json").read_text(encoding="utf-8"))
    rev = lambda m: {v: int(k) for k, v in m.items()}  # noqa: E731
    return {"ZONE": rev(d["Zone"]), "CARDTYPE": rev(d["CardType"])}


_ENUM = None


def _tags(raw: dict | None) -> dict[int, int]:
    """复盘包的标签（{"ZONE": "PLAY", "ATK": "3", "1196": "0"}）→ {标签号: 整数}。不是数字、也不是区域 / 卡牌类型的值（职业、种族名）跳过。"""
    global _ENUM
    if _ENUM is None:
        _ENUM = _enum_maps()
    names = _enums()[1]
    out = {}
    for k, v in (raw or {}).items():
        t = int(k) if str(k).lstrip("-").isdigit() else names.get(k)
        if t is None:
            continue
        s = str(v)
        if s.lstrip("-").isdigit():
            out[t] = int(s)
        elif k in _ENUM and s in _ENUM[k]:
            out[t] = _ENUM[k][s]
    return out


def _apply(st, patch):
    """JSON Patch（只用到 add / replace / remove）。"""
    for p in patch:
        parts = [x.replace("~1", "/").replace("~0", "~") for x in p["path"].split("/")[1:]]
        if not parts:
            continue
        cur = st
        for k in parts[:-1]:
            if isinstance(cur, list):
                cur = cur[int(k)]
            else:
                if cur.get(k) is None:                 # 金币一开始是 null，之后按字段打补丁
                    cur[k] = {}
                cur = cur[k]
        k, op = parts[-1], p["op"]
        if isinstance(cur, list):
            if op == "remove":
                cur.pop(int(k))
            elif k == "-":
                cur.append(p["value"])
            elif op == "add":
                cur.insert(int(k), p["value"])
            else:
                cur[int(k)] = p["value"]
        elif op == "remove":
            cur.pop(k, None)
        else:
            cur[k] = p["value"]


def states(z: zipfile.ZipFile):
    """[(时间线这一条, 这一条之后的局面)]（局面是同一个对象，一直在变：要留就自己复制）。"""
    st = None
    for line in z.read("data/timeline.jsonl").decode("utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("encoding") == "snapshot":
            st = copy.deepcopy(r["state"])
        elif r.get("encoding") == "patch" and st is not None:
            _apply(st, r.get("patch") or [])
        if st is not None:
            yield r, st


def outcome(manifest: dict) -> dict:
    s = str(manifest.get("outcome") or "")
    m, p = re.search(r"原分数\s*(\d+)", s), re.search(r"第\s*(\d+)\s*名", s)
    return {"mmr": int(m.group(1)) if m else 0, "place": int(p.group(1)) if p else 0}


def _record(rp: Replay, st: dict, lp: int, offered: list[dict] | None) -> dict | None:
    gold = st.get("gold") or {}
    pid = gold.get("player_entity_id")
    hero = (st.get("heroes") or {}).get("friendly") or {}
    rnd = int((st.get("context") or {}).get("round") or 0)
    if not pid or not hero.get("entity_id") or rnd <= 0:
        return None
    rp.ents = {}
    for eid, e in (st.get("entities") or {}).items():
        x = Ent(int(eid), e.get("card_id"))
        x.tags = _tags(e.get("tags"))
        rp.ents[x.id] = x
    pe = Ent(int(pid))
    pe.tags = _tags(gold.get("tags"))
    rp.ents[pe.id] = pe
    hid = int(hero["entity_id"])
    h = rp.ents.get(hid) or Ent(hid, hero.get("card_id"))
    h.card = h.card or hero.get("card_id")
    h.tags.update({T_CONTROLLER: lp, T_CARDTYPE: CT_HERO, T_ZONE: Z_PLAY, T_HEALTH: int(hero.get("max_health") or 30),
                   T_DAMAGE: int(hero.get("damage") or 0), T_ARMOR: int(hero.get("armor") or 0),
                   T_TECH: pe.g(T_TECH) or 1})
    rp.ents[hid] = h
    pe.tags[T_HERO_ENTITY] = hid
    rp.main_ent = pe.id
    g = Ent(-1)
    g.tags[T_TURN] = 2 * rnd - 1
    rp.ents[-1] = g
    rp.game_id = -1
    rp.offered = []
    for o in offered or []:
        x = Ent(int(o["entity_id"]), o.get("card_id"))
        x.tags = _tags(o.get("tags"))
        rp.ents[x.id] = x
        rp.offered.append(x.id)
    return rp._state("recruit")


def parse_pack(z: zipfile.ZipFile) -> tuple[list[dict], dict]:
    """一个复盘包 → (插件格式记录, {"mmr", "place", "hero", "rounds"})。"""
    man = json.loads(z.read("manifest.json").decode("utf-8"))
    if man.get("schema") != SCHEMA:
        raise ValueError(f"不认识的复盘包格式 {man.get('schema')}")
    match = man.get("match") or {}
    lp, sp = int(match["local_player"]), int(match["shop_player"])
    rp = Replay()
    rp.main_pid, rp.opp_pid = lp, sp
    out, offered, rows = [], None, []
    for r, st in states(z):
        kind = r.get("kind")
        phase = (st.get("context") or {}).get("phase")
        if kind == "choice_offered":
            offered = (r.get("details") or {}).get("options") or None
        if kind == "phase_start" and phase == "combat":          # 进战斗：这一回合最后的局面
            kind = "turn_end"
        elif kind not in EMIT or phase != "recruit":
            if kind == "choice_selected":
                offered = None
            continue
        s = _record(rp, st, lp, offered if kind == "choice_offered" else None)
        if kind == "choice_selected":
            offered = None
        if s is None:
            continue
        rows.append((r, s, (st.get("gold") or {}).get("available")))
        out.append(s)
    _upgrade_costs(rows)
    info = {**outcome(man), "hero": match.get("hero"), "rounds": match.get("rounds")}
    return out, info


def _upgrade_costs(rows):
    """包里没有升本按钮：某回合真的升了本（操作的来源是 TB_BaconShopTechUp0N_Button），就用那一步前后的金币差
    当这一回合（升本之前那些记录）的升本费用。"""
    by_round: dict[int, list[int]] = {}
    for i, (r, s, _) in enumerate(rows):
        by_round.setdefault(int(r.get("round") or 0), []).append(i)
    for rnd, idx in by_round.items():
        start = 0
        for j, i in enumerate(idx):
            r, s, g0 = rows[i]
            d = r.get("details") or {}
            src = (d.get("source") or {}).get("card_id") or ""
            if r.get("kind") != "operation_start" or "TechUp" not in src:
                continue
            g1 = next((rows[k][2] for k in idx[j + 1:] if rows[k][0].get("kind") == "operation_end"
                       and (rows[k][0].get("details") or {}).get("operation_id") == d.get("id")), None)
            if g0 is None or g1 is None or not 0 <= g0 - g1 <= 11:
                continue
            for k in idx[start:j + 1]:
                sh = rows[k][1]["player"]["shop"]
                sh.update(canUpgrade=True, upgradeCost=int(g0 - g1))
            start = j + 1


def iter_packs(path: str | Path, inner: str = ""):
    """[(包名, ZipFile)]：path 是复盘包本身，或者外层 zip（里面套着复盘包）；inner 指定外层 zip 里的哪一个。"""
    with zipfile.ZipFile(path) as z:
        if is_pack(z):
            yield Path(path).stem, z
            return
        names = [inner] if inner else [n for n in z.namelist() if n.endswith(".zip") and not n.endswith(".xml.zip")]
        for n in names:
            try:
                zz = zipfile.ZipFile(io.BytesIO(z.read(n)))
            except zipfile.BadZipFile:
                continue
            if is_pack(zz):
                yield Path(n).stem, zz
