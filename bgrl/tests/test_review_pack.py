"""HDT 复盘包（tg/review_pack.py）：合成一个两回合的小包，检查记录、补丁、升本费用、发现待选项、prodemos 能认出来。

  python -X utf8 bgrl/tests/test_review_pack.py                  检查
  python -X utf8 bgrl/tests/test_review_pack.py --write-sample D 在 D 下写一个外层 zip（里面套一个复盘包），CI 拿它跑 prodemos
"""
from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

LP, SP, PID, HERO = 4, 12, 11, 119


def _minion(eid, card, ctrl, zone, pos, atk, hp, race="MURLOC", tier=1):
    return {"entity_id": eid, "card_id": card, "kind": "MINION", "position": pos, "attack": atk, "health": hp,
            "tags": {"CONTROLLER": str(ctrl), "CARDTYPE": "MINION", "ZONE": zone, "ZONE_POSITION": str(pos),
                     "ATK": str(atk), "HEALTH": str(hp), "CARDRACE": race, "TECH_LEVEL": str(tier), "ENTITY_ID": str(eid)}}


def _gold(avail, tier):
    return {"available": avail, "resources": avail, "used": 0, "temporary": 0, "player_entity_id": PID,
            "tags": {"PLAYER_ID": str(LP), "CONTROLLER": str(LP), "CARDTYPE": "PLAYER", "ZONE": "PLAY",
                     "HERO_ENTITY": str(HERO), "PLAYER_TECH_LEVEL": str(tier), "RESOURCES": str(avail)}}


def build_pack() -> bytes:
    """第 1 回合：酒馆两张随从，买第一张；第 2 回合：升本（5 金 → 1 金）、发现、进战斗。"""
    heroes = {"friendly": {"entity_id": HERO, "card_id": "TB_BaconShop_HERO_43", "max_health": 30, "damage": 0, "armor": 5},
              "opponent": {"entity_id": 63, "card_id": "TB_BaconShopBob", "max_health": 30, "damage": 0, "armor": 0}}
    s0 = {"context": {"round": 0, "phase": "setup"}, "entities": {}, "board": [], "hand": [], "shop": [], "trinkets": [],
          "gold": None, "heroes": heroes}
    a = _minion(302, "BG31_330", SP, "PLAY", 1, 2, 1, "NAGA")
    b = _minion(298, "BG32_330", SP, "PLAY", 2, 3, 3)
    rows = [{"kind": "choice_offered", "round": 0, "encoding": "snapshot", "state": s0},
            {"kind": "phase_start", "round": 1, "encoding": "patch", "patch": [
                {"op": "replace", "path": "/context", "value": {"round": 1, "phase": "recruit"}},
                {"op": "replace", "path": "/gold/available", "value": 3},
                {"op": "replace", "path": "/gold/player_entity_id", "value": PID},
                {"op": "replace", "path": "/gold/tags", "value": _gold(3, 1)["tags"]},
                {"op": "replace", "path": "/entities", "value": {"302": a, "298": b}},
                {"op": "replace", "path": "/shop", "value": [302, 298]}]},
            {"kind": "operation_start", "round": 1, "encoding": "patch", "patch": [],
             "details": {"id": "a1", "action": "buy", "source": {"card_id": "TB_BaconShop_DragBuy"}}},
            {"kind": "operation_end", "round": 1, "encoding": "patch", "details": {"operation_id": "a1"}, "patch": [
                {"op": "replace", "path": "/gold", "value": _gold(0, 1)},
                {"op": "replace", "path": "/entities/302", "value": _minion(302, "BG31_330", LP, "HAND", 1, 2, 1, "NAGA")},
                {"op": "remove", "path": "/shop/0"}, {"op": "add", "path": "/hand/-", "value": 302}]},
            {"kind": "phase_start", "round": 1, "encoding": "patch",
             "patch": [{"op": "replace", "path": "/context", "value": {"round": 1, "phase": "combat"}}]},
            {"kind": "phase_start", "round": 2, "encoding": "patch", "patch": [
                {"op": "replace", "path": "/context", "value": {"round": 2, "phase": "recruit"}},
                {"op": "replace", "path": "/gold", "value": _gold(5, 1)}]},
            {"kind": "operation_start", "round": 2, "encoding": "patch", "patch": [],
             "details": {"id": "a2", "action": "other_active", "source": {"card_id": "TB_BaconShopTechUp02_Button"}}},
            {"kind": "operation_end", "round": 2, "encoding": "patch", "details": {"operation_id": "a2"},
             "patch": [{"op": "replace", "path": "/gold", "value": _gold(1, 2)}]},
            {"kind": "choice_offered", "round": 2, "encoding": "patch", "patch": [],
             "details": {"choice_id": "c1", "options": [_minion(901, "BG36_200", LP, "SETASIDE", 0, 1, 4, "BEAST"),
                                                        _minion(902, "BG32_330", LP, "SETASIDE", 0, 3, 3)]}},
            {"kind": "choice_selected", "round": 2, "encoding": "patch", "patch": [], "details": {"selected": [901]}},
            {"kind": "phase_start", "round": 2, "encoding": "patch",
             "patch": [{"op": "replace", "path": "/context", "value": {"round": 2, "phase": "combat"}}]}]
    man = {"schema": "hdt-replay-3layer-v1", "outcome": "游戏结束，战绩：第2名，原分数11418，现分数11488",
           "match": {"local_player": LP, "shop_player": SP, "hero": "布莱恩", "rounds": 2}}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("manifest.json", json.dumps(man, ensure_ascii=False))
        z.writestr("data/timeline.jsonl", "\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
        z.writestr("tools/query.py", "raise SystemExit('包里的脚本不该被运行')\n")
    return buf.getvalue()


def write_sample(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    out = d / "packs.zip"
    with zipfile.ZipFile(out, "w") as z:
        for i in range(3):
            z.writestr(f"2026-10-1{i}_布莱恩_AI复盘包.zip", build_pack())
            z.writestr(f"2026-10-1{i}_布莱恩_AI复盘包.verification.json", "{}")
    return out


def main() -> int:
    if len(sys.argv) > 2 and sys.argv[1] == "--write-sample":
        print(write_sample(Path(sys.argv[2])))
        return 0
    from tg.review_pack import _apply, iter_packs, parse_pack
    st = {"gold": None, "board": [1, 2], "entities": {"5": {"a": 1}}}
    _apply(st, [{"op": "replace", "path": "/gold/available", "value": 3}, {"op": "remove", "path": "/board/0"},
                {"op": "add", "path": "/board/-", "value": 9}, {"op": "remove", "path": "/entities/5"}])
    assert st == {"gold": {"available": 3}, "board": [2, 9], "entities": {}}, st
    z = zipfile.ZipFile(io.BytesIO(build_pack()))
    recs, info = parse_pack(z)
    assert info["mmr"] == 11418 and info["place"] == 2, info
    p = [r["player"] for r in recs]
    assert [r["turn"] for r in recs] == [1, 1, 1, 2, 2, 2, 2, 2], [r["turn"] for r in recs]
    assert [x["gold"] for x in p] == [3, 0, 0, 5, 1, 1, 1, 1], [x["gold"] for x in p]
    assert [c["cardId"] for c in recs[0]["tavern"]] == ["BG31_330", "BG32_330"]
    assert [c["cardId"] for c in p[1]["hand"]] == ["BG31_330"] and recs[1]["tavern"][0]["cardId"] == "BG32_330"
    assert p[0]["heroCardId"] == "TB_BaconShop_HERO_43" and p[0]["armor"] == 5 and p[0]["health"] == 30
    assert [x["tavernTier"] for x in p] == [1, 1, 1, 1, 2, 2, 2, 2]
    assert p[3]["shop"]["canUpgrade"] and p[3]["shop"]["upgradeCost"] == 4, p[3]["shop"]   # 升本那一步：5 → 1 金
    assert not p[0]["shop"]["canUpgrade"] and not p[4]["shop"]["canUpgrade"]               # 第 1 回合没升、升完以后不知道
    off = recs[5]["offered"]
    assert off and off["kind"] == "discover" and [c["cardId"] for c in off["cards"]] == ["BG36_200", "BG32_330"], off
    assert recs[6]["offered"] is None
    assert set(recs[2]["availableRaces"]) >= {"NAGA", "MURLOC"}, recs[2]["availableRaces"]   # 本局酒馆里见过的种族
    from tg.logs import load_records, segment
    kinds = [a.kind for a in segment(load_records(recs, "x"))]
    assert "buy" in kinds and "upgrade" in kinds, kinds
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "runs" / "_packs_ci"
    from tools.make_pro_demos import _jobs
    write_sample(d)
    jobs = _jobs(d)
    assert len(jobs) == 3 and all(j[1].endswith("复盘包.zip") for j in jobs), jobs
    assert len(list(iter_packs(d / "packs.zip"))) == 3
    print(f"review_pack OK：{len(recs)} 条记录、动作 {kinds}、升本费用 {p[3]['shop']['upgradeCost']}、发现 {len(off['cards'])} 张")
    return 0


if __name__ == "__main__":
    sys.exit(main())
