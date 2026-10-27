"""酒馆战旗卡牌数据：从 HDT 下载的 CardDefs.base.xml 导出 bg_cards.json。

HDT 每次启动都会把最新的卡牌数据存到 %AppData%\\HearthstoneDeckTracker\\CardDefs\\CardDefs.base.xml，
这里只挑出酒馆战旗相关的卡（随从池、衍生随从、金色版、酒馆法术、英雄、英雄技能、饰品、任务、畸变），
整理成训练环境直接能用的格式。版本更新后重新运行即可。

用法：python -m bgsim.cards [--carddefs CardDefs.base.xml] [--out bgrl/data/bg_cards.json]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

BGRL = Path(__file__).resolve().parent.parent
DEFAULT_OUT = BGRL / "data" / "bg_cards.json"

CARDTYPE = {3: "hero", 4: "minion", 5: "spell", 6: "enchantment", 10: "hero_power", 42: "battleground_spell",
            43: "battleground_anomaly", 44: "battleground_trinket", 40: "battleground_quest_reward"}

# HearthDb.Enums.Race（CARDRACE 的值）
RACE = {11: "UNDEAD", 14: "MURLOC", 15: "DEMON", 17: "MECHANICAL", 18: "ELEMENTAL", 20: "BEAST", 23: "PIRATE",
        24: "DRAGON", 26: "ALL", 43: "QUILBOAR", 92: "NAGA", 126: "ABERRATION"}
# 第二种族：HearthDb.Enums.RaceUtils.TagRaceMap（tag enumID -> 种族）
RACE_TAG = {2534: "UNDEAD", 2536: "MURLOC", 2537: "DEMON", 2539: "MECHANICAL", 2540: "ELEMENTAL", 2542: "BEAST",
            2522: "PIRATE", 2523: "DRAGON", 2546: "QUILBOAR", 2553: "NAGA", 4756: "ABERRATION"}

# 关键词（随从身上的状态）
KEYWORDS = {"TAUNT": "taunt", "DIVINE_SHIELD": "divineShield", "POISONOUS": "poisonous", "VENOMOUS": "venomous",
            "WINDFURY": "windfury", "MEGA_WINDFURY": "megaWindfury", "REBORN": "reborn", "STEALTH": "stealth",
            "CLEAVE": "cleave", "MAGNETIC": "magnetic"}
# 效果机制（用来做“效果标签”特征和效果模板分类）。名字来自 XML 的 name 属性；没有名字的用 enumID
MECHANICS = {
    "BATTLECRY": "battlecry", "DEATHRATTLE": "deathrattle", "TRIGGER_VISUAL": "trigger", "AURA": "aura",
    "DISCOVER": "discover", "CHOOSE_ONE": "chooseOne", "START_OF_COMBAT": "startOfCombat",
    "AVENGE": "avenge", "END_OF_TURN_TRIGGER": "endOfTurn", "BACON_RALLY": "rally",
    "BACON_SPELLCRAFT_ID": "spellcraft", "BACON_BLOOD_GEM_TOOLTIP": "bloodGem", "BACON_TRIGGER_XY": "afterX",
    "BACON_TRIGGER_UPBEAT": "upbeat", "MODULAR": "magnetic", "OVERHEAL": "overheal", "INFUSE": "infuse",
    "BACON_BOUNTY": "bounty", "BACON_TIMEWARPED": "timewarped",
}
NUM_RE = re.compile(r"\d+")
MARKUP_RE = re.compile(r"</?[a-zA-Z]+>|\[x\]|[$#]")


def clean_text(text: str | None, sd: list[int]) -> str:
    if not text:
        return ""
    t = text
    for i, v in enumerate(sd):
        t = t.replace("{%d}" % i, str(v))
    t = MARKUP_RE.sub("", t).replace("\n", " ")
    return re.sub(r"\s+", " ", t).strip()


def parse_carddefs(path: str) -> tuple[str, dict[str, dict]]:
    """读取 CardDefs.base.xml，返回 (build, {cardId: 原始信息})。"""
    build = ""
    cards: dict[str, dict] = {}
    for event, el in ET.iterparse(path, events=("start", "end")):
        if event == "start" and el.tag == "CardDefs":
            build = el.get("build", "")
        if event != "end" or el.tag != "Entity":
            continue
        cid = el.get("CardID") or ""
        e = {"id": cid, "dbf": int(el.get("ID") or 0), "tags": {}, "tagIds": {}, "loc": {}, "refs": {}}
        for t in el.findall("Tag"):
            enum = int(t.get("enumID") or 0)
            name = t.get("name") or str(enum)
            ttype = t.get("type")
            if ttype == "LocString":
                e["loc"][name] = {"zhCN": (t.findtext("zhCN") or ""), "enUS": (t.findtext("enUS") or "")}
            elif ttype in ("Int", "Bool", "Card"):
                v = int(t.get("value") or 0)
                e["tags"][name] = v
                e["tagIds"][enum] = v
                if ttype == "Card" and t.get("cardID"):
                    e["refs"][name] = t.get("cardID")
        cards[cid] = e
        el.clear()
    return build, cards


def build_db(build: str, raw: dict[str, dict]) -> dict:
    by_dbf = {e["dbf"]: e for e in raw.values()}

    def races(e) -> list[str]:
        out = []
        r = RACE.get(e["tags"].get("CARDRACE", 0))
        if r:
            out.append(r)
        for enum, name in RACE_TAG.items():
            if e["tagIds"].get(enum) and name not in out:
                out.append(name)
        return out

    def entry(e, category: str) -> dict:
        tags = e["tags"]
        sd = [tags.get("TAG_SCRIPT_DATA_NUM_1", 0), tags.get("TAG_SCRIPT_DATA_NUM_2", 0)]
        name = e["loc"].get("CARDNAME", {})
        text = e["loc"].get("CARDTEXT", {})
        d = {
            "id": e["id"], "dbf": e["dbf"], "category": category,
            "type": CARDTYPE.get(tags.get("CARDTYPE", 0), str(tags.get("CARDTYPE", 0))),
            "name": name.get("zhCN") or name.get("enUS", ""), "nameEn": name.get("enUS", ""),
            "text": clean_text(text.get("zhCN"), sd), "textEn": clean_text(text.get("enUS"), sd),
            "tier": tags.get("TECH_LEVEL", 0), "atk": tags.get("ATK", 0), "hp": tags.get("HEALTH", 0),
            "cost": tags.get("COST", 0), "races": races(e),
            "keywords": sorted(v for k, v in KEYWORDS.items() if tags.get(k)),
            "mechanics": sorted({v for k, v in MECHANICS.items() if tags.get(k)}),
            "scriptData": sd if any(sd) else [],
            "golden": bool(tags.get("PREMIUM")),
            "duosOnly": bool(tags.get("IS_BACON_DUOS_EXCLUSIVE")),
        }
        d["numbers"] = [int(x) for x in NUM_RE.findall(d["textEn"])][:6]
        up = tags.get("BACON_TRIPLE_UPGRADE_MINION_ID")
        if up and up in by_dbf:
            d["goldenId"] = by_dbf[up]["id"]
        base = tags.get("BACON_TRIPLED_BASE_MINION_ID")
        if base and base in by_dbf:
            d["normalId"] = by_dbf[base]["id"]
        if tags.get("HERO_POWER") and e["refs"].get("HERO_POWER"):
            d["heroPower"] = e["refs"]["HERO_POWER"]
        if tags.get("BACON_SPELLCRAFT_ID") and tags["BACON_SPELLCRAFT_ID"] in by_dbf:
            d["spellcraft"] = by_dbf[tags["BACON_SPELLCRAFT_ID"]]["id"]
        if tags.get("HEALTH") and d["type"] == "hero":
            d["armor"] = tags.get("ARMOR", 0)
        return d

    db: dict[str, dict] = {}
    # 1. 随从池（当前版本可出现在酒馆里的随从）+ 金色版
    for e in raw.values():
        if e["tags"].get("IS_BACON_POOL_MINION"):
            db[e["id"]] = entry(e, "pool")
    for cid in list(db):
        g = db[cid].get("goldenId")
        if g and g in raw:
            db[g] = entry(raw[g], "golden")
    # 2. 酒馆法术、饰品、可选英雄及其技能、任务/奖励、畸变
    for e in raw.values():
        t = e["tags"]
        ctype = CARDTYPE.get(t.get("CARDTYPE", 0))
        if e["id"] in db:
            continue
        if t.get("IS_BACON_POOL_SPELL"):
            db[e["id"]] = entry(e, "spell")
        elif t.get("BACON_TRINKET") or ctype == "battleground_trinket":
            db[e["id"]] = entry(e, "trinket")
        elif t.get("BACON_HERO_CAN_BE_DRAFTED") and ctype == "hero":
            db[e["id"]] = entry(e, "hero")
        elif ctype == "battleground_anomaly":
            db[e["id"]] = entry(e, "anomaly")
        elif ctype == "battleground_quest_reward" or (t.get("QUEST") and e["id"].startswith(("BG", "TB_Bacon"))):
            db[e["id"]] = entry(e, "quest")
    for h in [d for d in db.values() if d["category"] == "hero"]:
        hp = h.get("heroPower")
        if hp and hp in raw and hp not in db:
            db[hp] = entry(raw[hp], "hero_power")
    # 3. 其它酒馆战旗随从（衍生物等）：有星级、BG 开头的随从
    for e in raw.values():
        if e["id"] in db:
            continue
        t = e["tags"]
        if CARDTYPE.get(t.get("CARDTYPE", 0)) == "minion" and t.get("TECH_LEVEL") and \
                e["id"].startswith(("BG", "TB_Bacon")):
            db[e["id"]] = entry(e, "token")
        elif CARDTYPE.get(t.get("CARDTYPE", 0)) in ("spell", "battleground_spell") and \
                e["id"].startswith(("BG", "TB_Bacon", "EBG", "BGS")):
            db[e["id"]] = entry(e, "token_spell")       # 三连奖励、鲜血宝石、香蕉、法术迸发等衍生法术
    counts = Counter(d["category"] for d in db.values())
    return {"build": build, "counts": dict(counts), "cards": db}


def default_carddefs() -> str:
    return os.path.join(os.environ.get("APPDATA", ""), "HearthstoneDeckTracker", "CardDefs", "CardDefs.base.xml")


def load(path: str | Path = DEFAULT_OUT) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--carddefs", default=default_carddefs())
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args(argv)
    if not os.path.exists(args.carddefs):
        print(f"找不到 {args.carddefs}（先启动一次 HDT，它会下载最新卡牌数据）")
        return 1
    build, raw = parse_carddefs(args.carddefs)
    db = build_db(build, raw)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(db, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"卡牌数据 build {build}：{db['counts']} → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
