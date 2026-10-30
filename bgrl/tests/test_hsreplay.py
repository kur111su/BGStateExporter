"""高分回放转换（tg/hsreplay.py）：用一份手写的小回放（结构和 Firestone 导出的一样，内容是编的）检查
记录切出来的动作是买 / 打出 / 升本、战斗副本不算进场面、断线重连（整局重新下发）后旧实体清掉，
并且重建成 TavernGym 局面后这些动作都合法。"""
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

XML = """<?xml version="1.0" encoding="utf-8"?><HSReplay><Game ts="0" buildNumber="1">
<GameEntity id="1"><Tag tag="202" value="1" /><Tag tag="20" value="0" /></GameEntity>
<Player id="2" playerID="5" isMainPlayer="true"><Tag tag="30" value="5" /><Tag tag="202" value="2" /><Tag tag="27" value="10" /><Tag tag="26" value="3" /><Tag tag="25" value="0" /></Player>
<Player id="3" playerID="9" isMainPlayer="false"><Tag tag="30" value="9" /><Tag tag="202" value="2" /></Player>
<FullEntity id="10" cardID="TB_BaconShop_HERO_34"><Tag tag="50" value="5" /><Tag tag="49" value="1" /><Tag tag="202" value="3" /><Tag tag="45" value="30" /><Tag tag="1377" value="1" /><Tag tag="1373" value="1" /><Tag tag="30" value="5" /></FullEntity>
<FullEntity id="20" cardID="TB_BaconShop_DragBuy"><Tag tag="50" value="5" /><Tag tag="49" value="1" /><Tag tag="202" value="5" /><Tag tag="48" value="3" /></FullEntity>
<FullEntity id="21" cardID="TB_BaconShop_8p_Reroll_Button"><Tag tag="50" value="5" /><Tag tag="49" value="1" /><Tag tag="202" value="5" /><Tag tag="48" value="1" /></FullEntity>
<FullEntity id="22" cardID="TB_BaconShopTechUp02_Button"><Tag tag="50" value="5" /><Tag tag="49" value="1" /><Tag tag="202" value="5" /><Tag tag="48" value="4" /></FullEntity>
<FullEntity id="23" cardID="TB_BaconShop_DragSell"><Tag tag="50" value="5" /><Tag tag="49" value="1" /><Tag tag="202" value="5" /></FullEntity>
<FullEntity id="30" cardID="BGS_004"><Tag tag="50" value="9" /><Tag tag="49" value="1" /><Tag tag="202" value="4" /><Tag tag="263" value="1" /><Tag tag="47" value="1" /><Tag tag="45" value="3" /></FullEntity>
<FullEntity id="31" cardID="BG26_146"><Tag tag="50" value="9" /><Tag tag="49" value="1" /><Tag tag="202" value="4" /><Tag tag="263" value="2" /><Tag tag="47" value="2" /><Tag tag="45" value="2" /></FullEntity>
<FullEntity id="40" cardID="BG26_146"><Tag tag="50" value="5" /><Tag tag="49" value="3" /><Tag tag="202" value="4" /><Tag tag="263" value="1" /><Tag tag="47" value="2" /><Tag tag="45" value="2" /></FullEntity>
<TagChange entity="1" tag="20" value="1" />
<Options id="1"><Option index="0" type="2" entity="0" /><Option index="1" type="3" entity="21" /><Option index="2" type="3" entity="20"><Target index="0" entity="30" /><Target index="1" entity="31" /></Option><Option index="3" type="3" entity="30" /><Option index="4" type="3" entity="31" /><Option index="5" type="3" entity="40" /></Options>
<Block entity="20" index="0" type="7" target="30"><TagChange entity="30" tag="50" value="5" /><TagChange entity="30" tag="49" value="3" /><TagChange entity="30" tag="263" value="2" /><TagChange entity="2" tag="25" value="3" /></Block>
<Options id="2"><Option index="0" type="2" entity="0" /><Option index="1" type="3" entity="40" /><Option index="2" type="3" entity="30" /><Option index="3" type="3" entity="31" /></Options>
<Block entity="40" index="0" type="7"><TagChange entity="40" tag="49" value="1" /><TagChange entity="40" tag="263" value="1" /><TagChange entity="30" tag="263" value="1" /></Block>
<Options id="3"><Option index="0" type="2" entity="0" /><Option index="1" type="3" entity="40" /><Option index="2" type="3" entity="30" /></Options>
<TagChange entity="1" tag="20" value="2" />
<FullEntity id="50" cardID="BG26_146"><Tag tag="50" value="5" /><Tag tag="49" value="1" /><Tag tag="202" value="4" /><Tag tag="263" value="1" /><Tag tag="47" value="2" /><Tag tag="45" value="2" /></FullEntity>
<TagChange entity="50" tag="49" value="6" />
<TagChange entity="1" tag="20" value="3" />
<TagChange entity="2" tag="26" value="4" /><TagChange entity="2" tag="25" value="0" />
<Options id="4"><Option index="0" type="2" entity="0" /><Option index="1" type="3" entity="22" /><Option index="2" type="3" entity="23"><Target index="0" entity="40" /></Option><Option index="3" type="3" entity="40" /><Option index="4" type="3" entity="31" /></Options>
<Block entity="22" index="0" type="7"><TagChange entity="10" tag="1377" value="2" /><TagChange entity="2" tag="25" value="4" /></Block>
<Options id="5"><Option index="0" type="2" entity="0" /></Options>
</Game></HSReplay>"""


def main() -> int:
    from tg.bridge import build_game
    from tg.hsreplay import parse
    from tg.logs import load_records, segment
    from tg.replay import translate

    recs = parse(XML.encode("utf-8"))
    states = load_records(recs, "test")
    acts = segment(states)
    got = [(a.turn, a.kind) for a in acts]
    assert got == [(1, "buy"), (1, "play"), (2, "upgrade")], got
    s = acts[2].before
    assert [c["entityId"] for c in s["player"]["board"]] == [40], s["player"]["board"]   # 战斗副本 50 战斗后离场，不算
    assert s["player"]["gold"] == 4 and s["player"]["shop"]["upgradeCost"] == 4 and s["player"]["shop"]["canUpgrade"]
    assert [c["entityId"] for c in s["tavern"]] == [31], s["tavern"]
    for i, a in enumerate(acts):
        g, _ = build_game(a.before, seed=i)
        m = np.frombuffer(g.legal_mask(0), dtype=np.uint8).astype(bool)
        assert any(m[x] for x in translate(a)), (a.kind, translate(a), np.flatnonzero(m))

    # 断线重连：GameEntity 再出现时整局重新下发，旧实体清掉；玩家标签并在 GameEntity 里
    redump = XML.replace("</Game>", """<GameEntity id="1"><Tag tag="202" value="1" /><Tag tag="20" value="3" />
<Tag tag="50" value="5" /><Tag tag="202" value="2" /><Tag tag="30" value="5" /><Tag tag="53" value="2" /><Tag tag="27" value="10" /><Tag tag="26" value="4" /><Tag tag="25" value="4" />
<Tag tag="50" value="9" /><Tag tag="202" value="2" /><Tag tag="30" value="9" /><Tag tag="53" value="3" /><Tag tag="20" value="99" /></GameEntity>
<FullEntity id="10" cardID="TB_BaconShop_HERO_34"><Tag tag="50" value="5" /><Tag tag="49" value="1" /><Tag tag="202" value="3" /><Tag tag="45" value="30" /><Tag tag="1377" value="2" /><Tag tag="1373" value="1" /><Tag tag="30" value="5" /></FullEntity>
<FullEntity id="60" cardID="BG26_146"><Tag tag="50" value="5" /><Tag tag="49" value="1" /><Tag tag="202" value="4" /><Tag tag="263" value="1" /><Tag tag="47" value="5" /><Tag tag="45" value="5" /></FullEntity>
<Options id="9"><Option index="0" type="2" entity="0" /></Options>
</Game>""")
    last = [r for r in parse(redump.encode("utf-8")) if r["phase"] == "recruit"][-1]
    assert last["turn"] == 2, last["turn"]                                          # 游戏 TURN 3，不被对手的 99 覆盖
    assert [c["entityId"] for c in last["player"]["board"]] == [60], last["player"]["board"]
    print("hsreplay OK", got)
    return 0


def write_sample(out_dir: str) -> int:
    """CI 用：把上面这份编的回放写成分享包同样的 xml.zip（给 bg.py prodemos 冒烟）。"""
    import zipfile
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    for k in range(3):
        with zipfile.ZipFile(d / f"sample-{k}.xml.zip", "w") as z:
            z.writestr("replay.xml", XML)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--write-sample":
        sys.exit(write_sample(sys.argv[2]))
    sys.exit(main())
