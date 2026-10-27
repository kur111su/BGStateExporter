"""酒馆战旗规则参数。

数值来源：
  - “记录”= 用户 15 局对局记录统计（2026-10-25 ~ 10-27，calib 见 bgrl/README.md）；
  - “假设”= 记录里看不到、按通行规则设定，之后有数据再校准。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Rules:
    players: int = 8
    races_per_game: int = 5                       # 记录：每局 5 个种族
    start_gold: int = 3                           # 记录：第 1 回合 3 金，每回合 +1，最多 10
    max_gold: int = 10
    hero_health: int = 30
    default_armor: int = 10                       # 卡牌数据里没有护甲时用
    buy_cost: int = 3                             # 记录
    sell_value: int = 1
    refresh_cost: int = 1                         # 记录
    board_size: int = 7
    hand_size: int = 10                           # 记录：手牌最多 10
    max_tier: int = 6                             # 记录：最高 6 本（7 本只在特定畸变下出现，暂不支持）
    # 升到 n 本的初始费用，每回合 -1（记录：5 / 7 / 8 / 11 / 11）
    upgrade_cost: dict = field(default_factory=lambda: {2: 5, 3: 7, 4: 8, 5: 11, 6: 11})
    # 各等级酒馆随从格数（记录）和法术格数（记录：1）
    shop_minions: dict = field(default_factory=lambda: {1: 3, 2: 4, 3: 4, 4: 5, 5: 5, 6: 6})
    shop_spells: int = 1
    # 每张随从在共享卡池里的份数（假设）
    pool_copies: dict = field(default_factory=lambda: {1: 16, 2: 15, 3: 13, 4: 11, 5: 9, 6: 7, 7: 5})
    # 战斗伤害上限：{回合上限: 伤害上限}，按回合取第一个 >= 当前回合的键；空 = 不设（假设，待插件导出后校准）
    damage_cap: dict = field(default_factory=dict)
    triple_reward_max_tier: int = 6
    recent_opponents_avoid: int = 3               # 匹配时尽量避开最近几个对手（假设）
    max_actions_per_turn: int = 40

    def cap_for_turn(self, turn: int) -> int:
        for t in sorted(int(k) for k in self.damage_cap):
            if turn <= t:
                return int(self.damage_cap[t] if t in self.damage_cap else self.damage_cap[str(t)])
        return 0

    @staticmethod
    def load(path: str | Path | None = None) -> "Rules":
        r = Rules()
        if path and Path(path).exists():
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            for k, v in data.items():
                if hasattr(r, k):
                    if isinstance(getattr(r, k), dict):
                        v = {int(a): b for a, b in v.items()}
                    setattr(r, k, v)
        return r

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=1)
