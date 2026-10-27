"""对局里的实体：随从、法术牌、玩家。"""
from __future__ import annotations

from dataclasses import dataclass, field

KEYWORDS = ("taunt", "divineShield", "poisonous", "venomous", "windfury", "megaWindfury", "reborn", "stealth",
            "cleave")


@dataclass(eq=False)
class Minion:
    uid: int
    card_id: str
    atk: int
    hp: int
    tier: int
    races: list[str]
    golden: bool = False
    kw: set[str] = field(default_factory=set)        # 关键词
    sd: list[int] = field(default_factory=lambda: [0, 0, 0, 0])   # 计数器（对应 TAG_SCRIPT_DATA_NUM_1..4）
    temp_atk: int = 0                                  # 只在本回合有效的加成（回合结束清掉）
    temp_hp: int = 0
    frozen: bool = False                               # 只对酒馆里的随从有意义
    triple_reward: bool = False                        # 三连合成的金色随从：打出时给三连奖励
    attached: list[str] = field(default_factory=list)  # 磁力吸附上的机械 / 附魔名（只作记录）
    data: dict = field(default_factory=dict)           # 卡牌实现自用的状态

    @property
    def card(self) -> str:
        return self.card_id

    def has(self, kw: str) -> bool:
        return kw in self.kw

    def is_race(self, race: str) -> bool:
        return race in self.races or "ALL" in self.races

    def copy(self, uid: int) -> "Minion":
        m = Minion(uid, self.card_id, self.atk, self.hp, self.tier, list(self.races), self.golden, set(self.kw),
                   list(self.sd), self.temp_atk, self.temp_hp, False, False, list(self.attached), dict(self.data))
        return m

    def __repr__(self) -> str:
        g = "金" if self.golden else ""
        return f"<{g}{self.card_id} {self.atk}/{self.hp}>"


@dataclass(eq=False)
class Spell:
    uid: int
    card_id: str
    cost: int = 0
    tier: int = 0
    data: dict = field(default_factory=dict)           # 例如三连奖励的星级

    def __repr__(self) -> str:
        return f"<法术 {self.card_id}>"


Card = Minion | Spell


@dataclass(eq=False)
class Choice:
    """待玩家做的选择（发现、抉择、选饰品……）。options 是候选卡；on_pick(game, player, index) 完成后续。"""
    kind: str
    options: list[Card]
    on_pick: object = None
    source: str = ""


@dataclass(eq=False)
class Player:
    pid: int
    hero_id: str
    health: int
    armor: int
    tier: int = 1
    gold: int = 0
    max_gold: int = 0
    extra_gold_next: int = 0                           # 下回合额外金币
    upgrade_cost: int = 5
    board: list[Minion] = field(default_factory=list)  # 从左到右
    hand: list[Card] = field(default_factory=list)
    shop: list[Minion] = field(default_factory=list)
    shop_spells: list[Spell] = field(default_factory=list)
    frozen: bool = False
    free_refreshes: int = 0
    pending: list[Choice] = field(default_factory=list)
    hero_power_used: bool = False
    hero_power_uses: int = 0
    trinkets: list[str] = field(default_factory=list)
    alive: bool = True
    place: int = 0
    dead_turn: int = 0
    recent_opponents: list[int] = field(default_factory=list)
    seen_boards: dict = field(default_factory=dict)    # 对手 pid -> (回合, 当时的阵容快照)
    actions_this_turn: int = 0
    # 全局计数（卡牌效果用）
    counters: dict = field(default_factory=lambda: {
        "spells_cast": 0, "spells_cast_turn": 0, "minions_bought": 0, "minions_sold": 0, "refreshes": 0,
        "gold_spent_turn": 0, "blood_gem_atk": 1, "blood_gem_hp": 1, "tavern_atk": 0, "tavern_hp": 0,
        "tavern_spell_atk": 0, "tavern_spell_hp": 0, "elementals_played": 0, "battlecries": 0,
        "friendly_deaths": 0, "discards": 0, "buy_cost_mod": 0, "refresh_cost_mod": 0, "upgrade_discount": 0,
    })
    data: dict = field(default_factory=dict)           # 英雄技能、饰品等自用的状态

    @property
    def total_health(self) -> int:
        return self.health + self.armor

    def find(self, uid: int) -> Card | None:
        for zone in (self.board, self.hand, self.shop, self.shop_spells):
            for c in zone:
                if c.uid == uid:
                    return c
        return None
