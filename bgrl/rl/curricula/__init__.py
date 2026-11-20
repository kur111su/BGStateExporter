"""训练课程（rl/collector.py 的采样里按配方调用）：

  boost.py       加强对手：机器人换成同回合的高分真人阵容 / 按真人画像放大（ProBoardBooster、BotBooster、AdaptiveStrength）
  seed_comps.py  成型阵容课程：学习者到了某一回合，场面换成成型的 S / A 阵容（CompSeeder）
  gifts.py       送牌 / 送整套阵容（CardGifts、CompGifts）
  tribes.py      种族课程 + 起步牌（TribeLobbies、TribeStarter）

都通过 BatchEnv.save / load 改局面，只改机器人或学习者这一局，不碰引擎规则。
"""
from .boost import (AdaptiveStrength, BotBooster, ProBoardBooster, load_pro_boards, merge_pro_boards, ramp, target_curve,
                    target_for)
from .gifts import CardGifts, CompGifts
from .seed_comps import CompSeeder
from .tribes import STRONG, WEAK, TribeLobbies, TribeStarter, lobby_has_any, new_comps, parse_comps, parse_types

__all__ = ["AdaptiveStrength", "BotBooster", "ProBoardBooster", "load_pro_boards", "merge_pro_boards", "ramp", "target_curve",
           "target_for", "CardGifts", "CompGifts", "CompSeeder", "STRONG", "WEAK", "TribeLobbies", "TribeStarter",
           "lobby_has_any", "new_comps", "parse_comps", "parse_types"]
