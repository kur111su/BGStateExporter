"""训练课程（rl/collector.py 的采样里按配方调用）：

  boost.py       加强对手：机器人换成同回合的高分真人阵容 / 按真人画像放大（ProBoardBooster、BotBooster、AdaptiveStrength）
  seed_comps.py  成型阵容课程：学习者到了某一回合，场面换成成型的 S / A 阵容（CompSeeder）
  gifts.py       送牌课程（CardGifts）
  lobbies.py     指定酒馆种族（LobbyForcer）：单种族专家只在必开某个种族的酒馆里练

都通过 BatchEnv.save / load 改局面，只改机器人或学习者这一局，不碰引擎规则。
第 33328 轮删掉的：送整套阵容（CompGifts，第 23600 轮关掉）、种族课程和起步牌（TribeLobbies / TribeStarter，第 28935 轮关掉），
为什么关掉见 docs/recipe_history.md。
"""
from .boost import (AdaptiveStrength, BotBooster, ProBoardBooster, load_pro_boards, merge_pro_boards, ramp, target_curve,
                    target_for)
from .gifts import CardGifts
from .lobbies import LobbyForcer, parse_types
from .seed_comps import CompSeeder

__all__ = ["AdaptiveStrength", "BotBooster", "ProBoardBooster", "load_pro_boards", "merge_pro_boards", "ramp", "target_curve",
           "target_for", "CardGifts", "CompSeeder", "LobbyForcer", "parse_types"]
