using System.Collections.Generic;

namespace BGStateExporter
{
	// 输出 JSON 的数据结构（属性名会被序列化为 camelCase）。
	// 字段说明见 README.md 的“JSON 格式”一节。

	public class BgState
	{
		public int Schema { get; set; } = 1;
		public PluginInfo? Plugin { get; set; }
		public LearningStatus? Learning { get; set; }
		public bool InGame { get; set; }
		public bool IsBattlegrounds { get; set; }
		public string Mode { get; set; } = "none";        // solo | duos | none
		public string Phase { get; set; } = "menu";       // menu | hero_select | recruit | combat | game_over
		public int Turn { get; set; }
		public List<string> AvailableRaces { get; set; } = new();
		public int? AnomalyDbfId { get; set; }
		public PlayerState? Player { get; set; }
		public List<CardState> Tavern { get; set; } = new();      // 仅招募阶段：酒馆里可购买的随从/法术
		public OpponentState? Opponent { get; set; }              // 仅战斗阶段：本回合对手
		public List<LeaderboardEntry> Leaderboard { get; set; } = new();
		public int NextOpponentPlayerId { get; set; }              // 下回合对手的 PLAYER_ID（0 = 未知）
		public SimulationState? Simulation { get; set; }           // Bob's Buddy 模拟结果
	}

	public class LearningStatus
	{
		public bool Enabled { get; set; }
		public int GamesCollected { get; set; }            // 已记录学习样本的对局数
		public int SamplesCollected { get; set; }
		public int MinGames { get; set; }                  // 攒够多少局开始自动拟合
		public int SamplesThisGame { get; set; }
		public bool Fitting { get; set; }
		public string? LastFitTime { get; set; }
		public string? LastFitMessage { get; set; }
	}

	public class PluginInfo
	{
		public string? Version { get; set; }
		public string? LatestVersion { get; set; }        // GitHub 上的最新版本（null = 还没检查完，空 = 已是最新）
		public bool UpdateReady { get; set; }             // 新版本已下载，重启 HDT 生效
	}

	// ---------------- 对战模拟 ----------------

	public class SimulationState
	{
		public bool Available { get; set; }                // Bob's Buddy 是否可用
		public string? Message { get; set; }               // 不可用时的原因
		public LiveCombatSim? Live { get; set; }           // HDT 自己跑的本场（或上一场）战斗模拟
		public MatchupState Matchups { get; set; } = new();
		public OptimizerState Optimizer { get; set; } = new();
		public AdvisorState Advisor { get; set; } = new();
	}

	// ---------------- 第一层：搜索建议 ----------------

	public class AdvisorState
	{
		public string Status { get; set; } = "idle";       // idle | waiting | running | done | error
		public int Turn { get; set; }
		public string? Message { get; set; }               // 说明：用了哪个对手的阵容、没有阵容时怎么估值等
		public string? TargetHeroName { get; set; }        // 用来模拟的对手
		public int TargetBoardSeenTurn { get; set; }
		public int PlansConsidered { get; set; }           // 枚举了多少种方案
		public int PlansSimulated { get; set; }            // 其中多少种跑了模拟
		public long ElapsedMs { get; set; }
		public List<AdvisorPlan> Plans { get; set; } = new();   // 按得分从高到低，最多 3 个
		public List<WeightInfo> Weights { get; set; } = new();  // 这次计算用的各项权重
		public int DesiredTier { get; set; }                   // 升本节奏：这回合建议的酒馆等级
		public double HpFactor { get; set; }                   // 血量系数（战力特征已乘上）
		public int RunnerUps { get; set; }                     // 前三名之外还有多少个有效方案
	}

	public class AdvisorPlan
	{
		public int Rank { get; set; }
		public double Score { get; set; }
		public List<PlanStep> Steps { get; set; } = new();
		public List<string> Reasons { get; set; } = new();
		public SimResult? Sim { get; set; }                 // 打完这套操作后，对目标对手的模拟结果
		public int GoldLeft { get; set; }
		public List<int> FinalBoard { get; set; } = new();  // 结束时场面的 entityId，从左到右
		public Dictionary<string, double> ScoreParts { get; set; } = new();   // 各项得分，方便调权重
		public Dictionary<string, double> Features { get; set; } = new();     // 原始特征（键同 weights[].key），得分 = 权重 × 特征
	}

	public class WeightInfo
	{
		public string Key { get; set; } = "";
		public string Name { get; set; } = "";
		public double Value { get; set; }                    // 当前使用的权重
		public double Default { get; set; }                  // 默认权重（自动学习前）
	}

	public class PlanStep
	{
		public string Type { get; set; } = "";              // upgrade | sell | buy | play | refresh | freeze
		public string Text { get; set; } = "";
		public int EntityId { get; set; }                   // 相关卡牌（没有就是 0）
		public int Gold { get; set; }                       // 这一步之后剩余金币
	}

	public class SimResult
	{
		public double WinRate { get; set; }
		public double TieRate { get; set; }
		public double LossRate { get; set; }
		public double WinLethal { get; set; }              // 我方打死对手的概率
		public double LossLethal { get; set; }             // 我方被打死的概率
		public double AvgDamageDealt { get; set; }         // 赢的时候平均造成伤害
		public double AvgDamageTaken { get; set; }         // 输的时候平均承受伤害
		public int SimulationCount { get; set; }
	}

	public class LiveCombatSim : SimResult
	{
		public int Turn { get; set; }
		public string? State { get; set; }                 // Bob's Buddy 状态，如 Combat / Shopping
		public string? ErrorState { get; set; }            // None 表示正常
		public bool HasResult { get; set; }
	}

	public class MatchupState
	{
		public string Status { get; set; } = "idle";       // idle | waiting | running | done | error
		public int Turn { get; set; }
		public string? Error { get; set; }
		public List<MatchupResult> Results { get; set; } = new();
	}

	public class MatchupResult : SimResult
	{
		public int OpponentPlayerId { get; set; }
		public string? HeroCardId { get; set; }
		public string? HeroName { get; set; }
		public bool IsNextOpponent { get; set; }
		public int BoardSeenTurn { get; set; }             // 这份对手阵容是第几回合看到的
		public int OpponentMinionCount { get; set; }
		public string? Error { get; set; }
	}

	public class OptimizerState
	{
		public string Status { get; set; } = "idle";       // idle | running | done | error
		public int RequestId { get; set; }
		public int Turn { get; set; }
		public int OpponentPlayerId { get; set; }
		public string? OpponentHeroName { get; set; }
		public int BoardSeenTurn { get; set; }
		public SimResult? Current { get; set; }            // 当前站位
		public SimResult? Best { get; set; }               // 找到的最佳站位
		public List<int> CurrentOrder { get; set; } = new();   // entityId，从左到右
		public List<int> BestOrder { get; set; } = new();
		public List<string> BestOrderNames { get; set; } = new();
		public int Evaluated { get; set; }                 // 试过多少种站位
		public string? Message { get; set; }
	}

	public class PlayerState
	{
		public string? HeroCardId { get; set; }
		public string? HeroName { get; set; }
		public int Health { get; set; }
		public int Armor { get; set; }
		public int TavernTier { get; set; }
		public int Gold { get; set; }
		public int MaxGold { get; set; }
		public int ExtraGoldNextTurn { get; set; }
		public int LeaderboardPlace { get; set; }
		public List<HeroPowerState> HeroPowers { get; set; } = new();
		public List<CardState> Board { get; set; } = new();
		public List<CardState> Hand { get; set; } = new();
		public List<CardState> Trinkets { get; set; } = new();
		public List<CardState> QuestRewards { get; set; } = new();
		public ShopInfo? Shop { get; set; }
	}

	public class ShopInfo
	{
		public int BuyCost { get; set; } = 3;              // 购买随从的价格
		public bool BuyCostKnown { get; set; }             // false = 没读到按钮，用的默认值
		public int SpellBuyCost { get; set; }
		public int RefreshCost { get; set; } = 1;
		public bool RefreshCostKnown { get; set; }
		public int FreeRefreshes { get; set; }
		public int FreezeCost { get; set; }
		public bool CanUpgrade { get; set; }               // 还能不能升本（满级或没读到升级按钮时为 false）
		public int UpgradeCost { get; set; }               // 升本价格（每回合会降 1）
	}

	public class OpponentState
	{
		public string? HeroCardId { get; set; }
		public string? HeroName { get; set; }
		public int Health { get; set; }
		public int Armor { get; set; }
		public int TavernTier { get; set; }
		public List<CardState> Board { get; set; } = new();
	}

	public class HeroPowerState
	{
		public int EntityId { get; set; }
		public string? CardId { get; set; }
		public string? Name { get; set; }
		public int Cost { get; set; }
		public bool Used { get; set; }
	}

	public class CardState
	{
		public int EntityId { get; set; }
		public string? CardId { get; set; }
		public int DbfId { get; set; }
		public string? Name { get; set; }
		public string Type { get; set; } = "minion";      // minion | spell | other
		public int Position { get; set; }                  // ZONE_POSITION，场面从左到右 1..7
		public int Attack { get; set; }
		public int Health { get; set; }
		public int Tier { get; set; }
		public int Cost { get; set; }
		public List<string> Races { get; set; } = new();
		public bool Golden { get; set; }
		public bool Taunt { get; set; }
		public bool DivineShield { get; set; }
		public bool Poisonous { get; set; }
		public bool Venomous { get; set; }
		public bool Reborn { get; set; }
		public bool Windfury { get; set; }
		public bool Stealth { get; set; }
		public bool Deathrattle { get; set; }
		public bool Frozen { get; set; }
	}

	public class LeaderboardEntry
	{
		public int Place { get; set; }
		public string? HeroCardId { get; set; }
		public string? HeroName { get; set; }
		public int Health { get; set; }
		public int Armor { get; set; }
		public int TavernTier { get; set; }
		public int PlayerId { get; set; }                  // 与 nextOpponentPlayerId、模拟结果里的 opponentPlayerId 对应
		public bool IsPlayer { get; set; }
		public bool Dead { get; set; }
	}
}
