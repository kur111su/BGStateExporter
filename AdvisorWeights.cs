using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Hearthstone_Deck_Tracker.Utility.Logging;
using Newtonsoft.Json;

namespace BGStateExporter
{
	/// <summary>
	/// 搜索建议器的评分权重。保存在 %AppData%\HearthstoneDeckTracker\BGStateExporter\advisor_weights.json，
	/// 改完保存即可，下一次计算会自动读取新值，不用重启 HDT。
	/// </summary>
	public class AdvisorWeights
	{
		[JsonProperty("_说明")]
		public string Note { get; set; } =
			"总分 = Σ 权重 × 特征（战力、升本、三连、对子、种族、随从效果、购买、法术、剩余金币）。改完保存即可生效。";

		/// <summary>文件格式版本：1 = v0.4～v0.7；2 = v0.8（新增效果/购买/法术，升本节奏改为按实际对局统计）。</summary>
		[JsonProperty("_版本")] public int SchemaVersion { get; set; } = 2;

		// ---- 战力（Bob's Buddy 模拟 或 没有对手阵容时的随从估值） ----
		[JsonProperty("战力权重")] public double Sim { get; set; } = 1.0;
		[JsonProperty("伤害权重")] public double Damage { get; set; } = 0.01;          // 平均伤害对战力分的加成
		[JsonProperty("满血时战力系数")] public double HpFactorFull { get; set; } = 0.8; // 血量高时战力没那么急
		[JsonProperty("残血时战力系数")] public double HpFactorLow { get; set; } = 1.8;  // 血量低时更看重这回合能不能赢
		[JsonProperty("残血线")] public int LowHp { get; set; } = 12;
		[JsonProperty("无对手阵容时估值权重")] public double Heuristic { get; set; } = 0.15;

		// ---- 升本 ----
		[JsonProperty("升本权重")] public double Upgrade { get; set; } = 0.35;
		[JsonProperty("不升本落后惩罚")] public double BehindCurve { get; set; } = 0.12;
		/// <summary>回合 → 建议达到的酒馆等级。落后时升本加分，超前时升本减分。</summary>
		/// 默认值来自 v0.7 时 11 局实际对局（前 2 名对局的升本回合中位数）。
		[JsonProperty("升本节奏")] public Dictionary<int, int> Curve { get; set; } = NewDefaultCurve();

		private static Dictionary<int, int> NewDefaultCurve() => new() { [1] = 1, [2] = 2, [5] = 3, [7] = 4, [8] = 5, [10] = 6 };

		private static readonly Dictionary<int, int> OldDefaultCurve = new()
		{
			[1] = 1, [2] = 2, [3] = 2, [4] = 3, [5] = 3, [6] = 4, [7] = 4, [8] = 5, [9] = 5, [10] = 5, [11] = 6,
		};

		// ---- 三连 / 种族 ----
		[JsonProperty("三连权重")] public double Triple { get; set; } = 0.45;
		[JsonProperty("对子权重")] public double Pair { get; set; } = 0.06;
		[JsonProperty("主种族权重")] public double Tribe { get; set; } = 0.04;

		// ---- 随从效果 / 购买 / 法术（v0.8 新增：招募阶段效果模拟不了，用这几项近似） ----
		[JsonProperty("随从效果权重")] public double Effect { get; set; } = 0.08;    // 每买一个有战吼/回合结束/复仇/战斗开始等效果的随从（按星级折算）
		[JsonProperty("购买权重")] public double Buys { get; set; } = 0.03;          // 每买一个随从
		[JsonProperty("法术权重")] public double Spell { get; set; } = 0.06;         // 买一个酒馆法术（按星级折算）

		// ---- 金币 ----
		[JsonProperty("每次刷新价值")] public double Refresh { get; set; } = 0.015;
		[JsonProperty("最多计入刷新次数")] public int MaxRefreshValue { get; set; } = 3;

		// ---- 搜索规模 ----
		[JsonProperty("候选随从数")] public int Candidates { get; set; } = 4;     // 酒馆里先按估值挑前几个
		[JsonProperty("候选法术数")] public int SpellCandidates { get; set; } = 2;
		[JsonProperty("每方案最多购买")] public int MaxBuys { get; set; } = 3;
		[JsonProperty("最多模拟方案数")] public int MaxSimPlans { get; set; } = 12;
		[JsonProperty("每方案模拟次数")] public int SimIterations { get; set; } = 1000;
		[JsonProperty("每方案模拟毫秒")] public int SimMaxMs { get; set; } = 200;

		// ---- 自动学习 ----
		[JsonProperty("自动学习")] public bool AutoLearn { get; set; } = true;
		[JsonProperty("自动学习最少局数")] public int AutoLearnMinGames { get; set; } = 30;
		[JsonProperty("自动学习强度")] public double AutoLearnStrength { get; set; } = 1.0;   // 越大越敢偏离默认权重

		// ---------- 特征 ↔ 权重 ----------
		// 每个候选方案先算出这些原始特征，总分 = Σ 权重 × 特征。自动学习只调这几个权重。
		public static readonly string[] FeatureKeys =
			{ "sim", "heur", "upgrade", "behind", "triple", "pair", "tribe", "effect", "buys", "spell", "refresh" };

		public static readonly Dictionary<string, string> FeatureNames = new()
		{
			["sim"] = "战力", ["heur"] = "估值战力", ["upgrade"] = "升本", ["behind"] = "落后节奏",
			["triple"] = "三连", ["pair"] = "对子", ["tribe"] = "种族", ["effect"] = "随从效果",
			["buys"] = "购买", ["spell"] = "法术", ["refresh"] = "剩余金币",
		};

		public double Get(string key) => key switch
		{
			"sim" => Sim, "heur" => Heuristic, "upgrade" => Upgrade, "behind" => BehindCurve,
			"triple" => Triple, "pair" => Pair, "tribe" => Tribe, "effect" => Effect,
			"buys" => Buys, "spell" => Spell, "refresh" => Refresh, _ => 0,
		};

		public void Set(string key, double v)
		{
			v = Math.Round(Math.Max(0, v), 4);
			switch(key)
			{
				case "sim": Sim = v; break;
				case "heur": Heuristic = v; break;
				case "upgrade": Upgrade = v; break;
				case "behind": BehindCurve = v; break;
				case "triple": Triple = v; break;
				case "pair": Pair = v; break;
				case "tribe": Tribe = v; break;
				case "refresh": Refresh = v; break;
				case "effect": Effect = v; break;
				case "buys": Buys = v; break;
				case "spell": Spell = v; break;
			}
		}

		public double Score(Dictionary<string, double> f)
		{
			var s = 0.0;
			foreach(var kv in f)
				s += Get(kv.Key) * kv.Value;
			return s;
		}

		public void Save()
		{
			if(FilePath == null)
				return;
			File.WriteAllText(FilePath, JsonConvert.SerializeObject(this, Formatting.Indented));
			_cached = this;
			_cachedTime = File.GetLastWriteTimeUtc(FilePath);
		}

		public int DesiredTier(int turn)
		{
			var best = 1;
			foreach(var kv in Curve)
				if(kv.Key <= turn && kv.Key >= 0)
					best = Math.Max(best, kv.Value);
			return best;
		}

		private static bool SameCurve(Dictionary<int, int> a, Dictionary<int, int> b) =>
			a.Count == b.Count && a.All(kv => b.TryGetValue(kv.Key, out var v) && v == kv.Value);

		// ---------- 读写 ----------

		public static string? FilePath { get; set; }
		private static AdvisorWeights? _cached;
		private static DateTime _cachedTime;

		public static AdvisorWeights Load()
		{
			if(FilePath == null)
				return _cached ??= new AdvisorWeights();
			try
			{
				if(!File.Exists(FilePath))
				{
					var def = new AdvisorWeights();
					File.WriteAllText(FilePath, JsonConvert.SerializeObject(def, Formatting.Indented));
					_cached = def;
					_cachedTime = File.GetLastWriteTimeUtc(FilePath);
					return def;
				}
				var t = File.GetLastWriteTimeUtc(FilePath);
				if(_cached == null || t != _cachedTime)
				{
					// 缺少的字段保留默认值，所以旧文件升级后也能用
					var w = new AdvisorWeights { SchemaVersion = 1 };   // 旧文件没有“_版本”字段 → 视为 1
					JsonConvert.PopulateObject(File.ReadAllText(FilePath), w,
						new JsonSerializerSettings { ObjectCreationHandling = ObjectCreationHandling.Replace });
					if(w.SchemaVersion < 2)
					{
						// 升级：升本节奏还是旧默认值（没手动改过）就换成新默认值；新权重字段已经取默认值
						if(SameCurve(w.Curve, OldDefaultCurve))
							w.Curve = NewDefaultCurve();
						w.SchemaVersion = 2;
						File.WriteAllText(FilePath, JsonConvert.SerializeObject(w, Formatting.Indented));
						t = File.GetLastWriteTimeUtc(FilePath);
						Log.Info("[BGStateExporter] advisor_weights.json 已升级到第 2 版");
					}
					_cached = w;
					_cachedTime = t;
				}
				return _cached;
			}
			catch(Exception ex)
			{
				Log.Warn($"[BGStateExporter] 读取 advisor_weights.json 失败，使用默认权重：{ex.Message}");
				return _cached ??= new AdvisorWeights();
			}
		}
	}
}
