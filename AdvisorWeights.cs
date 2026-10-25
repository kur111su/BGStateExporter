using System;
using System.Collections.Generic;
using System.IO;
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
			"总分 = 战力分 × 战力权重 × 血量系数 + 升本分 + 三连分 + 种族分 + 剩余金币分。改完保存即可生效。";

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
		[JsonProperty("升本节奏")] public Dictionary<int, int> Curve { get; set; } = new()
		{
			[1] = 1, [2] = 2, [3] = 2, [4] = 3, [5] = 3, [6] = 4, [7] = 4, [8] = 5, [9] = 5, [10] = 5, [11] = 6,
		};

		// ---- 三连 / 种族 ----
		[JsonProperty("三连权重")] public double Triple { get; set; } = 0.45;
		[JsonProperty("对子权重")] public double Pair { get; set; } = 0.06;
		[JsonProperty("主种族权重")] public double Tribe { get; set; } = 0.04;

		// ---- 金币 ----
		[JsonProperty("每次刷新价值")] public double Refresh { get; set; } = 0.015;
		[JsonProperty("最多计入刷新次数")] public int MaxRefreshValue { get; set; } = 3;

		// ---- 搜索规模 ----
		[JsonProperty("候选随从数")] public int Candidates { get; set; } = 4;     // 酒馆里先按估值挑前几个
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
		public static readonly string[] FeatureKeys = { "sim", "heur", "upgrade", "behind", "triple", "pair", "tribe", "refresh" };

		public static readonly Dictionary<string, string> FeatureNames = new()
		{
			["sim"] = "战力", ["heur"] = "估值战力", ["upgrade"] = "升本", ["behind"] = "落后节奏",
			["triple"] = "三连", ["pair"] = "对子", ["tribe"] = "种族", ["refresh"] = "剩余金币",
		};

		public double Get(string key) => key switch
		{
			"sim" => Sim, "heur" => Heuristic, "upgrade" => Upgrade, "behind" => BehindCurve,
			"triple" => Triple, "pair" => Pair, "tribe" => Tribe, "refresh" => Refresh, _ => 0,
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
					var w = new AdvisorWeights();
					JsonConvert.PopulateObject(File.ReadAllText(FilePath), w,
						new JsonSerializerSettings { ObjectCreationHandling = ObjectCreationHandling.Replace });
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
