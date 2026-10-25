using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using HearthDb.Enums;
using Hearthstone_Deck_Tracker.Hearthstone;
using Hearthstone_Deck_Tracker.Hearthstone.Entities;

namespace BGStateExporter
{
	/// <summary>
	/// 第一层：规则 + 搜索。
	/// 招募阶段枚举“升不升本 × 买哪几个 × 卖谁”的组合，算出结束时的场面，
	/// 用 Bob's Buddy 对目标对手（下个对手上次的阵容）模拟，再加上升本节奏、三连、种族、剩余金币等启发式打分，
	/// 输出得分最高的 3 个方案。没有对手阵容时（前几回合）只用随从估值。
	///
	/// 已知不考虑：战吼 / 回合结束等招募阶段效果、酒馆法术、英雄技能、刷新后会出什么。
	/// </summary>
	internal sealed partial class SimManager
	{
		private const int AdvisorDebounceMs = 400;
		private const int BoardLimit = 7;

		private volatile AdvisorState _advisor = new();
		private volatile AdvisorRun? _lastRun;
		private int _advisorRunSeq;

		/// <summary>最近一次计算的完整记录（所有有效候选方案的特征），给自动学习用。</summary>
		public AdvisorRun? LastRun { get => _lastRun; private set => _lastRun = value; }
		private string _advisorSignature = "";
		private DateTime _advisorChangedAt = DateTime.MinValue;
		private bool _advisorRunning;
		private bool _advisorDirty;

		private sealed class AdvisorInput
		{
			public int Turn;
			public int Gold;
			public int Tier;
			public int Hp;                          // 血量 + 护甲
			public ShopInfo Shop = new();
			public SimContext Ctx = new();
			public SimSide PlayerTemplate = new();  // 英雄技能、血量等；Minions 会被每个方案替换
			public List<Entity> Board = new();      // 当前场面，从左到右
			public List<Entity> Hand = new();       // 手牌里的随从（打出免费）
			public List<Entity> Tavern = new();     // 酒馆里的随从
			public int TavernSpells;
			public string? DominantRace;
			public CapturedOpponent? Target;
			public bool TargetIsNext;
		}

		/// <summary>方案里的一个随从位置。</summary>
		private sealed class Slot
		{
			public Entity E = null!;
			public string Src = "";                 // board | hand | buy | triple
			public double V;
			public string BaseId = "";
			public bool Golden;
		}

		private sealed class Candidate
		{
			public bool Upgrade;
			public List<Entity> Buys = new();
			public List<Slot> Final = new();
			public List<Entity> Sells = new();
			public List<string> Triples = new();    // 凑成三连的随从名
			public int Pairs;
			public int TribeHits;
			public int GoldLeft;
			public double Power;                    // 随从估值之和
			public Dictionary<string, double> F = new();       // 原始特征，总分 = Σ 权重 × 特征
			public bool Valid = true;                          // 有对手阵容却没模拟到的方案不参与排名
			public Dictionary<string, double> Parts = new();   // 显示用：权重 × 特征（中文名）
			public double Score;
			public SimResult? Sim;
			public List<PlanStep> Steps = new();
			public List<string> Reasons = new();
		}

		// ---------- 调度（UI 线程） ----------

		private void TickAdvisor(GameV2 game, BgState state)
		{
			var sig = AdvisorSignature(state);
			if(sig != _advisorSignature)
			{
				_advisorSignature = sig;
				_advisorChangedAt = DateTime.Now;
				if(_advisorRunning)
					_advisorDirty = true;
				else
					_advisor = new AdvisorState
					{
						Status = "waiting", Turn = state.Turn,
						// 同一回合内先保留旧方案，算完再替换，面板不会闪空
						Plans = _advisor.Turn == state.Turn ? _advisor.Plans : new List<AdvisorPlan>(),
					};
			}

			if(_advisorRunning || _advisor.Status != "waiting")
				return;
			if((DateTime.Now - _advisorChangedAt).TotalMilliseconds < AdvisorDebounceMs)
				return;

			AdvisorInput input;
			try
			{
				input = CaptureAdvisorInput(game, state);
			}
			catch(Exception ex)
			{
				_advisor = new AdvisorState { Status = "error", Turn = state.Turn, Message = Unwrap(ex) };
				return;
			}
			_advisorRunning = true;
			_ = RunAdvisorAsync(input);
		}

		private static string AdvisorSignature(BgState s)
		{
			var p = s.Player!;
			var sb = new StringBuilder();
			sb.Append(s.Turn).Append('|').Append(p.Gold).Append('|').Append(p.TavernTier).Append('|')
				.Append(p.Health + p.Armor).Append('|').Append(s.NextOpponentPlayerId).Append('|');
			if(p.Shop != null)
				sb.Append(p.Shop.BuyCost).Append(',').Append(p.Shop.RefreshCost).Append(',')
					.Append(p.Shop.CanUpgrade ? p.Shop.UpgradeCost : -1).Append('|');
			foreach(var m in p.Board)
				sb.Append(m.EntityId).Append(':').Append(m.Attack).Append('/').Append(m.Health)
					.Append(m.DivineShield ? "D" : "").Append(m.Taunt ? "T" : "").Append(',');
			sb.Append('|');
			foreach(var m in p.Hand)
				sb.Append(m.EntityId).Append(',');
			sb.Append('|');
			foreach(var m in s.Tavern)
				sb.Append(m.EntityId).Append(m.Frozen ? "F" : "").Append(',');
			return sb.ToString();
		}

		private AdvisorInput CaptureAdvisorInput(GameV2 game, BgState state)
		{
			var p = state.Player!;
			var input = new AdvisorInput
			{
				Turn = state.Turn,
				Gold = p.Gold,
				Tier = p.TavernTier,
				Hp = p.Health + p.Armor,
				Shop = p.Shop ?? new ShopInfo(),
				Ctx = CaptureContext(game, state),
				PlayerTemplate = CapturePlayer(game),
			};
			input.Board = input.PlayerTemplate.Minions.ToList();
			input.Hand = game.Player.Hand.Where(e => e.IsMinion && !string.IsNullOrEmpty(e.CardId))
				.OrderBy(e => e.ZonePosition).Select(e => e.Clone()).ToList();
			var shop = game.Opponent.Board.ToList();
			input.Tavern = shop.Where(e => e.IsMinion && !string.IsNullOrEmpty(e.CardId))
				.OrderBy(e => e.ZonePosition).Select(e => e.Clone()).ToList();
			input.TavernSpells = shop.Count(e => e.IsBattlegroundsSpell || e.IsSpell);
			input.DominantRace = DominantRace(input.Board.Concat(input.Hand));

			var opponents = CaptureOpponents(game);
			var next = opponents.FirstOrDefault(o => o.PlayerId == state.NextOpponentPlayerId);
			input.TargetIsNext = next != null;
			input.Target = next ?? opponents
				.OrderByDescending(o => o.Side.Minions.Sum(m => (double)m.Attack + m.Health))
				.FirstOrDefault();
			return input;
		}

		// ---------- 搜索（后台线程） ----------

		private async Task RunAdvisorAsync(AdvisorInput input)
		{
			var sw = Stopwatch.StartNew();
			var w = AdvisorWeights.Load();
			try
			{
				var candidates = Enumerate(input, w);
				foreach(var c in candidates)
					ComputeFeatures(c, input, w);

				// 先用估值排序，只模拟最有希望的若干个；“什么都不买”的方案总是模拟，作为对照
				var ordered = candidates.OrderByDescending(c => c.Score).ToList();
				var toSim = new List<Candidate>();
				if(input.Target != null)
				{
					toSim = ordered.Take(Math.Max(1, w.MaxSimPlans)).ToList();
					var baseline = candidates.FirstOrDefault(c => !c.Upgrade && c.Buys.Count == 0);
					if(baseline != null && !toSim.Contains(baseline))
						toSim.Add(baseline);
				}

				var simulated = 0;
				foreach(var c in toSim)
				{
					if(!_recruitPhase)
						break;
					try
					{
						var side = new SimSide
						{
							Friendly = true,
							Health = input.PlayerTemplate.Health,
							DamageTaken = input.PlayerTemplate.DamageTaken,
							Tier = input.Tier + (c.Upgrade ? 1 : 0),
							HeroPowers = input.PlayerTemplate.HeroPowers,
							Attached = input.PlayerTemplate.Attached,
							Minions = c.Final.Select(s => s.E).ToList(),
						};
						var sim = SimBridge.BuildInput(input.Ctx, side, input.Target!.Side);
						c.Sim = await SimBridge.RunAsync(sim, w.SimIterations, w.SimMaxMs);
						simulated++;
					}
					catch(Exception ex)
					{
						c.Reasons.Add("模拟失败：" + Unwrap(ex));
					}
				}

				// 有模拟结果的方案用模拟战力；没有对手阵容时用估值战力；有对手阵容却没模拟到的不参与排名
				var hpFactor = HpFactor(input.Hp, w);
				foreach(var c in candidates)
				{
					if(c.Sim != null)
					{
						var r = c.Sim;
						var power = r.WinRate - r.LossRate + w.Damage * (r.AvgDamageDealt * r.WinRate - r.AvgDamageTaken * r.LossRate);
						c.F["sim"] = Math.Round(Math.Max(-1.5, Math.Min(1.5, power)) * hpFactor, 4);
						c.F.Remove("heur");
					}
					else if(input.Target != null)
						c.Valid = false;
					c.Score = w.Score(c.F);
					FillParts(c, w);
				}

				var valid = candidates.Where(c => c.Valid).ToList();
				var top = valid.OrderByDescending(c => c.Score).Take(3).ToList();
				LastRun = new AdvisorRun
				{
					Seq = ++_advisorRunSeq,
					Turn = input.Turn,
					Snapshot = SnapshotOf(input),
					Candidates = valid.Select(c => new AdvisorRun.Cand
					{
						Upgrade = c.Upgrade,
						Buys = c.Buys.Select(e => e.Id).ToList(),
						Sells = c.Sells.Select(e => e.Id).ToList(),
						F = new Dictionary<string, double>(c.F),
					}).ToList(),
				};
				foreach(var c in top)
					Describe(c, input, w);

				var msg = new List<string>();
				if(input.Target == null)
					msg.Add("还没见过任何对手的阵容，这回合只按随从估值排序");
				else if(!input.TargetIsNext)
					msg.Add("下个对手的阵容未知，用目前见过最强的阵容模拟");
				if(input.TavernSpells > 0)
					msg.Add("酒馆法术和英雄技能没有计入");
				msg.Add("战吼、回合结束等招募阶段效果没有计入");

				_advisor = new AdvisorState
				{
					Status = _recruitPhase ? "done" : "error",
					Turn = input.Turn,
					TargetHeroName = input.Target?.HeroName,
					TargetBoardSeenTurn = input.Target?.BoardSeenTurn ?? 0,
					PlansConsidered = candidates.Count,
					PlansSimulated = simulated,
					ElapsedMs = sw.ElapsedMilliseconds,
					Message = _recruitPhase ? string.Join("；", msg) : "已进入战斗，计算中止",
					Plans = top.Select((c, i) => new AdvisorPlan
					{
						Rank = i + 1,
						Score = Math.Round(c.Score, 3),
						Steps = c.Steps,
						Reasons = c.Reasons,
						Sim = c.Sim,
						GoldLeft = c.GoldLeft,
						FinalBoard = c.Final.Select(s => s.E.Id).ToList(),
						ScoreParts = c.Parts.ToDictionary(kv => kv.Key, kv => Math.Round(kv.Value, 3)),
						Features = c.F.ToDictionary(kv => kv.Key, kv => Math.Round(kv.Value, 4)),
					}).ToList(),
					Weights = AdvisorWeights.FeatureKeys.Select(k => new WeightInfo
					{
						Key = k,
						Name = AdvisorWeights.FeatureNames[k],
						Value = Math.Round(w.Get(k), 4),
						Default = Math.Round(new AdvisorWeights().Get(k), 4),
					}).ToList(),
					DesiredTier = w.DesiredTier(input.Turn),
					HpFactor = Math.Round(hpFactor, 3),
					RunnerUps = Math.Max(0, valid.Count - top.Count),
				};
			}
			catch(Exception ex)
			{
				_advisor = new AdvisorState { Status = "error", Turn = input.Turn, Message = Unwrap(ex), ElapsedMs = sw.ElapsedMilliseconds };
			}
			finally
			{
				_advisorRunning = false;
				if(_advisorDirty)
				{
					_advisorDirty = false;
					_advisor = new AdvisorState { Status = "waiting", Turn = input.Turn, Plans = _advisor.Plans };
				}
			}
		}

		/// <summary>枚举所有“升不升本 × 买哪几个”的组合，算出卖谁、三连、最终场面和剩余金币。</summary>
		private static List<Candidate> Enumerate(AdvisorInput input, AdvisorWeights w)
		{
			var norm = StatNorm(input);
			Slot ToSlot(Entity e, string src) => new()
			{
				E = e, Src = src, V = Value(e, norm, input.DominantRace, input.Tier),
				BaseId = BaseId(e.CardId), Golden = e.HasTag(GameTag.PREMIUM),
			};

			// 酒馆里先按估值挑前几个，再加上能和已有随从凑对/凑三连的
			var owned = input.Board.Concat(input.Hand).Where(e => !e.HasTag(GameTag.PREMIUM))
				.GroupBy(e => BaseId(e.CardId)).ToDictionary(g => g.Key, g => g.Count());
			var pool = input.Tavern
				.OrderByDescending(e => owned.ContainsKey(BaseId(e.CardId)) ? 1 : 0)
				.ThenByDescending(e => Value(e, norm, input.DominantRace, input.Tier))
				.Take(Math.Max(1, w.Candidates))
				.ToList();

			var subsets = new List<List<Entity>> { new() };
			foreach(var e in pool)
				subsets.AddRange(subsets.Where(s => s.Count < w.MaxBuys).Select(s => s.Concat(new[] { e }).ToList()).ToList());

			var result = new List<Candidate>();
			var upgradeOptions = input.Shop.CanUpgrade && input.Gold >= input.Shop.UpgradeCost
				? new[] { false, true } : new[] { false };

			foreach(var up in upgradeOptions)
			foreach(var buys in subsets)
			{
				var c = new Candidate { Upgrade = up, Buys = buys };
				var gold = input.Gold - (up ? input.Shop.UpgradeCost : 0);

				// 场面 → 手牌随从 → 新买的，依次排在右边
				var slots = input.Board.Select(e => ToSlot(e, "board"))
					.Concat(input.Hand.Select(e => ToSlot(e, "hand")))
					.Concat(buys.Select(e => ToSlot(e, "buy")))
					.ToList();

				// 三连：同一随从（非金色）凑够 3 个就合成金色
				foreach(var g in slots.Where(s => !s.Golden).GroupBy(s => s.BaseId).Where(g => g.Count() >= 3).ToList())
				{
					var three = g.Take(3).ToList();
					if(!three.Any(s => s.Src == "buy"))
						continue;   // 不是这回合买出来的三连，游戏已经自动合成过了
					var merged = MergeTriple(three);
					var idx = slots.IndexOf(three[0]);
					foreach(var s in three)
						slots.Remove(s);
					slots.Insert(Math.Min(idx, slots.Count), ToSlot(merged, "triple"));
					c.Triples.Add(three[0].E.LocalizedName ?? three[0].E.CardId ?? "?");
				}

				// 对子：新买的随从和已有的凑成一对
				c.Pairs = buys.Select(e => BaseId(e.CardId)).Distinct()
					.Count(id => owned.TryGetValue(id, out var n) && n == 1);
				c.TribeHits = input.DominantRace == null ? 0
					: buys.Count(e => Races(e).Contains(input.DominantRace));

				// 位置不够：卖掉场上最弱的（不卖三连材料、不卖新买的）
				var overflow = slots.Count - BoardLimit;
				if(overflow > 0)
				{
					var sellable = slots.Where(s => s.Src == "board").OrderBy(s => s.V).Take(overflow).ToList();
					if(sellable.Count < overflow)
						continue;   // 新随从太多放不下
					var weakestNew = slots.Where(s => s.Src == "buy").Select(s => s.V).DefaultIfEmpty(double.MaxValue).Min();
					if(sellable.Max(s => s.V) > weakestNew)
						continue;   // 卖掉的比买的还强，这个方案没意义
					foreach(var s in sellable)
					{
						slots.Remove(s);
						c.Sells.Add(s.E);
						gold += SellValue(s.E);
					}
				}

				var cost = buys.Count * input.Shop.BuyCost;
				if(cost > gold)
					continue;
				c.GoldLeft = gold - cost;
				c.Final = slots;
				c.Power = slots.Sum(s => s.V);
				result.Add(c);
			}
			return result;
		}

		/// <summary>
		/// 算出方案的原始特征（与权重无关）。战力特征在模拟完成后再填：
		///   sim     = (胜率 − 负率 + 伤害项) × 血量系数        —— 有模拟结果时
		///   heur    = (随从估值之和 − 当前场面估值) × 血量系数   —— 没有对手阵容时代替 sim
		///   upgrade = 升本时：落后节奏的等级数（−1～2）
		///   behind  = 不升本且落后节奏时：−落后等级数
		///   triple / pair / tribe = 三连数 / 新对子数 / 主种族随从数
		///   refresh = 剩余金币够刷新几次（含免费刷新，有上限）
		/// </summary>
		private static void ComputeFeatures(Candidate c, AdvisorInput input, AdvisorWeights w)
		{
			var desired = w.DesiredTier(input.Turn);
			var gap = desired - input.Tier;
			if(c.Upgrade)
				c.F["upgrade"] = Math.Max(-1, Math.Min(2, gap));
			else if(gap > 0 && input.Shop.CanUpgrade)
				c.F["behind"] = -gap;

			if(c.Triples.Count > 0) c.F["triple"] = c.Triples.Count;
			if(c.Pairs > 0) c.F["pair"] = c.Pairs;
			if(c.TribeHits > 0) c.F["tribe"] = c.TribeHits;

			var refreshCost = Math.Max(1, input.Shop.RefreshCost);
			var refreshes = Math.Min(w.MaxRefreshValue, c.GoldLeft / refreshCost + input.Shop.FreeRefreshes);
			if(refreshes > 0) c.F["refresh"] = refreshes;

			var basePower = input.Board.Sum(e => Value(e, StatNorm(input), input.DominantRace, input.Tier));
			c.F["heur"] = Math.Round((c.Power - basePower) * HpFactor(input.Hp, w), 4);
			c.Score = w.Score(c.F);
		}

		private static void FillParts(Candidate c, AdvisorWeights w)
		{
			c.Parts.Clear();
			foreach(var kv in c.F)
			{
				var part = w.Get(kv.Key) * kv.Value;
				if(Math.Abs(part) > 1e-6)
					c.Parts[AdvisorWeights.FeatureNames[kv.Key]] = part;
			}
		}

		/// <summary>生成步骤文字和理由。</summary>
		private static void Describe(Candidate c, AdvisorInput input, AdvisorWeights w)
		{
			var gold = input.Gold;
			string N(Entity e) => e.LocalizedName ?? e.CardId ?? "?";

			if(c.Upgrade)
			{
				gold -= input.Shop.UpgradeCost;
				c.Steps.Add(new PlanStep { Type = "upgrade", Text = $"升到 {input.Tier + 1} 本（{input.Shop.UpgradeCost} 金）", Gold = gold });
			}
			foreach(var e in c.Sells)
			{
				gold += SellValue(e);
				c.Steps.Add(new PlanStep { Type = "sell", Text = $"卖掉 {N(e)}（{e.Attack}/{e.Health}）", EntityId = e.Id, Gold = gold });
			}
			foreach(var e in c.Buys)
			{
				gold -= input.Shop.BuyCost;
				var pos = input.Tavern.IndexOf(e) + 1;
				c.Steps.Add(new PlanStep { Type = "buy", Text = $"买酒馆第 {pos} 个：{N(e)}（{e.Attack}/{e.Health}）", EntityId = e.Id, Gold = gold });
			}
			foreach(var name in c.Triples)
				c.Steps.Add(new PlanStep { Type = "play", Text = $"三连：{name} 合成金色后打出", Gold = gold });
			var played = c.Final.Where(s => s.Src == "hand" || s.Src == "buy").Select(s => N(s.E)).ToList();
			if(played.Count > 0)
				c.Steps.Add(new PlanStep { Type = "play", Text = "打出：" + string.Join("、", played), Gold = gold });

			// 剩余金币：酒馆里还有想要但买不起的就冻结，否则刷新
			var unbought = input.Tavern.Where(e => !c.Buys.Contains(e)).ToList();
			var owned = c.Final.Where(s => !s.Golden).GroupBy(s => s.BaseId).ToDictionary(g => g.Key, g => g.Count());
			var keep = unbought.FirstOrDefault(e => owned.TryGetValue(BaseId(e.CardId), out var n) && n >= 2)
				?? unbought.FirstOrDefault(e => owned.ContainsKey(BaseId(e.CardId)));
			var refreshCost = Math.Max(1, input.Shop.RefreshCost);
			if(keep != null && c.GoldLeft < input.Shop.BuyCost)
				c.Steps.Add(new PlanStep { Type = "freeze", Text = $"冻结酒馆，留着 {N(keep)} 下回合买", EntityId = keep.Id, Gold = c.GoldLeft });
			else if(c.GoldLeft >= refreshCost || input.Shop.FreeRefreshes > 0)
			{
				var n = c.GoldLeft / refreshCost + input.Shop.FreeRefreshes;
				c.Steps.Add(new PlanStep { Type = "refresh", Text = $"剩 {c.GoldLeft} 金：可以刷新 {n} 次找更好的", Gold = c.GoldLeft });
			}
			if(c.Steps.Count == 0)
				c.Steps.Add(new PlanStep { Type = "pass", Text = "什么都不做，直接结束回合", Gold = gold });

			// 理由
			if(c.Sim != null && input.Target != null)
				c.Reasons.Add($"对 {input.Target.HeroName}（第 {input.Target.BoardSeenTurn} 回合阵容）胜 {c.Sim.WinRate:P0} / 负 {c.Sim.LossRate:P0}");
			var desired = w.DesiredTier(input.Turn);
			if(c.Upgrade)
				c.Reasons.Add(desired > input.Tier ? $"升本节奏：第 {input.Turn} 回合建议 {desired} 本" : "比节奏提前升本");
			else if(desired > input.Tier && input.Shop.CanUpgrade)
				c.Reasons.Add($"没升本，落后节奏（第 {input.Turn} 回合建议 {desired} 本）");
			if(c.Triples.Count > 0) c.Reasons.Add("凑成三连：" + string.Join("、", c.Triples));
			if(c.Pairs > 0) c.Reasons.Add($"凑成 {c.Pairs} 个对子");
			if(c.TribeHits > 0) c.Reasons.Add($"买了 {c.TribeHits} 个主种族（{RaceName(input.DominantRace)}）随从");
		}

		private static TurnSnapshot SnapshotOf(AdvisorInput input)
		{
			var snap = new TurnSnapshot { Tier = input.Tier };
			foreach(var e in input.Board.Concat(input.Hand))
			{
				snap.Owned[e.Id] = TurnSnapshot.BaseId(e.CardId);
				if(e.HasTag(GameTag.PREMIUM)) snap.GoldenBases.Add(TurnSnapshot.BaseId(e.CardId));
			}
			foreach(var e in input.Tavern)
				snap.Tavern[e.Id] = TurnSnapshot.BaseId(e.CardId);
			return snap;
		}

		// ---------- 估值工具 ----------

		/// <summary>场面随从的平均身材（攻+血），用来把不同回合的身材归一化。</summary>
		private static double StatNorm(AdvisorInput input)
		{
			var all = input.Board.Concat(input.Hand).Concat(input.Tavern).Where(e => e.IsMinion).ToList();
			if(all.Count == 0)
				return 4;
			return Math.Max(4, all.Average(e => (double)e.Attack + Math.Max(0, e.Health)));
		}

		private static double Value(Entity e, double norm, string? dominantRace, int tier)
		{
			var v = (e.Attack + Math.Max(0, e.Health)) / norm;
			v += 0.15 * e.GetTag(GameTag.TECH_LEVEL);
			if(e.HasTag(GameTag.PREMIUM)) v += 0.8;
			if(e.HasTag(GameTag.DIVINE_SHIELD)) v += 0.5;
			if(e.HasTag(GameTag.TAUNT)) v += 0.1;
			if(e.HasTag(GameTag.REBORN)) v += 0.3;
			if(e.HasTag(GameTag.VENOMOUS) || e.HasTag(GameTag.POISONOUS)) v += 0.5;
			if(e.HasTag(GameTag.WINDFURY) || e.HasTag(GameTag.MEGA_WINDFURY)) v += 0.2;
			if(e.HasTag(GameTag.DEATHRATTLE)) v += 0.1;
			if(dominantRace != null && Races(e).Contains(dominantRace)) v += 0.3;
			return v;
		}

		private static double HpFactor(int hp, AdvisorWeights w)
		{
			if(hp <= w.LowHp) return w.HpFactorLow;
			if(hp >= 30) return w.HpFactorFull;
			var t = (30.0 - hp) / Math.Max(1, 30 - w.LowHp);
			return w.HpFactorFull + t * (w.HpFactorLow - w.HpFactorFull);
		}

		private static int SellValue(Entity e)
		{
			var v = e.GetTag(GameTag.BACON_SELL_VALUE);
			return v > 0 ? v : 1;
		}

		/// <summary>金色随从的卡牌 ID 一般是普通版加 _G，统一成普通版。</summary>
		private static string BaseId(string? cardId)
		{
			if(string.IsNullOrEmpty(cardId)) return "";
			return cardId!.EndsWith("_G") ? cardId.Substring(0, cardId.Length - 2) : cardId;
		}

		private static List<string> Races(Entity e)
		{
			var list = new List<string>();
			var r1 = e.Card.RaceEnum;
			var r2 = e.Card.SecondaryRaceEnum;
			if(r1 != null && r1 != Race.INVALID && r1 != Race.BLANK) list.Add(r1.Value.ToString());
			if(r2 != null && r2 != Race.INVALID && r2 != Race.BLANK) list.Add(r2.Value.ToString());
			return list;
		}

		private static string? DominantRace(IEnumerable<Entity> minions)
		{
			var best = minions.SelectMany(Races).Where(r => r != "ALL")
				.GroupBy(r => r).OrderByDescending(g => g.Count()).FirstOrDefault();
			return best != null && best.Count() >= 2 ? best.Key : null;
		}

		private static string RaceName(string? race) => race switch
		{
			"BEAST" => "野兽", "DEMON" => "恶魔", "DRAGON" => "龙", "ELEMENTAL" => "元素", "MECHANICAL" => "机械",
			"MURLOC" => "鱼人", "PIRATE" => "海盗", "QUILBOAR" => "野猪人", "NAGA" => "纳迦", "UNDEAD" => "亡灵",
			_ => race ?? "",
		};

		/// <summary>
		/// 近似金色随从：取身材最好的一个复制，攻击 / 生命取两个最强副本之和（金色是两倍基础身材 + 保留各副本的增益），
		/// 并打上金色标记。招募阶段的三连奖励（发现）不计入战力，用三连权重体现。
		/// </summary>
		private static Entity MergeTriple(List<Slot> three)
		{
			var byStats = three.OrderByDescending(s => s.E.Attack + s.E.Health).ToList();
			var golden = byStats[0].E.Clone();
			golden.SetTag(GameTag.ATK, byStats[0].E.Attack + byStats[1].E.Attack);
			golden.SetTag(GameTag.HEALTH, byStats[0].E.Health + byStats[1].E.Health);
			golden.SetTag(GameTag.DAMAGE, 0);
			golden.SetTag(GameTag.PREMIUM, 1);
			return golden;
		}
	}
}
