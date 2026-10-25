using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using HearthDb.Enums;
using Hearthstone_Deck_Tracker.Hearthstone;
using Hearthstone_Deck_Tracker.Hearthstone.Entities;
using Hearthstone_Deck_Tracker.Utility.Logging;

namespace BGStateExporter
{
	/// <summary>
	/// 管理所有对战模拟：
	///  1. 读取 HDT 自己跑的本场战斗模拟结果（Bob's Buddy 面板上的数字）
	///  2. 招募阶段：用你当前的场面，对每个对手“上次见到的阵容”自动模拟
	///  3. 站位优化：按请求搜索更好的站位
	/// Tick() 在 HDT UI 线程调用；模拟本身在后台线程跑。
	/// </summary>
	internal sealed class SimManager
	{
		// 自动对位模拟
		private const int MatchupIterations = 3000;
		private const int MatchupMaxMs = 700;
		private const int DebounceMs = 800;            // 场面停止变化多久后才开始模拟
		// 站位优化
		private const int SearchIterations = 1500;
		private const int SearchMaxMs = 250;
		private const int FinalIterations = 6000;
		private const int FinalMaxMs = 1500;
		private const int MaxEvaluations = 80;
		private const double MinImprovement = 0.02;    // 低于这个差距视为模拟误差

		private readonly string? _unavailableReason = SimBridge.CheckAvailable();

		// 后台任务完成时整体替换引用，序列化时不会读到一半的数据
		private volatile LiveCombatSim? _live;
		private volatile MatchupState _matchups = new();
		private volatile OptimizerState _optimizer = new();

		private string _lastSignature = "";
		private DateTime _signatureChangedAt = DateTime.MinValue;
		private bool _matchupRunning;
		private bool _matchupDirty;
		private DateTime _lastLiveRead = DateTime.MinValue;
		private Guid? _gameId;

		private volatile bool _recruitPhase;                    // 进入战斗时让后台模拟尽快停下，不和 HDT 抢 CPU
		private volatile int _pendingOptimizeOpponent = -1;   // -1 无请求，0 默认下个对手
		private int _optimizeRequestId;
		private bool _optimizerRunning;

		public SimulationState Snapshot() => new()
		{
			Available = _unavailableReason == null,
			Message = _unavailableReason,
			Live = _live,
			Matchups = _matchups,
			Optimizer = _optimizer,
		};

		/// <summary>由 HTTP 线程调用。opponentPlayerId=0 表示下个对手。</summary>
		public string RequestOptimize(int opponentPlayerId)
		{
			if(_unavailableReason != null)
				return "{\"accepted\":false,\"message\":\"Bob's Buddy 不可用\"}";
			if(_optimizerRunning)
				return "{\"accepted\":false,\"message\":\"上一次优化还没结束\"}";
			_pendingOptimizeOpponent = Math.Max(0, opponentPlayerId);
			return "{\"accepted\":true}";
		}

		public void Tick(GameV2 game, BgState state)
		{
			if(_unavailableReason != null)
				return;

			if(!state.IsBattlegrounds || state.Player == null)
			{
				if(!state.InGame)
					Reset();
				return;
			}

			var gameId = game.CurrentGameStats?.GameId;
			if(gameId != _gameId)
			{
				Reset();
				_gameId = gameId;
			}

			ReadLive(game, state, gameId);

			_recruitPhase = state.Phase == "recruit";
			if(!_recruitPhase)
			{
				_pendingOptimizeOpponent = -1;
				return;
			}

			TickMatchups(game, state);
			TickOptimizer(game, state);
		}

		private void Reset()
		{
			_live = null;
			_matchups = new MatchupState();
			_optimizer = new OptimizerState();
			_lastSignature = "";
			_matchupDirty = false;
			_gameId = null;
		}

		// ---------- 1. HDT 的本场战斗模拟 ----------

		private void ReadLive(GameV2 game, BgState state, Guid? gameId)
		{
			if(gameId == null || (DateTime.Now - _lastLiveRead).TotalMilliseconds < 400)
				return;
			_lastLiveRead = DateTime.Now;
			// 战斗中看本回合；招募阶段看上一回合的战斗（预测 vs 实际）
			var turn = state.Phase == "combat" ? state.Turn : state.Turn - 1;
			if(turn <= 0)
				return;
			try
			{
				_live = SimBridge.ReadHdtCombatSim(gameId.Value, turn) ?? _live;
			}
			catch(Exception ex)
			{
				Log.Debug($"[BGStateExporter] 读取 Bob's Buddy 结果失败：{ex.Message}");
			}
		}

		// ---------- 2. 招募阶段自动对位模拟 ----------

		private void TickMatchups(GameV2 game, BgState state)
		{
			var opponents = CaptureOpponents(game);
			var sig = Signature(game, state, opponents);
			if(sig != _lastSignature)
			{
				_lastSignature = sig;
				_signatureChangedAt = DateTime.Now;
				if(_matchupRunning)
					_matchupDirty = true;
				else
					_matchups = new MatchupState { Status = "waiting", Turn = state.Turn, Results = _matchups.Results };
			}

			if(_matchupRunning || _matchups.Status != "waiting")
				return;
			if((DateTime.Now - _signatureChangedAt).TotalMilliseconds < DebounceMs)
				return;
			if(opponents.Count == 0)
			{
				_matchups = new MatchupState { Status = "done", Turn = state.Turn, Error = "还没有见过任何对手的阵容" };
				return;
			}

			var ctx = CaptureContext(game, state);
			var player = CapturePlayer(game);
			var next = state.NextOpponentPlayerId;
			_matchupRunning = true;
			_ = RunMatchupsAsync(ctx, player, opponents, next, state.Turn);
		}

		private async Task RunMatchupsAsync(SimContext ctx, SimSide player, List<CapturedOpponent> opponents, int next, int turn)
		{
			var results = new List<MatchupResult>();
			// 重新模拟期间先保留上一轮的数字，算出一个替换一个，面板不会闪空
			var previous = _matchups.Results ?? new List<MatchupResult>();
			List<MatchupResult> Merged() => results
				.Concat(previous.Where(old => results.All(done => done.OpponentPlayerId != old.OpponentPlayerId)))
				.ToList();
			try
			{
				_matchups = new MatchupState { Status = "running", Turn = turn, Results = Merged() };
				foreach(var opp in opponents.OrderByDescending(o => o.PlayerId == next).ThenBy(o => o.Place))
				{
					if(!_recruitPhase)
						break;
					var r = new MatchupResult
					{
						OpponentPlayerId = opp.PlayerId,
						HeroCardId = opp.HeroCardId,
						HeroName = opp.HeroName,
						IsNextOpponent = opp.PlayerId == next,
						BoardSeenTurn = opp.BoardSeenTurn,
						OpponentMinionCount = opp.Side.Minions.Count,
					};
					try
					{
						var input = SimBridge.BuildInput(ctx, player, opp.Side);
						Copy(await SimBridge.RunAsync(input, MatchupIterations, MatchupMaxMs), r);
					}
					catch(Exception ex)
					{
						r.Error = Unwrap(ex);
					}
					results.Add(r);
					_matchups = new MatchupState { Status = "running", Turn = turn, Results = Merged() };
				}
				// 进入战斗而提前停下时，没算到的对手保留上一轮结果（只保留仍在场的对手）
				var alive = new HashSet<int>(opponents.Select(o => o.PlayerId));
				_matchups = new MatchupState { Status = "done", Turn = turn, Results = Merged().Where(x => alive.Contains(x.OpponentPlayerId)).ToList() };
			}
			catch(Exception ex)
			{
				_matchups = new MatchupState { Status = "error", Turn = turn, Error = Unwrap(ex), Results = results };
			}
			finally
			{
				_matchupRunning = false;
				if(_matchupDirty)
				{
					// 模拟期间场面又变了，下一个 Tick 重新跑
					_matchupDirty = false;
					_matchups = new MatchupState { Status = "waiting", Turn = turn, Results = _matchups.Results };
				}
			}
		}

		// ---------- 3. 站位优化 ----------

		private void TickOptimizer(GameV2 game, BgState state)
		{
			var requested = _pendingOptimizeOpponent;
			if(requested < 0 || _optimizerRunning)
				return;
			_pendingOptimizeOpponent = -1;
			var id = ++_optimizeRequestId;

			var opponents = CaptureOpponents(game);
			var targetId = requested > 0 ? requested : state.NextOpponentPlayerId;
			var target = opponents.FirstOrDefault(o => o.PlayerId == targetId) ?? (requested == 0 ? opponents.FirstOrDefault() : null);
			var player = CapturePlayer(game);

			var opt = new OptimizerState { RequestId = id, Turn = state.Turn };
			if(target == null)
			{
				opt.Status = "error";
				opt.Message = "还没见过这个对手的阵容，没法模拟";
				_optimizer = opt;
				return;
			}
			if(player.Minions.Count < 2)
			{
				opt.Status = "error";
				opt.Message = "场上不到两个随从，不需要调整站位";
				_optimizer = opt;
				return;
			}

			opt.Status = "running";
			opt.OpponentPlayerId = target.PlayerId;
			opt.OpponentHeroName = target.HeroName;
			opt.BoardSeenTurn = target.BoardSeenTurn;
			opt.CurrentOrder = player.Minions.Select(m => m.Id).ToList();
			_optimizer = opt;
			_optimizerRunning = true;
			_ = RunOptimizerAsync(CaptureContext(game, state), player, target, opt);
		}

		private async Task RunOptimizerAsync(SimContext ctx, SimSide player, CapturedOpponent target, OptimizerState template)
		{
			OptimizerState Next(string status) => new()
			{
				Status = status, RequestId = template.RequestId, Turn = template.Turn,
				OpponentPlayerId = template.OpponentPlayerId, OpponentHeroName = template.OpponentHeroName,
				BoardSeenTurn = template.BoardSeenTurn, CurrentOrder = template.CurrentOrder,
			};

			try
			{
				var evaluated = 0;
				async Task<SimResult> Eval(List<Entity> order, int it, int ms)
				{
					evaluated++;
					return await SimBridge.RunAsync(SimBridge.BuildInput(ctx, player, target.Side, order), it, ms);
				}

				var current = player.Minions.ToList();
				var best = current;
				var bestScore = Score(await Eval(current, SearchIterations, SearchMaxMs));

				// 局部搜索：每次尝试把一个随从挪到另一个位置，有明显提升就接受
				var improved = true;
				while(improved && evaluated < MaxEvaluations && _recruitPhase)
				{
					improved = false;
					for(var i = 0; i < best.Count && !improved && evaluated < MaxEvaluations && _recruitPhase; i++)
					{
						for(var j = 0; j < best.Count && evaluated < MaxEvaluations && _recruitPhase; j++)
						{
							if(i == j)
								continue;
							var candidate = best.ToList();
							var moved = candidate[i];
							candidate.RemoveAt(i);
							candidate.Insert(j, moved);
							var score = Score(await Eval(candidate, SearchIterations, SearchMaxMs));
							var accept = score > bestScore + MinImprovement;
							if(accept)
							{
								best = candidate;
								bestScore = score;
								improved = true;
							}
							if(accept || evaluated % 5 == 0)
							{
								// 定期更新进度，面板上能看到“已试 N 种站位”
								var progress = Next("running");
								progress.Evaluated = evaluated;
								progress.BestOrder = best.Select(m => m.Id).ToList();
								progress.BestOrderNames = best.Select(Name).ToList();
								_optimizer = progress;
							}
							if(accept)
								break;
						}
					}
				}

				if(!_recruitPhase)
				{
					var stopped = Next("error");
					stopped.Message = "已进入战斗，站位优化中止";
					_optimizer = stopped;
					return;
				}

				// 用更多次数重新评估当前站位和最佳站位，给出可信的数字
				var final = Next("done");
				final.Current = await Eval(current, FinalIterations, FinalMaxMs);
				final.Best = ReferenceEquals(best, current) ? final.Current : await Eval(best, FinalIterations, FinalMaxMs);
				final.BestOrder = best.Select(m => m.Id).ToList();
				final.BestOrderNames = best.Select(Name).ToList();
				final.Evaluated = evaluated;
				var gain = Score(final.Best) - Score(final.Current);
				final.Message = ReferenceEquals(best, current) || gain < MinImprovement / 2
					? "当前站位已经是最好的（其他站位的差距都在模拟误差内）"
					: $"调整后（胜率 - 负率）提升 {gain * 100:0.#} 个百分点";
				if(final.Message.StartsWith("当前"))
				{
					final.Best = final.Current;
					final.BestOrder = final.CurrentOrder;
					final.BestOrderNames = current.Select(Name).ToList();
				}
				_optimizer = final;
			}
			catch(Exception ex)
			{
				var err = Next("error");
				err.Message = Unwrap(ex);
				_optimizer = err;
			}
			finally
			{
				_optimizerRunning = false;
			}
		}

		// 胜率减负率为主，平均伤害做很小的加权
		private static double Score(SimResult r) =>
			r.WinRate - r.LossRate + 0.002 * (r.AvgDamageDealt * r.WinRate - r.AvgDamageTaken * r.LossRate);

		// ---------- 采集（UI 线程） ----------

		private sealed class CapturedOpponent
		{
			public int PlayerId;
			public int Place;
			public string? HeroCardId;
			public string? HeroName;
			public int BoardSeenTurn;
			public SimSide Side = new();
		}

		private static SimContext CaptureContext(GameV2 game, BgState state)
		{
			var ctx = new SimContext { Turn = state.Turn };
			var ge = game.GameEntity;
			if(ge != null && ge.GetTag(GameTag.BACON_COMBAT_DAMAGE_CAP_ENABLED) > 0)
				ctx.DamageCap = ge.GetTag(GameTag.BACON_COMBAT_DAMAGE_CAP);
			ctx.Races = BattlegroundsUtils.GetAvailableRaces()?.ToList() ?? new List<Race>();
			var anomaly = BattlegroundsUtils.GetBattlegroundsAnomalyDbfId(ge);
			if(anomaly.HasValue)
				ctx.AnomalyCardId = Database.GetCardFromDbfId(anomaly.Value, false)?.Id;
			return ctx;
		}

		private static SimSide CapturePlayer(GameV2 game)
		{
			var hero = game.Player.Hero;
			var board = game.Player.Board.ToList();
			var side = new SimSide
			{
				Friendly = true,
				Health = (hero?.Health ?? 0) + (hero?.GetTag(GameTag.ARMOR) ?? 0),
				DamageTaken = hero?.GetTag(GameTag.DAMAGE) ?? 0,
				Tier = hero?.GetTag(GameTag.PLAYER_TECH_LEVEL) ?? 0,
				Minions = board.Where(e => e.IsMinion && !string.IsNullOrEmpty(e.CardId))
					.OrderBy(e => e.ZonePosition).Select(e => e.Clone()).ToList(),
				HeroPowers = board.Where(e => e.IsHeroPower).Take(2).Select(e => e.Clone()).ToList(),
			};
			var ids = new HashSet<int>(side.Minions.Select(m => m.Id));
			foreach(var e in game.Entities.Values.ToList())
			{
				var target = e.GetTag(GameTag.ATTACHED);
				if(target == 0 || !ids.Contains(target) || !(e.IsInPlay || e.IsInSetAside || e.IsInGraveyard))
					continue;
				if(!side.Attached.TryGetValue(target, out var list))
					side.Attached[target] = list = new List<Entity>();
				list.Add(e.Clone());
			}
			return side;
		}

		private static List<CapturedOpponent> CaptureOpponents(GameV2 game)
		{
			var result = new List<CapturedOpponent>();
			var heroes = game.Entities.Values.ToList()
				.Where(e => e.IsHero && e.GetTag(GameTag.PLAYER_LEADERBOARD_PLACE) > 0 && !e.IsControlledBy(game.Player.Id))
				.GroupBy(e => e.GetTag(GameTag.PLAYER_ID))
				.Where(g => g.Key > 0)
				.Select(g => g.OrderByDescending(e => e.Id).First());

			foreach(var hero in heroes)
			{
				if(hero.Health <= 0)
					continue;   // 已出局
				var snapshot = game.GetBattlegroundsBoardStateFor(hero.Id);
				if(snapshot == null)
					continue;   // 还没交过手
				result.Add(new CapturedOpponent
				{
					PlayerId = hero.GetTag(GameTag.PLAYER_ID),
					Place = hero.GetTag(GameTag.PLAYER_LEADERBOARD_PLACE),
					HeroCardId = hero.CardId,
					HeroName = hero.LocalizedName,
					BoardSeenTurn = snapshot.Turn,
					Side = new SimSide
					{
						Friendly = false,
						Health = hero.Health + hero.GetTag(GameTag.ARMOR),
						DamageTaken = hero.GetTag(GameTag.DAMAGE),
						Tier = hero.GetTag(GameTag.PLAYER_TECH_LEVEL),
						Minions = snapshot.Entities.Where(e => e.IsMinion).OrderBy(e => e.ZonePosition).ToList(),
					},
				});
			}
			return result;
		}

		/// <summary>场面指纹：变了才重新模拟。</summary>
		private static string Signature(GameV2 game, BgState state, List<CapturedOpponent> opponents)
		{
			var sb = new StringBuilder();
			sb.Append(state.Turn).Append('|').Append(state.NextOpponentPlayerId).Append('|').Append(state.Player?.TavernTier).Append('|');
			foreach(var m in state.Player!.Board)
				sb.Append(m.EntityId).Append(':').Append(m.CardId).Append(':').Append(m.Attack).Append('/').Append(m.Health)
					.Append(m.Taunt ? "T" : "").Append(m.DivineShield ? "D" : "").Append(m.Reborn ? "R" : "")
					.Append(m.Venomous || m.Poisonous ? "P" : "").Append(m.Windfury ? "W" : "").Append(',');
			sb.Append('|');
			foreach(var hp in state.Player.HeroPowers)
				sb.Append(hp.CardId).Append(hp.Used ? "u" : "").Append(',');
			sb.Append('|');
			foreach(var o in opponents.OrderBy(o => o.PlayerId))
				sb.Append(o.PlayerId).Append(':').Append(o.BoardSeenTurn).Append(':').Append(o.Side.Health).Append(',');
			return sb.ToString();
		}

		private static string Name(Entity e) => e.LocalizedName ?? e.CardId ?? "?";

		private static void Copy(SimResult from, SimResult to)
		{
			to.WinRate = from.WinRate;
			to.TieRate = from.TieRate;
			to.LossRate = from.LossRate;
			to.WinLethal = from.WinLethal;
			to.LossLethal = from.LossLethal;
			to.AvgDamageDealt = from.AvgDamageDealt;
			to.AvgDamageTaken = from.AvgDamageTaken;
			to.SimulationCount = from.SimulationCount;
		}

		private static string Unwrap(Exception ex)
		{
			while(ex is AggregateException { InnerException: not null } || ex is System.Reflection.TargetInvocationException { InnerException: not null })
				ex = ex.InnerException!;
			// Bob's Buddy 不支持的卡牌会抛 UnsupportedInteractionException
			return ex.GetType().Name.Contains("Unsupported") ? "包含模拟器还不支持的卡牌或效果：" + ex.Message : ex.Message;
		}
	}
}
