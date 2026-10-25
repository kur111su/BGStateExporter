using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;
using BobsBuddy;
using BobsBuddy.Simulation;
using HearthDb.Enums;
using Hearthstone_Deck_Tracker;
using Hearthstone_Deck_Tracker.Hearthstone.Entities;
// BobsBuddy.Simulation 和 HearthDb.Enums 里也有同名类型，明确指定用哪个
using Entity = Hearthstone_Deck_Tracker.Hearthstone.Entities.Entity;
using Type = System.Type;

namespace BGStateExporter
{
	/// <summary>一方的模拟输入，全部是实体的克隆，可以离开 UI 线程使用。</summary>
	internal sealed class SimSide
	{
		public bool Friendly;
		public List<Entity> Minions = new();                                  // 从左到右
		public Dictionary<int, List<Entity>> Attached = new();                // 随从 id -> 附着的附魔
		public List<Entity> HeroPowers = new();
		public int Health;                                                    // 已包含护甲
		public int DamageTaken;
		public int Tier;
	}

	internal sealed class SimContext
	{
		public int Turn;
		public int DamageCap;
		public List<Race> Races = new();
		public string? AnomalyCardId;
	}

	/// <summary>
	/// 封装对 Bob's Buddy（HDT 自带的 BobsBuddy.dll）和 HDT 内部工具方法的调用。
	/// HDT 的 BobsBuddyInvoker / BobsBuddyUtils 是 internal，只能用反射访问。
	/// </summary>
	internal static class SimBridge
	{
		private const BindingFlags AnyStatic = BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic;

		private static readonly Assembly HdtAssembly = typeof(Core).Assembly;
		private static readonly Type? InvokerType = HdtAssembly.GetType("Hearthstone_Deck_Tracker.BobsBuddy.BobsBuddyInvoker");
		private static readonly Type? UtilsType = HdtAssembly.GetType("Hearthstone_Deck_Tracker.BobsBuddy.BobsBuddyUtils");
		private static readonly MethodInfo? GetInstanceMethod = InvokerType?.GetMethod("GetInstance", AnyStatic);
		private static readonly MethodInfo? GetMinionMethod = FindGetMinion();

		public static int ThreadCount => Math.Max(1, Environment.ProcessorCount / 2);

		/// <summary>检查 BobsBuddy.dll 能否加载。</summary>
		public static string? CheckAvailable()
		{
			try
			{
				_ = typeof(SimulationRunner).Assembly.GetName().Version;
				return null;
			}
			catch(Exception ex)
			{
				return "找不到 BobsBuddy.dll：" + ex.Message;
			}
		}

		private static MethodInfo? FindGetMinion()
		{
			var m = UtilsType?.GetMethod("GetMinionFromEntity", AnyStatic);
			// 预期签名：(Simulator, bool, Entity, IEnumerable<Entity>, IReadOnlyDictionary<int, Entity>?)
			if(m == null || m.GetParameters().Length < 4)
				return null;
			return m;
		}

		// ---------- 读取 HDT 自己跑的本场战斗模拟 ----------

		public static LiveCombatSim? ReadHdtCombatSim(Guid gameId, int turn)
		{
			if(GetInstanceMethod == null || InvokerType == null)
				return null;
			// 第三个参数 false：没有就不创建，避免干扰 HDT
			var inst = GetInstanceMethod.Invoke(null, new object[] { gameId, turn, false });
			if(inst == null)
				return null;

			var live = new LiveCombatSim
			{
				Turn = turn,
				State = InvokerType.GetProperty("State")?.GetValue(inst)?.ToString(),
				ErrorState = InvokerType.GetProperty("ErrorState")?.GetValue(inst)?.ToString(),
			};
			if(InvokerType.GetProperty("Output")?.GetValue(inst) is Output output)
			{
				Fill(live, output);
				live.HasResult = true;
			}
			return live;
		}

		// ---------- 自己构造输入并模拟 ----------

		public static Input BuildInput(SimContext ctx, SimSide player, SimSide opponent, IList<Entity>? playerOrder = null)
		{
			var simulator = new Simulator();
			var input = new Input();
			input.availableRaces = ctx.Races.ToList();
			if(ctx.DamageCap > 0)
				input.DamageCap = ctx.DamageCap;

			FillPlayer(simulator, input.Player, player, playerOrder ?? player.Minions);
			FillPlayer(simulator, input.Opponent, opponent, opponent.Minions);

			if(ctx.AnomalyCardId != null)
			{
				try { input.Anomaly = simulator.AnomalyFactory.Create(ctx.AnomalyCardId); }
				catch { /* 不认识的畸变就忽略 */ }
			}
			input.SetTurn(ctx.Turn);
			return input;
		}

		private static void FillPlayer(Simulator sim, BobsBuddy.Simulation.Player target, SimSide side, IList<Entity> order)
		{
			target.Health = side.Friendly ? side.Health : (side.Health <= 0 ? 1000 : side.Health);
			target.DamageTaken = side.DamageTaken;
			target.Tier = side.Tier;

			foreach(var hp in side.HeroPowers)
			{
				if(string.IsNullOrEmpty(hp.CardId))
					continue;
				var activated = hp.HasTag(GameTag.EXHAUSTED) || hp.HasTag(GameTag.BACON_HERO_POWER_ACTIVATED);
				target.AddHeroPower(hp.CardId!, side.Friendly, activated,
					hp.GetTag(GameTag.TAG_SCRIPT_DATA_NUM_1), hp.GetTag(GameTag.TAG_SCRIPT_DATA_NUM_2),
					hp.GetTag(GameTag.TAG_SCRIPT_DATA_NUM_3), null, hp.Id);
			}

			var sideList = target.Side;
			var add = sideList.GetType().GetMethods().First(m => m.Name == "Add" && m.GetParameters().Length == 1);
			foreach(var e in order)
			{
				side.Attached.TryGetValue(e.Id, out var attached);
				var minion = CreateMinion(sim, side.Friendly, e, attached ?? new List<Entity>());
				add.Invoke(sideList, new[] { minion });
			}
		}

		private static object CreateMinion(Simulator sim, bool friendly, Entity e, List<Entity> attached)
		{
			if(GetMinionMethod != null)
			{
				// 首选 HDT 自己的转换逻辑：附魔、模块化机械等都会处理
				var args = GetMinionMethod.GetParameters().Length >= 5
					? new object?[] { sim, friendly, e, attached, null }
					: new object?[] { sim, friendly, e, attached };
				var result = GetMinionMethod.Invoke(null, args);
				if(result != null)
					return result;
			}

			// 兜底：只设置基础属性
			var cardId = e.Info.LatestCardId ?? e.CardId ?? "Unknown";
			var m = sim.MinionFactory.CreateFromCardId(cardId, friendly);
			m.PrimaryRace = (Race)e.GetTag(GameTag.CARDRACE);
			m.baseAttack = e.GetTag(GameTag.ATK);
			m.baseHealth = e.GetTag(GameTag.HEALTH) - e.GetTag(GameTag.DAMAGE);
			m.maxAttack = e.GetTag(GameTag.ATK);
			m.maxHealth = e.GetTag(GameTag.HEALTH);
			m.taunt = e.HasTag(GameTag.TAUNT);
			m.div = e.GetTag(GameTag.DIVINE_SHIELD);
			m.poisonous = e.HasTag(GameTag.POISONOUS);
			m.venomous = e.HasTag(GameTag.VENOMOUS);
			m.windfury = e.HasTag(GameTag.WINDFURY);
			m.megaWindfury = e.HasTag(GameTag.MEGA_WINDFURY);
			m.stealth = e.HasTag(GameTag.STEALTH);
			m.golden = e.HasTag(GameTag.PREMIUM);
			m.tier = e.GetTag(GameTag.TECH_LEVEL);
			m.reborn = e.HasTag(GameTag.REBORN);
			m.ScriptDataNum1 = e.GetTag(GameTag.TAG_SCRIPT_DATA_NUM_1);
			m.ScriptDataNum2 = e.GetTag(GameTag.TAG_SCRIPT_DATA_NUM_2);
			return m;
		}

		public static async Task<SimResult> RunAsync(Input input, int iterations, int maxTimeMs)
		{
			var output = await Task.Run(() => new SimulationRunner().SimulateMultiThreaded(input, iterations, ThreadCount, maxTimeMs));
			var r = new SimResult();
			Fill(r, output);
			return r;
		}

		private static void Fill(SimResult r, Output o)
		{
			r.WinRate = Round(o.winRate);
			r.TieRate = Round(o.tieRate);
			r.LossRate = Round(o.lossRate);
			r.WinLethal = Round(o.theirDeathRate);
			r.LossLethal = Round(o.myDeathRate);
			r.SimulationCount = (int)o.simulationCount;

			// damageResults：正数 = 我方造成的伤害，负数 = 我方承受的伤害（与 HDT 面板一致）
			var dmg = o.damageResults?.ToList() ?? new List<int>();
			var dealt = dmg.Where(x => x > 0).ToList();
			var taken = dmg.Where(x => x < 0).ToList();
			r.AvgDamageDealt = dealt.Count > 0 ? Math.Round(dealt.Average(), 1) : 0;
			r.AvgDamageTaken = taken.Count > 0 ? Math.Round(-taken.Average(), 1) : 0;
		}

		private static double Round(double v) => Math.Round(v, 4);
	}
}
