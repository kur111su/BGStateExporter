using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using BobsBuddy;
using BobsBuddy.Factory;
using BobsBuddy.Simulation;
using HearthDb.Enums;
using Newtonsoft.Json;
using Player = BobsBuddy.Simulation.Player;

namespace BGSimHost
{
	// ---------------- 输入 / 输出（JSON 字段名就是这里的字段名） ----------------

	public sealed class CardIn
	{
		public string id = "";          // cardId（金色随从用金色的 cardId）
		public int atk, hp, tier = 1;
		public bool golden, taunt, ds, poison, venom, wf, mwf, stealth, reborn;
		public int[]? sd;               // TAG_SCRIPT_DATA_NUM_1..4（成长计数等），没有就省略
	}

	public sealed class HeroPowerIn
	{
		public string id = "";
		public bool used;
		public int[]? data;             // TAG_SCRIPT_DATA_NUM_1..3
	}

	public sealed class SideIn
	{
		public int health = 30;         // 血量 + 护甲
		public int damageTaken;
		public int tier = 1;
		public List<HeroPowerIn>? heroPowers;
		public List<string>? trinkets;  // 饰品 cardId
		public List<CardIn> board = new();   // 从左到右
	}

	public sealed class CombatIn
	{
		public int turn;
		public List<string>? races;     // HearthDb Race 枚举名，如 "BEAST"
		public string? anomaly;         // 畸变 cardId
		public int anomalyDbf;          // 或者畸变 dbfId
		public int damageCap;
		public SideIn me = new();
		public SideIn opp = new();
	}

	public sealed class ResultOut
	{
		public double w, t, l;          // 胜 / 平 / 负
		public double wl, ll;           // 打死对手 / 被打死
		public double dealt, taken;     // 赢时平均造成 / 输时平均承受
		public int n;                   // 实际模拟次数
		public List<int>? dmg;          // 模拟次数 ≤ 32 时给出每次的伤害（正 = 我方造成，负 = 我方承受）
		public string? err;
	}

	public static class Host
	{
		public const int ApiVersion = 1;

		private static string? _hdtDir;
		private static readonly object InitLock = new();

		/// <summary>
		/// 必须先调用：注册程序集解析（从 HDT 目录找依赖）。
		/// 在 .NET 里注册，这样模拟线程需要加载程序集时不用回调 Python。
		/// </summary>
		public static string Init(string hdtDir)
		{
			lock(InitLock)
			{
				if(_hdtDir == null)
				{
					_hdtDir = hdtDir;
					AppDomain.CurrentDomain.AssemblyResolve += Resolve;
				}
			}
			ThreadPool.GetMinThreads(out var worker, out var io);
			ThreadPool.SetMinThreads(Math.Max(worker, Environment.ProcessorCount * 2), io);
			return $"BGSimHost {typeof(Host).Assembly.GetName().Version}, BobsBuddy {typeof(Simulator).Assembly.GetName().Version}";
		}

		private static Assembly? Resolve(object? sender, ResolveEventArgs e)
		{
			var name = new AssemblyName(e.Name).Name;
			foreach(var dir in new[] { _hdtDir, Path.GetDirectoryName(typeof(Host).Assembly.Location) })
			{
				if(string.IsNullOrEmpty(dir))
					continue;
				foreach(var ext in new[] { ".dll", ".exe" })
				{
					var p = Path.Combine(dir!, name + ext);
					if(File.Exists(p))
						return Assembly.LoadFrom(p);
				}
			}
			return null;
		}

		/// <summary>
		/// 批量模拟。combatsJson 是 CombatIn 数组，返回同样长度的 ResultOut 数组（JSON）。
		/// mode = "runner"：每场用 SimulationRunner（HDT 的用法，最稳妥），所有场次同时提交给线程池；
		/// mode = "direct"：每场直接 new Simulator().SimulateForInput，按 threads 并行。
		/// </summary>
		public static string Simulate(string combatsJson, int iterations, int threads, int maxMsEach, string mode)
		{
			var combats = JsonConvert.DeserializeObject<List<CombatIn>>(combatsJson) ?? new List<CombatIn>();
			var results = mode == "direct"
				? SimulateDirect(combats, iterations, threads, maxMsEach)
				: SimulateRunner(combats, iterations, maxMsEach);
			return JsonConvert.SerializeObject(results);
		}

		/// <summary>只构造输入不模拟，用来测构造速度。返回每秒构造场数。</summary>
		public static double BenchBuild(string combatsJson, int repeat)
		{
			var combats = JsonConvert.DeserializeObject<List<CombatIn>>(combatsJson) ?? new List<CombatIn>();
			var sw = Stopwatch.StartNew();
			var n = 0;
			for(var r = 0; r < repeat; r++)
				foreach(var c in combats)
				{
					Build(new Simulator(), c);
					n++;
				}
			return n / Math.Max(1e-9, sw.Elapsed.TotalSeconds);
		}

		private static ResultOut[] SimulateRunner(List<CombatIn> combats, int iterations, int maxMs)
		{
			var results = new ResultOut[combats.Count];
			var tasks = new Task<Output>?[combats.Count];
			for(var i = 0; i < combats.Count; i++)
			{
				try
				{
					var input = Build(new Simulator(), combats[i]);
					tasks[i] = new SimulationRunner().SimulateMultiThreaded(input, iterations, 1, maxMs);
				}
				catch(Exception ex)
				{
					results[i] = Error(ex);
				}
			}
			for(var i = 0; i < combats.Count; i++)
			{
				var t = tasks[i];
				if(t == null)
					continue;
				try
				{
					results[i] = Summarize(t.GetAwaiter().GetResult(), iterations);
				}
				catch(Exception ex)
				{
					results[i] = Error(ex);
				}
			}
			return results;
		}

		private static ResultOut[] SimulateDirect(List<CombatIn> combats, int iterations, int threads, int maxMs)
		{
			var results = new ResultOut[combats.Count];
			var options = new ParallelOptions { MaxDegreeOfParallelism = Math.Max(1, threads) };
			Parallel.For(0, combats.Count, options, i =>
			{
				try
				{
					var input = Build(new Simulator(), combats[i]);
					results[i] = Summarize(new Simulator().SimulateForInput(input, iterations, maxMs), iterations);
				}
				catch(Exception ex)
				{
					results[i] = Error(ex);
				}
			});
			return results;
		}

		private static ResultOut Error(Exception ex)
		{
			if(ex is AggregateException ae && ae.InnerException != null)
				ex = ae.InnerException;
			return new ResultOut { err = ex.GetType().Name + ": " + ex.Message };
		}

		private static ResultOut Summarize(Output o, int iterations)
		{
			var dmg = o.damageResults ?? new List<int>();
			var dealt = dmg.Where(x => x > 0).ToList();
			var taken = dmg.Where(x => x < 0).ToList();
			return new ResultOut
			{
				w = Math.Round(o.winRate, 4),
				t = Math.Round(o.tieRate, 4),
				l = Math.Round(o.lossRate, 4),
				wl = Math.Round(o.theirDeathRate, 4),
				ll = Math.Round(o.myDeathRate, 4),
				n = o.simulationCount > 0 ? o.simulationCount : dmg.Count,   // direct 方式不填 simulationCount
				dealt = dealt.Count > 0 ? Math.Round(dealt.Average(), 2) : 0,
				taken = taken.Count > 0 ? Math.Round(-taken.Average(), 2) : 0,
				dmg = iterations <= 32 ? dmg.ToList() : null,
			};
		}

		// ---------------- 构造 Bob's Buddy 输入（参照 HDT 的 BobsBuddyInvoker / BobsBuddyUtils） ----------------

		private static Input Build(Simulator sim, CombatIn c)
		{
			var input = new Input();
			var races = new List<Race>();
			foreach(var r in c.races ?? new List<string>())
				if(Enum.TryParse<Race>(r, true, out var race))
					races.Add(race);
			input.availableRaces = races;
			if(c.damageCap > 0)
				input.DamageCap = c.damageCap;

			FillSide(sim, input.Player, c.me, true);
			FillSide(sim, input.Opponent, c.opp, false);

			var anomaly = c.anomaly;
			if(string.IsNullOrEmpty(anomaly) && c.anomalyDbf > 0)
				anomaly = HearthDb.Cards.GetFromDbfId(c.anomalyDbf, false)?.Id;
			if(!string.IsNullOrEmpty(anomaly))
			{
				try { input.Anomaly = sim.AnomalyFactory.Create(anomaly!); }
				catch { /* 不认识的畸变就忽略 */ }
			}
			input.SetTurn(c.turn);
			return input;
		}

		private static void FillSide(Simulator sim, Player p, SideIn s, bool friendly)
		{
			p.Health = friendly || s.health > 0 ? s.health : 1000;
			p.DamageTaken = s.damageTaken;
			p.Tier = s.tier;
			foreach(var hp in s.heroPowers ?? new List<HeroPowerIn>())
			{
				if(string.IsNullOrEmpty(hp.id))
					continue;
				var d = hp.data ?? Array.Empty<int>();
				p.AddHeroPower(hp.id, friendly, hp.used, At(d, 0), At(d, 1), At(d, 2), null, 0);
			}
			foreach(var t in s.trinkets ?? new List<string>())
			{
				try { p.Trinkets.Add(sim.TrinketFactory.Create(t, friendly)); }
				catch { /* 不认识的饰品就忽略 */ }
			}
			foreach(var card in s.board)
				p.Side.Add(MakeMinion(sim, friendly, card));
		}

		private static int At(int[] a, int i) => i < a.Length ? a[i] : 0;

		private static Minion MakeMinion(Simulator sim, bool friendly, CardIn c)
		{
			var m = sim.MinionFactory.CreateFromCardId(c.id, friendly);
			m.baseAttack = c.atk;
			m.maxAttack = c.atk;
			m.baseHealth = c.hp;
			m.maxHealth = c.hp;
			m.taunt = c.taunt;
			m.div = c.ds ? 1 : 0;
			m.cleave = MinionFactory.cardIDsWithCleave.Contains(m.CardID);
			m.poisonous = c.poison;
			m.venomous = c.venom;
			m.windfury = c.wf;
			m.megaWindfury = c.mwf || MinionFactory.cardIdsWithMegaWindfury.Contains(c.id);
			m.stealth = c.stealth;
			m.golden = c.golden;
			m.tier = c.tier;
			m.reborn = c.reborn;
			if(c.sd != null)
			{
				m.ScriptDataNum1 = At(c.sd, 0);
				m.ScriptDataNum2 = At(c.sd, 1);
				m.ScriptDataNum3 = At(c.sd, 2);
				m.ScriptDataNum4 = At(c.sd, 3);
			}
			return m;
		}
	}
}
