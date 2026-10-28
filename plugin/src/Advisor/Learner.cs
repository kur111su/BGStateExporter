using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Hearthstone_Deck_Tracker.Utility.Logging;
using Newtonsoft.Json;

namespace BGStateExporter
{
	/// <summary>建议器一次计算的完整记录：当时的局面摘要 + 所有有效候选方案的特征。</summary>
	internal sealed class AdvisorRun
	{
		public int Seq;
		public int Turn;
		public TurnSnapshot Snapshot = new();
		public List<Cand> Candidates = new();
		public string TopSummary = "";          // 推荐方案一句话（回合记录用）
		public bool TopUpgrade;
		public List<int> TopBuys = new();

		public sealed class Cand
		{
			public bool Upgrade;
			public List<int> Buys = new();
			public int Spell;                     // 买的法术 entityId，0 = 不买
			public List<int> Sells = new();
			public Dictionary<string, double> F = new();
		}
	}

	/// <summary>判断“玩家实际做了什么”所需的局面摘要。</summary>
	internal sealed class TurnSnapshot
	{
		public int Tier;
		public Dictionary<int, string> Owned = new();     // 场上 + 手牌随从：entityId → 普通版卡牌 ID
		public HashSet<string> GoldenBases = new();       // 拥有的金色随从（普通版卡牌 ID）
		public Dictionary<int, string> Tavern = new();    // 酒馆随从：entityId → 普通版卡牌 ID
		public HashSet<int> TavernSpells = new();         // 酒馆法术 entityId
		public Dictionary<int, string> Names = new();     // entityId → 名字（回合记录用）

		public static string BaseId(string? cardId)
		{
			if(string.IsNullOrEmpty(cardId)) return "";
			return cardId!.EndsWith("_G") ? cardId.Substring(0, cardId.Length - 2) : cardId;
		}

		public static TurnSnapshot From(BgState s)
		{
			var snap = new TurnSnapshot { Tier = s.Player?.TavernTier ?? 0 };
			if(s.Player != null)
				foreach(var m in s.Player.Board.Concat(s.Player.Hand).Where(m => m.Type == "minion"))
				{
					snap.Owned[m.EntityId] = BaseId(m.CardId);
					snap.Names[m.EntityId] = m.Name ?? m.CardId ?? "?";
					if(m.Golden) snap.GoldenBases.Add(BaseId(m.CardId));
				}
			foreach(var m in s.Tavern)
			{
				snap.Names[m.EntityId] = m.Name ?? m.CardId ?? "?";
				if(m.Type == "minion") snap.Tavern[m.EntityId] = BaseId(m.CardId);
				else if(m.Type == "spell") snap.TavernSpells.Add(m.EntityId);
			}
			return snap;
		}
	}

	/// <summary>
	/// 自动学习建议器权重。
	///  1. 记录：建议器每算一次（AdvisorRun），等到下一次计算或进入战斗时，比较前后局面，
	///     看玩家实际升没升本、买了谁、买没买法术，对上候选方案里的哪一个。
	///     这段时间里玩家什么都没做（只刷新、只看）不记录——那不是在候选方案之间做选择。
	///  2. 每局结束：按最终名次给这局的记录加权（第 1 名 1.0，第 4 名 0.14，第 5～8 名 0），追加到 training/samples_v2.jsonl
	///     （v0.8 起特征变了，v0.7 及以前的 samples.jsonl 保留但不再使用）。
	///  另外每个回合记一条“回合开始时推荐了什么 / 实际做了什么”，给网页面板显示。
	///  3. 拟合：攒够局数后，每局结束重新拟合一次。模型是条件 logit：P(选方案 j) ∝ exp(β·Σ 权重×特征)，
	///     “战力”权重固定作为尺度，β 单独拟合（代表选择有多稳定），其余权重向默认值收缩（局数越多收缩越弱）。
	///     按对局划出 1/4 做验证集，新权重在验证集上解释得更好才写入，旧权重自动备份。
	/// </summary>
	internal sealed class Learner
	{
		private readonly string _dir;
		private readonly string _samplesPath;
		private readonly string _logPath;

		private string? _gameKey;
		private readonly List<string> _pending = new();   // 本局的样本（JSON，名次待定）
		private AdvisorRun? _openRun;
		private TurnSnapshot? _lastRecruitSnap;
		private int _lastRecruitTurn;
		private int _lastPlace;
		private bool _fitting;

		private volatile LearningStatus _status = new();
		public LearningStatus Status => _status;

		// 回合记录
		private volatile List<TurnRecord> _turnLog = new();
		public List<TurnRecord> TurnLog => _turnLog;
		private int _curTurn;
		private TurnSnapshot? _turnStart;
		private TurnSnapshot? _prevSnap;
		private readonly Dictionary<int, string> _turnNames = new();
		private readonly List<string> _turnSpells = new();
		private readonly HashSet<int> _turnTavernSeen = new();   // 这回合在酒馆里出现过的随从
		private AdvisorRun? _turnAdvice;

		public Learner(string outDir)
		{
			_dir = Path.Combine(outDir, "training");
			Directory.CreateDirectory(_dir);
			_samplesPath = Path.Combine(_dir, "samples_v2.jsonl");
			_logPath = Path.Combine(_dir, "fit_log.txt");
			RefreshStatus(null);
		}

		// ---------- 记录（UI 线程，每帧调用） ----------

		public void OnState(BgState s, AdvisorRun? run)
		{
			try
			{
				if(s.InGame && s.IsBattlegrounds)
				{
					if(_gameKey == null)
					{
						_gameKey = DateTime.Now.ToString("yyyyMMdd_HHmmss");
						_pending.Clear();
						_openRun = null;
						_lastRecruitSnap = null;
						_lastPlace = 0;
						_turnLog = new List<TurnRecord>();
						_curTurn = 0;
						RefreshStatus(null);
					}
					if(s.Player != null && s.Player.LeaderboardPlace > 0)
						_lastPlace = s.Player.LeaderboardPlace;

					if(s.Phase == "recruit" && s.Player != null)
					{
						if(run != null && run.Turn == s.Turn && (_openRun == null || run.Seq != _openRun.Seq))
						{
							// 新的一次计算：上一次计算之后玩家做的事 = 两次计算的输入局面之差
							if(_openRun != null && _openRun.Turn == s.Turn)
								Label(_openRun, run.Snapshot);
							_openRun = run;
						}
						_lastRecruitSnap = TurnSnapshot.From(s);
						_lastRecruitTurn = s.Turn;
						TrackTurn(s, _lastRecruitSnap, run);
					}
					else if(s.Phase == "combat" || s.Phase == "game_over")
					{
						CloseTurn();
						// 招募结束：最后一次计算之后玩家做的事 = 它的输入局面 vs 招募阶段最后的局面
						if(_openRun != null && _lastRecruitSnap != null && _openRun.Turn == _lastRecruitTurn)
							Label(_openRun, _lastRecruitSnap);
						_openRun = null;
					}
					if(s.Phase == "game_over")
						Finish();
				}
				else if(_gameKey != null)
					Finish();
			}
			catch(Exception ex)
			{
				Log.Debug($"[BGStateExporter] 学习记录出错：{ex.Message}");
			}
		}

		private void Label(AdvisorRun run, TurnSnapshot after)
		{
			if(run.Candidates.Count < 2)
				return;
			var before = run.Snapshot;
			var upgraded = before.Tier >= 1 && after.Tier > before.Tier;   // 开局第 1 回合会从 0 变成 1，不算升本
			var newGolden = new HashSet<string>(after.GoldenBases.Except(before.GoldenBases));

			// 买了谁：酒馆里的随从现在归玩家了；或者它消失了但同名随从刚合成金色（三连）
			var bought = new HashSet<int>(before.Tavern
				.Where(kv => after.Owned.ContainsKey(kv.Key) || newGolden.Contains(kv.Value))
				.Select(kv => kv.Key));
			// 卖了谁：原来拥有、现在没了，且不是因为合成金色
			var sold = new HashSet<int>(before.Owned
				.Where(kv => !after.Owned.ContainsKey(kv.Key) && !newGolden.Contains(kv.Value))
				.Select(kv => kv.Key));
			// 买了哪个法术：酒馆里的法术不见了，而酒馆没有整体刷新（还有原来的卡）
			var tavernKept = before.Tavern.Keys.Any(after.Tavern.ContainsKey) || before.TavernSpells.Any(after.TavernSpells.Contains);
			var spellsBought = tavernKept ? before.TavernSpells.Where(id => !after.TavernSpells.Contains(id)).ToList() : new List<int>();

			// 这段时间什么都没做（只刷新 / 只看）：不是在候选方案之间做选择，不记录
			if(!upgraded && bought.Count == 0 && sold.Count == 0 && spellsBought.Count == 0)
				return;

			var best = -1;
			var bestOverlap = -1;
			for(var i = 0; i < run.Candidates.Count; i++)
			{
				var c = run.Candidates[i];
				if(c.Upgrade != upgraded || !new HashSet<int>(c.Buys).SetEquals(bought))
					continue;
				if((c.Spell != 0) != (spellsBought.Count > 0) || (c.Spell != 0 && !spellsBought.Contains(c.Spell)))
					continue;
				var overlap = c.Sells.Count(sold.Contains);
				if(overlap > bestOverlap)
				{
					best = i;
					bestOverlap = overlap;
				}
			}
			if(best < 0)
				return;   // 玩家的操作不在候选方案里（比如买了估值排名靠后的随从），这次不学

			var sample = new
			{
				v = 2,
				turn = run.Turn,
				chosen = best,
				cands = run.Candidates.Select(c => c.F).ToList(),
			};
			_pending.Add(JsonConvert.SerializeObject(sample));
			RefreshStatus(null);
		}

		private void TrackTurn(BgState s, TurnSnapshot snap, AdvisorRun? run)
		{
			if(s.Turn != _curTurn)
			{
				CloseTurn();
				_curTurn = s.Turn;
				_turnStart = snap;
				_prevSnap = null;
				_turnNames.Clear();
				_turnSpells.Clear();
				_turnTavernSeen.Clear();
				_turnAdvice = null;
			}
			foreach(var kv in snap.Names)
				_turnNames[kv.Key] = kv.Value;
			foreach(var id in snap.Tavern.Keys)
				_turnTavernSeen.Add(id);
			if(_turnStart != null && _turnStart.Tavern.Count == 0 && snap.Tavern.Count > 0)
				_turnStart = snap;   // 回合开始那一刻酒馆还是空的，用酒馆刷出来后的局面做起点
			if(_prevSnap != null)
			{
				var kept = _prevSnap.Tavern.Keys.Any(snap.Tavern.ContainsKey) || _prevSnap.TavernSpells.Any(snap.TavernSpells.Contains);
				if(kept)
					foreach(var id in _prevSnap.TavernSpells.Where(id => !snap.TavernSpells.Contains(id)))
						_turnSpells.Add(_turnNames.TryGetValue(id, out var n) ? n : "?");
			}
			_prevSnap = snap;
			// 回合开始后的第一份建议（酒馆刷出来以后算的）
			if(_turnAdvice == null && run != null && run.Turn == s.Turn && run.Snapshot.Tavern.Count > 0)
				_turnAdvice = run;
		}

		private void CloseTurn()
		{
			if(_curTurn <= 0 || _turnStart == null || _lastRecruitSnap == null || _lastRecruitTurn != _curTurn)
			{
				_curTurn = 0;
				return;
			}
			var start = _turnStart;
			var end = _lastRecruitSnap;
			var newGolden = new HashSet<string>(end.GoldenBases.Except(start.GoldenBases));
			string Name(int id) => _turnNames.TryGetValue(id, out var n) ? n : "?";

			var upgraded = start.Tier >= 1 && end.Tier > start.Tier;
			// 这回合见过的所有酒馆随从里，最后归你的（或合成了金色的）= 买的
			var bought = _turnTavernSeen.Where(end.Owned.ContainsKey).ToList();
			var sold = start.Owned.Where(kv => !end.Owned.ContainsKey(kv.Key) && !newGolden.Contains(kv.Value)).Select(kv => kv.Key).ToList();

			var actual = new List<string>();
			if(upgraded) actual.Add($"升到 {end.Tier} 本");
			if(bought.Count > 0) actual.Add("买 " + string.Join("、", bought.Select(Name)));
			if(_turnSpells.Count > 0) actual.Add("法术 " + string.Join("、", _turnSpells));
			if(sold.Count > 0) actual.Add("卖 " + string.Join("、", sold.Select(Name)));

			var rec = new TurnRecord
			{
				Turn = _curTurn,
				Tier = end.Tier,
				Actual = actual.Count > 0 ? string.Join("；", actual) : "没买也没升本",
				ActualUpgrade = upgraded,
			};
			if(_turnAdvice != null)
			{
				rec.Advice = _turnAdvice.TopSummary;
				rec.AdviceUpgrade = _turnAdvice.TopUpgrade;
				var adviceBuys = new HashSet<string>(_turnAdvice.TopBuys.Select(Name));
				var actualBuys = new HashSet<string>(bought.Select(Name));
				var sameUp = rec.AdviceUpgrade == upgraded;
				rec.Match = sameUp && adviceBuys.SetEquals(actualBuys) ? "same"
					: sameUp || adviceBuys.Overlaps(actualBuys) ? "partial" : "diff";
			}
			else
				rec.Match = "none";

			var log = _turnLog.Where(r => r.Turn != rec.Turn).ToList();
			log.Add(rec);
			_turnLog = log;
			_curTurn = 0;
		}

		private void Finish()
		{
			CloseTurn();
			var key = _gameKey;
			_gameKey = null;
			_openRun = null;
			if(key == null || _pending.Count == 0 || _lastPlace <= 0)
			{
				_pending.Clear();
				return;
			}

			var sb = new StringBuilder();
			foreach(var json in _pending)
				// {"game":..,"place":..,  + 原样本字段
				sb.Append("{\"game\":\"").Append(key).Append("\",\"place\":").Append(_lastPlace).Append(',').Append(json.Substring(1)).Append('\n');
			File.AppendAllText(_samplesPath, sb.ToString(), new UTF8Encoding(false));
			Log.Info($"[BGStateExporter] 本局记录 {_pending.Count} 个学习样本，名次 {_lastPlace}");
			_pending.Clear();

			var w = AdvisorWeights.Load();
			RefreshStatus(null);
			if(w.AutoLearn && _status.GamesCollected >= w.AutoLearnMinGames && !_fitting)
			{
				_fitting = true;
				_ = Task.Run(() =>
				{
					try { RefreshStatus(Fit()); }
					catch(Exception ex) { RefreshStatus("拟合出错：" + ex.Message); Log.Error(ex); }
					finally { _fitting = false; }
				});
			}
		}

		private void RefreshStatus(string? fitMessage)
		{
			var w = AdvisorWeights.Load();
			var games = 0;
			var samples = 0;
			try
			{
				if(File.Exists(_samplesPath))
				{
					var keys = new HashSet<string>();
					foreach(var line in File.ReadLines(_samplesPath))
					{
						var i = line.IndexOf("\"game\":\"", StringComparison.Ordinal);
						if(i < 0) continue;
						var j = line.IndexOf('"', i + 8);
						if(j > i) keys.Add(line.Substring(i + 8, j - i - 8));
						samples++;
					}
					games = keys.Count;
				}
			}
			catch { /* 文件被占用时下次再数 */ }

			var old = _status;
			_status = new LearningStatus
			{
				Enabled = w.AutoLearn,
				GamesCollected = games,
				SamplesCollected = samples,
				MinGames = w.AutoLearnMinGames,
				SamplesThisGame = _pending.Count,
				Fitting = _fitting,
				LastFitMessage = fitMessage ?? old.LastFitMessage,
				LastFitTime = fitMessage != null ? DateTime.Now.ToString("yyyy-MM-dd HH:mm") : old.LastFitTime,
			};
		}

		// ---------- 拟合（后台线程） ----------

		private sealed class Sample
		{
			public string Game = "";
			public double A;                 // 名次权重
			public int Chosen;
			public double[][] X = Array.Empty<double[]>();
		}

		private static readonly string[] Keys = AdvisorWeights.FeatureKeys;
		private const string AnchorKey = "sim";

		private string Fit()
		{
			var cur = AdvisorWeights.Load();
			var defaults = new AdvisorWeights();
			var samples = LoadSamples().Where(s => s.A > 0).ToList();
			var games = samples.Select(s => s.Game).Distinct().OrderBy(g => g).ToList();
			if(games.Count < 4)
				return $"名次前 4 的对局只有 {games.Count} 局，暂不拟合";

			// 每 4 局取 1 局做验证集（按对局 ID 的哈希固定划分）
			var hold = new HashSet<string>(games.Where(g => (StableHash(g) & 3) == 0));
			if(hold.Count == 0) hold.Add(games.Last());
			var train = samples.Where(s => !hold.Contains(s.Game)).ToList();
			var test = samples.Where(s => hold.Contains(s.Game)).ToList();
			if(train.Count == 0 || test.Count == 0)
				return "数据不够划分验证集";

			var w0 = Keys.Select(defaults.Get).ToArray();
			var wCur = Keys.Select(cur.Get).ToArray();
			var trainGames = train.Select(s => s.Game).Distinct().Count();

			var (wNew, bNew) = Optimize(train, wCur, w0, true, trainGames, cur.AutoLearnStrength);
			var (_, bCur) = Optimize(train, wCur, w0, false, trainGames, cur.AutoLearnStrength);
			var llNew = Evaluate(test, wNew, bNew).ll;
			var llCur = Evaluate(test, wCur, bCur).ll;

			var sb = new StringBuilder();
			sb.AppendLine($"==== {DateTime.Now:yyyy-MM-dd HH:mm:ss}  对局 {games.Count}（验证 {hold.Count}）  样本 {samples.Count}");
			sb.AppendLine($"验证集对数似然：当前 {llCur:F4} → 新 {llNew:F4}");
			string msg;
			if(llNew <= llCur + 1e-3)
			{
				msg = $"新权重在验证集上没有更好（{llCur:F3} → {llNew:F3}），保持原权重";
				sb.AppendLine(msg);
			}
			else
			{
				var (wAll, bAll) = Optimize(samples, wCur, w0, true, games.Count, cur.AutoLearnStrength);
				var (llBefore, accBefore) = Evaluate(samples, wCur, bCur);
				var (llAfter, accAfter) = Evaluate(samples, wAll, bAll);
				if(AdvisorWeights.FilePath is string path && File.Exists(path))
					File.Copy(path, Path.Combine(Path.GetDirectoryName(path)!, "advisor_weights.backup.json"), true);
				for(var k = 0; k < Keys.Length; k++)
				{
					sb.AppendLine($"  {AdvisorWeights.FeatureNames[Keys[k]],-6} {wCur[k],8:F4} → {wAll[k],8:F4}");
					cur.Set(Keys[k], wAll[k]);
				}
				cur.Save();
				msg = $"已更新权重：用 {games.Count} 局，方案 1 与你实际选择一致的比例 {accBefore:P0} → {accAfter:P0}";
				sb.AppendLine(msg + $"（β={bAll:F2}，全体对数似然 {llBefore:F3} → {llAfter:F3}）");
			}
			File.AppendAllText(_logPath, sb.ToString(), new UTF8Encoding(false));
			Log.Info("[BGStateExporter] " + msg);
			return msg;
		}

		/// <summary>Adam 优化：参数 = 非锚点权重（按默认值缩放）+ log β。fitWeights=false 时只拟合 β。</summary>
		private static (double[] w, double beta) Optimize(List<Sample> data, double[] wInit, double[] w0, bool fitWeights, int games, double strength)
		{
			var k = Keys.Length;
			var anchor = Array.IndexOf(Keys, AnchorKey);
			var sc = w0.Select(v => Math.Max(v, 0.05)).ToArray();
			var lam = 2.0 / (Math.Max(0.05, strength) * Math.Max(1, games));

			var th = new double[k + 1];
			for(var i = 0; i < k; i++) th[i] = wInit[i] / sc[i];
			th[k] = Math.Log(5.0);
			var m = new double[k + 1];
			var v = new double[k + 1];

			double[] W() { var w = new double[k]; for(var i = 0; i < k; i++) w[i] = i == anchor || !fitWeights ? wInit[i] : Math.Max(0, th[i] * sc[i]); return w; }

			for(var t = 1; t <= 800; t++)
			{
				var w = W();
				var beta = Math.Exp(th[k]);
				var (gw, gb) = Gradient(data, w, beta);
				var g = new double[k + 1];
				for(var i = 0; i < k; i++)
					g[i] = fitWeights && i != anchor ? gw[i] * sc[i] - 2 * lam * (w[i] - w0[i]) / sc[i] : 0;
				g[k] = gb * beta;
				for(var i = 0; i <= k; i++)
				{
					m[i] = 0.9 * m[i] + 0.1 * g[i];
					v[i] = 0.999 * v[i] + 0.001 * g[i] * g[i];
					th[i] += 0.03 * (m[i] / (1 - Math.Pow(0.9, t))) / (Math.Sqrt(v[i] / (1 - Math.Pow(0.999, t))) + 1e-8);
				}
				for(var i = 0; i < k; i++) th[i] = Math.Max(0, th[i]);
				th[k] = Math.Max(Math.Log(0.05), Math.Min(Math.Log(200), th[k]));
			}
			return (W(), Math.Exp(th[k]));
		}

		private static (double[] gw, double gb) Gradient(List<Sample> data, double[] w, double beta)
		{
			var k = w.Length;
			var gw = new double[k];
			var gb = 0.0;
			var total = 0.0;
			foreach(var s in data)
			{
				var (u, p) = Softmax(s.X, w, beta);
				var ubar = 0.0;
				var xbar = new double[k];
				for(var j = 0; j < u.Length; j++)
				{
					ubar += p[j] * u[j];
					for(var i = 0; i < k; i++) xbar[i] += p[j] * s.X[j][i];
				}
				for(var i = 0; i < k; i++) gw[i] += s.A * beta * (s.X[s.Chosen][i] - xbar[i]);
				gb += s.A * (u[s.Chosen] - ubar);
				total += s.A;
			}
			if(total <= 0) return (gw, 0);
			for(var i = 0; i < k; i++) gw[i] /= total;
			return (gw, gb / total);
		}

		private static (double ll, double acc) Evaluate(List<Sample> data, double[] w, double beta)
		{
			double ll = 0, acc = 0, total = 0;
			foreach(var s in data)
			{
				var (u, p) = Softmax(s.X, w, beta);
				ll += s.A * Math.Log(p[s.Chosen] + 1e-12);
				var arg = 0;
				for(var j = 1; j < u.Length; j++) if(u[j] > u[arg]) arg = j;
				if(arg == s.Chosen) acc += s.A;
				total += s.A;
			}
			return total > 0 ? (ll / total, acc / total) : (0, 0);
		}

		private static (double[] u, double[] p) Softmax(double[][] x, double[] w, double beta)
		{
			var n = x.Length;
			var u = new double[n];
			var p = new double[n];
			var max = double.MinValue;
			for(var j = 0; j < n; j++)
			{
				for(var i = 0; i < w.Length; i++) u[j] += x[j][i] * w[i];
				max = Math.Max(max, beta * u[j]);
			}
			var sum = 0.0;
			for(var j = 0; j < n; j++) { p[j] = Math.Exp(beta * u[j] - max); sum += p[j]; }
			for(var j = 0; j < n; j++) p[j] /= sum;
			return (u, p);
		}

		private List<Sample> LoadSamples()
		{
			var list = new List<Sample>();
			if(!File.Exists(_samplesPath))
				return list;
			foreach(var line in File.ReadLines(_samplesPath))
			{
				if(string.IsNullOrWhiteSpace(line)) continue;
				try
				{
					var raw = JsonConvert.DeserializeObject<RawSample>(line);
					if(raw?.Cands == null || raw.Cands.Count < 2 || raw.Chosen < 0 || raw.Chosen >= raw.Cands.Count)
						continue;
					list.Add(new Sample
					{
						Game = raw.Game ?? "",
						A = Math.Max(0, 4.5 - raw.Place) / 3.5,
						Chosen = raw.Chosen,
						X = raw.Cands.Select(f => Keys.Select(key => f != null && f.TryGetValue(key, out var val) ? val : 0).ToArray()).ToArray(),
					});
				}
				catch { /* 跳过损坏的行 */ }
			}
			return list;
		}

		private sealed class RawSample
		{
			[JsonProperty("game")] public string? Game { get; set; }
			[JsonProperty("place")] public int Place { get; set; }
			[JsonProperty("chosen")] public int Chosen { get; set; }
			[JsonProperty("cands")] public List<Dictionary<string, double>>? Cands { get; set; }
		}

		private static int StableHash(string s)
		{
			unchecked
			{
				var h = 23;
				foreach(var ch in s) h = h * 31 + ch;
				return h & 0x7fffffff;
			}
		}
	}
}
