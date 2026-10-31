using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Windows.Controls;
using Hearthstone_Deck_Tracker;
using Hearthstone_Deck_Tracker.Plugins;
using Hearthstone_Deck_Tracker.Utility.Logging;
using Newtonsoft.Json;
using Newtonsoft.Json.Serialization;

namespace BGStateExporter
{
	public class BGStateExporterPlugin : IPlugin
	{
		private const int Port = 8765;          // /state.json、/events、/optimize（没有网页面板了）
		private const int ShadowPort = 8766;    // bgrl 影子模式的页面：模型建议 + 这里的模拟结果

		private static readonly JsonSerializerSettings JsonSettings = new()
		{
			// 属性名用 camelCase，但字典的键（标签名、中文特征名）保持原样
			ContractResolver = new DefaultContractResolver
			{
				NamingStrategy = new CamelCaseNamingStrategy { ProcessDictionaryKeys = false },
			},
			Formatting = Formatting.None,
		};

		private StateServer? _server;
		private SimManager? _sim;
		private MenuItem? _menuItem;
		private bool _menuNoticeShown;
		private string _lastContent = "";
		private long _seq;
		private string _outDir = "";
		private string? _gameLogPath;      // 当前这局的 jsonl 记录文件
		private DateTime _lastErrorLog = DateTime.MinValue;

		public string Name => "BG State Exporter";
		public string Description => "实时导出酒馆战旗局面和对局记录、Bob's Buddy 对战模拟（本场胜率、对位胜率、站位优化）。建议来自 PPO 模型的影子模式：bgrl\\shadow.bat，页面 http://localhost:8766/";
		public string ButtonText => "打开影子模式页面";
		public string Author => "炉石酒馆项目";
		public Version Version => UpdateChecker.CurrentVersion;

		public MenuItem MenuItem
		{
			get
			{
				if(_menuItem == null)
				{
					_menuItem = new MenuItem { Header = "影子模式（模型建议）" };
					_menuItem.Click += (_, _) => OpenPanel();
				}
				return _menuItem;
			}
		}

		public void OnLoad()
		{
			_outDir = Path.Combine(
				Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
				"HearthstoneDeckTracker", "BGStateExporter");
			Directory.CreateDirectory(Path.Combine(_outDir, "games"));
			try
			{
				_sim = new SimManager();
			}
			catch(Exception ex)
			{
				// 找不到 BobsBuddy.dll 时只关闭模拟功能，局面导出照常工作
				_sim = null;
				Log.Error($"[BGStateExporter] 对战模拟不可用：{ex.Message}");
			}
			_server = new StateServer(Port, opponent => _sim?.RequestOptimize(opponent) ?? "{\"accepted\":false}");
			_server.Start();
			Log.Info($"[BGStateExporter] 已加载，输出目录 {_outDir}");

			// 后台检查 GitHub 上有没有新版本
			_ = UpdateChecker.CheckAsync();
		}

		public void OnUnload()
		{
			_server?.Dispose();
			_server = null;
			_sim = null;
		}

		public void OnButtonPress() => OpenPanel();

		public void OnUpdate()
		{
			try
			{
				var state = StateReader.Read();
				state.Plugin = new PluginInfo
				{
					Version = UpdateChecker.CurrentVersion.ToString(3),
					LatestVersion = UpdateChecker.LatestVersion,
					UpdateReady = UpdateChecker.UpdateReady,
					UpdateError = UpdateChecker.Error,
				};
				if(UpdateChecker.UpdateReady && _menuItem != null && !_menuNoticeShown)
				{
					_menuNoticeShown = true;
					_menuItem.Header = $"影子模式（新版本 {UpdateChecker.LatestVersion} 已下载，重启 HDT 生效）";
				}
				if(_sim != null)
				{
					// 模拟出错不能影响局面导出
					try
					{
						_sim.Tick(Core.Game, state);
						if(state.IsBattlegrounds)
							state.Simulation = _sim.Snapshot();
					}
					catch(Exception ex)
					{
						if((DateTime.Now - _lastErrorLog).TotalSeconds > 10)
						{
							_lastErrorLog = DateTime.Now;
							Log.Warn($"[BGStateExporter] 模拟出错：{ex.Message}");
						}
					}
				}
				var content = JsonConvert.SerializeObject(state, JsonSettings);
				if(content == _lastContent)
					return;
				_lastContent = content;
				_seq++;

				// 在最前面插入序号和时间戳：{"seq":12,"ts":1729760000000, ...局面字段}
				var ts = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
				var json = "{\"seq\":" + _seq + ",\"ts\":" + ts + "," + content.Substring(1);

				_server?.Publish(json);
				WriteStateFile(json);
				AppendGameLog(state, json);
			}
			catch(Exception ex)
			{
				// 日志线程正在改实体时偶尔会读失败，跳过这一帧即可；错误日志最多 10 秒一条
				if((DateTime.Now - _lastErrorLog).TotalSeconds > 10)
				{
					_lastErrorLog = DateTime.Now;
					Log.Warn($"[BGStateExporter] 读取局面失败：{ex.Message}");
				}
			}
		}

		private void WriteStateFile(string json)
		{
			var target = Path.Combine(_outDir, "state.json");
			var tmp = target + ".tmp";
			File.WriteAllText(tmp, json, new UTF8Encoding(false));
			if(File.Exists(target))
				File.Replace(tmp, target, null);
			else
				File.Move(tmp, target);
		}

		/// <summary>每局战旗把每次局面变化追加到 games/时间.jsonl，以后可作为训练数据。</summary>
		private void AppendGameLog(BgState state, string json)
		{
			if(!state.IsBattlegrounds)
			{
				_gameLogPath = null;
				return;
			}
			_gameLogPath ??= Path.Combine(_outDir, "games", DateTime.Now.ToString("yyyyMMdd_HHmmss") + ".jsonl");
			File.AppendAllText(_gameLogPath, json + "\n", new UTF8Encoding(false));
		}

		private void OpenPanel()
		{
			try
			{
				Process.Start(new ProcessStartInfo($"http://localhost:{ShadowPort}/") { UseShellExecute = true });
			}
			catch(Exception ex)
			{
				Log.Error($"[BGStateExporter] 无法打开浏览器：{ex.Message}");
			}
		}
	}
}
