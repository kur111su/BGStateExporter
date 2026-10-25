using System;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading.Tasks;
using Hearthstone_Deck_Tracker.Utility.Logging;
using Newtonsoft.Json.Linq;

namespace BGStateExporter
{
	/// <summary>
	/// 启动时检查 GitHub Releases 有没有新版本；有就下载 zip，把 DLL 放进
	/// %AppData%\HearthstoneDeckTracker\Plugins\BGStateExporter\，下次启动 HDT 时生效。
	/// （HDT 实际加载的是它自己目录下 Plugins 里的副本，所以这里覆盖文件是安全的。）
	///
	/// 不用 HDT 自带的 GitHub 工具：它用系统默认编码（中文 Windows 上是 GBK）解码 GitHub 返回的 UTF-8，
	/// 发布说明里有中文时会把 JSON 解析坏（v0.3.0～v0.6.0 因此一直检查失败）。这里固定用 UTF-8。
	/// </summary>
	internal static class UpdateChecker
	{
		public const string GitHubUser = "kur111su";
		public const string GitHubRepo = "BGStateExporter";

		public static Version CurrentVersion => typeof(UpdateChecker).Assembly.GetName().Version;

		/// <summary>null = 还没检查完；空字符串 = 已是最新；其它 = 最新版本号。</summary>
		public static string? LatestVersion { get; private set; }

		/// <summary>新版本已下载好，重启 HDT 生效。</summary>
		public static bool UpdateReady { get; private set; }

		/// <summary>检查或下载失败的原因（显示在网页面板上）。</summary>
		public static string? Error { get; private set; }

		private static WebClient NewClient()
		{
			var wc = new WebClient { Encoding = Encoding.UTF8 };
			wc.Headers.Add(HttpRequestHeader.UserAgent, "BGStateExporter/" + CurrentVersion.ToString(3));
			return wc;
		}

		public static async Task CheckAsync()
		{
			try
			{
				ServicePointManager.SecurityProtocol |= SecurityProtocolType.Tls12;

				string json;
				using(var wc = NewClient())
				{
					wc.Headers.Add(HttpRequestHeader.Accept, "application/vnd.github+json");
					json = await wc.DownloadStringTaskAsync($"https://api.github.com/repos/{GitHubUser}/{GitHubRepo}/releases/latest");
				}
				var release = JObject.Parse(json);
				var tag = (string?)release["tag_name"] ?? "";
				if(!Version.TryParse(tag.TrimStart('v', 'V'), out var latest))
				{
					Error = $"看不懂版本号 {tag}";
					return;
				}
				// 只比较前三位（Release 是 0.7.0，程序集版本是 0.7.0.0）
				var current = new Version(CurrentVersion.Major, CurrentVersion.Minor, Math.Max(0, CurrentVersion.Build));
				if(latest <= current)
				{
					LatestVersion = "";
					Log.Info($"[BGStateExporter] 已是最新版本（{current}）");
					return;
				}
				LatestVersion = tag;

				var asset = (release["assets"] as JArray)?
					.FirstOrDefault(a => ((string?)a["name"])?.EndsWith(".zip", StringComparison.OrdinalIgnoreCase) == true);
				var url = (string?)asset?["browser_download_url"];
				if(url == null)
				{
					Error = $"{tag} 没有可下载的 zip";
					return;
				}

				var temp = Path.Combine(Path.GetTempPath(), "BGStateExporter_update");
				if(Directory.Exists(temp))
					Directory.Delete(temp, true);
				Directory.CreateDirectory(temp);
				var zipPath = Path.Combine(temp, "update.zip");
				using(var wc = NewClient())
					await wc.DownloadFileTaskAsync(url, zipPath);

				var extracted = Path.Combine(temp, "x");
				ZipFile.ExtractToDirectory(zipPath, extracted);
				var dll = Directory.GetFiles(extracted, "BGStateExporter.dll", SearchOption.AllDirectories).FirstOrDefault();
				if(dll == null)
				{
					Error = $"{tag} 的压缩包里没有 BGStateExporter.dll";
					return;
				}

				var target = Path.Combine(
					Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
					"HearthstoneDeckTracker", "Plugins", "BGStateExporter");
				Directory.CreateDirectory(target);
				File.Copy(dll, Path.Combine(target, "BGStateExporter.dll"), true);
				UpdateReady = true;
				Error = null;
				Log.Info($"[BGStateExporter] 新版本 {tag} 已下载，重启 HDT 后生效");
			}
			catch(Exception ex)
			{
				Error = "检查更新失败：" + ex.Message;
				Log.Warn($"[BGStateExporter] {Error}");
			}
		}
	}
}
