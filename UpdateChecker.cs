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

		/// <summary>
		/// 用 github.com 的“最新版本”跳转拿版本号：/releases/latest 会 302 到 /releases/tag/vX.Y.Z。
		/// 不走 api.github.com——它按 IP 限制匿名请求（每小时 60 次），共用出口 IP 时很容易 403。
		/// </summary>
		private static async Task<string?> LatestTagViaRedirect()
		{
			var req = (HttpWebRequest)WebRequest.Create($"https://github.com/{GitHubUser}/{GitHubRepo}/releases/latest");
			req.AllowAutoRedirect = false;
			req.UserAgent = "BGStateExporter/" + CurrentVersion.ToString(3);
			req.Timeout = 15000;
			using var resp = (HttpWebResponse)await req.GetResponseAsync();
			var location = resp.Headers["Location"];
			if(string.IsNullOrEmpty(location) || !location!.Contains("/releases/tag/"))
				return null;
			return location.Substring(location.LastIndexOf('/') + 1);
		}

		/// <summary>备用：GitHub API（可能被限流）。</summary>
		private static async Task<(string? tag, string? url)> LatestViaApi()
		{
			string json;
			using(var wc = NewClient())
			{
				wc.Headers.Add(HttpRequestHeader.Accept, "application/vnd.github+json");
				json = await wc.DownloadStringTaskAsync($"https://api.github.com/repos/{GitHubUser}/{GitHubRepo}/releases/latest");
			}
			var release = JObject.Parse(json);
			var asset = (release["assets"] as JArray)?
				.FirstOrDefault(a => ((string?)a["name"])?.EndsWith(".zip", StringComparison.OrdinalIgnoreCase) == true);
			return ((string?)release["tag_name"], (string?)asset?["browser_download_url"]);
		}

		public static async Task CheckAsync()
		{
			try
			{
				ServicePointManager.SecurityProtocol |= SecurityProtocolType.Tls12;

				string? tag = null;
				string? url = null;
				string? redirectError = null;
				try
				{
					tag = await LatestTagViaRedirect();
					if(tag != null)
						// Release 里的附件名由 CI 固定为 BGStateExporter-vX.Y.Z.zip
						url = $"https://github.com/{GitHubUser}/{GitHubRepo}/releases/download/{tag}/BGStateExporter-{tag}.zip";
				}
				catch(Exception ex)
				{
					redirectError = ex.Message;
				}
				if(tag == null)
				{
					try
					{
						(tag, url) = await LatestViaApi();
					}
					catch(Exception ex)
					{
						Error = "检查更新失败：" + (redirectError ?? ex.Message);
						Log.Warn($"[BGStateExporter] {Error}（备用接口：{ex.Message}）");
						return;
					}
				}
				tag ??= "";
				if(!Version.TryParse(tag.TrimStart('v', 'V'), out var latest))
				{
					Error = $"看不懂版本号 {tag}";
					return;
				}
				// 只比较前三位（Release 是 0.8.0，程序集版本是 0.8.0.0）
				var current = new Version(CurrentVersion.Major, CurrentVersion.Minor, Math.Max(0, CurrentVersion.Build));
				if(latest <= current)
				{
					LatestVersion = "";
					Log.Info($"[BGStateExporter] 已是最新版本（{current}）");
					return;
				}
				LatestVersion = tag;
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
				var dest = Path.Combine(target, "BGStateExporter.dll");
				File.Copy(dll, dest, true);
				// HDT 启动时只在“源文件比它自己目录里的副本新”（按修改时间）时才复制插件。
				// 解压出来的文件带的是压缩包里的时间（CI 编译时间，而且按 UTC 记、被当成本地时间读），
				// 常常比旧副本还“旧”，结果 HDT 一直加载旧版本（v0.9.2 就没更新上）。这里把时间改成现在。
				File.SetLastWriteTimeUtc(dest, DateTime.UtcNow);
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
