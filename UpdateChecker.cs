using System;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Threading.Tasks;
using Hearthstone_Deck_Tracker.Utility;
using Hearthstone_Deck_Tracker.Utility.Logging;

namespace BGStateExporter
{
	/// <summary>
	/// 启动时检查 GitHub Releases 有没有新版本；有就下载 zip，把 DLL 放进
	/// %AppData%\HearthstoneDeckTracker\Plugins\BGStateExporter\，下次启动 HDT 时生效。
	/// （HDT 实际加载的是它自己目录下 Plugins 里的副本，所以这里覆盖文件是安全的。）
	/// </summary>
	internal static class UpdateChecker
	{
		public const string GitHubUser = "kur111su";
		public const string GitHubRepo = "BGStateExporter";

		public static Version CurrentVersion => typeof(UpdateChecker).Assembly.GetName().Version;

		/// <summary>null = 还没检查完；空字符串 = 已是最新。</summary>
		public static string? LatestVersion { get; private set; }

		/// <summary>新版本已下载好，重启 HDT 生效。</summary>
		public static bool UpdateReady { get; private set; }

		public static async Task CheckAsync()
		{
			try
			{
				var release = await GitHub.CheckForUpdate(GitHubUser, GitHubRepo, CurrentVersion);
				if(release == null)
				{
					LatestVersion = "";
					return;
				}
				LatestVersion = release.Tag;

				var temp = Path.Combine(Path.GetTempPath(), "BGStateExporter_update");
				if(Directory.Exists(temp))
					Directory.Delete(temp, true);
				Directory.CreateDirectory(temp);

				var zipPath = await GitHub.DownloadRelease(release, temp);
				if(zipPath == null)
					return;

				string? dll;
				if(zipPath.EndsWith(".zip", StringComparison.OrdinalIgnoreCase))
				{
					var extracted = Path.Combine(temp, "x");
					ZipFile.ExtractToDirectory(zipPath, extracted);
					dll = Directory.GetFiles(extracted, "BGStateExporter.dll", SearchOption.AllDirectories).FirstOrDefault();
				}
				else
					dll = zipPath.EndsWith("BGStateExporter.dll", StringComparison.OrdinalIgnoreCase) ? zipPath : null;
				if(dll == null)
				{
					Log.Warn("[BGStateExporter] 新版本的压缩包里没找到 BGStateExporter.dll");
					return;
				}

				var target = Path.Combine(
					Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
					"HearthstoneDeckTracker", "Plugins", "BGStateExporter");
				Directory.CreateDirectory(target);
				File.Copy(dll, Path.Combine(target, "BGStateExporter.dll"), true);
				UpdateReady = true;
				Log.Info($"[BGStateExporter] 新版本 {release.Tag} 已下载，重启 HDT 后生效");
			}
			catch(Exception ex)
			{
				Log.Warn($"[BGStateExporter] 检查更新失败：{ex.Message}");
			}
		}
	}
}
