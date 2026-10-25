using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Reflection;
using System.Text;
using System.Threading;
using Hearthstone_Deck_Tracker.Utility.Logging;

namespace BGStateExporter
{
	/// <summary>
	/// 本机 HTTP 服务（只监听 localhost）：
	///   GET /            网页面板
	///   GET /state.json  当前局面
	///   GET /events      Server-Sent Events，局面变化时推送
	///   GET /optimize    请求站位优化
	/// </summary>
	internal sealed class StateServer : IDisposable
	{
		private readonly int _port;
		private readonly Func<int, string> _onOptimize;
		private readonly HttpListener _listener = new();
		private readonly List<HttpListenerResponse> _sseClients = new();
		private readonly object _lock = new();
		private readonly AutoResetEvent _changed = new(false);
		private readonly byte[] _indexHtml;
		private volatile string _currentJson = "{}";
		private volatile bool _running;
		private Thread? _acceptThread;
		private Thread? _pushThread;

		public StateServer(int port, Func<int, string> onOptimize)
		{
			_port = port;
			_onOptimize = onOptimize;
			_indexHtml = LoadIndexHtml();
		}

		public string Url => $"http://localhost:{_port}/";

		public bool Start()
		{
			try
			{
				_listener.Prefixes.Add(Url);
				_listener.Start();
			}
			catch(Exception ex)
			{
				Log.Error($"[BGStateExporter] HTTP 服务启动失败（端口 {_port} 可能被占用）：{ex.Message}");
				return false;
			}
			_running = true;
			_acceptThread = new Thread(AcceptLoop) { IsBackground = true, Name = "BGStateExporter-HTTP" };
			_pushThread = new Thread(PushLoop) { IsBackground = true, Name = "BGStateExporter-SSE" };
			_acceptThread.Start();
			_pushThread.Start();
			Log.Info($"[BGStateExporter] 面板地址 {Url}");
			return true;
		}

		/// <summary>由插件在局面变化时调用（UI 线程），真正的推送在后台线程完成。</summary>
		public void Publish(string json)
		{
			_currentJson = json;
			_changed.Set();
		}

		private void AcceptLoop()
		{
			while(_running)
			{
				HttpListenerContext ctx;
				try
				{
					ctx = _listener.GetContext();
				}
				catch
				{
					break; // 监听器已关闭
				}
				ThreadPool.QueueUserWorkItem(_ => Handle(ctx));
			}
		}

		private void Handle(HttpListenerContext ctx)
		{
			var res = ctx.Response;
			try
			{
				res.AddHeader("Access-Control-Allow-Origin", "*");
				res.AddHeader("Cache-Control", "no-store");
				var path = ctx.Request.Url.AbsolutePath.TrimEnd('/');
				switch(path)
				{
					case "":
					case "/index.html":
						Write(res, "text/html; charset=utf-8", _indexHtml);
						break;
					case "/optimize":
						// 站位优化：/optimize?opponent=对手PLAYER_ID（不填则用下个对手），结果出现在 state.json 的 simulation.optimizer
						int.TryParse(ctx.Request.QueryString["opponent"], out var opp);
						Write(res, "application/json; charset=utf-8", Encoding.UTF8.GetBytes(_onOptimize(opp)));
						break;
					case "/state.json":
						Write(res, "application/json; charset=utf-8", Encoding.UTF8.GetBytes(_currentJson));
						break;
					case "/events":
						res.ContentType = "text/event-stream; charset=utf-8";
						res.SendChunked = true;
						res.KeepAlive = true;
						SendEvent(res, _currentJson); // 连上就先发一次当前局面
						lock(_lock)
							_sseClients.Add(res);
						return; // 保持连接，不关闭
					default:
						res.StatusCode = 404;
						Write(res, "text/plain; charset=utf-8", Encoding.UTF8.GetBytes("not found"));
						break;
				}
			}
			catch(Exception ex)
			{
				Log.Debug($"[BGStateExporter] 请求处理失败：{ex.Message}");
				try { res.Abort(); } catch { }
			}
		}

		private static void Write(HttpListenerResponse res, string contentType, byte[] body)
		{
			res.ContentType = contentType;
			res.ContentLength64 = body.Length;
			res.OutputStream.Write(body, 0, body.Length);
			res.OutputStream.Close();
		}

		private void PushLoop()
		{
			while(_running)
			{
				// 有变化立即推送；没有变化则每 15 秒发一次心跳，防止连接被断开
				var changed = _changed.WaitOne(TimeSpan.FromSeconds(15));
				if(!_running)
					break;
				List<HttpListenerResponse> clients;
				lock(_lock)
					clients = new List<HttpListenerResponse>(_sseClients);

				var json = _currentJson;
				foreach(var c in clients)
				{
					try
					{
						if(changed)
							SendEvent(c, json);
						else
							SendRaw(c, ": ping\n\n");
					}
					catch
					{
						lock(_lock)
							_sseClients.Remove(c);
						try { c.Abort(); } catch { }
					}
				}
			}
		}

		private static void SendEvent(HttpListenerResponse res, string json) => SendRaw(res, "data: " + json + "\n\n");

		private static void SendRaw(HttpListenerResponse res, string text)
		{
			var bytes = Encoding.UTF8.GetBytes(text);
			res.OutputStream.Write(bytes, 0, bytes.Length);
			res.OutputStream.Flush();
		}

		private static byte[] LoadIndexHtml()
		{
			using var stream = Assembly.GetExecutingAssembly().GetManifestResourceStream("BGStateExporter.index.html");
			if(stream == null)
				return Encoding.UTF8.GetBytes("<h1>index.html 没有被打包进插件</h1>");
			using var ms = new MemoryStream();
			stream.CopyTo(ms);
			return ms.ToArray();
		}

		public void Dispose()
		{
			_running = false;
			_changed.Set();
			lock(_lock)
			{
				foreach(var c in _sseClients)
					try { c.Abort(); } catch { }
				_sseClients.Clear();
			}
			try { _listener.Stop(); _listener.Close(); } catch { }
		}
	}
}
