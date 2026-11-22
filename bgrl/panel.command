#!/bin/bash
# Mac：双击打开面板（训练看板、所有操作、竞技场对局）。第一次先在“终端”里运行 ./setup.sh。
# 面板在后台独立运行；打开以后这个终端窗口自己关掉（“终端”只剩这一个窗口时直接退出“终端”）。
# 想留着终端窗口：在终端里运行  PANEL_KEEP_TERMINAL=1 ./panel.command
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "还没装环境：在终端里运行  cd \"$(pwd)\" && ./setup.sh"
  read -n 1 -s -r -p "按任意键关闭"
  exit 1
fi
mkdir -p reports
# 面板放进新的会话里跑（setsid，和终端完全脱开）：nohup 只挡 SIGHUP，退出“终端”时整个会话里的进程还是会被结束
.venv/bin/python -c 'import subprocess, sys
subprocess.Popen([sys.executable, "-X", "utf8", "bg.py", "panel"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                 stderr=open("reports/panel_stderr.txt", "w"), start_new_session=True, close_fds=True)'
clear && printf '\033[3J'            # 清屏，连滚动记录一起清掉
if [ "$TERM_PROGRAM" = "Apple_Terminal" ] && [ -z "$PANEL_KEEP_TERMINAL" ]; then
  # 等这个脚本和它的 shell 退出以后再关窗口（否则“终端”会问要不要结束正在运行的进程）。
  # 只剩这一个窗口：退出“终端”；还有别的窗口：只关这一个，别的窗口和里面跑的东西不动。
  me="$(tty)"
  .venv/bin/python -c 'import subprocess, sys
scpt = """delay 1
tell application "Terminal"
  if (count of windows) is 1 then
    quit
  else
    repeat with w in windows
      if tty of selected tab of w is "%s" then close w
    end repeat
  end if
end tell""" % sys.argv[1]
subprocess.Popen(["osascript", "-e", scpt], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                 stderr=subprocess.DEVNULL, start_new_session=True, close_fds=True)' "$me"
fi
exit 0
