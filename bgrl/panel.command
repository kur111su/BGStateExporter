#!/bin/bash
# Mac：双击打开面板（训练看板、所有操作、竞技场对局）。第一次先在“终端”里运行 ./setup.sh。
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
clear && printf '\033[3J'            # 清屏，连滚动记录一起清掉（面板在后台跑，这个窗口可以直接关）
exit 0
