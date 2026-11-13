#!/bin/bash
# Mac：双击打开面板（训练看板、所有操作、竞技场对局）。第一次先在“终端”里运行 ./setup.sh。
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "还没装环境：在终端里运行  cd \"$(pwd)\" && ./setup.sh"
  read -n 1 -s -r -p "按任意键关闭"
  exit 1
fi
mkdir -p reports
nohup .venv/bin/python -X utf8 bg.py panel >/dev/null 2>reports/panel_stderr.txt &
disown
exit 0
