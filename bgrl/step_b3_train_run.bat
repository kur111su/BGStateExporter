@echo off
chcp 65001 >nul
rem Step B3: long training run. Safe to close the window and run again later: it resumes from runs\main\latest.pt.
rem Logs: bgrl\runs\main\log.jsonl ; snapshots for the opponent library: bgrl\runs\main\snapshots
setlocal
cd /d "%~dp0"
.venv\Scripts\python -X utf8 tools\train.py --run runs\main --resume --iters 100000 --threads %NUMBER_OF_PROCESSORS% %*
pause
