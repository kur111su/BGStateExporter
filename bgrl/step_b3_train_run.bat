@echo off
chcp 65001 >nul
rem Step B3: long training run. Safe to close the window and run again later: it resumes from runs\v2\latest.pt.
rem Logs: bgrl\runs\v2\log.jsonl ; snapshots for the opponent library: bgrl\runs\v2\snapshots
rem runs\v2 starts from the healthy iteration-300 weights of the first run (runs\main went bad after ~310).
setlocal
cd /d "%~dp0"
set "INIT="
if not exist runs\v2\latest.pt if exist runs\main\snapshots\it00300.pt set "INIT=--init runs\main\snapshots\it00300.pt"
.venv\Scripts\python -X utf8 tools\train.py --run runs\v2 --resume %INIT% --iters 100000 --threads %NUMBER_OF_PROCESSORS% %*
pause
