@echo off
chcp 65001 >nul
rem Training. Safe to close the window and run again later: it resumes from runs\v2\latest.pt.
rem Log: bgrl\runs\v2\log.jsonl ; opponent snapshots: bgrl\runs\v2\snapshots
rem Other settings: train.bat --help   (e.g. train.bat --run runs\v3 --init runs\v2\snapshots\it00300.pt)
setlocal
cd /d "%~dp0"
set "INIT="
if not exist runs\v2\latest.pt if exist runs\main\snapshots\it00300.pt set "INIT=--init runs\main\snapshots\it00300.pt"
.venv\Scripts\python -X utf8 bg.py train --run runs\v2 --resume %INIT% --iters 100000 --threads %NUMBER_OF_PROCESSORS% %*
pause
