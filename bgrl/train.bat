@echo off
chcp 65001 >nul
rem Training, stage 2 (runs\v3): bots are boosted to the strength of the real players in your games
rem (see bgrl\rl\calibrate.py), so the model has to learn to level up and build a late-game board.
rem Starts from the stage-1 model (runs\v2\latest.pt). Safe to close the window and run again later: it resumes.
rem Log: bgrl\runs\v3\log.jsonl ; opponent snapshots: bgrl\runs\v3\snapshots
rem Other settings: train.bat --help   (e.g. train.bat --threads 4 if the game stutters)
setlocal
cd /d "%~dp0"
.venv\Scripts\python -c "import orjson" 2>nul || .venv\Scripts\python -m pip install -q orjson 2>nul
set "INIT="
if not exist runs\v3\latest.pt if exist runs\v2\latest.pt set "INIT=--init runs\v2\latest.pt"
.venv\Scripts\python -X utf8 bg.py train --run runs\v3 --resume %INIT% --iters 100000 --threads %NUMBER_OF_PROCESSORS% --boost-frac 0.75 --ent-coef 0.01 %*
pause
