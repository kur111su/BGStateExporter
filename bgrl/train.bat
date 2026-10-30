@echo off
chcp 65001 >nul
rem Training, stage 2 (runs\v3): bots are boosted to the strength of the real players in your games
rem (see bgrl\rl\calibrate.py), so the model has to learn to level up and build a late-game board.
rem Starts from the stage-1 model (runs\v2\latest.pt). Safe to close the window and run again later: it resumes.
rem Also imitates your own moves (bc-coef; upgrades, sells and buys weighted up): runs\demos.npz is rebuilt from your games when there are new ones.
rem Tempo bonus for levelling up while behind the human tier curve (fades out over 1500 iterations) and a small penalty for refreshes followed by no buy.
rem Bots get real 6500+ boards (runs\pro_boards.json from bg.py prolib); without that file they are scaled up as before.
rem Also imitates 6500+ players (runs\pro_demos from bg.py prodemos, about 90k decisions from 361 replays).
rem Log: bgrl\runs\v3\log.jsonl ; opponent snapshots: bgrl\runs\v3\snapshots
rem Other settings: train.bat --help   (e.g. train.bat --threads 4 if the game stutters)
setlocal
cd /d "%~dp0"
.venv\Scripts\python -c "import orjson" 2>nul || .venv\Scripts\python -m pip install -q orjson 2>nul
.venv\Scripts\python -X utf8 bg.py demos --if-stale
set "INIT="
if not exist runs\v3\latest.pt if exist runs\v2\latest.pt set "INIT=--init runs\v2\latest.pt"
.venv\Scripts\python -X utf8 bg.py train --run runs\v3 --resume %INIT% --iters 100000 --threads %NUMBER_OF_PROCESSORS% --boost-frac 0.75 --boost-kind pro --boost-from-turn 5 --tempo-curve pro --bc-coef 0.05 --bc-weights "upgrade:3,sell:3,buy:2" --bc-train-cap 0.6 --bc-smooth 0.1 --pro-coef 0.1 --pro-weights "upgrade:3,end:2,buy:1.5" --tempo-bonus 0.03 --tempo-deficit 0.01 --refresh-waste-penalty 0.015 %*
pause
