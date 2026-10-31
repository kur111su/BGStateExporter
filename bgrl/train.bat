@echo off
chcp 65001 >nul
rem Training, stage 2 (runs\v3): bots are boosted to the strength of the real players in your games
rem (see bgrl\rl\calibrate.py), so the model has to learn to level up and build a late-game board.
rem Starts from the stage-1 model (runs\v2\latest.pt). Safe to close the window and run again later: it resumes.
rem Also imitates your own moves (bc-coef; upgrades, sells and buys weighted up): runs\demos.npz is rebuilt from your games when there are new ones.
rem Tempo bonus for levelling up while behind the human tier curve (fades out over 1500 iterations) and a small penalty for refreshes followed by no buy.
rem Bots get real 6500+ boards (runs\pro_boards.json from bg.py prolib); without that file they are scaled up as before.
rem Also imitates 6500+ players (runs\pro_demos from bg.py prodemos, about 90k decisions from 361 replays).
rem Gold flow (bg.py goldflow): while behind the 6500+ tier curve, a step that spends the upgrade money on something else is penalised (buys half), and so is ending the turn with upgrade money; turns 2-5 of the 6500+ imitation count double.
rem Pro imitation strength halves automatically when the model starts memorising the pro samples (train hit rate more than 0.08 above held-out games).
rem Also imitates 13k-MMR games (runs\h13_demos from bg.py nativedemos) and uses their boards for the boosted bots (runs\h13_boards.json).
rem Pro imitation weights: buying a key card or an economy card (rl\card_value.json from bg.py cardvalue) counts double, selling 1.5x,
rem buying a minion of a tribe already on the board (2+ of them) 2x, buying a card that pairs with one on the board (13k synergy pairs) 3x.
rem Log: bgrl\runs\v3\log.jsonl ; opponent snapshots: bgrl\runs\v3\snapshots
rem Other settings: train.bat --help   (e.g. train.bat --threads 4 if the game stutters)
setlocal
cd /d "%~dp0"
.venv\Scripts\python -c "import orjson" 2>nul || .venv\Scripts\python -m pip install -q orjson 2>nul
.venv\Scripts\python -X utf8 bg.py demos --if-stale
set "INIT="
if not exist runs\v3\latest.pt if exist runs\v2\latest.pt set "INIT=--init runs\v2\latest.pt"
.venv\Scripts\python -X utf8 bg.py train --run runs\v3 --resume %INIT% --iters 100000 --threads %NUMBER_OF_PROCESSORS% --boost-frac 0.75 --boost-kind pro --boost-from-turn 5 --tempo-curve pro --bc-coef 0.05 --bc-weights "upgrade:3,sell:3,buy:2" --bc-train-cap 0.6 --bc-smooth 0.1 --pro-coef 0.1 --pro-batch 64 --pro-weights "upgrade:3,end:2,buy:1.5,sell:1.5,key:2,eco:2,tribe:2,syn:3,t2-5:2" --tempo-bonus 0.03 --tempo-deficit 0.03 --refresh-waste-penalty 0.015 --upgrade-starve-penalty 0.08 --starve-gift-weight 2 --starve-buy-weight 0 --unspent-gold-penalty 0.005 --pro-gap-cap 0.12 --pro-data "runs\pro_demos;runs\h13_demos" --pro-boards "runs\pro_boards.json;runs\h13_boards.json" %*
pause
