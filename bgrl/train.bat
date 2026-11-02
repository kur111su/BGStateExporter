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
rem Comp bonus: from turn 6, each end of turn earns up to 0.08 for a formed board (main tribe share, 13k synergy pairs, key cards).
rem Log: bgrl\runs\v3\log.jsonl ; opponent snapshots: bgrl\runs\v3\snapshots
rem All training settings are in configs\v3.json (what each one does is under "notes" there).
rem Other settings: train.bat --help ; anything given here overrides the file (e.g. train.bat --threads 4 if the game stutters)
setlocal
cd /d "%~dp0"
.venv\Scripts\python -c "import orjson" 2>nul || .venv\Scripts\python -m pip install -q orjson 2>nul
.venv\Scripts\python -X utf8 bg.py demos --if-stale
set "INIT="
if not exist runs\v3\latest.pt if exist runs\v2\latest.pt set "INIT=--init runs\v2\latest.pt"
.venv\Scripts\python -X utf8 bg.py train --recipe configs\v3.json --resume %INIT% --threads %NUMBER_OF_PROCESSORS% %*
pause
