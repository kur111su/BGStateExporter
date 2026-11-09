@echo off
chcp 65001 >nul
rem Lookahead test: plays the same 128 comp-test games twice with the same model (against real high-MMR boards, full strength):
rem   1) normally (writes runs\v3\comptest.json)
rem   2) from turn 9, every decision first tries the model's top 6 moves + end turn on a copy of the game (random stream
rem      replaced, so it cannot see the real next shop), lets the model finish the turn, measures the end-of-turn board
rem      against 12 same-turn high-MMR boards and keeps the strongest (writes runs\v3\comptest_look.json)
rem and prints both side by side: if the lookahead grows the late board clearly better, it becomes the late-game
rem teacher for training and the late-game advisor. About 10 minutes.
rem Other settings: comptest_lookahead.bat --ckpt runs\v3\best.pt   or   --games 64
setlocal
cd /d "%~dp0"
.venv\Scripts\python -X utf8 bg.py goldflow --boosted --games 128 %*
.venv\Scripts\python -X utf8 bg.py goldflow --boosted --games 128 --lookahead-from 9 %*
pause
