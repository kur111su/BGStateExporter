@echo off
chcp 65001 >nul
rem Comp test with formed comps: same as comptest.bat (against real high-MMR boards, full strength), but in every game
rem the model's board is replaced at turn 8, 9 or 10 by a formed S/A comp a high-MMR player had at that turn
rem (one that this lobby's tribes allow); then the model plays on by itself. 256 games, about 1-2 minutes.
rem Shows how far the model gets once it HAS a formed comp (per comp, and board strength per turn).
rem Result goes to runs\v3\comptest_seeded.json (comptest.json is left alone).
rem Other settings: comptest_seeded.bat --seed-comps 9,10,11   or   comptest_seeded.bat --ckpt runs\v3\best.pt
setlocal
cd /d "%~dp0"
.venv\Scripts\python -X utf8 bg.py goldflow --boosted --games 256 --seed-comps 8,9,10 %*
pause
