@echo off
chcp 65001 >nul
rem Comp test: the current model (runs\v3\latest.pt) against bots playing real high-MMR boards (6500+ replays and 13k games),
rem 64 games. Prints gold flow per turn, final boards (tribes, synergy pairs, key cards, golden) next to 13k players,
rem and how the model picks cards compared with high-MMR players (key cards, economy cards).
rem Other settings: comptest.bat --games 128   or   comptest.bat --ckpt runs\v3\snapshots\it07900.pt
setlocal
cd /d "%~dp0"
.venv\Scripts\python -X utf8 bg.py goldflow --boosted %*
pause
