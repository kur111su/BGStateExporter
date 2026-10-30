@echo off
chcp 65001 >nul
rem Early-turn gold flow: the current model (runs\v3\latest.pt, 64 games vs bots) next to 6500+ players.
rem How much gold goes to upgrading, buying, refreshing, activating, Dark Gifts each turn; upgrade rate when affordable.
rem Other settings: goldflow.bat --games 128   or   goldflow.bat --ckpt runs\v3\snapshots\it04350.pt
setlocal
cd /d "%~dp0"
.venv\Scripts\python -X utf8 bg.py goldflow %*
pause
