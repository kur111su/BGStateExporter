@echo off
chcp 65001 >nul
rem Shadow mode: the model's plan for your current recruit turn, shown in the plugin panel (http://localhost:8765/)
rem and on its own page (http://localhost:8766/). Read-only: it never touches the game. Close this window to stop.
rem Uses bgrl\runs\v2\latest.pt and switches to a newer one between games.
rem Another checkpoint: shadow.bat --ckpt runs\v2\snapshots\it00300.pt
setlocal
cd /d "%~dp0"
.venv\Scripts\python -X utf8 bg.py shadow %*
pause
