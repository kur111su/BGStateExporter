@echo off
chcp 65001 >nul
rem Step C1 shadow mode: shows the model's plan for your current recruit turn at http://localhost:8766/
rem Read-only: it never touches the game. Keep HDT (with the plugin) running. Close this window to stop.
rem Uses bgrl\runs\main\latest.pt (the training run) and switches to a newer one between games.
rem Another checkpoint: step_c1_shadow.bat --ckpt runs\main\snapshots\it00050.pt
setlocal
cd /d "%~dp0"
start "" http://localhost:8766/
.venv\Scripts\python -X utf8 tools\shadow.py %*
pause
