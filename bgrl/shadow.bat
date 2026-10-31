@echo off
chcp 65001 >nul
rem Shadow mode: the PPO model's plan for your current recruit turn, plus the plugin's Bob's Buddy results
rem (win rate, win rate vs each opponent, positioning button), on http://localhost:8766/ (the plugin has no panel of its own).
rem Read-only: it never touches the game. Close this window to stop.
rem Uses bgrl\runs\v3\latest.pt and switches to a newer one between games.
rem Another checkpoint: shadow.bat --ckpt runs\v3\snapshots\it07900.pt
setlocal
cd /d "%~dp0"
.venv\Scripts\python -X utf8 bg.py shadow %*
pause
