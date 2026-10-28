@echo off
chcp 65001 >nul
rem Step A5: replay every step of your recorded games (plugin v0.9+) in TavernGym and report where it disagrees.
rem Progress is shown per game. Report: bgrl\reports\a5_replay.txt (summary) and a5_replay.json (details).
rem Only the last N games: step_a5_replay.bat --last 5
setlocal
cd /d "%~dp0"
if not exist reports mkdir reports
.venv\Scripts\python -X utf8 tools\tg_replay.py "%APPDATA%\HearthstoneDeckTracker\BGStateExporter\games" --out reports\a5_replay.json --txt reports\a5_replay.txt %*
pause
