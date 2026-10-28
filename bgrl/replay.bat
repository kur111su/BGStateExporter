@echo off
chcp 65001 >nul
rem Replay check (A5): replays every step of your recorded games (plugin v0.9+) in TavernGym and reports where it disagrees.
rem Progress is shown per game. Report: bgrl\reports\a5_replay.txt (summary) and a5_replay.json (details).
rem Only the last N games: replay.bat --last 5
setlocal
cd /d "%~dp0"
.venv\Scripts\python -X utf8 bg.py replay %*
pause
