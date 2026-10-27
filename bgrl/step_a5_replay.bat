@echo off
chcp 65001 >nul
rem Step A5: replay every step of your recorded games in TavernGym and report where it disagrees.
rem Report: bgrl\reports\a5_replay.json (+ summary printed below)
setlocal
cd /d "%~dp0"
if not exist reports mkdir reports
.venv\Scripts\python -X utf8 tools\tg_replay.py "%APPDATA%\HearthstoneDeckTracker\BGStateExporter\games" --out reports\a5_replay.json > reports\a5_replay.txt 2>&1
type reports\a5_replay.txt
pause
