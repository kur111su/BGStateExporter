@echo off
chcp 65001 >nul
rem Local viewer for arena games: http://127.0.0.1:8767/ (results, Elo, step-by-step records, live game). Close the window to stop it.
cd /d "%~dp0"
set "PY=arena\.venv\Scripts\python.exe"
if not exist %PY% set "PY=.venv\Scripts\python.exe"
start "" http://127.0.0.1:8767/
%PY% -X utf8 arena\viewer.py
pause
