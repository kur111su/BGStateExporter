@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
rem 训练看板：画一次 runs\v3\dashboard.md 并用默认程序打开（训练时每 10 轮也会自动重画）
.venv\Scripts\python -X utf8 bg.py board --open %*
if errorlevel 1 pause
