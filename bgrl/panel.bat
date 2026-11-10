@echo off
setlocal
cd /d "%~dp0"
rem The panel: a local web page with the training dashboard and a button for every task
rem (training, comp tests, data, arena, cleanup). Opens http://127.0.0.1:8770 in the browser; this PC only.
rem Closing this window closes the page; training and other running tasks keep going. Run it again to see them.
if not exist .venv\Scripts\python.exe (
  echo Run setup.bat first.
  pause
  exit /b 1
)
.venv\Scripts\python -c "import streamlit" 2>nul
if errorlevel 1 (
  echo Installing Streamlit, one time, about 200 MB ...
  .venv\Scripts\python -m pip install -q "streamlit>=1.37"
  if errorlevel 1 .venv\Scripts\python -m pip install -q "streamlit>=1.37" -i https://pypi.tuna.tsinghua.edu.cn/simple
)
.venv\Scripts\python -c "import orjson" 2>nul || .venv\Scripts\python -m pip install -q orjson 2>nul
.venv\Scripts\python -X utf8 bg.py panel %*
if errorlevel 1 pause
