@echo off
setlocal
cd /d "%~dp0"
rem The panel: a desktop program with the training dashboard, a button for every task
rem (training, comp tests, data, cleanup) and the arena games. No browser, no web page.
rem Closing the window does not stop training or other running tasks; open it again to see them.
rem (The old web version is still there: .venv\Scripts\python bg.py panel --web)
if not exist .venv\Scripts\python.exe (
  echo Run setup.bat first.
  pause
  exit /b 1
)
.venv\Scripts\python -c "import PySide6, pyqtgraph" 2>nul
if errorlevel 1 (
  echo Installing the window toolkit PySide6 and pyqtgraph, one time, about 200 MB ...
  .venv\Scripts\python -m pip install -q "PySide6>=6.6" "pyqtgraph>=0.13"
  if errorlevel 1 .venv\Scripts\python -m pip install -q "PySide6>=6.6" "pyqtgraph>=0.13" -i https://pypi.tuna.tsinghua.edu.cn/simple
)
.venv\Scripts\python -c "import orjson" 2>nul || .venv\Scripts\python -m pip install -q orjson 2>nul
.venv\Scripts\python -X utf8 -c "import panel.gui"
if errorlevel 1 (
  echo The panel could not start - send me this window.
  pause
  exit /b 1
)
start "" .venv\Scripts\pythonw.exe -X utf8 bg.py panel
