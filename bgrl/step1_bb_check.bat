@echo off
rem Step 1: check that Bob's Buddy runs outside HDT and measure its speed.
rem Report goes to bgrl\reports\bb_check_*.txt
setlocal
cd /d "%~dp0"
set "PY="
for %%V in (3.12 3.13 3.11 3.10) do (
  if not defined PY ( py -%%V -c "import sys" >nul 2>nul && set "PY=py -%%V" )
)
if not defined PY ( python -c "import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>nul && set "PY=python" )
if not defined PY (
  echo Python 3.10-3.13 not found. Install Python 3.12 from https://www.python.org/downloads/ and tick "Add python.exe to PATH".
  pause
  exit /b 1
)
echo Using %PY%
if not exist .venv\Scripts\python.exe (
  %PY% -m venv .venv
  if errorlevel 1 ( echo Failed to create .venv & pause & exit /b 1 )
)
.venv\Scripts\python -m pip install -q -r requirements.txt
if errorlevel 1 (
  echo Retrying with Tsinghua mirror...
  .venv\Scripts\python -m pip install -q -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
  if errorlevel 1 ( echo pip install failed & pause & exit /b 1 )
)
.venv\Scripts\python -X utf8 tools\bb_check.py %*
pause
