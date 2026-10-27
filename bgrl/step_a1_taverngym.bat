@echo off
rem Step A1: install the TavernGym wheel (built by our GitHub Actions from MCG-NJU/AutoTavern, MIT) and benchmark it.
rem Report goes to bgrl\reports\tg_bench_*.txt
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo .venv not found - please run step1_bb_check.bat once first.
  pause
  exit /b 1
)
for /f "usebackq delims=" %%T in (`.venv\Scripts\python -c "import sys;print('cp'+str(sys.version_info[0])+str(sys.version_info[1]))"`) do set "PYTAG=%%T"
for /f "usebackq tokens=2 delims==" %%C in (`findstr /b "commit=" taverngym\PIN.txt`) do set "TGCOMMIT=%%C"
set "TGTAG=taverngym-%TGCOMMIT:~0,7%"
set "WHL=taverngym-1.0.0-%PYTAG%-%PYTAG%-win_amd64.whl"
if not exist wheels mkdir wheels
if not exist "wheels\%WHL%" (
  echo Downloading %WHL% from release %TGTAG% ...
  powershell -NoProfile -Command "Invoke-WebRequest -UseBasicParsing -Uri https://github.com/kur111su/BGStateExporter/releases/download/%TGTAG%/%WHL% -OutFile wheels\%WHL%"
  if errorlevel 1 ( echo Download failed. & pause & exit /b 1 )
)
.venv\Scripts\python -m pip install -q numpy
if errorlevel 1 .venv\Scripts\python -m pip install -q numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
.venv\Scripts\python -m pip install -q --force-reinstall --no-deps "wheels\%WHL%"
if errorlevel 1 ( echo Wheel install failed. & pause & exit /b 1 )
.venv\Scripts\python -X utf8 tools\tg_bench.py %*
pause
