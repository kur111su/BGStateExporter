@echo off
chcp 65001 >nul
rem One-time setup (safe to run again): Python venv, TavernGym simulator, PyTorch (CUDA), then tests and a tiny training run.
rem   setup.bat          install / update and check
rem   setup.bat bench    also measure training speed for the small / medium / large model (about 10 minutes)
rem Everything goes into bgrl\.venv and bgrl\wheels; the log is bgrl\reports\setup.txt (send it if something fails).
setlocal
cd /d "%~dp0"
if not exist reports mkdir reports
set "LOG=reports\setup.txt"
echo setup %DATE% %TIME% > %LOG%

rem --- 1. Python 3.10-3.13 and the venv ---
if not exist .venv\Scripts\python.exe (
  set "PY="
  for %%V in (3.12 3.13 3.11 3.10) do (
    if not defined PY ( py -%%V -c "import sys" >nul 2>nul && set "PY=py -%%V" )
  )
  if not defined PY ( python -c "import sys; sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,13) else 1)" >nul 2>nul && set "PY=python" )
  if not defined PY (
    echo Python 3.10-3.13 not found. Install Python 3.12 from https://www.python.org/downloads/ and tick "Add python.exe to PATH".
    pause
    exit /b 1
  )
  echo Creating .venv ...
  call %%PY%% -m venv .venv
  if errorlevel 1 ( echo Failed to create .venv & pause & exit /b 1 )
)
set "VPY=.venv\Scripts\python"
%VPY% -m pip install -q -r requirements.txt >> %LOG% 2>&1
if errorlevel 1 (
  echo Retrying pip with the Tsinghua mirror...
  %VPY% -m pip install -q -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple >> %LOG% 2>&1
  if errorlevel 1 ( echo pip install failed - see bgrl\%LOG% & pause & exit /b 1 )
)

rem --- 2. TavernGym wheel (built by our GitHub Actions from MCG-NJU/AutoTavern, MIT; commit pinned in taverngym\PIN.txt) ---
for /f "usebackq tokens=2 delims==" %%C in (`findstr /b "commit=" taverngym\PIN.txt`) do set "TGCOMMIT=%%C"
set "TGTAG=taverngym-%TGCOMMIT:~0,7%"
set "WHL=taverngym-1.0.0-cp310-abi3-win_amd64.whl"
if not exist wheels mkdir wheels
if not exist "wheels\%TGTAG%-%WHL%" (
  echo Downloading TavernGym %TGTAG% ...
  powershell -NoProfile -Command "Invoke-WebRequest -UseBasicParsing -Uri https://github.com/kur111su/BGStateExporter/releases/download/%TGTAG%/%WHL% -OutFile 'wheels\%TGTAG%-%WHL%'"
  if errorlevel 1 ( echo Download failed. & pause & exit /b 1 )
  copy /y "wheels\%TGTAG%-%WHL%" "wheels\%WHL%" >nul
  %VPY% -m pip install -q --force-reinstall --no-deps "wheels\%WHL%" >> %LOG% 2>&1
  if errorlevel 1 ( echo TavernGym install failed - see bgrl\%LOG% & pause & exit /b 1 )
)
%VPY% -c "import taverngym" 2>nul
if errorlevel 1 (
  copy /y "wheels\%TGTAG%-%WHL%" "wheels\%WHL%" >nul
  %VPY% -m pip install -q --force-reinstall --no-deps "wheels\%WHL%" >> %LOG% 2>&1
)

rem --- 3. PyTorch with CUDA ---
%VPY% -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>nul
if errorlevel 1 (
  echo Installing PyTorch with CUDA 12.4 (about 2.5 GB download^) ...
  %VPY% -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu124
  if errorlevel 1 ( echo PyTorch install failed. & pause & exit /b 1 )
)

rem --- 4. checks ---
%VPY% -X utf8 bg.py info
%VPY% -X utf8 bg.py info >> %LOG% 2>&1
echo.
echo === tests ===
%VPY% -X utf8 bg.py test > reports\setup_step.txt 2>&1
set "RC=%ERRORLEVEL%"
type reports\setup_step.txt & type reports\setup_step.txt >> %LOG%
if not "%RC%"=="0" ( echo Tests failed - see bgrl\%LOG% & pause & exit /b 1 )
echo.
echo === tiny training run ===
%VPY% -X utf8 bg.py train --smoke > reports\setup_step.txt 2>&1
set "RC=%ERRORLEVEL%"
type reports\setup_step.txt & type reports\setup_step.txt >> %LOG%
if not "%RC%"=="0" ( echo Smoke run failed - see bgrl\%LOG% & pause & exit /b 1 )

if /i "%~1"=="bench" (
  echo.
  echo === speed test: small / medium / large, 3 iterations each ===
  for %%S in (small medium large) do (
    echo --- %%S ---
    %VPY% -X utf8 bg.py train --bench --size %%S --threads %NUMBER_OF_PROCESSORS% > reports\setup_step.txt 2>&1
    type reports\setup_step.txt & type reports\setup_step.txt >> %LOG%
  )
)
echo.
echo Setup OK. Next: train.bat (training), shadow.bat (model advice in the plugin panel), replay.bat (check the simulator).
pause
