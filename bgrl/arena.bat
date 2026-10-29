@echo off
chcp 65001 >nul
rem TavernArena (the arena runs the open TavernGym engine; our model runs here and only actions are sent).
rem   arena.bat check      package the current model (runs\v3\latest.pt), verify the arena contract, run the official --check
rem   arena.bat practice   one unrated practice game (needs the Agent Key, see below)
rem   arena.bat check --ckpt runs\v3\snapshots\xxx.pt   package another model
rem Agent Key: set ARENA_API_KEY in this window, or put it alone in bgrl\arena\key.txt (git-ignored).
rem It is never printed, logged or put on a command line. Arena URL: bgrl\arena\url.txt overrides the default.
rem The connector runs in bgrl\arena\.venv (Python 3.12+, numpy, PyTorch CPU, deliberately without TavernGym).
setlocal
cd /d "%~dp0"
set "URL=https://rizling.top"
if exist arena\url.txt set /p URL=<arena\url.txt
set "MODE=%~1"
if "%MODE%"=="" set "MODE=check"
if /i not "%MODE%"=="check" if /i not "%MODE%"=="practice" (
  echo Unknown mode "%MODE%". Use: arena.bat check  or  arena.bat practice
  pause
  exit /b 1
)

rem --- arena\.venv ---
if not exist arena\.venv\Scripts\python.exe (
  set "PY="
  for %%V in (3.12 3.13 3.14) do (
    if not defined PY ( py -%%V -c "import sys" >nul 2>nul && set "PY=py -%%V" )
  )
  if not defined PY ( python -c "import sys; sys.exit(0 if sys.version_info[:2]>=(3,12) else 1)" >nul 2>nul && set "PY=python" )
  if not defined PY (
    echo The arena connector needs Python 3.12 or newer.
    pause
    exit /b 1
  )
  echo Creating arena\.venv ...
  call %%PY%% -m venv arena\.venv
)
set "APY=arena\.venv\Scripts\python"
%APY% -c "import numpy, torch" >nul 2>nul
if errorlevel 1 (
  echo Installing numpy and PyTorch CPU into arena\.venv ^(about 200 MB^) ...
  %APY% -m pip install -q numpy
  %APY% -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu
)
%APY% -c "import importlib.util,sys; sys.exit(1 if importlib.util.find_spec('taverngym') else 0)"
if errorlevel 1 ( echo arena\.venv must not contain TavernGym. Delete arena\.venv and run again. & pause & exit /b 1 )

if /i not "%MODE%"=="check" goto :have_bundle
echo [1/3] Packaging the model ...
.venv\Scripts\python -X utf8 arena\export.py %2 %3
if errorlevel 1 ( echo Packaging failed. & pause & exit /b 1 )
echo [2/3] Checking the arena contract and card vocabulary ...
%APY% -X utf8 arena\prep.py --url %URL%
if errorlevel 1 ( echo Contract check failed - send me bgrl\arena\compat.json & pause & exit /b 1 )

:have_bundle
if not exist arena\compat.json ( echo Run arena.bat check first. & pause & exit /b 1 )
for /f "usebackq delims=" %%H in (`%APY% -c "import json;print(json.load(open('arena/compat.json',encoding='utf-8'))['catalog_sha256'])"`) do set "CAT=%%H"
for /f "usebackq delims=" %%V in (`%APY% -c "import json;print(json.load(open('arena/compat.json',encoding='utf-8'))['model_version'])"`) do set "MV=%%V"
if not exist arena\results mkdir arena\results
if /i not "%MODE%"=="check" goto :practice
echo [3/3] Official connector --check (no queue) ...
%APY% -X utf8 arena\arena_client.py --url %URL% --check --policy agent:act --catalog-sha256 %CAT% > arena\results\check.json
set "RC=%ERRORLEVEL%"
type arena\results\check.json
pause
exit /b %RC%

:practice
rem --- needs the Agent Key ---
if not defined ARENA_API_KEY if exist arena\key.txt set /p ARENA_API_KEY=<arena\key.txt
if not defined ARENA_API_KEY (
  echo No Agent Key. Put it alone in bgrl\arena\key.txt, or run:  set ARENA_API_KEY=...  then arena.bat practice
  pause
  exit /b 1
)
echo Practice game (unrated) with model %MV% ...
%APY% -X utf8 arena\arena_client.py --url %URL% --mode practice --policy agent:act --catalog-sha256 %CAT% --model-version %MV% > arena\results\practice_last.json
set "RC=%ERRORLEVEL%"
type arena\results\practice_last.json
%APY% -X utf8 -c "import pathlib;p=pathlib.Path('arena/results/practice_last.json');t=p.read_text(encoding='utf-8').strip();open('arena/results/history.jsonl','a',encoding='utf-8').write(t+'\n') if t else None"
pause
exit /b %RC%
