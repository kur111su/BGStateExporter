@echo off
chcp 65001 >nul
rem Continuous RATED TavernArena games (--mode arena; rule bots fill empty seats), one game after another,
rem until you close this window or press Ctrl+C (a game already running finishes on the server; reconnect within 45 s or it forfeits).
rem   arena_loop.bat          always use the model packaged by arena.bat check
rem   arena_loop.bat update   before a game, re-package runs\v3\latest.pt if it is more than 2 hours newer than the packaged one
rem Results: arena\results\history.jsonl ; step-by-step records: arena\records\ ; watch them with arena_view.bat
setlocal EnableDelayedExpansion
cd /d "%~dp0"
set "URL=https://rizling.top"
if exist arena\url.txt set /p URL=<arena\url.txt
set "APY=arena\.venv\Scripts\python"
if not exist arena\.venv\Scripts\python.exe ( echo Run arena.bat check first. & pause & exit /b 1 )
if not exist arena\compat.json ( echo Run arena.bat check first. & pause & exit /b 1 )
if not defined ARENA_API_KEY if exist arena\key.txt set /p ARENA_API_KEY=<arena\key.txt
if not defined ARENA_API_KEY ( echo No Agent Key. Put it alone in bgrl\arena\key.txt & pause & exit /b 1 )
if not exist arena\results mkdir arena\results
set /a N=0
set /a FAILS=0

:loop
if /i "%~1"=="update" (
  %APY% -c "import os,sys; a='runs/v3/latest.pt'; b='arena/model_bundle/model.pt'; sys.exit(0 if os.path.exists(a) and (not os.path.exists(b) or os.path.getmtime(a) > os.path.getmtime(b) + 7200) else 1)"
  if not errorlevel 1 (
    echo Packaging the newer model ...
    .venv\Scripts\python -X utf8 arena\export.py
    if errorlevel 1 ( echo Packaging failed. & pause & exit /b 1 )
    %APY% -X utf8 arena\prep.py --url %URL%
    if errorlevel 1 ( echo Contract check failed - stopping. & pause & exit /b 1 )
  )
)
for /f "usebackq delims=" %%H in (`%APY% -c "import json;print(json.load(open('arena/compat.json',encoding='utf-8'))['catalog_sha256'])"`) do set "CAT=%%H"
for /f "usebackq delims=" %%V in (`%APY% -c "import json;print(json.load(open('arena/compat.json',encoding='utf-8'))['model_version'])"`) do set "MV=%%V"
set /a N+=1
echo.
echo ===== Rated game !N! with model !MV! (%DATE% %TIME%) =====
%APY% -E -s -X utf8 arena\arena_client.py --url %URL% --mode arena --games 1 --policy agent:act --catalog-sha256 !CAT! --model-version !MV! > arena\results\arena_last.json
set "RC=!ERRORLEVEL!"
%APY% -X utf8 -c "import pathlib;t=pathlib.Path('arena/results/arena_last.json').read_text(encoding='utf-8').strip();open('arena/results/history.jsonl','a',encoding='utf-8').write(t+'\n') if t else None"
%APY% -X utf8 arena\viewer.py --print-last arena\results\arena_last.json
if "!RC!"=="0" ( set /a FAILS=0 & goto loop )
if "!RC!"=="130" exit /b 130
set /a FAILS+=1
if !FAILS! GEQ 3 ( echo Three failures in a row - stopping. Send me this window. & pause & exit /b 1 )
echo The connector stopped with an error; trying again in 60 seconds ...
timeout /t 60 /nobreak >nul
goto loop
