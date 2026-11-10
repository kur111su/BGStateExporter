@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
rem Cleanup of old files. Close the training window first (train.bat), then run this.
rem Deletes:
rem   runs\_bench, runs\_smoke, runs\main, runs\v2, runs\pro_demos_old   old runs and test runs
rem   runs\v3\snapshots   snapshots before iteration 19000 (every 1000th one is kept; cleanup.bat 20000 = another cut-off)
rem   leftovers of removed features (shadow mode, replay check, your-own-game demos, plugin sources)
rem Keeps: runs\v3 (latest.pt, best.pt, log, dashboard), teacher.pt, pro_demos, h13_demos, the board libraries.
rem The next train.bat drops the deleted snapshots from the opponent pool by itself.
set KEEP_FROM=19000
if not "%~1"=="" set KEEP_FROM=%~1

echo This deletes old runs (_bench, _smoke, main, v2, pro_demos_old), runs\v3 snapshots before
echo iteration %KEEP_FROM% (every 1000th is kept) and files of removed features.
echo Close the training window first.
choice /c YN /m "Delete now"
if errorlevel 2 exit /b 0

for %%D in (_bench _smoke main v2 pro_demos_old) do if exist "runs\%%D" rmdir /s /q "runs\%%D"
for %%F in (runs\comptest.json runs\demos.npz runs\v3\best_it15450.pt runs\v3\comptest_look.json runs\v3\comptest_seeded_look.json) do if exist "%%F" del /q "%%F"

set KEPT=0
set GONE=0
for %%F in (runs\v3\snapshots\it?????.pt) do (
  set "N=%%~nF"
  set "N=!N:~2!"
  set /a NUM=1!N! - 100000
  set /a MOD=NUM %% 1000
  if !NUM! GEQ %KEEP_FROM% (
    set /a KEPT+=1
  ) else if !MOD! EQU 0 (
    set /a KEPT+=1
  ) else (
    del /q "%%F"
    set /a GONE+=1
  )
)
echo Snapshots: deleted !GONE!, kept !KEPT!.

for %%F in (shadow.bat replay.bat comptest_lookahead.bat rl\advisor.py tests\test_shadow.py tests\fixtures\replay_v3.jsonl.gz tools\shadow.py tools\shadow_eval.py tools\tg_replay.py tools\make_demos.py tools\make_fixtures.py tools\make_replay_fixture.py reports\a5_replay.json reports\a5_replay.txt) do if exist "%%F" del /q "%%F"
if exist reports\shadow rmdir /s /q reports\shadow
for %%D in (plugin docs examples) do if exist "..\%%D" rmdir /s /q "..\%%D"
echo Done.
pause
