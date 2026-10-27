@echo off
rem Step B3 setup: install PyTorch (CUDA) into bgrl\.venv, run the RL tests, a tiny training run and a speed test.
rem Needs step_a1_taverngym.bat to have been run once (TavernGym wheel installed).
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo .venv not found - please run step1_bb_check.bat and step_a1_taverngym.bat first.
  pause
  exit /b 1
)
.venv\Scripts\python -c "import taverngym" 2>nul
if errorlevel 1 ( echo TavernGym not installed - run step_a1_taverngym.bat first. & pause & exit /b 1 )
.venv\Scripts\python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>nul
if errorlevel 1 (
  echo Installing PyTorch with CUDA 12.4 (about 2.5 GB download^) ...
  .venv\Scripts\python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu124
  if errorlevel 1 ( echo PyTorch install failed. & pause & exit /b 1 )
)
.venv\Scripts\python -c "import torch; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
if not exist reports mkdir reports
echo.
echo === 1/3 tests ===
.venv\Scripts\python -X utf8 tests\test_rl.py
if errorlevel 1 ( echo Tests failed. & pause & exit /b 1 )
echo.
echo === 2/3 tiny training run ===
.venv\Scripts\python -X utf8 tools\train.py --smoke
if errorlevel 1 ( echo Smoke run failed. & pause & exit /b 1 )
echo.
echo === 3/3 speed test (3 iterations, default size) ===
.venv\Scripts\python -X utf8 tools\train.py --bench --threads %NUMBER_OF_PROCESSORS% > reports\b3_bench.txt 2>&1
type reports\b3_bench.txt
echo.
echo Saved to bgrl\reports\b3_bench.txt
pause
