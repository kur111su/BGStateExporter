@echo off
chcp 65001 >nul
rem TavernBench (C3): the 368-task evaluation set of the AutoTavern paper (bgrl\third_party\tavernbench, MIT).
rem Answers it with our model and compares with TavernAgent, language models and Jev. CPU only; fine while training runs.
rem Report: bgrl\reports\tavernbench\<model>\report.txt
rem Another model: tavernbench.bat --ckpt runs\v2\latest.pt
setlocal
cd /d "%~dp0"
.venv\Scripts\python -X utf8 bg.py tavernbench %*
pause
