@echo off
chcp 65001 >nul
rem Teacher model: imitates only the high-MMR samples (runs\pro_demos + runs\h13_demos), learning from scratch,
rem and keeps the version that does best on held-out games -> runs\teacher.pt (stays on this PC; runs\ is not in Git).
rem train.bat then uses it (teacher / teacher_coef in configs\v3.json): on the positions the model reaches itself,
rem it is pulled a little toward what the teacher would do. About 15-25 minutes on the GPU. Stop train.bat first.
rem Other settings: teacher.bat --steps 8000 ; teacher.bat --init runs\v3\latest.pt (start from the training model; it already memorised the samples, not recommended)
setlocal
cd /d "%~dp0"
.venv\Scripts\python -X utf8 bg.py teacher %*
pause
