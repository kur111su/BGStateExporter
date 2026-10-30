@echo off
chcp 65001 >nul
rem Build the 6500+ decision samples (runs\pro_demos) from the Firestone share package.
rem   prodemos.bat                    reads bgrl\data\firestone (put the 5 zip files there, no need to unzip)
rem   prodemos.bat D:\some\folder     reads that folder instead
rem Takes about 10-20 minutes. The samples stay on this PC (runs\ is not in Git).
setlocal
cd /d "%~dp0"
set "SRC=%~1"
if "%SRC%"=="" set "SRC=data\firestone"
.venv\Scripts\python -X utf8 bg.py prodemos "%SRC%" %2 %3 %4
pause
