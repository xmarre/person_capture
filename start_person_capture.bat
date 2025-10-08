@echo off
setlocal enableextensions
cd /d "%~dp0"

if not exist "env\Scripts\pythonw.exe" (
  echo [!] venv missing. Run setup_person_capture.bat first.
  exit /b 1
)

REM Launch GUI with pythonw so no terminal remains
start "" /min "env\Scripts\pythonw.exe" "person_capture\gui_app.py"
exit /b 0
