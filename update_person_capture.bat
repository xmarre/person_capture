@echo off
setlocal enableextensions
cd /d "%~dp0"

REM Pull latest code if this is a git repo
if exist ".git" (
  echo [*] Updating repo...
  git pull --rebase || (echo [!] git pull failed & exit /b 1)
)

REM Ensure venv present
if not exist "env\Scripts\python.exe" (
  echo [!] venv missing. Run setup_person_capture.bat first.
  exit /b 1
)

call "env\Scripts\activate.bat" || (echo [!] Failed to activate venv & exit /b 1)

if exist "requirements.txt" (
  echo [*] Updating deps...
  pip install -r requirements.txt --upgrade || (echo [!] pip upgrade failed & exit /b 1)
)

echo [OK] Update complete.
exit /b 0
