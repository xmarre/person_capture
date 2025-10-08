@echo off
setlocal enableextensions
cd /d "%~dp0"

REM Create venv if missing
if not exist "env\Scripts\python.exe" (
  echo [*] Creating venv "env"...
  py -3 -m venv env || (echo [!] Failed to create venv & exit /b 1)
)

echo [*] Activating venv...
call "env\Scripts\activate.bat" || (echo [!] Failed to activate venv & exit /b 1)

echo [*] Upgrading pip...
python -m pip install --upgrade pip wheel

REM Install deps (prefer requirements.txt if present)
if exist "requirements.txt" (
  echo [*] Installing requirements.txt...
  pip install -r requirements.txt || (echo [!] pip install failed & exit /b 1)
) else (
  echo [*] Installing baseline deps (no requirements.txt found)...
  pip install PySide6 opencv-python ultralytics open-clip-torch --upgrade
  REM If you need torch with CUDA, install the right build manually later.
)

echo [OK] Setup complete.
exit /b 0
