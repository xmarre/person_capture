@echo off
setlocal
set "REPO=%~dp0"
pushd "%REPO%"

if not exist ".git" (
  echo Not a git repo: %REPO%
  exit /b 1
)

rem Update repo (rebase onto upstream, keep your local changes stashed automatically)
git fetch --all --tags --prune
git pull --rebase --autostash
git submodule update --init --recursive

rem Update Python deps in the existing venv
set "PIP=%REPO%env\Scripts\pip.exe"
if not exist "%PIP%" (
  echo venv missing. Run setup_person_capture.bat
  exit /b 1
)
"%PIP%" install --upgrade -r requirements.txt

popd
endlocal
