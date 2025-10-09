@echo off
setlocal
set "REPO=%~dp0"
pushd "%REPO%"

if not exist env (
  py -3 -m venv env || (
    echo Failed to create venv
    exit /b 1
  )
)

set "PIP=%REPO%env\Scripts\pip.exe"
"%PIP%" install --upgrade pip
"%PIP%" install -r requirements.txt

popd
endlocal
