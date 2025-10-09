@echo off
setlocal
set "REPO=%~dp0"
pushd "%REPO%"
set "VENV=%REPO%env"
set "PYTHONW=%VENV%\Scripts\pythonw.exe"

if not exist "%PYTHONW%" (
  echo Missing venv python: "%PYTHONW%"
  exit /b 1
)

rem Add CUDA/TensorRT runtime DLL folders from venv wheels
set "PATH=%VENV%\Lib\site-packages\torch\lib;%PATH%"
set "PATH=%VENV%\Lib\site-packages\tensorrt;%PATH%"
for /d %%D in ("%VENV%\Lib\site-packages\nvidia\*") do (
  if exist "%%D\bin" set "PATH=%%D\bin;%PATH%"
  if exist "%%D\lib" set "PATH=%%D\lib;%PATH%"
)

"%PYTHONW%" -m person_capture.gui_app
popd
endlocal
