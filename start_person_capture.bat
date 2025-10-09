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

rem === add NVIDIA/TensorRT runtime DLL dirs from venv wheels ===
set "PATH=%VENV%\Lib\site-packages\torch\lib;%PATH%"
set "PATH=%VENV%\Lib\site-packages\tensorrt;%PATH%"
set "PATH=%VENV%\Lib\site-packages\tensorrt_libs;%PATH%"
for /R "%VENV%\Lib\site-packages\nvidia" %%F in (nvinfer*.dll nvonnxparser*.dll nvinfer_plugin*.dll) do (
  set "PATH=%%~dpF;%PATH%"
)
for /R "%VENV%\Lib\site-packages\tensorrt_libs" %%F in (nvinfer*.dll nvonnxparser*.dll nvinfer_plugin*.dll) do (
  set "PATH=%%~dpF;%PATH%"
)

"%PYTHONW%" -m person_capture.gui_app
popd
endlocal
