@echo off
setlocal EnableExtensions
set "REPO=%~dp0"
pushd "%REPO%"
set "PERSON_CAPTURE_ROOT=%REPO%"

:: (Optional) Lift GitHub API rate limits for update checks.
:: set "GH_TOKEN=ghp_your_token_here"

set "VENV=%REPO%env"
set "TRT_LIB_DIR=D:\tensorrt\TensorRT-10.13.3.9\lib"
set "LOG=%REPO%last_run.log"
set "PYEXE=%VENV%\Scripts\python.exe"
if not exist "%PYEXE%" set "PYEXE=python"

set ORT_DISABLE_ALL_DEFAULT_EP=1
set ORT_LOG_SEVERITY_LEVEL=0
set ORT_TENSORRT_VERBOSE_LOGGING=1
set ORT_TENSORRT_DUMP_SUBGRAPHS=1

> "%LOG%" echo ==== PersonCapture %DATE% %TIME% ====

rem --- venv ---
if not exist "%VENV%\Scripts\activate.bat" (
  echo ERROR: venv missing at "%VENV%\Scripts\activate.bat" >>"%LOG%"
  goto FAIL
)
call "%VENV%\Scripts\activate.bat"

rem --- runtime DLLs (TRT first, then CUDA/cuDNN from wheels) ---
set "PATH=%TRT_LIB_DIR%;%VENV%\Lib\site-packages\torch\lib;%PATH%"
for %%D in (cublas cudnn cuda_runtime cuda_nvrtc cufft curand cusolver cusparse) do (
  if exist "%VENV%\Lib\site-packages\nvidia\%%D\bin" set "PATH=%VENV%\Lib\site-packages\nvidia\%%D\bin;%PATH%"
)
rem --- conservative TensorRT defaults ---
set PC_TRT_FP16=0
set PC_TRT_OPT=3
set PC_TRT_WS=4294967296

:: Force HDR pipe to output BGR instead of NV12 so we skip the manual NV12?BGR conversion.
set PC_PIPE_PIXFMT=bgr24

:: Prefer a BGRA readback format from libplacebo for BGR pipes.
set PC_LP_SW_FMT=bgra

rem --- must-have TRT DLLs ---
for %%F in (nvinfer.dll nvinfer_plugin.dll) do (
  if not exist "%TRT_LIB_DIR%\%%F" goto FAIL_MISS
)

rem --- optional (log if present, but do not fail) ---
for %%F in (nvonnxparser.dll nvonnxparser_10.dll) do (
  if exist "%TRT_LIB_DIR%\%%F" echo found %%F>>"%LOG%"
)

rem --- ORT must expose TensorrtExecutionProvider ---
"%PYEXE%" -c "import onnxruntime as ort,sys; p=ort.get_available_providers(); print('ORT providers:',p); sys.exit(0 if 'TensorrtExecutionProvider' in p else 42)" 1>>"%LOG%" 2>&1 || goto FAIL

rem --- run GUI; nonzero exit == fail ---
"%PYEXE%" -u -Xfaulthandler -m person_capture.gui_app 1>>"%LOG%" 2>&1 || goto FAIL

echo OK >>"%LOG%"
type "%LOG%"
goto END

:FAIL_MISS
echo ERROR: Missing required TensorRT DLLs in %TRT_LIB_DIR% >>"%LOG%"
:FAIL
type "%LOG%"

:END
popd
pause
endlocal
