@echo off
setlocal
set "REPO=%~dp0"
pushd "%REPO%"

rem --- TensorRT absolute path ---
set "TRT_LIB_DIR=D:\tensorrt\TensorRT-10.13.3.9\lib"
set "PATH=%TRT_LIB_DIR%;%PATH%"

rem --- activate venv ---
set "VENV=%REPO%env"
call "%VENV%\Scripts\activate.bat"

rem --- CUDA/cuDNN DLLs from wheels (needed before ORT import) ---
set "PATH=%VENV%\Lib\site-packages\torch\lib;%PATH%"
for %%D in (cublas cudnn cuda_runtime cuda_nvrtc cufft curand cusolver cusparse) do (
  if exist "%VENV%\Lib\site-packages\nvidia\%%D\bin" set "PATH=%VENV%\Lib\site-packages\nvidia\%%D\bin;%PATH%"
)
if exist "%VENV%\Lib\site-packages\tensorrt" set "PATH=%VENV%\Lib\site-packages\tensorrt;%PATH%"
if exist "%VENV%\Lib\site-packages\tensorrt_libs" set "PATH=%VENV%\Lib\site-packages\tensorrt_libs;%PATH%"

rem --- quick sanity: required TRT parser DLLs present? ---
if not exist "%TRT_LIB_DIR%\nvonnxparser.dll" if not exist "%TRT_LIB_DIR%\nvonnxparser_10.dll" (
  echo ERROR: nvonnxparser*.dll missing in %TRT_LIB_DIR%
  popd & endlocal & exit /b 2
)

rem --- ORT providers must include TensorRT ---
python - <<^
import onnxruntime as ort
print("ORT providers:", ort.get_available_providers())
^

rem --- run GUI ---
python -Xfaulthandler -m person_capture.gui_app
popd
endlocal
