# Unofficial ONNX Runtime 1.24-dev (TensorRT EP) — Windows

**Summary:** Unofficial ONNX Runtime `1.24.0` snapshot built from source with **TensorRT 10.13.3.9**, **CUDA 12.8**, **cuDNN 9**, **Python 3.12**, on **Windows 10/11**, tested on **RTX 5090**. Includes CUDA and TensorRT execution providers. This is **not** an official Microsoft release.

> Download wheel: https://pixeldrain.com/u/SFVVf9KU

---

## Quick start

1. **Activate your venv** (`env`):
   ```bat
   cd /d C:\Users\marre\source\repos\person_capture
   call env\Scripts\activate
   ```

2. **Install the wheel** (remove conflicting packages first):
   ```bat
   pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-gpu-tensorrt
   pip install -U <path-to-wheel>\onnxruntime\build\Windows\Release\Release\dist\onnxruntime_gpu-1.24.0-cp312-cp312-win_amd64.whl
   ```

3. **Tell the app where TensorRT is:**
   - **Preferred:** Download the NVIDIA ZIP for **TensorRT 10.13.3.9**, extract to e.g. `D:\tensorrt\TensorRT-10.13.3.9`, then in **PersonCapture** GUI set **Settings → Backends → TensorRT folder** to that path.
   - **Alternative:** Install NVIDIA’s **pip wheel** for TensorRT and point the GUI to your venv’s `...\Lib\site-packages\tensorrt\lib` folder.
   - **Note:** The pip-wheel route is **untested by the author**. In all cases, keep TensorRT **exactly** at **10.13.3.9** to match the ORT build.

4. **Run the app:**
   ```bat
   python person_capture\gui_app.py
   ```

---

## Why this exists

- PyPI wheels (`onnxruntime`, `onnxruntime-gpu`) do **not** ship the TensorRT EP.
- New stacks (CUDA 12.8 + cuDNN 9 + TensorRT 10.13) are not covered by older official builds.
- This wheel is a convenience snapshot for 50‑series GPUs.

---

## Versions

- **ORT:** `1.24.0` (development snapshot)
- **TensorRT:** `10.13.3.9` (pin this exact version)
- **CUDA Toolkit:** `12.8`
- **cuDNN:** `9.x` for CUDA 12.x
- **Python:** `3.12`
- **OS:** Windows 10/11 x64

> ABI and behavior may change before an official `v1.24.0` tag. Pin commit SHA if rebuilding.

---

## Verify install

```bat
python -c "import onnxruntime as ort; print('ORT', ort.__version__); print('providers', ort.get_available_providers())"
```
Expected:
```
['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

Optional build info:
```bat
python -c "import onnxruntime.capi._pybind_state as C; print(C.get_build_info())"
```

---

## TensorRT sources

**A) NVIDIA ZIP (recommended, tested)**  
1) Create/sign in to an NVIDIA Developer account.  
2) https://developer.nvidia.com/tensorrt/download/10x → download **TensorRT 10.13.3.9** for **Windows x86_64** (CUDA 12.8).  
3) Extract to e.g. `D:\tensorrt\TensorRT-10.13.3.9`.  
4) In **PersonCapture** GUI set **TensorRT folder** to that path.

**B) NVIDIA pip wheel (optional, untested by author)**  
- Install TensorRT from pip.  
- In **PersonCapture** GUI set **TensorRT folder** to your venv’s `...\Lib\site-packages\tensorrt\lib`.  
- Ensure the installed wheel version is **10.13.3.9**, to match this ORT build.

---

## Rebuild recipe (condensed)

For developers who want to reproduce the wheel:

```bat
git clone --recursive https://github.com/microsoft/onnxruntime.git
cd onnxruntime
git checkout <COMMIT_SHA_USED>

set "TRT_HOME=D:\tensorrt\TensorRT-10.13.3.9"
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "CUDNN_HOME=C:\Program Files\NVIDIA\CUDNN\cuda"
set "PATH=%TRT_HOME%\lib;%CUDA_HOME%\bin;%CUDNN_HOME%\bin;%PATH%"

python tools\ci_build\build.py ^
  --build_dir build ^
  --config Release ^
  --build_wheel ^
  --enable_pybind ^
  --use_cuda --cuda_version=12.8 --cudnn_home="%CUDNN_HOME%" ^
  --use_tensorrt --tensorrt_home="%TRT_HOME%" ^
  --parallel

dir build\Windows\Release\dist
```

---

## Legal

- ONNX Runtime is MIT-licensed. This wheel is unofficial.
- **Do not redistribute** NVIDIA binaries (TensorRT, CUDA, cuDNN). Share instructions only.
- Users must download TensorRT from NVIDIA under its license.

---

## Troubleshooting

- **`nvinfer.dll not found`**: In the GUI, set **TensorRT folder** to the directory containing `lib\nvinfer*.dll`.
- **`no available providers`**: Update NVIDIA driver and use Python 3.12. Reinstall the wheel in a clean venv.
- **Crashes on load**: Version mismatch between CUDA/cuDNN/TRT. Align to the versions listed above.
- **TensorRT engine won't rebuild**: Delete cached engines after toggling FP16 or resizing limits, e.g. `rd /s /q trt_cache` from a Developer Command Prompt.
