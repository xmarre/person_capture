# PersonCapture — target‑person crops from video

**Goal:** Build clean image datasets of **one specific person** from video. Best identity precision is with **ArcFace‑only**.

- **Intended mode:** `Match mode = face_only`, **ArcFace = on**, **Disable ReID = on**.
- **Acceleration (optional):** ONNX Runtime 1.24 + TensorRT EP. See `README_onnxruntime_1.24_trt_windows.md`. A TensorRT **ZIP** install is preferred; a **pip** TensorRT may also work (untested here).

---

## What it does

1) **Pre‑scan**: a fast, low‑duty pass to find **time spans** that likely contain the target and to optionally **grow** a strong **face reference bank** under strict gates.  
2) **Main pass**: processes only the kept spans at your frame stride, validates identity with ArcFace, and writes well‑framed crops at chosen ratios.

Outputs:
- Crops → `output/crops`
- Optional annotated previews → `output/annot`
- Optional debug JSONL → `output/debug/…`

---

## How it works

### Detection
- **Person detector**: YOLOv8 persons. Controls: `min_det_conf`, device (CPU/CUDA). Half precision is used on CUDA when safe.
- **Face detector**: InsightFace SCRFD (default) or YOLOv8‑face inside each person ROI. Controls: `face_det_conf` (default 0.5), `face_det_pad` (padding so the jawline is not cut).
  > On the first run, InsightFace auto-downloads SCRFD and ArcFace models into its cache directory. YOLOv8-face runs typically use `face_det_conf` in the **0.15–0.30** range.

### Identity (intended best‑ID path)
- **ArcFace ONNX** embeddings compared by cosine distance against a **normalized reference bank**.
- Frames without a reliable **visible face** are **skipped** in intended mode to avoid drift.
- `face_thresh` controls strictness. Lower = stricter. Typical range: **0.28–0.38** depending on material.

> Optional ReID (OpenCLIP body features) exists but is not part of the intended best‑ID configuration.

### Matching and stability
- **Face‑only gate**: accept only if a valid face is present **and** ArcFace distance ≤ `face_thresh`.
- **Locking**: short temporal lock after a hit to dampen jitter.
- **De‑dup**: `only_best`, `min_gap_sec`, and `iou_gate` reduce near‑duplicate crops around the same moment.

### Cropper and ratio choice
For each accepted person detection:
1. Try every ratio in your list (e.g., `2:3,1:1,3:2`) by expanding the person box.
2. Score = **area growth** + **penalty**. The penalty discourages
   - side‑cut faces (insufficient lateral margin around the face),
   - excessive **top headroom**,
   - missing **lower torso**.
3. Guards: `face_max_frac_in_crop`, `crop_min_height_frac`, optional `face_min_frac_in_crop` prevent over‑zoom/under‑zoom.
4. Bias the crop **downwards** from the face center (`face_anchor_down_frac`) to include torso.
5. Clamp to frame and write the crop at exact ratio.

---

## Pre‑scan in depth

**Purpose:** Make the main pass faster and more accurate by pruning dead time and improving the reference bank before full processing.

**Behavior**
- Samples every `prescan_stride` frames and may downscale to `prescan_max_width`.
- Uses **face detect + ArcFace** with **hysteresis**:
  - **ENTER** a span when best distance ≤ `prescan_fd_enter`.
  - **EXIT** the span when distance ≥ `prescan_fd_exit` (stricter), or after sustained negatives.
- **Bank growth** (optional): add the current face vector when distance ≤ `prescan_fd_add` **and** `face_quality_min` is met. Dedupe by cosine similarity. A cooldown prevents flooding. Cap the bank.
- Pad each span by `prescan_pad_sec`, require `prescan_min_segment_sec`, and **bridge** short gaps with `prescan_bridge_gap_sec` to avoid over‑fragmentation.
- Result: `(spans, updated_ref_bank)`. The main pass runs **only** on `spans`.

**Benefits**
- **Speed**: Skip irrelevant sections. Fewer full‑res evaluations.
- **Accuracy**: Stronger face bank reduces false accepts in the main pass.
- **Stability**: Hysteresis + padding smooths choppy open/close around borderline frames.

**Recommended policy**
- Build your bank **during pre‑scan** under strict gates. Keep **runtime bank growth off** for the main pass to avoid drift. If you enable runtime adds, apply the **same gates** as pre‑scan.

---

## Quick start (Windows, Python 3.12, venv `env`)

```bat
cd /d C:\Users\marre\source\repos\person_capture
py -3.12 -m venv env
call env\Scripts\activate
python -m pip install -U pip
pip install -r requirements.txt
pip install <path-to-wheel>\onnxruntime\build\Windows\Release\Release\dist\onnxruntime_gpu-1.24.0-cp312-cp312-win_amd64.whl
python person_capture\gui_app.py
```

> Replace `<path-to-wheel>` with the folder from `README_onnxruntime_1.24_trt_windows.md` after downloading the packaged build.

CLI example:
```bat
python -m person_capture.main ^
  --video path\to\video.mp4 ^
  --ref path\to\person.jpg ^
  --out out_dir ^
  --ratio 2:3 ^
  --frame-stride 2 ^
  --min-det-conf 0.35 ^
  --face-thresh 0.32 ^
  --device cuda
```

---

## Performance: ONNX Runtime 1.24 + TensorRT (optional)

- Follow `README_onnxruntime_1.24_trt_windows.md` for the **unofficial ORT 1.24** wheel and version matrix.
- **TensorRT source**: Prefer NVIDIA’s ZIP **10.13.3.9**. Extract and set **Settings → Backends → TensorRT folder** to that directory.  
- **Alternative**: A **TensorRT pip wheel** may work. If you use it, point the GUI to `env\Lib\site-packages\tensorrt\lib`. **Untested by the author.**

Verification:
```bat
python -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())"
# expect: ['TensorrtExecutionProvider','CUDAExecutionProvider','CPUExecutionProvider']
```

---

## Key controls

- **Ratio list** (`2:3,1:1,3:2`), **frame stride**, **min_det_conf**.
- **Face visible gate**, **face_quality_min**, **face_thresh**.
- **De‑dup**: `only_best`, `min_gap_sec`, `iou_gate`.
- **Crop heuristics**: `crop_face_side_margin_frac`, `crop_top_headroom_max_frac`, `crop_bottom_min_face_heights`, `face_max_frac_in_crop`, `crop_min_height_frac`, `face_anchor_down_frac`.
- **Pre‑scan knobs**: `prescan_stride`, `prescan_max_width`, `prescan_face_conf`, `prescan_fd_enter`, `prescan_fd_add`, `prescan_fd_exit`, `prescan_pad_sec`, `prescan_bridge_gap_sec`, `prescan_min_segment_sec`.

---

## Troubleshooting

- **`nvinfer.dll not found`**: In the GUI set **TensorRT folder** to the directory that contains `lib\nvinfer*.dll` (e.g., `D:\tensorrt\TensorRT-10.13.3.9`). No PATH edits needed.
- **CPU‑only providers**: Install the ORT+TRT wheel and confirm compatible NVIDIA driver.
- **First seek pauses**: ORT session warm‑up or TensorRT engine build. Later seeks are faster.
- **Missed faces on extreme close‑ups**: Raise `face_det_conf` slightly and check `face_max_frac_in_crop`.


### HDR passthrough debug checklist

1. **Confirm P010 planes carry signal.** Enable DEBUG logging (`set PERSON_CAPTURE_LOG_LEVEL=DEBUG`) and watch the `HDR: P010 … min/max` lines emitted by `HDRPreviewWidget`. If Y/UV min & max both read `0`, FFmpeg’s passthrough pipe is feeding zeros and the shader will only show the clear color.
2. **Validate stride + plane sizes.** The preview widget now warns when the provided width/height do not match the numpy plane shapes, or when the UV plane is too small. Fix the upstream dimensions before calling into `pc_hdr_upload_p010`.
3. **Trace the DLL upload.** Set `PC_HDR_TRACE_UPLOAD=1` before launching the GUI to emit `OutputDebugStringA` markers before/after the CPU repack inside `uploadP010ToBuffers`. If the function throws because of a bad stride or size, the trace shows where it failed.
4. **Check the Vulkan draw path.** If the upload looks good, confirm the fullscreen-triangle path in `hdr_preview/pc_hdr_vulkan.cpp` renders: `recordCommandBuffer` clears to black, binds the descriptors, pushes the width/height constants, and issues `vkCmdDraw(cmd, 3, 1, 0, 0)`. When that draw does not run you will only see the clear color.

---

## Legal

You are responsible for content you process. InsightFace, ONNX Runtime, TensorRT, CUDA, and YOLO models are licensed by their owners. Do not redistribute NVIDIA binaries.
