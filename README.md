# PersonCapture — target‑person crops from video

**Purpose:** Find one specific person in a video and auto‑save clean, framed crops at fixed ratios (e.g., 2:3) for downstream editing.

## Key points

- **Intended mode:** Use **ArcFace** for identity and turn ReID off. This gives the strongest identity precision when a face is visible. In the GUI this is the default: `Match mode = face_only`, `Use ArcFace = on`, `Disable ReID = on`.
- **Backends:** Face detection uses YOLOv8‑face. Identity uses ArcFace ONNX by default. The app can run on CPU or CUDA. Optional: ONNX Runtime TensorRT EP for speed on NVIDIA.
- **Outputs:** Crops to `output/crops`, optional annotated previews to `output/annot`, index CSV with timestamps and scores.

## Quick start (Windows, Python 3.12, venv named `env`)

1. Open a terminal in your repo folder.
2. Create and activate the venv:
   ```bat
   py -3.12 -m venv env
   call env\Scripts\activate
   python -m pip install -U pip
   pip install -r requirements.txt
   ```
3. Run the GUI:
   ```bat
   python person_capture\gui_app.py
   ```

> Tip: You can also run the CLI tool if you prefer batch mode. See **CLI** below.

## Recommended identity setup (ArcFace‑only)

Use ArcFace as the single identity signal and gate crops on face presence/quality.

- **GUI → Settings → Matching**
  - **Match mode:** `face_only`
  - **Use ArcFace:** `on`
  - **Disable ReID:** `on`
  - **Face threshold:** start at `0.28–0.38` (lower is stricter). The app clamps to a safe maximum internally.
- **Reference image(s):** Provide one or more clear frontal shots of the target. Multiple refs help robustness. The app builds a small deduplicated feature bank.
- **When no face is visible:** The default behavior is to **skip** to avoid false crops. If you need continuity on profile/occluded frames, enable the optional “faceless fallback” in advanced settings, but this is not the intended best‑ID mode.

## Performance: ONNX Runtime 1.24 + TensorRT (optional)

For RTX 50‑series or newer NVIDIA GPUs you can accelerate ArcFace ONNX with ONNX Runtime’s TensorRT Execution Provider.

- Read: **`README_onnxruntime_1.24_trt_windows.md`** in this repo for the wheel and exact steps.
- After installing that wheel, download **TensorRT 10.13.3.9** from NVIDIA and **extract** it.
- In the GUI set **Settings → Backends → TensorRT folder** to your extracted directory, e.g. `D:\tensorrt\TensorRT-10.13.3.9`. No PATH edits required.

## Typical workflow

1. **Load video** and **reference image(s)** of the person.
2. Set **ratio list** (default `2:3,1:1,3:2`). The cropper chooses the best framing per hit with penalties to avoid half‑faces or excess headroom.
3. Start processing. Use the timeline and seek controls to review. Prescan can limit work to relevant segments.
4. Collect results from `output/crops`. Optional previews in `output/annot`. Debug JSONL in `output/debug/debug.jsonl` if enabled.

## Controls that matter

- **Frame stride:** Analyze every Nth frame. Higher is faster but may miss brief moments.
- **Min detection conf:** YOLO person threshold.
- **Face visible gate:** Require a solid face detection before considering a match.
- **Ratio list:** Comma‑separated e.g. `2:3,1:1,3:2`.
- **Crop placement:** Heuristics keep side margins, cap headroom, and bias downward to include torso. Anti‑over‑zoom guards keep the face from filling the whole crop.

## CLI (optional)

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

Notes:
- The GUI and CLI share the same detection and identity code paths.
- Thresholds are data‑dependent. Tighten or loosen based on your material.

## Troubleshooting

- **`nvinfer.dll not found`** when using TensorRT EP: In the GUI set **TensorRT folder** to the directory that contains `lib\nvinfer*.dll` (e.g., `D:\tensorrt\TensorRT-10.13.3.9`). No manual PATH edits needed.
- **Providers show CPU‑only:** Ensure you installed the ORT wheel with CUDA/TRT EP support and a matching NVIDIA driver. See the ORT+TRT README.
- **ArcFace model download failed:** The app tries multiple mirrors. Check connectivity or place `arcface_r100.onnx` next to the executable.
- **Startup warnings about unused initializers (ORT logs):** Benign graph‑clean messages from ONNX Runtime.

## Files and architecture (high level)

- Person detection: `person_capture/detectors.py` (YOLOv8 persons).
- Face identity: `person_capture/face_embedder.py` (YOLOv8‑face + ArcFace ONNX; optional ORT TensorRT path).
- ReID identity (optional): `person_capture/reid_embedder.py` (OpenCLIP). Disabled in intended mode.
- GUI and pipeline: `person_capture/gui_app.py` (processing thread, prescan, cropping, heuristics).
- Utilities: `person_capture/utils.py` (ratio math, black‑border detect, etc.).

## Legal

- You are responsible for the content you process.
- InsightFace, ONNX Runtime, TensorRT, CUDA, and YOLO models are licensed by their respective owners.
- Do not redistribute NVIDIA binaries. Download them from NVIDIA.



## Performance: ONNX Runtime 1.24 + TensorRT (optional)

For RTX 50-series or newer NVIDIA GPUs you can accelerate ArcFace ONNX with ONNX Runtime’s TensorRT Execution Provider.

- Read: **`README_onnxruntime_1.24_trt_windows.md`** in this repo for the wheel and exact steps.
- **TensorRT source:** Prefer the NVIDIA ZIP for **10.13.3.9** and just **extract** it.
- **Alternative:** NVIDIA’s **TensorRT pip wheel** may work. If used, point the GUI to your venv’s `...\Lib\site-packages\tensorrt\lib` folder. **Untested by the author.**
- In the GUI set **Settings → Backends → TensorRT folder** to the chosen directory, e.g. `D:\tensorrt\TensorRT-10.13.3.9`. No PATH edits required.
