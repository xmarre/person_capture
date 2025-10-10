# PersonCapture — target-person crops from video

**Goal:** Detect and crop **one specific person** from a video into clean image datasets (e.g., 2:3, 1:1, 3:2). Best identity precision is achieved in **ArcFace-only** mode.

- **Intended mode:** `Match mode = face_only`, **ArcFace = on**, **Disable ReID = on`. :contentReference[oaicite:0]{index=0}
- **Recommended:** Use the optional **ONNX Runtime 1.24 + TensorRT EP** for faster ArcFace on NVIDIA. See **`README_onnxruntime_1.24_trt_windows.md`**.
- **Outputs:** Crops in `output/crops`, optional previews in `output/annot`, and debug JSONL in `output/debug` when enabled. :contentReference[oaicite:1]{index=1}

---

## What the application does

End-to-end on each video:

1. **Pre-scan** (fast pass, enabled by default): sample frames sparsely to find time **spans** that likely contain the target and to **grow a face reference bank** from confident hits. :contentReference[oaicite:2]{index=2}
2. **Main pass**: process only the kept spans at the chosen frame stride. For each frame:
   - Detect **persons**. :contentReference[oaicite:3]{index=3}
   - Inside each person region, detect **face** and compute an **ArcFace** embedding when a face is present. :contentReference[oaicite:4]{index=4}
   - Compare to the **reference bank**; require face visibility/quality; accept if ArcFace distance ≤ threshold. :contentReference[oaicite:5]{index=5}
   - Pick the best **crop ratio** and place the crop to avoid half-faces, excessive headroom, and missing torso. 
   - Save crop and optional annotated preview; de-duplicate with IoU and time-gap gates. :contentReference[oaicite:7]{index=7}

---

## How it works (pipeline details)

### Detectors and embedders
- **Person detection:** YOLOv8 persons; model fused; uses FP16 on CUDA when possible; threshold `min_det_conf`. :contentReference[oaicite:8]{index=8}
- **Face detection:** YOLOv8-face per person ROI with padding `face_det_pad` and confidence `face_det_conf`. :contentReference[oaicite:9]{index=9}
- **Identity (intended):** **ArcFace ONNX** cosine distance against a normalized bank. First-run auto-download of `arcface_r100.onnx`; Windows adds CUDA/cuDNN/TensorRT DLL dirs; TensorRT EP can be used via ONNX Runtime. :contentReference[oaicite:10]{index=10}
- **Optional ReID:** OpenCLIP (default ViT-L-14) body embedding; disabled in intended mode. :contentReference[oaicite:11]{index=11}

### Matching and gating
- **Match mode:** `face_only` means a valid face must be detected and **ArcFace** distance ≤ `face_thresh`. :contentReference[oaicite:12]{index=12}
- **Quality gate:** frames with face quality below `face_quality_min` are rejected for matching/bank adds. :contentReference[oaicite:13]{index=13}
- **Locking & stability:** short-lived lock with IoU and timing to avoid jitter; `only_best`, `min_gap_sec`, `iou_gate` reduce near-dupes. :contentReference[oaicite:14]{index=14}
- **No-face frames:** in intended mode they are skipped to keep identity purity (`require_face_if_visible=True`). :contentReference[oaicite:15]{index=15}

### Crop placement and ratio choice
- Evaluate each candidate ratio in `ratio` (e.g., `2:3,1:1,3:2`) by expanding the person box, then score:
  - **Score = area growth + λ × placement penalty**. Penalty discourages side-cut faces (insufficient side margins), excess top headroom, and missing lower torso. Tuned by `crop_penalty_weight`, `crop_face_side_margin_frac`, `crop_top_headroom_max_frac`, `crop_bottom_min_face_heights`. :contentReference[oaicite:16]{index=16}
- Enforce **anti-zoom** guards: `face_max_frac_in_crop`, optional `face_min_frac_in_crop`, and `crop_min_height_frac`. :contentReference[oaicite:17]{index=17}
- Bias center **downward** from face center by `face_anchor_down_frac` to include torso. :contentReference[oaicite:18]{index=18}
- Clamp to frame and maintain exact ratio (`expand_box_to_ratio`). :contentReference[oaicite:19]{index=19}

---

## Pre-scan: purpose, behavior, benefits

**Purpose:** accelerate and stabilize the main pass via a cheap identity sweep that also improves the reference bank.

**Behavior:**
- Sample interval `prescan_stride` with optional downscale to `prescan_max_width`. :contentReference[oaicite:20]{index=20}
- Use face detect + ArcFace with hysteresis:
  - **ENTER** segment if best face distance ≤ `prescan_fd_enter`.
  - **EXIT** segment when distance ≥ `prescan_fd_exit` or after negatives accumulate. :contentReference[oaicite:21]{index=21}
- **Bank growth:** add a normalized face vector when distance ≤ `prescan_fd_add`, quality ≥ `face_quality_min`, and cooldown via `prescan_add_cooldown_samples`; dedupe by cosine similarity threshold; cap bank size. :contentReference[oaicite:22]{index=22}
- Pad each span by `prescan_pad_sec`, require `prescan_min_segment_sec`, and **bridge** small gaps via `prescan_bridge_gap_sec`. :contentReference[oaicite:23]{index=23}
- Return `(spans, updated_ref_bank)`; main pass processes only those spans. :contentReference[oaicite:24]{index=24}

**Benefits:**
- **Speed:** skip dead regions; fewer full-res evaluations. :contentReference[oaicite:25]{index=25}
- **Accuracy:** stronger **ArcFace bank** before the main pass reduces false accepts. :contentReference[oaicite:26]{index=26}
- **Stability:** hysteresis + padding reduce choppy open/close on borderline frames. :contentReference[oaicite:27]{index=27}

---

## Recommended identity setup (ArcFace-only)

- **GUI → Settings → Matching**
  - `face_only`, **ArcFace on**, **Disable ReID on**. Start `face_thresh` ≈ **0.28–0.38**; lower is stricter. :contentReference[oaicite:28]{index=28}
- Provide several clean reference photos. Avoid blur and heavy occlusion.
- No-face frames are intentionally skipped in this mode. :contentReference[oaicite:29]{index=29}

---

## Performance: ONNX Runtime 1.24 + TensorRT (optional)

- Use the ORT 1.24 build with TensorRT EP; follow **`README_onnxruntime_1.24_trt_windows.md`**. TensorRT ZIP 10.13.3.9 is preferred; just extract and set the GUI **TensorRT folder**. Pip wheels for TensorRT may also work; untested here. :contentReference[oaicite:30]{index=30}

Verify providers:
```bat
python -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())"
# expect: ['TensorrtExecutionProvider','CUDAExecutionProvider','CPUExecutionProvider']
