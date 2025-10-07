# agents.md — ChatGPT Code Agent Brief

## Mission
Deliver small, reviewable diffs that improve **PersonCapture**: detect a target person in video and export ratio‑true crops plus an index CSV. Keep behavior deterministic.

## Repo layout
- `detectors.py` — Ultralytics YOLOv8 wrapper. Person class only.
- `face_embedder.py` — Face detector + identity embeddings (ArcFace ONNX if available, else OpenCLIP).
- `reid_embedder.py` — Body embeddings via OpenCLIP. Returns L2‑normalized vectors.
- `utils.py` — Geometry, ratio boxing, borders, and helpers.
- `main.py` — CLI pipeline.
- `gui_app.py` — PySide6 GUI wrapper around the same pipeline.
- `preset.json` — Example GUI preset.

## Public API invariants
- `PersonDetector.detect(bgr, conf) -> List[{'xyxy': [x1,y1,x2,y2], 'conf': float, 'cls': 0}]`.
- `FaceEmbedder.extract(bgr) -> List[{'bbox': np.int32[4], 'feat': np.float32[D]|None, 'quality': float}]` + `FaceEmbedder.best_face(list)`.
- `ReIDEmbedder.extract([bgr,...]) -> List[np.float32[D]]` where each is L2‑normalized.
- `utils.expand_box_to_ratio(x1,y1,x2,y2, rw,rh, W,H, anchor=None, head_bias=0.0) -> (x1,y1,x2,y2)` exact ratio inside frame.

Do not silently change these; if you must, bump minor version and update all call sites.

## Tasks the agent should prioritize
1. Add tests (`pytest`) for: ratio expansion edge cases, border detection, score combining, and gating logic.
2. Add config validation and helpful error messages.
3. Keep imports robust for both flat‑file and package usage.
4. Maintain fast failure if heavy deps are missing. Do not auto‑install packages in CI.

## Coding rules
- Python 3.10+. Type hints + docstrings.
- Format with `black` (120 cols). Lint with `ruff`. Keep functions small and pure where possible.
- No noisy prints in library code; use GUI `status` signal or CLI prints only when helpful.

## CLI usage
```
python main.py --video /path/video.mp4 --ref /path/ref.jpg --out out_dir --ratio 2:3 --frame-stride 2
```

## GUI usage
```
python gui_app.py
```
Fill the fields or provide `preset.json` with matching keys.

## Matching logic summary
- Build reference: best face feature (if available) and body ReID feature (largest box or full image).
- Per frame: detect persons -> crop candidates -> face features per candidate -> ReID features batch.
- Score: combine face/reid distances using `min|avg|face_priority`.
- Thresholds: reject only if **both** distances exceed their thresholds.
- Expand to exact ratio that minimally enlarges the detection. Save crop and row in `index.csv`.

## Conventions
- Outputs go to `<out>/crops/` and `<out>/index.csv`. Optional `<out>/annot/` if annotation is enabled.
- Determinism: sort when ties are likely. Prefer minimal‑area expansion for ratio selection.
- Keep CUDA checks honest: never select CUDA if `torch.cuda.is_available()` is false.
