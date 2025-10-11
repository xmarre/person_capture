# agents.md — ChatGPT Codex Agent Guide

## Mission
Own code-quality and feature work for **PersonCapture**: a video-to-crops tool that finds a target person and exports 2:3 crops with a CSV index and optional GUI control.

Deliver small, safe, reviewable diffs. Keep behavior deterministic and reproducible.

## Repository map
- `detectors.py`: `PersonDetector` wrapper around Ultralytics YOLOv8. Focus: robust weight loading, person-only detect, CUDA/CPU selection.
- `face_embedder.py`: `FaceEmbedder` with YOLOv8-face + embedding backends (OpenCLIP or ArcFace ONNX). Includes auto-download mirrors, face quality scoring, and `best_face` helper.
- `reid_embedder.py`: `ReIDEmbedder` using OpenCLIP for body embeddings; returns L2-normalized features.
- `utils.py`: geometry and I/O helpers such as `expand_box_to_ratio`, black border detection, normalization, etc.
- `main.py`: CLI pipeline: reference prep → per‑frame detection → face/ReID scoring → ratio‑consistent crop → CSV.
- `gui_app.py`: PySide6 GUI front-end that wraps the same pipeline with presets, live preview, cadence/lock logic, and status throttling.

## Core pipeline (reference)
1. Load models: `PersonDetector`, `FaceEmbedder`, `ReIDEmbedder`.
2. Build reference features:
   - Face: `FaceEmbedder.extract(...)` → pick `FaceEmbedder.best_face(...)` → `feat`.
   - ReID: detect person in the ref image, crop largest person region, `ReIDEmbedder.extract([...])`.
3. For each video frame (with stride):
   - Detect persons → build candidate crops.
   - Face features per candidate (optional, filtered by quality).
   - ReID features batch.
   - Combine with mode `min | avg | face_priority`, threshold with `face_thresh` / `reid_thresh`.
   - Expand chosen box to exact aspect ratio via `expand_box_to_ratio` and save crop + CSV.
4. (GUI) Optional lock logic: after N hits, tighten thresholds and add IoU gate; emit preview images and status updates.

## Coding standards
- Python 3.10+. Use type hints and concise docstrings.
- Follow EAFP style. Handle missing heavy deps with clear `RuntimeError` messages that point to `requirements.txt`.
- Maintain **both** import modes: package and flat-file (`_imp()` pattern in `gui_app.py`).
- Keep backends pluggable and device-aware (`cuda` if available else `cpu`).
- Use small, pure helpers with unit-testable logic. Avoid global state.
- Logging: prefer structured, throttled status updates in GUI; avoid noisy prints in libraries.
- Determinism: avoid nondeterministic ordering; sort candidates when ties are likely.

## Style and tooling
- Lint: `ruff` with default rules. Auto-format with `black` (120 columns). Type-check with `mypy` on library code.
- Tests: `pytest`. Unit-test `utils.py` geometry and black-border detection; smoke-test scoring and selection logic with small synthetic arrays. Stub heavy modules in tests (`torch`, `ultralytics`, `open_clip`, `onnxruntime`) via `pytest` monkeypatch or dummy classes.
- Docstrings: Google-style or NumPy-style. Include ranges and units for thresholds.

## Git workflow
- Branch from `feat/<slug>`, `fix/<slug>`, or `refactor/<slug>`.
- Conventional commits:
  - `feat(gui): add IoU gate slider`
  - `fix(utils): correct ratio shrink path for border clamp`
  - `refactor(face): factor quality computation`
- One logical change per PR. Keep PRs under ~300 LOC when feasible.
- Include before/after behavior notes in the PR description and a short “Validation” section.

## Public API invariants
- `PersonDetector.detect(frame, conf)` → `List[dict]` with `xyxy`, `conf`, `cls=0`.
- `FaceEmbedder.extract(bgr)` → `List[{'bbox': np.int32[4], 'feat': np.float32[D], 'quality': float}]` sorted by `(quality, area)` descending. Provide `best_face(...)`.
- `ReIDEmbedder.extract(bgr_list)` → `List[np.float32[D]]` L2‑normalized.
- `utils.expand_box_to_ratio(...)` returns an exact‑ratio box inside the frame, respecting optional `anchor` and `head_bias`.
- GUI signals in `Processor`: `setup(int,float)`, `progress(int)`, `status(str)`, `preview(QImage)`, `hit(str)`, `finished(bool,str)`.

Do not silently change these without bumping minor version and updating call sites.

## Guardrails for common edits
- **Model loading**: keep quarantine/recovery logic for corrupt `.pt` paths; default to hub names when needed.
- **Device selection**: never claim CUDA if `torch.cuda.is_available()` is false. Allow `device='cpu'` from config.
- **Quality gates**: any face‑first policies must respect `face_quality_min` and `prefer_face_when_available` flags.
- **Tie‑breaks**: when scores tie within `score_margin`, prefer single best candidate to reduce duplicates.
- **Aspect ratio**: always finalize crops to the exact target ratio before writing.
- **I/O**: write crops to `<out>/crops` and index CSV with consistent columns. Only create `annot/` when requested.

## Safe refactors and targets
- Extract common scoring into a shared helper to remove duplication between CLI and GUI.
- Add unit tests for:
  - `expand_box_to_ratio` edge cases near frame borders and tiny boxes.
  - IoU gate, lock momentum EMA, and cadence timing.
  - Black-border detection correctness on synthetic frames.
- Improve config serde for GUI presets (`SessionConfig.to_json/from_json`) and validate on load.
- Make `FaceEmbedder` backend selection explicit and introspectable (`backend in {'clip','arcface'}`).
- Add lightweight progress callback hooks across detectors/embedders.

## Performance notes
- Batch operations where possible (ReID embeddings already batched).
- Avoid repeated color conversions and memory copies in tight loops.
- Use integer boxing and pre-allocate buffers where practical in the GUI path.
- Keep OpenCLIP and ONNX on `.eval()` or provider-optimized inference paths.

## PR checklist (copy into every PR)
- [ ] Public API unchanged or documented with version bump.
- [ ] Unit tests added or updated. `pytest -q` green locally.
- [ ] `ruff --fix` and `black` applied. `mypy` clean on library code.
- [ ] GUI still runs and signals fire. No console spam. No blocking calls on the UI thread.
- [ ] CSV schema and crop paths verified on a short sample.

## Minimal code examples

**Detection**

```python
from detectors import PersonDetector
det = PersonDetector(model_name="yolov8n.pt", device="cuda")
people = det.detect(frame_bgr, conf=0.35)
```

**Face features**

```python
from face_embedder import FaceEmbedder
face = FaceEmbedder(ctx="cuda", yolo_model="yolov8l-face.pt", use_arcface=False)
faces = face.extract(crop_bgr)
best = FaceEmbedder.best_face(faces)
```

**ReID features**

```python
from reid_embedder import ReIDEmbedder
reid = ReIDEmbedder(device="cuda", model_name="ViT-L-14", pretrained="laion2b_s32b_b82k")
feats = reid.extract([crop1_bgr, crop2_bgr])
```

**Box expansion to ratio**

```python
from utils import expand_box_to_ratio
ex1, ey1, ex2, ey2 = expand_box_to_ratio(x1, y1, x2, y2, 2, 3, frame_w, frame_h, anchor=None, head_bias=0.12)
```

## Issue templates
When opening an issue or task, include:
- **Type**: bug | perf | feat | refactor | docs
- **Scope**: detectors | face | reid | utils | cli | gui
- **Input**: sample frame or synthetic repro
- **Expected vs actual**: concise bullets
- **Risk**: low | medium | high
- **Acceptance test**: short checklist

## License and credits
Respect upstream model licenses for YOLOv8, YOLOv8-face, OpenCLIP, and ArcFace weights. Keep mirror URLs configurable.

---

This guide is for the ChatGPT Codex agent working in this repo. Keep edits scoped. Ship clean diffs.
