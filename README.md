# PersonCapture: Target-person finder and 2:3 crops from video

Pipeline:
- Detect persons in video frames (YOLOv8).
- Compute two identity signals per candidate:
  - Face embedding when a face is visible (InsightFace).
  - Full-body person ReID embedding (TorchReID OSNet).
- Compare to a reference image you supply.
- Save crops around matches at a fixed aspect ratio (e.g., 2:3).

## Quick start (Windows + Visual Studio 2022)

1) Install Python 3.10 or 3.11 (64‑bit). In Visual Studio Installer, add the **Python development** workload.
2) Install a CUDA runtime if you want GPU (NVIDIA). CUDA 12.x is typical for RTX 40/50 series.
3) Create and select a virtual environment from VS or PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio  # GPU
pip install -r requirements.txt
```

4) Open **Visual Studio 2022** → *Open a project or solution* → select this folder.
   - Add a **Python Application** project if VS asks. Set `person_capture/main.py` as startup.
5) Run:

```powershell
python -m person_capture.main --video path\to\video.mp4 --ref path\to\person.jpg --out out_dir --ratio 2:3
```

Optional flags:
```
--frame-stride 3            # analyze every 3rd frame
--min-det-conf 0.35         # YOLO confidence threshold
--face-thresh 0.32          # cosine distance for face match (lower is stricter)
--reid-thresh 0.38          # cosine distance for reid match (lower is stricter)
--combine 'min'             # combine scores: min | avg | face_priority
--device cuda               # cuda | cpu
--save-annot                # also save annotated frames
```

Outputs: crops in `out_dir/crops` and an index CSV with timestamps and scores.

## Notes
- Reference image can be a frame grab. If it has a clear face, face matching will dominate.
- When face is not visible, ReID keeps tracking using clothes/shape cues.
- Thresholds are dataset dependent. Start with defaults, then tighten/loosen as needed.
