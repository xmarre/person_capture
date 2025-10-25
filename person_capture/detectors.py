import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import torch
    from ultralytics import YOLO as YOLOType

class PersonDetector:
    def __init__(self, model_name='yolov8n.pt', device='cuda', progress=None):
        try:
            import torch as _torch
            # Ensure home/settings before Ultralytics reads them
            _pkg_dir = Path(__file__).resolve().parent
            _repo_root = _pkg_dir.parent if _pkg_dir.name == "person_capture" else _pkg_dir
            os.environ.setdefault("ULTRALYTICS_HOME", str(_repo_root / ".ultralytics"))
            os.environ.setdefault(
                "ULTRALYTICS_SETTINGS", str(_repo_root / ".ultralytics" / "settings.yaml")
            )
            Path(os.environ["ULTRALYTICS_HOME"]).mkdir(parents=True, exist_ok=True)
            settings_path = Path(os.environ["ULTRALYTICS_SETTINGS"])
            settings_path.parent.mkdir(parents=True, exist_ok=True)
            settings_path.touch(exist_ok=True)
            from ultralytics import YOLO as _YOLO
            from ultralytics.utils import SETTINGS
            # Point all Ultralytics dirs into repo root to avoid global/user/cache drift
            _home = Path(os.environ["ULTRALYTICS_HOME"])
            _weights = _home / "weights"
            _datasets = _home / "datasets"
            _downloads = _home / "downloads"
            for d in (_weights, _datasets, _downloads):
                d.mkdir(parents=True, exist_ok=True)
            dirty = False
            if str(SETTINGS.get("weights_dir", "")) != str(_weights):
                SETTINGS["weights_dir"] = str(_weights)
                dirty = True
            if str(SETTINGS.get("datasets_dir", "")) != str(_datasets):
                SETTINGS["datasets_dir"] = str(_datasets)
                dirty = True
            if str(SETTINGS.get("downloads_dir", "")) != str(_downloads):
                SETTINGS["downloads_dir"] = str(_downloads)
                dirty = True
            if dirty:
                try:
                    SETTINGS.save()
                except Exception:
                    pass
            logging.getLogger(__name__).info("YOLO paths: home=%s weights=%s", _home, _weights)
        except Exception as e:  # pragma: no cover - executed only when deps missing
            raise RuntimeError(
                "Heavy dependencies not installed; install requirements.txt to run detection."
            ) from e

        self._torch = _torch
        self._YOLO = _YOLO
        self.device = 'cuda' if (str(device).startswith('cuda') and _torch.cuda.is_available()) else 'cpu'
        self.progress = progress
        self.model = self._load_model(model_name)
        try:
            self.model.fuse()
            if self.device == 'cuda':
                self.model.model.half()
        except Exception:
            pass

    def _load_model(self, model_name: str):
        weights_root = Path(os.environ.get("ULTRALYTICS_HOME", ".")) / "weights"
        weights_root.mkdir(parents=True, exist_ok=True)
        m = Path(model_name)
        model_arg = str(m) if m.is_file() else str(weights_root / m.name)
        try:
            return self._YOLO(model_arg)
        except Exception as e:
            if self.progress:
                self.progress(f"YOLO load failed ({e}). Recovering...")
            # If user pointed to a local file, quarantine it and try hub name
            try:
                p = Path(model_arg)
                if p.is_file():
                    bad = p.with_suffix(p.suffix + '.bad')
                    p.rename(bad)
                    if self.progress:
                        self.progress(f"Quarantined corrupt weights: {bad.name}")
            except Exception:
                pass
            # Derive a clean hub model name
            base = Path(model_name).name.lower()
            hub = base if base.startswith('yolov8') and base.endswith('.pt') else 'yolov8n.pt'
            # Use hub name so Ultralytics resolves to SETTINGS["weights_dir"] (repo .ultralytics/weights)
            return self._YOLO(hub)

    def detect(self, frame, conf=0.35):
        """Return list of dicts for class=person only."""
        try:
            # Ultralytics auto-handles dtype from model; ensure fp16 path on CUDA model
            res = self.model.predict(
                frame,
                device=self.device,
                conf=float(conf),
                iou=0.45,
                classes=[0],
                verbose=False,
                max_det=40,
                imgsz=640,
            )[0]
        except Exception:
            return []

        out = []
        bxs = getattr(res, "boxes", None)
        if bxs is None:
            return out
        for i in range(len(bxs)):
            xyxy = bxs.xyxy[i].tolist()
            c = float(bxs.conf[i].item()) if getattr(bxs, "conf", None) is not None else 0.0
            out.append({"xyxy": xyxy, "conf": c, "cls": 0})
        return out
