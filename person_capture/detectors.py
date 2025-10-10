from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import torch
    from ultralytics import YOLO as YOLOType

class PersonDetector:
    def __init__(self, model_name='yolov8n.pt', device='cuda', progress=None):
        try:
            import torch as _torch
            from ultralytics import YOLO as _YOLO
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
        name = str(model_name)
        try:
            return self._YOLO(name)
        except Exception as e:
            if self.progress:
                self.progress(f"YOLO load failed ({e}). Recovering...")
            # If user pointed to a local file, quarantine it and try hub name
            try:
                p = Path(name)
                if p.is_file():
                    bad = p.with_suffix(p.suffix + '.bad')
                    p.rename(bad)
                    if self.progress:
                        self.progress(f"Quarantined corrupt weights: {bad.name}")
            except Exception:
                pass
            # Derive a clean hub model name
            base = Path(name).name.lower()
            hub = base if base.startswith('yolov8') and base.endswith('.pt') else 'yolov8n.pt'
            return self._YOLO(hub)

    def detect(self, frame, conf=0.35):
        """Return list of dicts for class=person only."""
        try:
            # Ultralytics auto-handles dtype from model; ensure fp16 path on CUDA model
            res = self.model.predict(
                frame, device=self.device, conf=float(conf), iou=0.45, classes=[0], verbose=False
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
