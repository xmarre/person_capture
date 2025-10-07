from typing import TYPE_CHECKING, List, Dict, Any

if TYPE_CHECKING:  # pragma: no cover
    import torch
    from ultralytics import YOLO as YOLOType

class PersonDetector:
    """Ultralytics YOLO wrapper restricted to the person class (id=0)."""
    def __init__(self, model_name: str = 'yolov8n.pt', device: str = 'cuda', progress=None):
        try:
            import torch as _torch
            from ultralytics import YOLO as _YOLO
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Heavy dependencies not installed; install requirements.txt to run detection.") from e
        self._torch = _torch
        self.device = 'cuda' if (str(device).startswith('cuda') and _torch.cuda.is_available()) else 'cpu'
        self.model = _YOLO(model_name)
        self.progress = progress

    def detect(self, bgr, conf: float = 0.35) -> List[Dict[str, Any]]:
        if bgr is None:
            return []
        try:
            res = self.model.predict(bgr, conf=float(conf), iou=0.45, classes=[0], verbose=False)[0]
        except Exception:
            return []
        out: List[Dict[str, Any]] = []
        bxs = getattr(res, "boxes", None)
        if bxs is None or len(bxs) == 0:
            return out
        for i in range(len(bxs)):
            xyxy = [float(v) for v in bxs.xyxy[i].tolist()]
            c = float(bxs.conf[i].item()) if getattr(bxs, "conf", None) is not None else 0.0
            out.append({"xyxy": xyxy, "conf": c, "cls": 0})
        return out
