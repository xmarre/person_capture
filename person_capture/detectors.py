
from ultralytics import YOLO
import torch

class PersonDetector:
    def __init__(self, model_name='yolov8n.pt', device='cuda'):
        self.model = YOLO(model_name)
        self.device = 'cuda' if (device.startswith('cuda') and torch.cuda.is_available()) else 'cpu'

    def detect(self, frame, conf=0.35):
        """
        Returns list of dicts: {xyxy: [x1,y1,x2,y2], conf: float, cls: int}
        Only class 0 (person).
        """
        try:
            res = self.model.predict(
                frame, device=self.device, conf=float(conf), iou=0.45, classes=[0], verbose=False
            )[0]
        except Exception:
            return []

        out = []
        if getattr(res, "boxes", None) is None:
            return out

        bxs = res.boxes
        for i in range(len(bxs)):
            xyxy = bxs.xyxy[i].tolist()
            c = float(bxs.conf[i].item()) if getattr(bxs, "conf", None) is not None else 0.0
            cls = int(bxs.cls[i].item()) if getattr(bxs, "cls", None) is not None else 0
            out.append({"xyxy": xyxy, "conf": c, "cls": cls})
        return out
