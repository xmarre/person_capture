from ultralytics import YOLO
import torch
import numpy as np

class PersonDetector:
    def __init__(self, model_name='yolov8n.pt', device='cuda'):
        self.model = YOLO(model_name)
        self.device = device if torch.cuda.is_available() and device.startswith('cuda') else 'cpu'

    def detect(self, frame, conf=0.35):
        # Returns list of dicts: {xyxy: [x1,y1,x2,y2], conf: float, cls: int}
        try:
            res = self.model.predict(frame, device=self.device, conf=conf, iou=0.45, classes=0, verbose=False)[0]
        except Exception:
            return []
        out = []
        for b in res.boxes:
            cls_id = int(b.cls[0].item())
            if cls_id != 0:  # person class only
                continue
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            out.append({'xyxy':[x1,y1,x2,y2],'conf':float(b.conf[0].item()),'cls':cls_id})
        return out
