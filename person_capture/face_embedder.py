import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
import open_clip


class FaceEmbedder:
    """
    Face detection: YOLOv8-face
    Face embedding: OpenCLIP ViT-B/32
    Returns list of dicts: {'bbox': np.int32[x1,y1,x2,y2], 'kps': None, 'feat': np.float32[D]}
    """

    def __init__(self, ctx: str = 'cuda', yolo_model: str = 'yolov8n-face.pt', conf: float = 0.30):
        self.device = 'cuda' if (ctx == 'cuda' and torch.cuda.is_available()) else 'cpu'
        self.det = YOLO(yolo_model)
        self.conf = conf
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k', device=self.device
        )
        self.model.eval()

    @torch.no_grad()
    def extract(self, img_bgr: np.ndarray):
        if img_bgr is None or img_bgr.size == 0:
            return []
        h, w = img_bgr.shape[:2]
        res = self.det.predict(img_bgr, conf=self.conf, device=self.device, verbose=False)[0]

        boxes, crops = [], []
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            x1 = int(max(0, x1)); y1 = int(max(0, y1))
            x2 = int(min(w - 1, x2)); y2 = int(min(h - 1, y2))
            if x2 <= x1 + 2 or y2 <= y1 + 2:
                continue
            boxes.append((x1, y1, x2, y2))
            crops.append(img_bgr[y1:y2, x1:x2])

        if not crops:
            return []

        tensors = []
        for c in crops:
            rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            tensors.append(self.preprocess(pil))
        batch = torch.stack(tensors).to(self.device, non_blocking=True)

        feats = self.model.encode_image(batch)
        feats = torch.nn.functional.normalize(feats, dim=1).cpu().numpy().astype(np.float32)

        out = []
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            out.append({'bbox': np.array([x1, y1, x2, y2], dtype=np.int32),
                        'kps': None,
                        'feat': feats[i]})
        return out

    @staticmethod
    def best_face(faces):
        if not faces:
            return None
        return max(faces, key=lambda f: (f['bbox'][2]-f['bbox'][0]) * (f['bbox'][3]-f['bbox'][1]))
