
import os
import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
import open_clip
from urllib.request import urlopen
import shutil
from urllib.error import URLError, HTTPError
import tempfile
from pathlib import Path

Y8F_DEFAULT = "yolov8n-face.pt"

# Candidate mirrors for yolov8n-face weights
Y8F_URLS = [
    # Lindevs release assets
    "https://github.com/lindevs/yolov8-face/releases/download/1.0.1/yolov8n-face.pt",
    "https://github.com/lindevs/yolov8-face/releases/download/1.0.0/yolov8n-face.pt",
    # Derronqi repo
    "https://github.com/derronqi/yolov8-face/raw/main/weights/yolov8n-face.pt",
    # HuggingFace mirror
    "https://huggingface.co/junjiang/GestureFace/resolve/main/yolov8n-face.pt",
]

def _ensure_file(path: str, urls=Y8F_URLS) -> str:
    """
    Ensure a local file exists. If missing, attempt to download from given URLs.
    Returns absolute path. Raises FileNotFoundError if all downloads fail.
    """
    p = Path(path)
    if p.exists():
        return str(p.resolve())

    # If path has parent directories, create them
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)

    last_err = None
    for url in urls:
        try:
            with urlopen(url, timeout=30) as r, tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(r.read())
                tmp_path = Path(tmp.name)
            shutil.move(str(tmp_path), str(p))
            return str(p.resolve())
        except (URLError, HTTPError, TimeoutError, OSError) as e:
            last_err = e
            continue

    raise FileNotFoundError(f"Could not obtain '{path}'. Tried: {', '.join(urls)}. Last error: {last_err}")

class FaceEmbedder:
    """
    Face detection: YOLOv8-face
    Face embedding: OpenCLIP ViT-B/32
    Returns list of dicts: {'bbox': np.int32[x1,y1,x2,y2], 'kps': None, 'feat': np.float32[D]}
    """

    def __init__(self, ctx: str = 'cuda', yolo_model: str = Y8F_DEFAULT, conf: float = 0.30):
        # Auto-download face detector weights if a bare filename like 'yolov8n-face.pt' is given
        yolo_path = yolo_model
        if not os.path.isabs(yolo_path) and os.path.basename(yolo_path).startswith("yolov8") and yolo_path.endswith(".pt"):
            try:
                yolo_path = _ensure_file(yolo_path)
            except FileNotFoundError as e:
                # Defer with a clearer error
                raise FileNotFoundError(f"{e}. You can also set a custom path to a face detector .pt via FaceEmbedder(yolo_model=...)") from e

        self.det = YOLO(yolo_path)
        self.device = 'cuda' if (ctx.startswith('cuda') and torch.cuda.is_available()) else 'cpu'
        self.conf = float(conf)

        # CLIP image encoder for facial embeddings
        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.model.eval().to(self.device)

    def extract(self, bgr_img: np.ndarray):
        """Detect faces and return embeddings for the largest faces first."""
        if bgr_img is None or bgr_img.size == 0:
            return []

        # YOLOv8-face detection
        res = self.det.predict(bgr_img, conf=self.conf, verbose=False, device=self.device)[0]
        boxes = []
        for b in res.boxes:
            x1, y1, x2, y2 = [int(v.item()) for v in b.xyxy[0]]
            boxes.append((x1, y1, x2, y2))

        if not boxes:
            return []

        # Crop faces and compute CLIP embeddings
        tensors = []
        for (x1, y1, x2, y2) in boxes:
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = max(x1+1, x2); y2 = max(y1+1, y2)
            face_bgr = bgr_img[y1:y2, x1:x2]
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(face_rgb)
            tensors.append(self.preprocess(pil))

        batch = torch.stack(tensors).to(self.device, non_blocking=True)
        with torch.inference_mode():
            feats = self.model.encode_image(batch)
            feats = torch.nn.functional.normalize(feats, dim=1).cpu().numpy().astype(np.float32)

        out = []
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            out.append({'bbox': np.array([x1, y1, x2, y2], dtype=np.int32),
                        'kps': None,
                        'feat': feats[i]})
        # Sort by area desc
        out.sort(key=lambda f: (f['bbox'][2]-f['bbox'][0])*(f['bbox'][3]-f['bbox'][1]), reverse=True)
        return out

    @staticmethod
    def best_face(faces):
        if not faces:
            return None
        return max(faces, key=lambda f: (f['bbox'][2]-f['bbox'][0]) * (f['bbox'][3]-f['bbox'][1]))
