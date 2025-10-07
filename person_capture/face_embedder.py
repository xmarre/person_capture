import os
import cv2
import numpy as np
import tempfile
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, List, Dict, Any, Optional, Tuple

from PIL import Image

if TYPE_CHECKING:  # pragma: no cover
    import torch
    from ultralytics import YOLO as YOLOType
    import open_clip as open_clip_type

try:
    import onnxruntime as ort
except Exception:
    ort = None  # optional

Y8F_DEFAULT = "yolov8n-face.pt"
Y8F_URLS = [
    "https://github.com/lindevs/yolov8-face/releases/download/1.0.1/yolov8n-face.pt",
    "https://github.com/lindevs/yolov8-face/releases/download/1.0.0/yolov8n-face.pt",
    "https://github.com/derronqi/yolov8-face/raw/main/weights/yolov8n-face.pt",
    "https://huggingface.co/junjiang/GestureFace/resolve/main/yolov8n-face.pt",
]

ARCFACE_ONNX = "arcface_r100.onnx"
ARCFACE_URLS = [
    "https://github.com/deepinsight/insightface/releases/download/v0.7/arcface_r100.onnx",
    "https://raw.githubusercontent.com/deepinsight/insightface/master/python-package/insightface/model_zoo/arcface_r100.onnx",
    "https://huggingface.co/MonsterMMORPG/Arcface/resolve/main/arcface_r100.onnx",
]

def _download_to(path: Path, url: str, progress=None) -> bool:
    try:
        from urllib.request import urlopen
        from urllib.error import URLError, HTTPError
        if progress: progress(f"Downloading: {url}")
        with urlopen(url, timeout=45) as r:
            total = int(r.headers.get("Content-Length") or 0)
            tmp = tempfile.NamedTemporaryFile(delete=False)
            downloaded = 0
            while True:
                chunk = r.read(1024 * 1024)
                if not chunk:
                    break
                tmp.write(chunk)
                downloaded += len(chunk)
                if progress and total:
                    progress(f"Downloading: {url}  {downloaded/total*100:.1f}%")
            tmp.close()
            shutil.move(tmp.name, str(path))
        if progress: progress(f"Saved: {path}")
        return True
    except Exception:
        return False

def _ensure_file(path: str, urls, progress=None) -> str:
    p = Path(path)
    if p.exists():
        return str(p.resolve())
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    for url in urls:
        if _download_to(p, url, progress=progress):
            return str(p.resolve())
    raise FileNotFoundError(f"Could not obtain '{path}'. Tried: {', '.join(urls)}")

class FaceEmbedder:
    """Face detector + identity embedding.

    Detection: YOLOv8-face (Ultralytics).
    Embedding: ArcFace ONNX if available else OpenCLIP image encoder.
    Output: List[{'bbox': np.int32[4], 'feat': np.float32[D] | None, 'quality': float}]
    """
    def __init__(self, ctx: str = 'cuda',
                 yolo_model: str = Y8F_DEFAULT,
                 conf: float = 0.25,
                 use_arcface: bool = False,
                 clip_model_name: str = 'ViT-L-14',
                 clip_pretrained: str = 'laion2b_s32b_b82k',
                 progress=None):
        try:
            import torch as _torch
            from ultralytics import YOLO as _YOLO
            import open_clip as _open_clip
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Heavy dependencies not installed; install requirements.txt to run face embedding.") from e

        self._torch = _torch
        # face detector weights: auto-download if given as a bare name
        yolo_path = yolo_model
        if not os.path.isabs(yolo_path) and os.path.basename(yolo_path).startswith("yolov8") and yolo_path.endswith(".pt"):
            yolo_path = _ensure_file(yolo_path, Y8F_URLS, progress=progress)
        self.det = _YOLO(yolo_path)
        self.conf = float(conf)
        self.device = 'cuda' if (str(ctx).startswith('cuda') and _torch.cuda.is_available()) else 'cpu'

        # Embedding backend
        self.backend = None
        self.use_arcface = bool(use_arcface) and (ort is not None)
        if self.use_arcface:
            try:
                onnx_path = _ensure_file(ARCFACE_ONNX, ARCFACE_URLS, progress=progress)
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
                self.arc_sess = ort.InferenceSession(onnx_path, providers=providers)
                self.arc_input = self.arc_sess.get_inputs()[0].name
                self.backend = 'arcface'
            except Exception:
                self.use_arcface = False

        if not self.use_arcface:
            self.model, _, self.preprocess = _open_clip.create_model_and_transforms(clip_model_name, pretrained=clip_pretrained)
            self.model.eval().to(self.device)
            self.backend = 'clip'

    # ---------- utilities ----------
    @staticmethod
    def best_face(faces: List[Dict[str, Any]] | None) -> Optional[Dict[str, Any]]:
        if not faces:
            return None
        def keyf(f):
            x1,y1,x2,y2 = f.get('bbox', [0,0,0,0])
            area = max(1, (x2-x1)*(y2-y1))
            return (float(f.get('quality', 0.0)), float(area))
        return sorted(faces, key=keyf, reverse=True)[0]

    def _lap_var(self, bgr: np.ndarray) -> float:
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(g, cv2.CV_64F).var())

    def _arc_pre(self, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (112,112), interpolation=cv2.INTER_LINEAR)
        arr = rgb.astype(np.float32) / 127.5 - 1.0
        arr = np.transpose(arr, (2,0,1))[None, ...]  # NCHW
        return arr

    def _arc_encode(self, crops: List[np.ndarray]) -> List[np.ndarray]:
        if not crops:
            return []
        batch = np.concatenate([self._arc_pre(c) for c in crops], axis=0)
        feats = self.arc_sess.run(None, {self.arc_input: batch})[0]
        # L2 normalize
        norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9
        feats = feats / norms
        return [f.astype(np.float32) for f in feats]

    def _clip_encode(self, crops: List[np.ndarray]) -> List[np.ndarray]:
        if not crops:
            return []
        tensors = []
        for bgr in crops:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            tensors.append(self.preprocess(Image.fromarray(rgb)))
        if not tensors:
            return []
        batch = self._torch.stack(tensors).to(self.device, non_blocking=True)
        with self._torch.inference_mode():
            emb = self.model.encode_image(batch)
            emb = self._torch.nn.functional.normalize(emb, dim=1)
        feats = emb.detach().cpu().numpy().astype(np.float32)
        return [f for f in feats]

    # ---------- public ----------
    def extract(self, bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Return list of faces with embeddings and quality."""
        try:
            res = self.det.predict(bgr, conf=self.conf, iou=0.45, classes=None, verbose=False)[0]
        except Exception:
            return []
        faces = []
        bxs = getattr(res, "boxes", None)
        if bxs is None or len(bxs) == 0:
            return faces
        H, W = bgr.shape[:2]
        # Build crops
        crops = []
        boxes = []
        for i in range(len(bxs)):
            xyxy = [int(v) for v in bxs.xyxy[i].tolist()]
            x1,y1,x2,y2 = xyxy
            x1 = max(0, x1); y1 = max(0, y1); x2 = min(W-1, x2); y2 = min(H-1, y2)
            if x2 <= x1+1 or y2 <= y1+1:
                continue
            crop = bgr[y1:y2, x1:x2]
            crops.append(crop); boxes.append((x1,y1,x2,y2))

        if not boxes:
            return []

        if self.backend == 'arcface':
            feats = self._arc_encode(crops)
        else:
            feats = self._clip_encode(crops)

        out = []
        for i, box in enumerate(boxes):
            q = self._lap_var(crops[i])
            feat = feats[i] if i < len(feats) else None
            out.append({"bbox": np.array(box, dtype=np.int32), "feat": feat, "quality": float(q)})
        # Sort by quality then area
        out.sort(key=lambda f: (float(f.get('quality', 0.0)),
                                (int(f['bbox'][2])-int(f['bbox'][0]))*(int(f['bbox'][3])-int(f['bbox'][1]))),
                 reverse=True)
        return out
