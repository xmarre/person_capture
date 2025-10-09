
import os
import cv2
import numpy as np
from urllib.request import urlopen
from urllib.error import URLError, HTTPError
import tempfile
from pathlib import Path
import shutil
from typing import TYPE_CHECKING, Optional, List, Tuple

from PIL import Image

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    import torch
    from ultralytics import YOLO as YOLOType
    import open_clip as open_clip_type

try:
    import onnxruntime as ort
except Exception:
    ort = None  # fallback later

Y8F_DEFAULT = "yolov8n-face.pt"

# Candidate mirrors for yolov8n-face weights
Y8F_URLS = [
    "https://github.com/lindevs/yolov8-face/releases/download/1.0.1/yolov8n-face.pt",
    "https://github.com/lindevs/yolov8-face/releases/download/1.0.0/yolov8n-face.pt",
    "https://github.com/derronqi/yolov8-face/raw/main/weights/yolov8n-face.pt",
    "https://huggingface.co/junjiang/GestureFace/resolve/main/yolov8n-face.pt",
]

# ArcFace ONNX mirrors (R100 model). Any that succeeds is fine.
ARCFACE_ONNX = "arcface_r100.onnx"
ARCFACE_URLS = [
    "https://github.com/deepinsight/insightface/releases/download/v0.7/arcface_r100.onnx",
    "https://raw.githubusercontent.com/deepinsight/insightface/master/python-package/insightface/model_zoo/arcface_r100.onnx",
    "https://huggingface.co/MonsterMMORPG/Arcface/resolve/main/arcface_r100.onnx",
]

def _ensure_file(path: str, urls, progress=None) -> str:
    p = Path(path)
    if p.exists():
        return str(p.resolve())
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for url in urls:
        try:
            if progress: progress(f"Downloading: {url}")
            with urlopen(url, timeout=45) as r:
                total = int(r.headers.get('Content-Length') or 0)
                tmp = tempfile.NamedTemporaryFile(delete=False)
                downloaded = 0
                while True:
                    chunk = r.read(1024*1024)
                    if not chunk:
                        break
                    tmp.write(chunk)
                    downloaded += len(chunk)
                    if progress and total:
                        progress(f"Downloading: {url}  {downloaded/total*100:.1f}%")
                tmp.close()
                tmp_path = Path(tmp.name)
            shutil.move(str(tmp_path), str(p))
            if progress: progress(f"Saved: {p}")
            return str(p.resolve())
        except (URLError, HTTPError, TimeoutError, OSError) as e:
            last_err = e
            continue
    raise FileNotFoundError(f"Could not obtain '{path}'. Tried: {', '.join(urls)}. Last error: {last_err}")

class FaceEmbedder:
    """
    Face detection: YOLOv8-face.
    Face embedding: OpenCLIP (strong backbone) or ArcFace ONNX for identity.
    Returns list of dicts: {'bbox': np.int32[x1,y1,x2,y2], 'feat': np.float32[D], 'quality': float}
    """

    def __init__(self, ctx: str = 'cuda', yolo_model: str = Y8F_DEFAULT, conf: float = 0.30,
                 use_arcface: bool = True,
                 clip_model_name: str = 'ViT-L-14',
                 clip_pretrained: str = 'laion2b_s32b_b82k',
                 progress=None):
        # Auto-download face detector weights if a bare filename is given
        yolo_path = yolo_model
        if not os.path.isabs(yolo_path) and os.path.basename(yolo_path).startswith("yolov8") and yolo_path.endswith(".pt"):
            yolo_path = _ensure_file(yolo_path, Y8F_URLS, progress=progress)

        try:
            import torch as _torch
            from ultralytics import YOLO as _YOLO
            import open_clip as _open_clip
        except Exception as e:  # pragma: no cover - executed only when deps missing
            raise RuntimeError(
                "Heavy dependencies not installed; install requirements.txt to run face embedding."
            ) from e

        self._torch = _torch
        self.det = _YOLO(yolo_path)
        self.use_arcface = bool(use_arcface)
        # Keep ArcFace enabled; we'll align by 5pts only if keypoints are available at inference time.
        self.device = 'cuda' if (ctx.startswith('cuda') and _torch.cuda.is_available()) else 'cpu'
        self.conf = float(conf)
        self.backend = None
        self.arc_sess = None
        self.arc_input = None
        if self.use_arcface and ort is not None:
            try:
                onnx_path = _ensure_file(ARCFACE_ONNX, ARCFACE_URLS, progress=progress)
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
                self.arc_sess = ort.InferenceSession(onnx_path, providers=providers)
                self.arc_input = self.arc_sess.get_inputs()[0].name
                self.backend = 'arcface'
            except Exception as _e:
                # Fallback to CLIP
                self.use_arcface = False
        else:
            if self.use_arcface:
                if progress:
                    progress("onnxruntime not available. Falling back to CLIP for face embeddings.")
                self.use_arcface = False
        if not self.use_arcface or self.backend is None:
            if False and progress: progress(f"Preparing OpenCLIP {clip_model_name} {clip_pretrained} (will download if missing)...")
            self.model, _, self.preprocess = _open_clip.create_model_and_transforms(
                clip_model_name, pretrained=clip_pretrained
            )
            self.model.eval().to(self.device)
            self.backend = 'clip'
            pass

    def _face_quality(self, bgr):
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(g, cv2.CV_64F).var())

    # ArcFace 112x112 landmark template (lfw standard)
    _ARC_DST = np.array([[38.2946,51.6963],[73.5318,51.5014],[56.0252,71.7366],[41.5493,92.3655],[70.7299,92.2041]], dtype=np.float32)

    def _arcface_preprocess(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != (112,112):
            rgb = cv2.resize(rgb, (112,112), interpolation=cv2.INTER_LINEAR)
        arr = rgb.astype(np.float32) / 127.5 - 1.0
        arr = np.transpose(arr, (2,0,1))[None, ...]  # NCHW
        return arr

    def _arcface_encode(self, bgr_list: List[np.ndarray]):
        if not bgr_list:
            return []
        chips = [self._arcface_preprocess(b) for b in bgr_list]
        chips_flip = [self._arcface_preprocess(cv2.flip(b, 1)) for b in bgr_list]
        batch = np.concatenate(chips + chips_flip, axis=0)
        feats = self.arc_sess.run(None, {self.arc_input: batch})[0]
        n = len(bgr_list)
        f = feats[:n] + feats[n:]
        f /= (np.linalg.norm(f, axis=1, keepdims=True) + 1e-9)
        return f.astype(np.float32)

    def _try_keypoints(self, res) -> Optional[np.ndarray]:
        kps = getattr(res, "keypoints", None)
        if kps is None:
            return None

        # Avoid directly truth-testing tensors as this raises runtime errors.
        data = getattr(kps, "xy", None)
        if data is None:
            data = getattr(kps, "data", None)
        if data is None:
            return None

        # Normalize the keypoint container to a numpy array with shape [N, K, 2].
        if isinstance(data, (list, tuple)):
            if not data:
                return None
            stacked = []
            for item in data:
                arr = item.cpu().numpy() if hasattr(item, "cpu") else np.asarray(item)
                if arr.ndim == 2 and arr.shape[-1] >= 2:
                    stacked.append(arr[..., :2])
            if not stacked:
                return None
            arr = np.stack(stacked, axis=0)
        else:
            arr = data.cpu().numpy() if hasattr(data, "cpu") else np.asarray(data)
            if arr.ndim == 2:
                arr = arr[None, ...]
            if arr.shape[-1] >= 2:
                arr = arr[..., :2]

        arr = np.asarray(arr, dtype=np.float32)

        if arr.ndim == 3 and arr.shape[1] >= 5:
            if not np.isfinite(arr).all():
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            return arr[:, :5, :2]
        return None

    def _align_by_5pts(self, bgr: np.ndarray, pts5: np.ndarray) -> np.ndarray:
        M, _ = cv2.estimateAffinePartial2D(pts5.astype(np.float32), self._ARC_DST, method=cv2.LMEDS)
        if M is None:
            M, _ = cv2.estimateAffinePartial2D(pts5[:3], self._ARC_DST[:3], method=cv2.LMEDS)
        if M is None:
            return cv2.resize(bgr, (112,112), interpolation=cv2.INTER_LINEAR)
        return cv2.warpAffine(bgr, M, (112,112), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    def _clip_encode(self, bgr_list):
        if not bgr_list:
            return []
        tensors = []
        for bgr in bgr_list:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            from PIL import Image
            pil = Image.fromarray(rgb)
            tensors.append(self.preprocess(pil))
        batch = self._torch.stack(tensors).to(self.device, non_blocking=True)
        with self._torch.inference_mode():
            feats = self.model.encode_image(batch)
            feats = self._torch.nn.functional.normalize(feats, dim=1)
        return feats.detach().cpu().numpy().astype(np.float32)

    def extract(self, bgr_img: np.ndarray):
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

        faces: List[Tuple[int, int, int, int, np.ndarray, float]] = []
        crops: List[np.ndarray] = []
        kps = self._try_keypoints(res)
        if kps is not None and len(kps) != len(boxes):
            kps = None  # fallback to unaligned chips when counts mismatch
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = max(x1+1, x2); y2 = max(y1+1, y2)
            face_bgr = bgr_img[y1:y2, x1:x2]
            use_landmarks = (
                self.use_arcface
                and kps is not None
                and i < len(kps)
                and np.isfinite(kps[i]).all()
            )
            if use_landmarks:
                pts = kps[i].astype(np.float32, copy=True)
                pts[:, 0] -= float(x1)
                pts[:, 1] -= float(y1)
                pts[:, 0] = np.clip(pts[:, 0], 0.0, max(0, face_bgr.shape[1] - 1))
                pts[:, 1] = np.clip(pts[:, 1], 0.0, max(0, face_bgr.shape[0] - 1))
                chip = self._align_by_5pts(face_bgr, pts)
            else:
                chip = face_bgr
            faces.append((x1,y1,x2,y2, chip, self._face_quality(chip)))
            crops.append(chip)

        if self.use_arcface:
            feats = self._arcface_encode(crops)
        else:
            feats = self._clip_encode(crops)

        out = []
        for i, (x1,y1,x2,y2, crop, q) in enumerate(faces):
            out.append({'bbox': np.array([x1,y1,x2,y2], dtype=np.int32),
                        'feat': feats[i],
                        'quality': float(q)})
        # Sort by quality then area
        out.sort(key=lambda f: (f['quality'], (f['bbox'][2]-f['bbox'][0])*(f['bbox'][3]-f['bbox'][1])), reverse=True)
        return out

    @staticmethod
    def best_face(faces):
        if not faces:
            return None
        return max(faces, key=lambda f: (f['quality'], (f['bbox'][2]-f['bbox'][0]) * (f['bbox'][3]-f['bbox'][1])))
