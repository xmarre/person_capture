
import os, io, sys, glob, site, ctypes, math
from pathlib import Path
import cv2
import numpy as np
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from socket import timeout as SocketTimeout
import tempfile
import shutil
from typing import TYPE_CHECKING, Optional, List, Tuple

from PIL import Image

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    import torch
    from ultralytics import YOLO as YOLOType
    import open_clip as open_clip_type

ort = None  # import after DLL dirs are added
Y8F_DEFAULT = "yolov8n-face.pt"

# Candidate mirrors for yolov8n-face weights
Y8F_URLS = [
    "https://github.com/lindevs/yolov8-face/releases/download/1.0.1/yolov8n-face.pt",
    "https://github.com/lindevs/yolov8-face/releases/download/1.0.0/yolov8n-face.pt",
    "https://github.com/derronqi/yolov8-face/raw/main/weights/yolov8n-face.pt",
    "https://huggingface.co/junjiang/GestureFace/resolve/main/yolov8n-face.pt",
]

# Recognition ONNX target path we write to (keep filename stable for the app).
ARCFACE_ONNX = "arcface_r100.onnx"
# Canonical, working sources (2024–2025):
# 1) antelopev2/glintr100.onnx (InsightFace package)
# 2) buffalo_l/w600k_r50.onnx  (InsightFace package)
ARCFACE_URLS = [
    # glintr100.onnx mirrors
    "https://huggingface.co/LPDoctor/insightface/resolve/25226b4048397eb2adc0fa5a3c21f416005fc228/models/antelopev2/glintr100.onnx",
    "https://huggingface.co/XuminYu/example_safetensors/resolve/0e9cb8b6ec530f64c20e69fa33e9da6a79895e85/insightface/models/antelopev2/glintr100.onnx",
    # buffalo_l recognition fallback
    "https://huggingface.co/fofr/comfyui/resolve/b24971859a2d244d5f746ce707c3dfa4dd574108/insightface/models/buffalo_l/w600k_r50.onnx",
    "https://huggingface.co/datasets/theanhntp/Liblib/resolve/ae4357741af379482690fe3e0f2fa6fd32ba33b4/insightface/models/buffalo_l/w600k_r50.onnx",
]
# Package ZIP fallback (contains glintr100.onnx + det_10g.onnx + 2d106det.onnx)
ANTELOPE_ZIPS = [
    "https://sourceforge.net/projects/insightface.mirror/files/v0.7/antelopev2.zip/download"
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
            req = Request(url, headers={"User-Agent": "Mozilla/5.0 PersonCapture"})
            with urlopen(req, timeout=90) as r:
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
            # minimal integrity: require > 50MB (real models are ~160–260MB)
            if p.stat().st_size < 50 * 1024 * 1024:
                raise OSError(f"Downloaded file too small: {p} ({p.stat().st_size} bytes)")
            return str(p.resolve())
        except (URLError, HTTPError, SocketTimeout, OSError) as e:
            last_err = e
            continue
    raise FileNotFoundError(f"Could not obtain '{path}'. Tried: {', '.join(urls)}. Last error: {last_err}")


def _ensure_from_zip(out_path: str, zip_urls, want_suffix="glintr100.onnx", progress=None) -> str:
    p = Path(out_path)
    if p.exists():
        return str(p.resolve())
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    tmpzip = None
    last_err = None
    for url in zip_urls:
        try:
            if progress: progress(f"Downloading package: {url}")
            req = Request(url, headers={"User-Agent": "Mozilla/5.0 PersonCapture"})
            with urlopen(req, timeout=120) as r:
                tmpzip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
                shutil.copyfileobj(r, tmpzip)
                tmpzip.close()
            import zipfile
            with zipfile.ZipFile(tmpzip.name, 'r') as zf:
                cand = [n for n in zf.namelist() if n.lower().endswith(want_suffix)]
                if not cand:
                    raise FileNotFoundError(f"{want_suffix} not found in ZIP")
                member = max(cand, key=len)
                if progress: progress(f"Extracting: {member}")
                with zf.open(member) as src, open(p, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                if p.stat().st_size < 50 * 1024 * 1024:
                    raise OSError(f"Extracted file too small: {p} ({p.stat().st_size} bytes)")
                return str(p.resolve())
        except (URLError, HTTPError, SocketTimeout, OSError, Exception) as e:
            last_err = e
            continue
        finally:
            try:
                if tmpzip:
                    os.unlink(tmpzip.name)
            except Exception:
                pass
    raise FileNotFoundError(
        f"Could not extract '{out_path}' from ZIPs. Tried: {', '.join(zip_urls)}. Last error: {last_err}"
    )

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
                 progress=None, trt_lib_dir: Optional[str] = None):
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

        self.progress = progress
        self._torch = _torch
        self.det = _YOLO(yolo_path)
        self.use_arcface = bool(use_arcface)
        # Keep ArcFace enabled; we'll align by 5pts only if keypoints are available at inference time.
        self.device = 'cuda' if (ctx.startswith('cuda') and _torch.cuda.is_available()) else 'cpu'
        self.conf = float(conf)
        self.backend = None
        self.arc_sess = None
        self.arc_input = None
        if self.use_arcface:
            try:
                try:
                    onnx_path = _ensure_file(ARCFACE_ONNX, ARCFACE_URLS, progress=progress)
                except FileNotFoundError:
                    # Fallback to package ZIP (antelopev2)
                    onnx_path = _ensure_from_zip(
                        ARCFACE_ONNX, ANTELOPE_ZIPS, want_suffix="glintr100.onnx", progress=progress
                    )
                # Ensure CUDA/cuDNN/TensorRT DLLs are discoverable on Windows
                if os.name == 'nt':
                    from pathlib import Path as _P

                    def _add_dir(p):
                        try:
                            if p and p.exists():
                                os.add_dll_directory(str(p))
                                if progress:
                                    progress(f"Added DLL dir: {p}")
                        except Exception:
                            pass

                    # Always add PyTorch CUDA/cuDNN bin
                    _add_dir(_P(_torch.__file__).parent / "lib")
                    # Priority: GUI -> env -> fallback
                    trt_root = (
                        trt_lib_dir
                        or os.environ.get("TRT_LIB_DIR")
                        or os.environ.get("TENSORRT_DIR")
                        or os.environ.get("TENSORRT_HOME")
                        or r"D:\tensorrt\TensorRT-10.13.3.9"
                    )
                    trt_root_p = _P(trt_root)
                    # Typical layouts: <root>\lib or the root already is lib
                    trt_lib_p = trt_root_p / "lib" if (trt_root_p / "lib").exists() else trt_root_p
                    _add_dir(trt_lib_p)
                    if progress:
                        progress(f"Using TensorRT dir: {trt_lib_p}")
                    # Hard sanity check so we fail fast if the wrong dir is used
                    try:
                        ctypes.WinDLL("nvinfer.dll")
                        ctypes.WinDLL("nvinfer_plugin.dll")
                    except OSError as e:
                        raise RuntimeError(f"TensorRT core DLLs not loadable from {trt_lib_p}: {e}")
                    # Parser is optional for ORT TRT-EP, but load if present
                    for opt in ("nvonnxparser.dll", "nvonnxparser_runtime.dll"):
                        try:
                            ctypes.WinDLL(opt)
                            break
                        except OSError:
                            pass
                # Import ORT only after DLL dirs are set
                global ort
                import onnxruntime as ort  # noqa
                if hasattr(ort, "preload_dlls"):
                    try:
                        ort.preload_dlls()
                    except Exception as e:
                        if progress: progress(f"ORT preload_dlls note: {e!r}")
                # === TensorRT ONLY ===
                avail = list(ort.get_available_providers())
                if 'TensorrtExecutionProvider' not in avail:
                    raise RuntimeError(f"TensorRT EP not available in ORT. avail={avail}")

                so = ort.SessionOptions()
                so.log_severity_level = 0  # VERBOSE
                go = getattr(ort, "GraphOptimizationLevel", None)
                if go is not None:
                    level = getattr(go, "ORT_ENABLE_ALL", None)
                    if level is not None:
                        so.graph_optimization_level = level
                # Tag logs
                so.add_session_config_entry("session.logid", "arcface_trt")

                # ---- Debug & options -------------------------------------------------
                debug_lines: list[str] = []

                def _logd(msg: str) -> None:
                    debug_lines.append(msg)
                    if progress:
                        try:
                            progress(msg)
                        except Exception:
                            pass

                if os.getenv("PERSON_CAPTURE_TRT_DEBUG", "0") not in ("0", "", "false", "False"):
                    _logd(
                        f"onnxruntime {getattr(ort, '__version__', '?')} file={getattr(ort, '__file__', '?')} avail={avail}"
                    )
                    try:
                        import tensorrt as _trt
                        _logd(f"tensorrt python module: {_trt.__version__}")
                    except Exception as e:
                        _logd(f"tensorrt import failed: {e!r}")
                    try:
                        # Probe common DLLs for TRT 8/10 and CUDA 11/12
                        dlls = [
                            "nvinfer.dll",
                            "nvinfer_plugin.dll",
                            "nvonnxparser.dll",
                            "cudart64_12.dll",
                            "cudart64_110.dll",
                            "cudart64_111.dll",
                            "cudart64_112.dll",
                        ]
                        for d in dlls:
                            try:
                                ctypes.WinDLL(d)
                                _logd(f"Loaded {d}")
                            except Exception as ee:
                                _logd(f"Load failed {d}: {ee!r}")
                        _logd(
                            f"PATH has TensorRT? {any('tensorrt' in p.lower() for p in os.getenv('PATH', '').split(os.pathsep))}"
                        )
                        _logd(
                            f"PATH has CUDA bin? {any(os.path.basename(p).lower() == 'bin' and 'cuda' in p.lower() for p in os.getenv('PATH', '').split(os.pathsep))}"
                        )
                    except Exception as e:
                        _logd(f"ctypes probe error: {e!r}")

                # Build provider chain: TRT -> CUDA -> CPU with **no fp16, no cache**, default opts
                prov = []
                if 'TensorrtExecutionProvider' in avail:
                    prov.append('TensorrtExecutionProvider')
                if 'CUDAExecutionProvider' in avail:
                    prov.append('CUDAExecutionProvider')
                prov.append('CPUExecutionProvider')

                prov_opts: list[dict] = [{} for _ in prov]

                last_err = None
                bound = []
                try:
                    _logd(f"Trying TRT session providers={prov} (no fp16, no cache)")
                    self.arc_sess = ort.InferenceSession(
                        onnx_path, sess_options=so, providers=prov, provider_options=prov_opts
                    )
                    bound = self.arc_sess.get_providers()
                    if bound[:1] != ['TensorrtExecutionProvider']:
                        raise RuntimeError(f"TRT not bound. bound={bound}")
                    try:
                        _logd(f"Resolved provider options: {self.arc_sess.get_provider_options()}")
                    except Exception:
                        pass
                    _logd(f"ArcFace(TRT): bound={bound} file={onnx_path}")
                except Exception as e:
                    last_err = e
                    _logd(f"Attempt failed: {e!r}")

                if bound[:1] != ['TensorrtExecutionProvider']:
                    # Hard fail with full debug trail
                    raise RuntimeError("TRT binding failed.\n" + "\n".join(debug_lines)) from last_err

                self.arc_input = self.arc_sess.get_inputs()[0].name
                self.backend = 'arcface'
                # sanity: output embedding should be 512-D
                out0 = self.arc_sess.get_outputs()[0]
                if hasattr(out0, "shape") and (out0.shape is not None) and (out0.shape[-1] not in (512,)):
                    raise RuntimeError(f"Unexpected ArcFace output dim: {out0.shape}")
            except Exception as _e:
                # Hard fail so the real reason is visible
                raise RuntimeError(f"ArcFace download/init failed: {_e!r}")
        else:
            if self.use_arcface:
                import sys as _sys
                raise RuntimeError(f"onnxruntime not importable; exe={_sys.executable}")
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
        batch = np.concatenate(chips + chips_flip, axis=0).astype(np.float32, copy=False)
        # Pad to fixed batch so TRT compiles once
        FIX = 32
        n = batch.shape[0]
        if n < FIX:
            pad = np.repeat(batch[:1], FIX - n, axis=0)
            batch = np.concatenate([batch, pad], axis=0)
        batch = np.ascontiguousarray(batch)
        feats = self.arc_sess.run(None, {self.arc_input: batch})[0]
        m = len(bgr_list)
        f = feats[:m] + feats[m:2*m]
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
                    stacked.append(arr[:, :2])
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

    @staticmethod
    def _canon_5pts(pts: np.ndarray) -> Optional[np.ndarray]:
        """
        Accepts shape (5, 2) in any order and returns ArcFace ordering:
        [left_eye, right_eye, nose, left_mouth, right_mouth].
        """
        if pts is None or pts.shape != (5, 2):
            return None
        pts = np.asarray(pts, dtype=np.float32)
        if not np.isfinite(pts).all():
            return None

        order_y = np.argsort(pts[:, 1])
        eyes_idx = order_y[:2]
        nose_idx = order_y[2]
        mouth_idx = order_y[3:]
        if mouth_idx.size != 2:
            return None

        eyes = pts[eyes_idx]
        mouth = pts[mouth_idx]

        leye, reye = eyes[np.argsort(eyes[:, 0])]
        lmouth, rmouth = mouth[np.argsort(mouth[:, 0])]
        nose = pts[nose_idx]

        if not (leye[0] < reye[0] and lmouth[0] < rmouth[0]):
            return None
        upper_eye = max(leye[1], reye[1])
        lower_mouth = min(lmouth[1], rmouth[1])
        if not (nose[1] > upper_eye and nose[1] < lower_mouth):
            return None

        return np.stack([leye, reye, nose, lmouth, rmouth], axis=0)

    def _align_by_5pts(self, bgr: np.ndarray, pts5: np.ndarray) -> np.ndarray:
        M, _ = cv2.estimateAffinePartial2D(pts5.astype(np.float32), self._ARC_DST, method=cv2.LMEDS)
        if M is None:
            M, _ = cv2.estimateAffinePartial2D(pts5[:3], self._ARC_DST[:3], method=cv2.LMEDS)
        if M is None:
            h, w = bgr.shape[:2]
            interp = cv2.INTER_AREA if max(h, w) > 112 else cv2.INTER_LINEAR
            return cv2.resize(bgr, (112, 112), interpolation=interp)
        return cv2.warpAffine(bgr, M, (112,112), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    def _redetect_align_on_rotations(self, face_bgr: np.ndarray) -> Optional[np.ndarray]:
        """If landmarks are missing, try YOLOv8-face on rotated crops and align."""
        if face_bgr is None or face_bgr.size == 0:
            return None
        h, w = face_bgr.shape[:2]
        if h < 32 or w < 32:
            return None
        # Use a tolerant conf for the tiny crop
        conf = 0.03
        # Prefer sideways hypotheses first; only 180° if those miss.
        for rot in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180):
            img = cv2.rotate(face_bgr, rot)
            try:
                H, W = img.shape[:2]
                dyn = int(min(1280, max(320, max(H, W))))
                if self.progress:
                    angle = (
                        "+90"
                        if rot == cv2.ROTATE_90_CLOCKWISE
                        else "-90"
                        if rot == cv2.ROTATE_90_COUNTERCLOCKWISE
                        else "180"
                    )
                    self.progress(f"roll-fallback: angle={angle}°, imgsz={dyn}")
                res = self.det.predict(
                    img,
                    conf=conf,
                    imgsz=dyn,
                    verbose=False,
                    device=self.device,
                    max_det=60,
                )[0]
            except Exception:
                continue
            kps = self._try_keypoints(res)
            if kps is None or len(kps) == 0:
                continue
            best_i = 0
            try:
                boxes = getattr(res, "boxes", None)
                if boxes is not None:
                    xywh = getattr(boxes, "xywh", None)
                    confs = getattr(boxes, "conf", None)
                    if xywh is not None:
                        centers = xywh
                        if hasattr(centers, "detach"):
                            centers = centers.detach()
                        if hasattr(centers, "cpu"):
                            centers = centers.cpu()
                        centers = np.asarray(centers)
                        if centers.ndim == 1:
                            centers = centers.reshape(1, -1)
                        if centers.size >= 2 and centers.shape[-1] >= 2:
                            centers = centers.reshape(-1, centers.shape[-1])
                            center_xy = centers[:, :2]
                            cx, cy = img.shape[1] / 2.0, img.shape[0] / 2.0
                            dist2 = (center_xy[:, 0] - cx) ** 2 + (center_xy[:, 1] - cy) ** 2
                            if dist2.size:
                                best_i = int(np.argmin(dist2))
                            if confs is not None and dist2.size:
                                conf_arr = confs
                                if hasattr(conf_arr, "detach"):
                                    conf_arr = conf_arr.detach()
                                if hasattr(conf_arr, "cpu"):
                                    conf_arr = conf_arr.cpu()
                                conf_arr = np.asarray(conf_arr).reshape(-1)
                                if conf_arr.size:
                                    diag = math.hypot(img.shape[1], img.shape[0])
                                    dist_norm = np.sqrt(dist2[: conf_arr.size])
                                    if diag > 0:
                                        dist_norm = dist_norm / diag
                                    else:
                                        dist_norm = np.zeros_like(dist_norm)
                                    min_len = min(conf_arr.size, dist_norm.size)
                                    if min_len > 0:
                                        scores = 0.7 * conf_arr[:min_len] - 0.3 * dist_norm[:min_len]
                                        idx = int(np.argmax(scores))
                                        if 0 <= idx < len(kps):
                                            best_i = idx
                                        else:
                                            best_i = max(0, min(len(kps) - 1, idx))
            except Exception:
                pass
            pts5 = kps[best_i][:5, :2].astype(np.float32)
            pts5[:, 0] = np.clip(pts5[:, 0], 0, img.shape[1] - 1)
            pts5[:, 1] = np.clip(pts5[:, 1], 0, img.shape[0] - 1)
            canon = self._canon_5pts(pts5)
            if canon is None:
                continue
            chip = self._align_by_5pts(img, canon)
            if chip is not None:
                if self.progress:
                    self.progress("roll-fallback: success on rotated crop")
                return chip
        return None

    def _upright_by_eye_roll(self, bgr: np.ndarray, pts5: np.ndarray) -> np.ndarray:
        if bgr is None or bgr.size == 0:
            return bgr

        h, w = bgr.shape[:2]
        if h == 0 or w == 0:
            return bgr

        def _resize_for_arc(img: np.ndarray) -> np.ndarray:
            ih, iw = img.shape[:2]
            interp = cv2.INTER_AREA if max(ih, iw) > 112 else cv2.INTER_LINEAR
            return cv2.resize(img, (112, 112), interpolation=interp)

        pts = np.asarray(pts5, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] < 5 or pts.shape[1] < 2:
            return _resize_for_arc(bgr)

        if not np.isfinite(pts[:5, :2]).all():
            return _resize_for_arc(bgr)

        coords = pts[:5, :2].copy()
        if not np.isfinite(coords).all():
            return _resize_for_arc(bgr)

        coords[:, 0] = np.clip(coords[:, 0], 0.0, max(0, w - 1))
        coords[:, 1] = np.clip(coords[:, 1], 0.0, max(0, h - 1))

        eye_vec = coords[1] - coords[0]
        if float(np.hypot(eye_vec[0], eye_vec[1])) >= 1e-3:
            vec = eye_vec
        else:
            mouth_vec = coords[4] - coords[3]
            if float(np.hypot(mouth_vec[0], mouth_vec[1])) >= 1e-3:
                vec = mouth_vec
            else:
                return _resize_for_arc(bgr)

        if float(np.hypot(vec[0], vec[1])) < 1e-3:
            return _resize_for_arc(bgr)

        angle = math.degrees(math.atan2(float(vec[1]), float(vec[0])))
        if angle < -90.0:
            angle += 180.0
        elif angle > 90.0:
            angle -= 180.0

        if abs(angle) < 8.0:
            return _resize_for_arc(bgr)
        if angle > 80.0:
            angle = 90.0
        elif angle < -80.0:
            angle = -90.0

        best_side = max(h, w)
        scale = 1.0 if best_side <= 256 else 256.0 / float(best_side)
        center = (w / 2.0, h / 2.0)
        if hasattr(self, "progress") and callable(self.progress):
            self.progress(f"roll-fallback: angle={angle:.1f}°, scale={scale:.3f}")
        M = cv2.getRotationMatrix2D(center, -angle, scale)
        rotated = cv2.warpAffine(
            bgr,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        pts_h = np.hstack([pts[:5, :2], np.ones((5, 1), dtype=np.float32)])
        pts_rot = (M @ pts_h.T).T.astype(np.float32)

        canon = self._canon_5pts(pts_rot)
        if canon is not None:
            return self._align_by_5pts(rotated, canon.astype(np.float32))

        # fallback if still not canonical
        ih, iw = rotated.shape[:2]
        interp = cv2.INTER_AREA if max(ih, iw) > 112 else cv2.INTER_LINEAR
        return cv2.resize(rotated, (112, 112), interpolation=interp)

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

    def extract(self, bgr_img: np.ndarray, *, imgsz: Optional[int] = None):
        if bgr_img is None or bgr_img.size == 0:
            return []

        # YOLOv8-face detection
        dyn_imgsz = int(imgsz) if (imgsz is not None and imgsz > 0) else 640
        res = self.det.predict(
            bgr_img,
            conf=self.conf,
            verbose=False,
            device=self.device,
            max_det=60,
            imgsz=dyn_imgsz,
        )[0]
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
                pts = kps[i].astype(np.float32, copy=False)
                five = self._canon_5pts(pts[:5])
                if five is not None:
                    pts = five.copy()
                    pts[:, 0] -= float(x1)
                    pts[:, 1] -= float(y1)
                    pts[:, 0] = np.clip(pts[:, 0], 0.0, max(0, face_bgr.shape[1] - 1))
                    pts[:, 1] = np.clip(pts[:, 1], 0.0, max(0, face_bgr.shape[0] - 1))
                    chip = self._align_by_5pts(face_bgr, pts)
                else:
                    pts = pts[:5].copy()
                    pts[:, 0] -= float(x1)
                    pts[:, 1] -= float(y1)
                    chip = self._upright_by_eye_roll(face_bgr, pts)
            else:
                # No landmarks: try re-detecting on ±90° rotated crops, then fall back.
                chip = self._redetect_align_on_rotations(face_bgr)
                if chip is None:
                    interp = cv2.INTER_AREA if max(face_bgr.shape[:2]) > 112 else cv2.INTER_LINEAR
                    chip = cv2.resize(face_bgr, (112, 112), interpolation=interp)
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
