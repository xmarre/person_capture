
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
Y8F_DEFAULT = "yolov8l-face.pt"  # default remains; pass "scrfd_10g_bnkps" to switch backends

# Candidate mirrors for YOLOv8-face weights (keyed by filename)
Y8F_URLS = {
    "yolov8n-face.pt": [
        "https://github.com/lindevs/yolov8-face/releases/download/1.0.1/yolov8n-face.pt",
        "https://github.com/lindevs/yolov8-face/releases/download/1.0.0/yolov8n-face.pt",
        "https://github.com/derronqi/yolov8-face/raw/main/weights/yolov8n-face.pt",
        "https://huggingface.co/junjiang/GestureFace/resolve/main/yolov8n-face.pt",
    ],
    "yolov8l-face.pt": [
        "https://huggingface.co/MonsterZero/yolov8-face/resolve/main/yolov8l-face.pt",
        "https://huggingface.co/vipermu/yolov8-face/resolve/main/yolov8l-face.pt",
    ],
    "yolov8x-face.pt": [
        "https://huggingface.co/MonsterZero/yolov8-face/resolve/main/yolov8x-face.pt",
        "https://huggingface.co/vipermu/yolov8-face/resolve/main/yolov8x-face.pt",
    ],
}

# Verified mirrors for SCRFD ONNX (Hugging Face "resolve" endpoints).
# 10g model is preferred; 2.5g is a smaller fallback.
SCRFD_URLS = {
    "scrfd_10g_bnkps.onnx": [
        "https://huggingface.co/ByteDance/InfiniteYou/resolve/main/supports/insightface/models/antelopev2/scrfd_10g_bnkps.onnx",
        "https://huggingface.co/Aitrepreneur/models-moved/resolve/main/antelopev2/scrfd_10g_bnkps.onnx",
        "https://huggingface.co/Charles-Elena/antelopev2/resolve/main/scrfd_10g_bnkps.onnx",
    ],
    "scrfd_2.5g_bnkps.onnx": [
        "https://huggingface.co/MonsterMMORPG/files1/resolve/main/scrfd_2.5g_bnkps.onnx",
        "https://huggingface.co/OwlMaster/AllFilesRope/resolve/main/models/insightface/models/buffalo_l/scrfd_2.5g_bnkps.onnx",
    ],
}

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


def _round32(x: int) -> int:
    return ((int(x) + 31) // 32) * 32


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
    Face detection: YOLOv8-face or SCRFD via InsightFace.
    Face embedding: OpenCLIP (strong backbone) or ArcFace ONNX for identity.
    Returns list of dicts: {'bbox': np.int32[x1,y1,x2,y2], 'feat': np.float32[D], 'quality': float}
    """

    def __init__(self, ctx: str = 'cuda', yolo_model: str = Y8F_DEFAULT, conf: float = 0.30,
                 use_arcface: bool = True,
                 clip_model_name: str = 'ViT-L-14',
                 clip_pretrained: str = 'laion2b_s32b_b82k',
                 progress=None, trt_lib_dir: Optional[str] = None):
        # Auto-download face detector weights if a bare filename is given
        self.detector_backend = "yolov8"
        self.det = None
        self.scrfd = None
        self.insight_app: Optional[object] = None  # FaceAnalysis fallback that auto-downloads models
        self.conf = float(conf)
        yolo_path = yolo_model
        if isinstance(yolo_model, str) and yolo_model.lower().startswith("scrfd"):
            self.detector_backend = "scrfd"
        elif not os.path.isabs(yolo_path) and os.path.basename(yolo_path).startswith("yolov8") and yolo_path.endswith(".pt"):
            mirrors = Y8F_URLS.get(os.path.basename(yolo_path))
            if mirrors:
                yolo_path = _ensure_file(yolo_path, mirrors, progress=progress)
            elif callable(progress):
                progress(
                    f"Skipping auto-download for {os.path.basename(yolo_path)}; no mirrors configured"
                )

        try:
            import torch as _torch
            _YOLO = None
            if self.detector_backend == "yolov8":
                from ultralytics import YOLO as _YOLO
            import open_clip as _open_clip
        except Exception as e:  # pragma: no cover - executed only when deps missing
            raise RuntimeError(
                "Heavy dependencies not installed; install requirements.txt to run face embedding."
            ) from e

        self.progress = progress
        self._torch = _torch
        # --- pre-scan controls ---
        self._fast_prescan: bool = False
        self._prescan_rr: int = 0
        self._prescan_rr_mode: str = "rr"
        self._prescan_escalate: bool = False
        self._probe_conf: float = 0.03
        self._high_90: int = 1536
        self._high_180: int = 1280
        self._prescan_period: int = 3       # rotate every N prescan samples when empty
        self._prescan_probe_imgsz: int = 384
        self._heavy_cap: int = 2048
        # --- adaptive rotation controls (speed-up on empty scenes) ---
        self._frame_idx: int = 0
        self._no_face_streak: int = 0
        self._last_face_idx: int = -10**9
        self._rot_cycle: int = 0
        # Do rotation passes only occasionally when we have not seen faces recently.
        self.rot_adaptive: bool = True       # set False to force rotations every frame
        self.rot_every_n: int = 12           # run rotations every N frames while empty
        self.rot_after_hit_frames: int = 8   # after a hit, allow rotations for this many frames
        self.fast_no_face_imgsz: int = 512   # shrink 0° pass when streak >= 3
        if self.detector_backend == "yolov8":
            self.det = _YOLO(yolo_path)
        else:
            try:
                from insightface.model_zoo.scrfd import SCRFD as _SCRFD
            except Exception as e:
                raise RuntimeError(
                    "SCRFD backend requires 'insightface'. Install or update the package and retry."
                ) from e
            # --- Force TensorRT for SCRFD (GPU-only) ---
            if not ctx.startswith('cuda'):
                raise RuntimeError("SCRFD requires a CUDA device (use device='cuda' or 'cuda:N').")
            if _torch is None:
                raise RuntimeError("PyTorch not importable; CUDA device resolution required.")
            if not _torch.cuda.is_available():
                raise RuntimeError("CUDA not available to PyTorch; check drivers and torch build.")
            try:
                ctx_id = int(ctx.split(':', 1)[1]) if (':' in ctx) else _torch.cuda.current_device()
            except Exception:
                ctx_id = 0

            # Prepare ONNX Runtime so any session created by SCRFD uses TensorRT EP
            try:
                global ort
                import onnxruntime as ort  # type: ignore
            except Exception as e:
                raise RuntimeError(f"onnxruntime not importable for SCRFD/TRT: {e!r}")
            avail = list(getattr(ort, "get_available_providers", lambda: [])())
            if 'TensorrtExecutionProvider' not in avail:
                raise RuntimeError(f"TensorRT EP not available in onnxruntime. avail={avail}")

            _default_so = getattr(ort, "SessionOptions")()
            _gopt = getattr(ort, "GraphOptimizationLevel", None)
            if _gopt is not None:
                level = getattr(_gopt, "ORT_ENABLE_ALL", None)
                if level is not None:
                    _default_so.graph_optimization_level = level
            _default_so.add_session_config_entry("session.logid", "scrfd_trt")

            _providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider']
            _prov_opts = [
                {"device_id": str(ctx_id)},
                {"device_id": str(ctx_id)},
            ]
            # keep ORT CPU threads low; TRT runs on GPU
            _default_so.intra_op_num_threads = 1
            _default_so.inter_op_num_threads = 1
            if not hasattr(ort, "_pc_trt_patched"):
                _orig_IS = ort.InferenceSession

                def _pc_InferenceSession(model_path, sess_options=None, providers=None, provider_options=None, *a, **kw):
                    try:
                        import os
                        mp = (os.fspath(model_path)).lower() if model_path is not None else ""
                    except Exception:
                        mp = ""
                    is_scrfd = ("scrfd" in mp)
                    if is_scrfd:
                        _p  = providers or _providers
                        _po = provider_options or _prov_opts
                        s = _orig_IS(model_path, sess_options or _default_so, providers=_p, provider_options=_po, *a, **kw)
                        provs = tuple(s.get_providers())
                        if 'TensorrtExecutionProvider' not in provs:
                            raise RuntimeError(f"SCRFD expected TRT EP; got {provs}")
                        if callable(self.progress):
                            self.progress(f"SCRFD providers={provs}")
                        return s
                    # Non-SCRFD sessions: do not touch providers or options.
                    return _orig_IS(model_path, sess_options, providers=providers, provider_options=provider_options, *a, **kw)

                ort.InferenceSession = _pc_InferenceSession  # type: ignore
                ort._pc_trt_patched = True  # type: ignore

            mdl = yolo_model if isinstance(yolo_model, str) else ""
            if not mdl or not mdl.lower().startswith("scrfd"):
                mdl = "scrfd_10g_bnkps.onnx"
            if mdl.lower().startswith("scrfd") and not mdl.lower().endswith(".onnx"):
                mdl = mdl + ".onnx"
            if not os.path.isabs(mdl) or not os.path.exists(mdl):
                mirrors = SCRFD_URLS.get(os.path.basename(mdl))
                if mirrors:
                    mdl = _ensure_file(os.path.basename(mdl), mirrors, progress=self.progress)
            self.scrfd = _SCRFD(mdl)
            # Robust TRT init on the chosen GPU. Retry smaller det_size once. Then warm up.
            _last_err = None
            for _det_size in ((640, 640), (512, 512)):
                try:
                    self.scrfd.prepare(ctx_id=ctx_id, det_size=_det_size)
                    if callable(self.progress):
                        self.progress(f"SCRFD(TRT) ready on cuda:{ctx_id} det={_det_size}")
                    try:
                        _dummy = np.zeros((_det_size[1], _det_size[0], 3), dtype=np.uint8)
                        _ = self.scrfd.detect(_dummy, threshold=max(self.conf, 0.30))
                    except Exception:
                        pass
                    break
                except Exception as e:
                    _last_err = e
            else:
                raise RuntimeError(f"SCRFD(TRT) init failed on cuda:{ctx_id}: {_last_err!r}")
            # Align threshold handling across SCRFD API variants
            try:
                self.scrfd.det_thresh = float(self.conf)
            except Exception:
                pass
            self.insight_app = None  # avoid RetinaFace/FaceAnalysis to prevent CUDA provider issues
        self.use_arcface = bool(use_arcface)
        # Keep ArcFace enabled; we'll align by 5pts only if keypoints are available at inference time.
        self.device = 'cuda' if (ctx.startswith('cuda') and _torch.cuda.is_available()) else 'cpu'
        self.backend = None
        self.arc_sess = None
        self._arc_fixed_batch = False
        self.arc_input = None
        self._arc_scratch: Optional[np.ndarray] = None  # reused fixed-batch buffer for TRT
        self._arc_fixed_batch_len: Optional[int] = None
        self._arc_feat_dim: Optional[int] = None
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
                    if callable(self.progress):
                        try:
                            self.progress(f"ArcFace providers={tuple(self.arc_sess.get_providers())}")
                        except Exception:
                            pass
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

                inp0 = self.arc_sess.get_inputs()[0]
                self.arc_input = inp0.name
                try:
                    shape0 = getattr(inp0, "shape", None)
                    dim0 = shape0[0] if isinstance(shape0, (list, tuple)) and shape0 else None
                    if isinstance(dim0, int) and dim0 > 0:
                        self._arc_fixed_batch_len = dim0
                except Exception:
                    self._arc_fixed_batch_len = None
                self.backend = 'arcface'
                # sanity: output embedding should be 512-D
                out0 = self.arc_sess.get_outputs()[0]
                feat_dim = None
                try:
                    shape = getattr(out0, "shape", None)
                    if isinstance(shape, (list, tuple)) and shape and isinstance(shape[-1], int) and shape[-1] > 0:
                        feat_dim = int(shape[-1])
                except Exception:
                    feat_dim = None
                if feat_dim is None:
                    feat_dim = 512
                self._arc_feat_dim = feat_dim
                try:
                    provs = [p.lower() for p in self.arc_sess.get_providers()]
                    self._arc_fixed_batch = any("tensorrt" in p for p in provs)
                except Exception:
                    self._arc_fixed_batch = False
                if self._arc_fixed_batch:
                    if not self._arc_fixed_batch_len:
                        self._arc_fixed_batch_len = 32
                else:
                    self._arc_fixed_batch_len = None
                    self._arc_scratch = None
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

    def set_prescan_fast(self, enable: bool, *, mode: str = "rr") -> None:
        """Enable/disable light-weight rotation policy for pre-scan."""

        self._fast_prescan = bool(enable)
        self._prescan_rr_mode = str(mode)
        # reset the round-robin counter when enabling fast mode
        if enable:
            self._prescan_rr = 0

    def set_prescan_hint(self, *, escalate: bool = False) -> None:
        """Per-call hint from caller: allow heavy pass on this sample."""

        self._prescan_escalate = bool(escalate)

    def configure_rotation_strategy(
        self,
        *,
        adaptive: Optional[bool] = None,
        every_n: Optional[int] = None,
        after_hit_frames: Optional[int] = None,
        fast_no_face_imgsz: Optional[int] = None,
    ) -> None:
        """Update rotation gating behaviour for SCRFD inference.

        Args:
            adaptive: Enable adaptive gating (False → rotate every frame).
            every_n: Run rotated passes every N frames during no-face streaks.
            after_hit_frames: Allow rotations for this many frames after a detection.
            fast_no_face_imgsz: Shrink the upright pass to this size after long streaks.
        """

        if adaptive is not None:
            self.rot_adaptive = bool(adaptive)
        if every_n is not None:
            try:
                self.rot_every_n = max(1, int(every_n))
            except Exception:
                pass
        if after_hit_frames is not None:
            try:
                self.rot_after_hit_frames = max(0, int(after_hit_frames))
            except Exception:
                pass
        if fast_no_face_imgsz is not None:
            try:
                self.fast_no_face_imgsz = max(0, int(fast_no_face_imgsz))
            except Exception:
                pass
        self._rot_cycle = 0

    def _face_quality(self, bgr):
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(g, cv2.CV_64F).var())

    # ArcFace 112x112 landmark template (lfw standard)
    _ARC_DST = np.array([[38.2946,51.6963],[73.5318,51.5014],[56.0252,71.7366],[41.5493,92.3655],[70.7299,92.2041]], dtype=np.float32)

    def _arcface_preprocess(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != (112,112):
            interp = cv2.INTER_AREA if (rgb.shape[0] > 112 or rgb.shape[1] > 112) else cv2.INTER_LINEAR
            rgb = cv2.resize(rgb, (112,112), interpolation=interp)
        arr = rgb.astype(np.float32) / 127.5 - 1.0
        arr = np.transpose(arr, (2,0,1))[None, ...]  # NCHW
        return arr

    def _arcface_encode(self, bgr_list: List[np.ndarray]):
        if not bgr_list:
            return []

        m = len(bgr_list)
        do_flip = (not getattr(self, "_fast_prescan", False)) or getattr(self, "_prescan_escalate", False)
        pairs: List[np.ndarray] = list(bgr_list)
        if do_flip:
            pairs.extend(cv2.flip(b, 1) for b in bgr_list)

        def _prep(bgr: np.ndarray) -> np.ndarray:
            arr = self._arcface_preprocess(bgr)
            return arr[0]

        X = np.empty((len(pairs), 3, 112, 112), dtype=np.float32)
        for i, bgr in enumerate(pairs):
            X[i] = _prep(bgr)

        X = np.ascontiguousarray(X, dtype=np.float32)
        if getattr(self, "_arc_fixed_batch", False):
            FIX = max(1, int(self._arc_fixed_batch_len or 32))
            if self._arc_scratch is None or self._arc_scratch.shape[0] != FIX:
                self._arc_scratch = np.zeros((FIX, 3, 112, 112), dtype=np.float32, order="C")
                assert self._arc_scratch.flags["C_CONTIGUOUS"]
            feat_dim = int(self._arc_feat_dim or 512)
            feats = np.empty((X.shape[0], feat_dim), dtype=np.float32)
            run = self.arc_sess.run
            inp = self.arc_input
            for start in range(0, X.shape[0], FIX):
                end = min(start + FIX, X.shape[0])
                n = end - start
                if n == FIX:
                    np.copyto(self._arc_scratch, X[start:end], casting="no")
                else:
                    self._arc_scratch.fill(0.0)
                    self._arc_scratch[:n] = X[start:end]
                out = run(None, {inp: self._arc_scratch})[0]
                feats[start:end] = out[:n]
        else:
            feats = self.arc_sess.run(None, {self.arc_input: X})[0]

        if feats.shape[0] != X.shape[0]:
            raise RuntimeError(f"ArcFace rows {feats.shape[0]} != input {X.shape[0]}")

        f = feats[:m].copy()
        if do_flip:
            f += feats[m:2 * m]
        norms = np.linalg.norm(f, axis=1, keepdims=True).astype(np.float32, copy=False)
        np.maximum(norms, 1e-6, out=norms)
        f /= norms
        return f.astype(np.float32, copy=False)

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
        H0, W0 = bgr_img.shape[:2]

        if self.detector_backend == "scrfd":
            return self._extract_with_scrfd(bgr_img, imgsz=imgsz)

        # YOLOv8-face detection
        dyn_imgsz = int(imgsz) if (imgsz is not None and imgsz > 0) else 640
        dyn_imgsz = max(320, dyn_imgsz)
        dyn_imgsz = _round32(dyn_imgsz)  # divisible by 32
        L = max(H0, W0)
        heavy_cap = max(int(getattr(self, "_heavy_cap", 2048)), dyn_imgsz)  # optional safety cap
        heavy_size_auto = min(_round32(max(dyn_imgsz, int(0.75 * L))), heavy_cap)
        heavy_size_auto_180 = min(_round32(max(dyn_imgsz, int(0.67 * L))), heavy_cap)
        try:
            res = self.det.predict(
                bgr_img,
                conf=self.conf,
                verbose=False,
                device=self.device,
                max_det=60,
                imgsz=dyn_imgsz,
                iou=0.30,
                augment=True,
            )[0]
        except Exception:
            res = self.det.predict(
                bgr_img,
                conf=self.conf,
                verbose=False,
                device=self.device,
                max_det=60,
                imgsz=dyn_imgsz,
                iou=0.30,
            )[0]
        boxes = []
        for b in res.boxes:
            x1, y1, x2, y2 = [int(v.item()) for v in b.xyxy[0]]
            boxes.append((x1, y1, x2, y2))

        if not boxes:
            # multi-scale TTA before rot-fallback
            if not self._fast_prescan:
                for s in (1.25, 1.5):
                    img_s = cv2.resize(bgr_img, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
                    imgsz_s = max(320, int(dyn_imgsz * s))
                    imgsz_s = ((imgsz_s + 31) // 32) * 32
                    try:
                        res_s = self.det.predict(
                            img_s,
                            conf=min(self.conf, 0.10),
                            verbose=False,
                            device=self.device,
                            max_det=80,
                            imgsz=imgsz_s,
                            iou=0.30,
                            augment=True,
                        )[0]
                    except Exception:
                        try:
                            res_s = self.det.predict(
                                img_s,
                                conf=min(self.conf, 0.10),
                                verbose=False,
                                device=self.device,
                                max_det=80,
                                imgsz=imgsz_s,
                                iou=0.30,
                            )[0]
                        except Exception:
                            continue
                    bxs_s = getattr(res_s, "boxes", None)
                    if bxs_s is None or len(bxs_s) == 0:
                        continue
                    confs_s = getattr(bxs_s, "conf", None)
                    for j, b in enumerate(bxs_s):
                        confv = 1.0
                        if confs_s is not None and len(confs_s) > j:
                            try:
                                confv = float(confs_s[j].item())
                            except Exception:
                                try:
                                    confv = float(confs_s[j])
                                except Exception:
                                    confv = 1.0
                        if confv < 0.05:
                            continue
                        x1s, y1s, x2s, y2s = [v.item() for v in b.xyxy[0]]
                        x1 = max(0, min(W0 - 1, int(round(x1s / s))))
                        y1 = max(0, min(H0 - 1, int(round(y1s / s))))
                        x2 = max(x1 + 1, min(W0, int(round(x2s / s))))
                        y2 = max(y1 + 1, min(H0, int(round(y2s / s))))
                        boxes.append((x1, y1, x2, y2))
                    if boxes:
                        break

        if not boxes:
            # Bootstrap with rotated full-frame when detector misses
            fullframe_sizes: List[int] = []
            if self._fast_prescan:
                fullframe_sizes = [dyn_imgsz]
            else:
                for base in (max(dyn_imgsz, 1280), max(dyn_imgsz, 1536)):
                    base = ((int(base) + 31) // 32) * 32
                    if base not in fullframe_sizes:
                        fullframe_sizes.append(base)
                if not fullframe_sizes:
                    fullframe_sizes = [max(dyn_imgsz, 896)]

            def _map_pts_back(xr, yr, rot):
                # inverse mappings from rotated -> original
                if rot == cv2.ROTATE_90_CLOCKWISE:         # x_o = y_r, y_o = H0-1-x_r
                    return (yr, H0 - 1 - xr)
                if rot == cv2.ROTATE_90_COUNTERCLOCKWISE:  # x_o = W0-1-y_r, y_o = x_r
                    return (W0 - 1 - yr, xr)
                # 180 rotation
                return (W0 - 1 - xr, H0 - 1 - yr)

            def _map_box_back(x1, y1, x2, y2, rot):
                xs = [x1, x2, x2, x1]
                ys = [y1, y1, y2, y2]
                pts = [_map_pts_back(x, y, rot) for x, y in zip(xs, ys)]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))

            if self._fast_prescan:
                rr = self._prescan_rr % 2
                if self._prescan_rr_mode == "rr":
                    rot_list = (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    rot_seq = (rot_list[rr],)
                    self._prescan_rr += 1
                else:
                    rot_seq = (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                rot_seq = (
                    cv2.ROTATE_90_CLOCKWISE,
                    cv2.ROTATE_90_COUNTERCLOCKWISE,
                    cv2.ROTATE_180,
                )
            for rot in rot_seq:
                img_r = cv2.rotate(bgr_img, rot)
                res_r = None
                try:
                    probe = self.det.predict(
                        img_r,
                        conf=self._probe_conf,
                        imgsz=dyn_imgsz,
                        verbose=False,
                        device=self.device,
                        max_det=40,
                        iou=0.40,
                        augment=False,
                    )[0]
                except Exception:
                    probe = None
                probe_boxes = getattr(probe, "boxes", None)
                probe_hits = int(getattr(probe_boxes, "__len__", lambda: 0)())

                do_heavy = (
                    (probe_hits > 0)
                    or (self._fast_prescan and self._prescan_escalate)
                    or (not self._fast_prescan)
                )
                if self._fast_prescan:
                    if rot == cv2.ROTATE_180:
                        heavy_size = heavy_size_auto_180
                        override = self._high_180
                    else:
                        heavy_size = heavy_size_auto
                        override = self._high_90
                    if override and override > 0:
                        heavy_size = max(heavy_size, _round32(int(override)))
                    heavy_size = min(heavy_size, heavy_cap)
                    det_sizes = [dyn_imgsz] if not do_heavy else [heavy_size]
                else:
                    det_sizes = fullframe_sizes if do_heavy else [dyn_imgsz]

                for det_size in det_sizes:
                    try:
                        res_r = self.det.predict(
                            img_r,
                            conf=min(self.conf, 0.10),
                            imgsz=det_size,
                            verbose=False,
                            device=self.device,
                            max_det=80,
                            iou=0.30,
                            augment=(not self._fast_prescan) or do_heavy,
                        )[0]
                    except Exception:
                        try:
                            res_r = self.det.predict(
                                img_r,
                                conf=min(self.conf, 0.10),
                                imgsz=det_size,
                                verbose=False,
                                device=self.device,
                                max_det=80,
                                iou=0.30,
                            )[0]
                        except Exception:
                            res_r = None
                            continue
                    bxs_r = getattr(res_r, "boxes", None)
                    if bxs_r is not None and len(bxs_r) > 0:
                        break
                    res_r = None
                if res_r is None:
                    continue
                bxs_r = getattr(res_r, "boxes", None)
                if bxs_r is None or len(bxs_r) == 0:
                    continue
                kps_r = self._try_keypoints(res_r)
                idx = 0
                try:
                    confs = getattr(bxs_r, "conf", None)
                    if confs is not None and len(confs) > 0:
                        idx = int(confs.argmax().item())
                except Exception:
                    pass
                bx = bxs_r.xyxy[idx].tolist()
                x1r, y1r, x2r, y2r = [int(v) for v in bx]
                Hr, Wr = img_r.shape[:2]
                x1r = max(0, min(Wr - 1, x1r))
                y1r = max(0, min(Hr - 1, y1r))
                x2r = max(x1r + 1, min(Wr, x2r))
                y2r = max(y1r + 1, min(Hr, y2r))
                chip = None
                if kps_r is not None and len(kps_r) > idx:
                    pts5 = kps_r[idx][:5, :2].astype(np.float32)
                    pts5[:, 0] = np.clip(pts5[:, 0], 0, img_r.shape[1] - 1)
                    pts5[:, 1] = np.clip(pts5[:, 1], 0, img_r.shape[0] - 1)
                    canon = self._canon_5pts(pts5)
                    if canon is not None:
                        chip = self._align_by_5pts(img_r, canon)
                if chip is None:
                    crop_r = img_r[y1r:y2r, x1r:x2r]
                    if crop_r.size == 0:
                        continue
                    interp = (
                        cv2.INTER_AREA
                        if max(crop_r.shape[:2]) > 112
                        else cv2.INTER_LINEAR
                    )
                    chip = cv2.resize(crop_r, (112, 112), interpolation=interp)
                x1o, y1o, x2o, y2o = _map_box_back(x1r, y1r, x2r, y2r, rot)
                x1o = max(0, min(W0 - 1, x1o))
                y1o = max(0, min(H0 - 1, y1o))
                x2o = max(x1o + 1, min(W0, x2o))
                y2o = max(y1o + 1, min(H0, y2o))
                if (x2o - x1o) * (y2o - y1o) < 32 * 32:
                    continue
                feats = (
                    self._arcface_encode([chip])
                    if self.use_arcface
                    else self._clip_encode([chip])
                )
                return [
                    {
                        "bbox": np.array([x1o, y1o, x2o, y2o], dtype=np.int32),
                        "feat": feats[0],
                        "quality": float(self._face_quality(chip)),
                    }
                ]

            # Try arbitrary angles via affine rotation (±45°, ±135°)
            if self._fast_prescan:
                return []

            def _affine_rotate(img, angle_deg):
                h, w = img.shape[:2]
                M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
                r = cv2.warpAffine(
                    img,
                    M,
                    (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(114, 114, 114),
                )
                return r, M

            def _map_box_back_affine(x1, y1, x2, y2, M):
                A = np.vstack([M, [0, 0, 1]]).astype(np.float32)
                Minv = np.linalg.inv(A)[:2, :]
                pts = np.array([[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]], dtype=np.float32).T
                back = Minv @ pts
                xs, ys = back[0], back[1]
                x1o, y1o = int(np.floor(xs.min())), int(np.floor(ys.min()))
                x2o, y2o = int(np.ceil(xs.max())), int(np.ceil(ys.max()))
                return x1o, y1o, x2o, y2o

            for ang in (45, -45, 135, -135):
                img_r, M = _affine_rotate(bgr_img, ang)
                res_r = None
                for det_size in fullframe_sizes:
                    try:
                        res_r = self.det.predict(
                            img_r,
                            conf=min(self.conf, 0.10),
                            imgsz=det_size,
                            verbose=False,
                            device=self.device,
                            max_det=80,
                            iou=0.30,
                            augment=True,
                        )[0]
                    except Exception:
                        try:
                            res_r = self.det.predict(
                                img_r,
                                conf=min(self.conf, 0.10),
                                imgsz=det_size,
                                verbose=False,
                                device=self.device,
                                max_det=80,
                                iou=0.30,
                            )[0]
                        except Exception:
                            res_r = None
                            continue
                    bxs_r = getattr(res_r, "boxes", None)
                    if bxs_r is not None and len(bxs_r) > 0:
                        break
                    res_r = None
                if res_r is None:
                    continue
                bxs_r = getattr(res_r, "boxes", None)
                if bxs_r is None or len(bxs_r) == 0:
                    continue
                kps_r = self._try_keypoints(res_r)
                idx = 0
                try:
                    confs = getattr(bxs_r, "conf", None)
                    if confs is not None and len(confs) > 0:
                        idx = int(confs.argmax().item())
                except Exception:
                    pass
                bx = bxs_r.xyxy[idx].tolist()
                x1r, y1r, x2r, y2r = [int(v) for v in bx]
                x1r = max(0, min(img_r.shape[1] - 1, x1r))
                y1r = max(0, min(img_r.shape[0] - 1, y1r))
                x2r = max(x1r + 1, min(img_r.shape[1], x2r))
                y2r = max(y1r + 1, min(img_r.shape[0], y2r))

                chip = None
                if kps_r is not None and len(kps_r) > idx:
                    pts5 = kps_r[idx][:5, :2].astype(np.float32)
                    pts5[:, 0] = np.clip(pts5[:, 0], 0, img_r.shape[1] - 1)
                    pts5[:, 1] = np.clip(pts5[:, 1], 0, img_r.shape[0] - 1)
                    canon = self._canon_5pts(pts5)
                    if canon is not None:
                        chip = self._align_by_5pts(img_r, canon)
                if chip is None:
                    crop_r = img_r[y1r:y2r, x1r:x2r]
                    if crop_r.size == 0:
                        continue
                    interp = cv2.INTER_AREA if max(crop_r.shape[:2]) > 112 else cv2.INTER_LINEAR
                    chip = cv2.resize(crop_r, (112, 112), interpolation=interp)

                x1o, y1o, x2o, y2o = _map_box_back_affine(x1r, y1r, x2r, y2r, M)
                x1o = max(0, min(W0 - 1, x1o)); y1o = max(0, min(H0 - 1, y1o))
                x2o = max(x1o + 1, min(W0, x2o)); y2o = max(y1o + 1, min(H0, y2o))
                if (x2o - x1o) * (y2o - y1o) < 32 * 32:
                    continue
                feats = self._arcface_encode([chip]) if self.use_arcface else self._clip_encode([chip])
                return [{
                    "bbox": np.array([x1o, y1o, x2o, y2o], dtype=np.int32),
                    "feat": feats[0],
                    "quality": float(self._face_quality(chip)),
                }]

            return []

        boxes = self._nms_boxes(boxes, iou_thr=0.45)

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

    def _extract_with_scrfd(self, bgr_img: np.ndarray, *, imgsz: Optional[int] = None):
        self._frame_idx += 1
        # prescan throttle for both SCRFD and insight_app backends
        if self._fast_prescan:
            period = max(1, int(getattr(self, "_prescan_period", 3)))
            if not (
                self._prescan_escalate
                or ((self._frame_idx + self._prescan_rr) % period) == 0
            ):
                return []
        if self.scrfd is not None:
            return self._extract_with_scrfd_raw(bgr_img, imgsz=imgsz)
        if self.insight_app is not None:
            H0, W0 = bgr_img.shape[:2]
            faces = self.insight_app.get(bgr_img)
            if not faces:
                return []
            kept = []
            for f in faces:
                if hasattr(f, "det_score") and float(f.det_score) < float(self.conf):
                    continue
                x1, y1, x2, y2 = [int(v) for v in f.bbox.astype(int)]
                x1 = max(0, min(W0 - 1, x1))
                y1 = max(0, min(H0 - 1, y1))
                x2 = max(x1 + 1, min(W0, x2))
                y2 = max(y1 + 1, min(H0, y2))
                pts = None
                if hasattr(f, "kps") and f.kps is not None:
                    pts = np.asarray(f.kps, dtype=np.float32)
                    if pts.ndim == 2 and pts.shape[0] >= 5:
                        pts = pts[:5]
                        pts[:, 0] = np.clip(pts[:, 0] - x1, 0, max(0, (x2 - x1) - 1))
                        pts[:, 1] = np.clip(pts[:, 1] - y1, 0, max(0, (y2 - y1) - 1))
                    else:
                        pts = None
                kept.append(((x1, y1, x2, y2), pts))
            crops = []
            metas = []
            for (x1, y1, x2, y2), kps in kept:
                face_bgr = bgr_img[y1:y2, x1:x2]
                if self.use_arcface and kps is not None:
                    canon = self._canon_5pts(kps)
                    chip = (
                        self._align_by_5pts(face_bgr, canon)
                        if canon is not None
                        else self._upright_by_eye_roll(face_bgr, kps)
                    )
                else:
                    interp = cv2.INTER_AREA if max(face_bgr.shape[:2]) > 112 else cv2.INTER_LINEAR
                    chip = cv2.resize(face_bgr, (112, 112), interpolation=interp)
                crops.append(chip)
                metas.append((x1, y1, x2, y2, float(self._face_quality(chip))))
            feats = self._arcface_encode(crops) if self.use_arcface else self._clip_encode(crops)
            out = [
                {
                    'bbox': np.array([m[0], m[1], m[2], m[3]], dtype=np.int32),
                    'feat': feats[i],
                    'quality': m[4],
                }
                for i, m in enumerate(metas)
            ]
            out.sort(
                key=lambda f: (
                    f['quality'],
                    (f['bbox'][2] - f['bbox'][0]) * (f['bbox'][3] - f['bbox'][1]),
                ),
                reverse=True,
            )
            return out
        raise RuntimeError("SCRFD not initialized.")

    def _extract_with_scrfd_raw(self, bgr_img: np.ndarray, *, imgsz: Optional[int] = None):

        def _rot_img(img, deg):
            if deg == 90:   return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            if deg == 180:  return cv2.rotate(img, cv2.ROTATE_180)
            if deg == 270:  return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            return img
        def _map_xy_from_rot(xr: float, yr: float, deg: int, W0: int, H0: int):
            if deg == 0:   return xr, yr
            if deg == 90:  return yr, H0 - 1 - xr
            if deg == 180: return W0 - 1 - xr, H0 - 1 - yr
            if deg == 270: return W0 - 1 - yr, xr
            return xr, yr
        def _detect_once(img, dyn, conf: float):
            # handle SCRFD API variants
            try:
                return self.scrfd.detect(img, thresh=float(conf), input_size=(dyn, dyn))
            except TypeError:
                try:
                    return self.scrfd.detect(img, det_thresh=float(conf), input_size=(dyn, dyn))
                except TypeError:
                    try:
                        self.scrfd.det_thresh = float(conf)
                    except Exception:
                        pass
                    try:
                        return self.scrfd.detect(img, input_size=(dyn, dyn))
                    except TypeError:
                        return self.scrfd.detect(img, (dyn, dyn))

        H0, W0 = bgr_img.shape[:2]
        dyn_req = int(imgsz) if (imgsz is not None and imgsz > 0) else 640
        dyn = dyn_req
        # speed-up: during a no-face streak, use a smaller first-pass size
        if self._no_face_streak >= 3:
            dyn = min(dyn, self.fast_no_face_imgsz)
        # in fast pre-scan, cap upright pass to probe size for empty scenes
        if self._fast_prescan:
            dyn = min(dyn, int(getattr(self, "_prescan_probe_imgsz", 384)))
        dyn = _round32(max(320, dyn))
        L = max(H0, W0)
        heavy_cap = max(int(getattr(self, "_heavy_cap", 2048)), dyn)  # optional safety cap
        heavy90 = min(_round32(max(dyn, int(0.75 * L))), heavy_cap)
        heavy180 = min(_round32(max(dyn, int(0.67 * L))), heavy_cap)

        # 1) Try 0° first
        bboxes, kpss = _detect_once(bgr_img, dyn, conf=float(self.conf))

        dets = []  # list of (box, kps, score)
        def _accumulate(bb, kp, deg):
            x1, y1, x2, y2 = [int(v) for v in bb[:4]]
            # map corners back to original
            x1o, y1o = _map_xy_from_rot(x1, y1, deg, W0, H0)
            x2o, y2o = _map_xy_from_rot(x2, y2, deg, W0, H0)
            xa1, ya1 = min(x1o, x2o), min(y1o, y2o)
            xa2, ya2 = max(x1o, x2o), max(y1o, y2o)
            xa1 = max(0, min(W0 - 1, xa1)); ya1 = max(0, min(H0 - 1, ya1))
            xa2 = max(xa1 + 1, min(W0, xa2)); ya2 = max(ya1 + 1, min(H0, ya2))
            if xa2 - xa1 <= 2 or ya2 - ya1 <= 2:
                return
            pts = None
            if kp is not None:
                pts = np.asarray(kp, dtype=np.float32)
                pts = pts.reshape(-1, 2)
                pts_mapped = []
                for (px, py) in pts:
                    ox, oy = _map_xy_from_rot(float(px), float(py), deg, W0, H0)
                    # convert to local crop coords for alignment step
                    pts_mapped.append([float(ox - xa1), float(oy - ya1)])
                if len(pts_mapped) >= 5:
                    pts = np.asarray(pts_mapped[:5], dtype=np.float32)
                else:
                    pts = None
            score = float(bb[4]) if len(bb) > 4 else 1.0
            dets.append(((xa1, ya1, xa2, ya2), pts, score))

        # collect 0°
        if bboxes is not None and len(bboxes) > 0:
            for i, bb in enumerate(bboxes):
                kp = None if (kpss is None or i >= len(kpss)) else kpss[i]
                _accumulate(bb, kp, 0)
        # decide whether to run rotations
        need_rot = False
        if not dets:
            # empty this frame
            self._no_face_streak += 1
            # adaptive policy: rotate only occasionally while empty
            if self.rot_adaptive:
                if (self._frame_idx - self._last_face_idx) <= self.rot_after_hit_frames:
                    need_rot = True  # just after a hit, allow rotations
                elif ((self._frame_idx + (id(self) & 7)) % self.rot_every_n) == 0:
                    need_rot = True  # periodic probe
            else:
                need_rot = True      # always rotate if configured so
        else:
            # we have faces; reset streak and remember index
            self._no_face_streak = 0
            self._last_face_idx = self._frame_idx
            self._rot_cycle = 0

        if self._fast_prescan:
            # rotate only periodically when empty; escalate when active
            period = max(1, int(getattr(self, "_prescan_period", 3)))
            need_rot = need_rot or self._prescan_escalate or (
                ((self._frame_idx + self._prescan_rr) % period) == 0
            )

        if self._fast_prescan and (not dets) and (not need_rot):
            return []

        # 2) If requested, try rotated views: 90, 270, then 180
        if (not dets) and need_rot:
            self._rot_cycle += 1
            if self._fast_prescan:
                rr = self._prescan_rr % 2
                if self._prescan_rr_mode == "rr":
                    rot_seq = (90, 270)
                    rot_seq = (rot_seq[rr],)
                    self._prescan_rr += 1
                else:
                    rot_seq = (90, 270)
            else:
                rot_seq = (90, 270, 180)
            for deg in rot_seq:
                rimg_probe = _rot_img(bgr_img, deg)
                probe_conf = max(0.02, float(getattr(self, "_probe_conf", 0.02)))
                probe_dyn = _round32(
                    max(320, min(dyn, int(getattr(self, "_prescan_probe_imgsz", 384))))
                )
                probe_boxes, _ = _detect_once(rimg_probe, probe_dyn, conf=probe_conf)
                probe_hits = len(probe_boxes) if probe_boxes is not None else 0

                do_heavy = (
                    (probe_hits > 0)
                    or (self._fast_prescan and self._prescan_escalate)
                    or (not self._fast_prescan)
                )
                if self._fast_prescan and (probe_boxes is None or len(probe_boxes) == 0):
                    continue  # no heavy pass on this rotation

                # heavy path needs padding to avoid edge loss
                pad = 24
                rimg = cv2.copyMakeBorder(rimg_probe, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

                if self._fast_prescan:
                    heavy = heavy180 if deg == 180 else heavy90
                    override = getattr(self, "_high_180" if deg == 180 else "_high_90", None)
                    if override and override > 0:
                        heavy = max(heavy, _round32(int(override)))
                    heavy = min(heavy, int(getattr(self, "_heavy_cap", 2048)))
                    det_sizes = [dyn] if not do_heavy else [heavy]
                else:
                    fullframe_sizes: List[int] = []
                    for base in (max(dyn, 1280), max(dyn, 1536)):
                        base = _round32(base)
                        if base not in fullframe_sizes:
                            fullframe_sizes.append(base)
                    if not fullframe_sizes:
                        fullframe_sizes = [_round32(max(dyn, 896))]
                    det_sizes = fullframe_sizes if do_heavy else [dyn]
                conf_deg = max(0.10, float(self.conf) * (0.8 if deg in (90, 270) else 0.6))
                rb = rk = None
                for det_size in det_sizes:
                    rb, rk = _detect_once(rimg, det_size, conf=conf_deg)
                    if rb is not None and len(rb) > 0:
                        break
                    rb = rk = None
                if rb is None or len(rb) == 0:
                    continue
                for i, bb in enumerate(rb):
                    kp = None if (rk is None or i >= len(rk)) else rk[i]
                    # remove padding offset before remap
                    if bb is not None:
                        bb = np.asarray(bb).copy()
                        bb[:4] -= np.array([pad, pad, pad, pad], dtype=bb.dtype)
                    if kp is not None:
                        kp = np.asarray(kp)
                        kp[..., 0] -= pad
                        kp[..., 1] -= pad
                    _accumulate(bb, kp, deg)
                if dets:
                    break

        if not dets:
            return []

        # NMS across rotations, keep highest-score boxes
        dets = sorted(dets, key=lambda t: (t[2], (t[0][2]-t[0][0])*(t[0][3]-t[0][1])), reverse=True)
        kept = []
        for box, pts, sc in dets:
            if all(self._iou(box, k[0]) < 0.45 for k in kept):
                kept.append((box, pts, sc))

        faces: List[Tuple[int, int, int, int, np.ndarray, float]] = []
        crops: List[np.ndarray] = []
        for (x1, y1, x2, y2), kps, _sc in kept:
            xi1 = max(0, min(W0 - 1, int(round(x1))))
            yi1 = max(0, min(H0 - 1, int(round(y1))))
            xi2 = max(xi1 + 1, min(W0,     int(round(x2))))
            yi2 = max(yi1 + 1, min(H0,     int(round(y2))))
            face_bgr = bgr_img[yi1:yi2, xi1:xi2]
            chip = None
            if self.use_arcface and kps is not None:
                pts = np.asarray(kps, dtype=np.float32)
                canon = self._canon_5pts(pts)
                chip = self._align_by_5pts(face_bgr, canon) if canon is not None else self._upright_by_eye_roll(face_bgr, pts)
            if chip is None:
                interp = cv2.INTER_AREA if max(face_bgr.shape[:2]) > 112 else cv2.INTER_LINEAR
                chip = cv2.resize(face_bgr, (112, 112), interpolation=interp)
            q = self._face_quality(chip)
            faces.append((xi1, yi1, xi2, yi2, chip, q))
            crops.append(chip)

        feats = self._arcface_encode(crops) if self.use_arcface else self._clip_encode(crops)
        out = [
            {
                'bbox': np.array([x1, y1, x2, y2], dtype=np.int32),
                'feat': feats[i],
                'quality': float(q),
            }
            for i, (x1, y1, x2, y2, _, q) in enumerate(faces)
        ]
        out.sort(
            key=lambda f: (
                f['quality'],
                (f['bbox'][2] - f['bbox'][0]) * (f['bbox'][3] - f['bbox'][1]),
            ),
            reverse=True,
        )
        # if still empty, streak already incremented; otherwise it's reset above
        return out

    @staticmethod
    def _iou(a, b):
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
        area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
        denom = area_a + area_b - inter
        return inter / denom if denom > 0 else 0.0

    @staticmethod
    def _nms_boxes(boxes, iou_thr=0.5):
        kept = []
        for b in sorted(boxes, key=lambda t: (t[2] - t[0]) * (t[3] - t[1]), reverse=True):
            if all(FaceEmbedder._iou(b, k) < iou_thr for k in kept):
                kept.append(b)
        return kept

    @staticmethod
    def best_face(faces):
        if not faces:
            return None
        return max(faces, key=lambda f: (f['quality'], (f['bbox'][2]-f['bbox'][0]) * (f['bbox'][3]-f['bbox'][1])))
