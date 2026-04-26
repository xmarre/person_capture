#!/usr/bin/env python3
"""
GUI for PersonCapture: target-person finder and 2:3 crops from video.

Requirements:
  pip install PySide6
  # plus the project's requirements (torch, ultralytics, open-clip-torch, opencv-python, etc.)

Run:
  python gui_app.py
"""

from __future__ import annotations
import time

import os
import sys
import subprocess
import shutil
import threading
import struct
import hashlib
import json
import csv
import traceback
import logging

# Ensure logs also go to stdout (console), not only to the Qt log widget.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
import cv2
import numpy as np
import bisect
import queue
import math
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List
from pathlib import Path

# ---------- Repo root + caches MUST be set before any model/video imports ----------
_PKG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parent if _PKG_DIR.name == "person_capture" else _PKG_DIR
# Ensure logging is live early so video_io can emit
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)
# Defaults; run() may override from cfg
os.environ.setdefault("PERSON_CAPTURE_TRT_CACHE_ROOT", str(_REPO_ROOT / "trt_cache"))

# Respect overrides: derive paths from env, then create those paths only
_UL_HOME = Path(os.environ.get("ULTRALYTICS_HOME", str(_REPO_ROOT / ".ultralytics")))
os.environ.setdefault("ULTRALYTICS_HOME", str(_UL_HOME))

_UL_SETTINGS = os.environ.get("ULTRALYTICS_SETTINGS")
if not _UL_SETTINGS:
    _UL_SETTINGS = str(_UL_HOME / "settings.yaml")
    os.environ["ULTRALYTICS_SETTINGS"] = _UL_SETTINGS

# Create exactly the chosen paths
_UL_HOME.mkdir(parents=True, exist_ok=True)
Path(_UL_SETTINGS).parent.mkdir(parents=True, exist_ok=True)
Path(_UL_SETTINGS).touch(exist_ok=True)
os.environ.setdefault("HF_HOME", str(_REPO_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_REPO_ROOT / ".cache" / "huggingface"))
_log.info("INIT Repo root=%s", _REPO_ROOT)
_log.info("INIT TRT cache root=%s", os.getenv("PERSON_CAPTURE_TRT_CACHE_ROOT"))
_log.info("INIT ULTRALYTICS_HOME=%s", os.getenv("ULTRALYTICS_HOME"))
_log.info("INIT ULTRALYTICS_SETTINGS=%s", os.getenv("ULTRALYTICS_SETTINGS"))
# -----------------------------------------------------------------------------------


# Robust getter for HDR helper (try package then flat before stubbing)
def _get_open_video_with_tonemap():
    lg = logging.getLogger(__name__)
    try:
        from .video_io import (
            open_video_with_tonemap as _fn,
            hdr_detect_reason as _reason,
            open_hdr_passthrough_reader as _open_hdr_reader,
        )  # type: ignore
        lg.info("HDR: using package video_io")
        return _fn, _reason, _open_hdr_reader
    except Exception as e1:
        lg.warning("HDR: package video_io import failed (%s), trying flat", e1)
        try:
            from video_io import (
                open_video_with_tonemap as _fn,
                hdr_detect_reason as _reason,
                open_hdr_passthrough_reader as _open_hdr_reader,
            )  # type: ignore
            lg.info("HDR: using flat video_io")
            return _fn, _reason, _open_hdr_reader
        except Exception as e2:
            lg.warning("HDR disabled: cannot import video_io (%s)", e2)
            return (
                lambda _path: None,
                lambda _path: "unknown",
                lambda _path: None,
            )


# Robust imports: support both package ("from .module") and flat files ("import module").
def _imp():
    try:
        # --- Updater (new) ---
        try:
            from .updater import UpdateManager  # type: ignore
        except Exception:
            UpdateManager = None  # type: ignore
        (
            open_video_with_tonemap,
            hdr_detect_reason,
            open_hdr_passthrough_reader,
        ) = _get_open_video_with_tonemap()

        from .detectors import PersonDetector  # type: ignore
        from .face_embedder import FaceEmbedder  # type: ignore
        from .reid_embedder import ReIDEmbedder  # type: ignore
        from .utils import (
            ensure_dir,
            parse_ratio,
            expand_box_to_ratio,
            phash_similarity,
            _phash_bits,
        )

        return (
            PersonDetector,
            FaceEmbedder,
            ReIDEmbedder,
            ensure_dir,
            parse_ratio,
            expand_box_to_ratio,
            phash_similarity,
            _phash_bits,
            open_video_with_tonemap,
            hdr_detect_reason,
            open_hdr_passthrough_reader,
            UpdateManager,
        )
    except Exception:
        # Add script dir to sys.path and try again as flat modules
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.insert(0, here)
        from detectors import PersonDetector  # type: ignore
        from face_embedder import FaceEmbedder  # type: ignore
        from reid_embedder import ReIDEmbedder  # type: ignore
        try:
            from updater import UpdateManager  # type: ignore
        except Exception:
            UpdateManager = None  # type: ignore
        (
            open_video_with_tonemap,
            hdr_detect_reason,
            open_hdr_passthrough_reader,
        ) = _get_open_video_with_tonemap()
        try:
            from .utils import ensure_dir, parse_ratio, expand_box_to_ratio, phash_similarity, _phash_bits  # type: ignore
        except Exception:
            from utils import ensure_dir, parse_ratio, expand_box_to_ratio, phash_similarity, _phash_bits  # type: ignore
        return (
            PersonDetector,
            FaceEmbedder,
            ReIDEmbedder,
            ensure_dir,
            parse_ratio,
            expand_box_to_ratio,
            phash_similarity,
            _phash_bits,
            open_video_with_tonemap,
            hdr_detect_reason,
            open_hdr_passthrough_reader,
            UpdateManager,
        )

(
    PersonDetector,
    FaceEmbedder,
    ReIDEmbedder,
    ensure_dir,
    parse_ratio,
    expand_box_to_ratio,
    phash_similarity,
    _phash_bits,
    open_video_with_tonemap,
    hdr_detect_reason,
    open_hdr_passthrough_reader,
    UpdateManager,
) = _imp()

try:
    from .utils import set_ffmpeg_env, resolve_ffmpeg_bins, ffmpeg_has_hdr_filters  # type: ignore
except Exception:  # pragma: no cover - fallback for flat layout
    from utils import set_ffmpeg_env, resolve_ffmpeg_bins, ffmpeg_has_hdr_filters  # type: ignore
# Optional Curate tab
try:
    from .gui_curate_tab import CurateTab  # type: ignore
except Exception:
    try:
        from gui_curate_tab import CurateTab  # type: ignore
    except Exception:
        CurateTab = None  # type: ignore
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtWidgets import QDockWidget


logger = logging.getLogger(__name__)


try:
    from .hdr_preview import hdr_passthrough_available as _hdr_passthrough_available  # type: ignore
except Exception:
    try:
        from hdr_preview import hdr_passthrough_available as _hdr_passthrough_available  # type: ignore
    except Exception:
        def _hdr_passthrough_available() -> bool:  # type: ignore
            return False


class FileList(QtWidgets.QListWidget):
    filesDropped = QtCore.Signal(list)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        if event.mimeData().hasUrls():
            self.filesDropped.emit([url.toLocalFile() for url in event.mimeData().urls()])
            event.acceptProposedAction()
            return
        if event.source() is self and event.proposedAction() == QtCore.Qt.CopyAction:
            event.setDropAction(QtCore.Qt.MoveAction)
        super().dropEvent(event)

# ---------------------- UI helpers ----------------------
class CollapsibleSection(QtWidgets.QWidget):
    toggled = QtCore.Signal(bool)

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._btn = QtWidgets.QToolButton(self)
        self._btn.setText(title)
        self._btn.setCheckable(True)
        self._btn.setChecked(False)
        self._btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self._btn.setArrowType(QtCore.Qt.RightArrow)
        self._btn.clicked.connect(self._on_clicked)
        self._content = QtWidgets.QWidget()
        self._content.setVisible(False)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lay.addWidget(self._btn)
        lay.addWidget(self._content)

    def setContentLayout(self, layout: QtWidgets.QLayout):
        self._content.setLayout(layout)

    def _on_clicked(self, checked: bool):
        self._btn.setArrowType(QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow)
        self._content.setVisible(checked)
        self.toggled.emit(checked)

APP_ORG = "PersonCapture"
APP_NAME = "PersonCapture GUI"

_SETTINGS_KEY_FFMPEG_DIR = "paths/ffmpeg_dir"
_SETTINGS_KEY_SDR_NITS_MIGRATED = "migrations/sdr_nits_default_100_v1"
_SETTINGS_KEY_CROP_HEAD_PAD_MIGRATED = "migrations/crop_head_pad_defaults_088_095_v1"

# ---------------------- Data & Settings ----------------------

@dataclass
class SessionConfig:
    video: str = ""
    ref: str = ""
    # --- seek behavior ---
    # Fast seek: jump to nearest previous keyframe and cap forward grabs,
    # then resume processing immediately. Set to False for exact frame seeks.
    seek_fast: bool = True
    # Max frames to grab forward after landing on a keyframe during seek.
    # ≤0 uses fps-derived auto cap (~1s of decoding) to avoid long stalls.
    seek_max_grabs: int = 12
    out_dir: str = "output"
    ratio: str = "1:1,2:3,3:2"
    frame_stride: int = 2
    min_det_conf: float = 0.35
    face_thresh: float = 0.45
    # FFmpeg hardware decode for HDR path:
    #   "off"  → CPU decode
    #   "cuda" → NVDEC (CUDA) decode + CUDA→Vulkan mapping
    ff_hwaccel: str = "cuda"
    reid_thresh: float = 0.42
    combine: str = "min"            # min | avg | face_priority
    match_mode: str = "face_only"        # either | both | face_only | reid_only
    only_best: bool = True
    min_sharpness: float = 0.0
    min_gap_sec: float = 1.5
    min_box_pixels: int = 8000
    auto_crop_borders: bool = True
    # HDR preview is optional. Final screencaps must not depend on this flag.
    hdr_passthrough: bool = False  # Vulkan HDR preview path
    # HDR output/export controls.
    hdr_screencap_fullres: bool = True   # write primary crops from original-resolution source frames
    hdr_archive_crops: bool = False      # additionally write source HDR crops to hdr_crops/
    hdr_crop_format: str = "mkv"        # mkv (FFV1 lossless) | avif (AV1 lossless)
    # Full-resolution HDR->SDR still-render quality controls. These affect only
    # primary crops/f*.jpg source export, not pre-scan/detection preview.
    hdr_sdr_quality: str = "madvr_like"  # madvr_like | resolve_like | balanced | fast
    hdr_sdr_tonemap: str = "auto"        # auto | bt.2390 | spline | st2094-40 | mobius | hable
    hdr_sdr_gamut_mapping: str = "clip"  # clip | perceptual | relative | saturation
    hdr_sdr_contrast_recovery: float = 0.30
    hdr_sdr_peak_detect: bool = True
    hdr_sdr_allow_inaccurate_fallback: bool = False
    hdr_export_timeout_sec: int = 300
    log_interval_sec: float = 1.0
    lock_after_hits: int = 1
    lock_face_thresh: float = 0.28
    lock_reid_thresh: float = 0.30
    score_margin: float = 0.03
    iou_gate: float = 0.05
    # --- HDR tonemap tuning ---
    sdr_nits: float = 100.0          # target SDR display brightness
    tm_desat: float = 0.25           # 0..1 chroma desaturation in highlights
    tm_param: float = 0.40           # Mobius shoulder softness
    hdr_tonemap_pref: str = "auto"   # auto | libplacebo | zscale | scale
    reid_backbone: str = "ViT-L-14"
    reid_pretrained: str = "laion2b_s32b_b82k"
    clip_face_backbone: str = "ViT-L-14"
    clip_face_pretrained: str = "laion2b_s32b_b82k"
    use_arcface: bool = True
    # crop scoring (per-frame, no ratio bias)
    face_target_close: float = 0.38          # close face target frac (face_area/crop_area)
    face_target_upper: float = 0.20          # head+upper torso target frac (ideal)
    face_target_cowboy: float = 0.08         # cowboy target frac
    face_target_body: float = 0.03           # full-body target frac
    face_target_tolerance: float = 0.04      # Huber delta around targets
    face_target_close_min_frac: float = 0.10   # min face_w / frame_w to allow close-ups
    w_close: float = 1.10                    # template weights
    w_upper: float = 1.00
    w_cowboy: float = 0.70
    w_body: float = 0.50
    lambda_facefrac: float = 2.0             # weight for face-fraction loss
    crop_center_weight: float = 0.8          # weight for face-center alignment
    # area vs. composition
    area_gamma: float = 0.60                 # <1 softens area growth
    area_face_scale_weight: float = 0.70     # down-weight area when face is large
    square_pull_face_min: float = 0.16       # activate square pull when fh/frame_h > this; 0..1
    square_pull_weight: float = 1.10         # strength of square pull
    tight_face_relax_thresh: float = 0.48    # if face_h / crop_h ≥ thresh, relax bottom
    tight_face_relax_scale: float = 0.5      # scale want_bottom by this when tight
    device: str = "cuda"            # cuda | cpu
    yolo_model: str = "yolov8n.pt"
    face_model: str = "scrfd_10g_bnkps"
    save_annot: bool = False
    preview_every: int = 3
    # I/O
    async_save: bool = True                # write crops/CSV on a background thread
    jpg_quality: int = 85                  # JPEG quality (lower = faster, smaller)
    # Face full-frame fallback cadence (frames). 0 disables.
    face_fullframe_cadence: int = 12
    prefer_face_when_available: bool = True
    face_quality_min: float = 70.0
    face_visible_uses_quality: bool = True      # if False, any detected face counts as "visible"
    face_det_conf: float = 0.5                  # Face detector confidence
    face_det_pad: float = 0.08                  # expand person box before face detect (fraction of w/h)
    face_margin_min: float = 0.05
    require_face_if_visible: bool = True
    drop_reid_if_any_face_match: bool = True
    learn_bank_runtime: bool = False            # learn into face bank during normal runtime (off by default)
    # --- crop placement heuristics ---
    crop_face_side_margin_frac: float = 0.30     # min(side margin) >= this * face_w
    crop_top_headroom_max_frac: float = 0.15     # max(top margin / crop_h)
    crop_bottom_min_face_heights: float = 1.5    # min bottom margin in face-heights
    crop_penalty_weight: float = 3.0             # weight for placement penalties vs area growth
    crop_head_side_pad_frac: float = 0.88        # protect hair/head around detected face width
    crop_head_top_pad_frac: float = 0.95         # protect hair/forehead above detected face height
    crop_head_bottom_pad_frac: float = 0.30      # protect chin/neck below detected face height
    wide_face_aspect_penalty_weight: float = 10.0# penalize landscape crops when face is prominent
    wide_face_min_frame_frac: float = 0.12       # prominent-face threshold for landscape penalty
    wide_face_aspect_limit: float = 1.05         # aspects above this are considered landscape
    # Final safety guards (post-trim)
    side_guard_drop_enable: bool = True          # drop frames that still violate side margin after all steps
    side_guard_drop_factor: float = 0.66         # require at least this * desired margin on both sides before saving
    face_anchor_down_frac: float = 1.1           # shift center downward by this * face_h (torso bias)
    compose_crop_enable: bool = True             # compose final dataset crop from identity boxes
    compose_detect_person_for_face: bool = True  # associate global face hits with YOLO person boxes
    compose_close_face_h_frac: float = 0.34      # target face_h / crop_h for close/head crops
    compose_upper_face_h_frac: float = 0.22      # target face_h / crop_h for portrait/upper-body crops
    compose_body_face_h_frac: float = 0.085      # target face_h / crop_h for full-body crops
    compose_landscape_face_penalty: float = 5.0  # discourage landscape crops for prominent faces
    compose_body_every_n: int = 6                # deterministic body-shot bias cadence when viable
    compose_person_detect_cadence: int = 6        # only run YOLO person association for face hits on body-suited cadence
    border_threshold: int = 22                   # grayscale threshold for border trimming
    border_scan_frac: float = 0.25               # scan depth as fraction of min(w,h)
    # --- smart crop ---
    smart_crop_enable: bool = True           # dynamic, per-frame
    smart_crop_steps: int = 6                # lateral search half-steps per side
    smart_crop_side_search_frac: float = 0.35# search ± this * crop_w
    smart_crop_use_grad: bool = True         # use gradient saliency (no extra deps)
    # --- anti-zoom guards ---
    face_max_frac_in_crop: float = 0.42          # face_h / crop_h ≤ this
    face_min_frac_in_crop: float = 0.18          # optional lower bound to avoid face too small
    crop_min_height_frac: float = 0.28           # crop_h ≥ this * frame_h
    # Face-only controls
    disable_reid: bool = True                    # force no ReID usage
    face_fullframe_when_missed: bool = True      # try full-frame face detect if per-person face=0
    face_fullframe_imgsz: int = 1408             # full-frame face detector size (0/None -> default)
    rot_adaptive: bool = True                    # gate rotated SCRFD passes when empty
    rot_every_n: int = 12                        # check rotated views every N frames in empty streaks
    rot_after_hit_frames: int = 8                # allow rotations for this many frames after a hit
    fast_no_face_imgsz: int = 512                # shrink 0° pass to this size during long streaks

    # Debug/diagnostics
    debug_dump: bool = True
    debug_dir: str = "debug"
    overlay_scores: bool = False
    overlay_face_fd: bool = True
    lock_momentum: float = 0.7
    suppress_negatives: bool = False
    neg_tolerance: float = 0.35
    max_negatives: int = 5         # emit preview every N processed frames
    # Preview controls
    preview_max_dim: int = 1280        # downscale UI preview; keeps 4K snappy
    preview_fps_cap: int = 20          # max UI preview fps (time-based throttle)
    seek_preview_peek_every: int = 16  # during fast-seek, retrieve every N grabs
    # --- faceless fallback controls ---
    allow_faceless_when_locked: bool = True
    faceless_reid_thresh: float = 0.40      # <= lock if ReID distance <= this
    faceless_iou_min: float = 0.30          # accept person box if IoU with last lock >= this
    faceless_persist_frames: int = 0        # disable carry to avoid background crops
    faceless_min_area_frac: float = 0.03    # min area vs frame
    faceless_max_area_frac: float = 0.55    # max area vs frame
    faceless_center_max_frac: float = 0.12  # max center drift vs diag
    faceless_min_motion_frac: float = 0.02  # min moving pixels in ROI
    # --- pre-scan ---
    prescan_enable: bool = True
    prescan_stride: int = 24               # sample every N frames
    prescan_max_width: int = 416           # downscale for prescan
    # Decoder-level prescan downscale (open a separate low-res reader only for prescan).
    prescan_decode_max_w: int = 384
    prescan_face_conf: float = 0.5         # face detector confidence during prescan
    prescan_fd_enter: float = 0.45         # ArcFace dist to ENTER (looser)
    prescan_fd_add: float = 0.22           # ArcFace dist to add to bank (tighter)
    prescan_fd_exit: float = 0.52          # ArcFace dist to EXIT  (hysteresis)
    prescan_add_cooldown_samples: int = 5  # add at most every N prescan samples
    prescan_rot_probe_period: int = 1      # probe rotations every sample
    prescan_probe_imgsz: int = 512         # slightly stronger probe
    prescan_probe_conf: float = 0.03
    prescan_heavy_90: int = 1536
    prescan_heavy_180: int = 1280
    prescan_min_segment_sec: float = 1.0
    prescan_pad_sec: float = 1.5
    prescan_bridge_gap_sec: float = 1.0    # merge spans separated by short gaps
    # --- pre-scan precision controls ---
    prescan_exit_cooldown_sec: float = 0.50    # was hardcoded 0.5s; now tunable
    prescan_boundary_refine_sec: float = 0.75  # rescan window around each edge
    prescan_refine_stride_min: int = 3         # small stride for edge refine
    prescan_trim_pad: bool = True              # remove pad if refine finds a tighter edge
    # --- pre-scan refine limits ---
    prescan_skip_trailing_refine: bool = True      # don’t refine spans that already hit EOF
    prescan_refine_budget_sec: float = 3.0         # max wall time for refine pass
    # --- pre-scan bank management ---
    prescan_bank_max: int = 64                 # target size
    prescan_diversity_dedup_cos: float = 0.968 # ≥ skip as duplicate
    prescan_replace_margin: float = 0.010      # new score must beat worst by this
    # Early-out on empty frames: skip heavy work when last best fd≈9.00
    prescan_fd9_skip: bool = True               # re-enable skip-gate to avoid misses
    prescan_fd9_grace: int = 1                  # start skipping after this many consecutive fd≈9 samples
    prescan_fd9_probe_period: int = 3           # while skipping, run a real probe every Nth sample
    prescan_weights: Tuple[float, float, float] = (0.70, 0.25, 0.05)  # (anchor, diversity, quality)
    # Persistent pre-scan reuse. Keyed only by video/ref identity and pre-scan-affecting settings,
    # so HDR/export-only changes do not invalidate it.
    prescan_cache_mode: str = "auto"  # auto | refresh | off
    prescan_cache_dir: str = "prescan_cache"
    # --- runtime paths ---
    trt_lib_dir: str = ""                  # preferred TensorRT lib directory
    # --- TensorRT / ORT (advanced) ---
    trt_fp16_enable: bool = True
    trt_timing_cache_enable: bool = True
    trt_engine_cache_enable: bool = True
    trt_cache_root: str = "trt_cache"
    trt_builder_optimization_level: int = 5
    trt_cuda_graph_enable: bool = True
    trt_context_memory_sharing_enable: bool = True
    trt_auxiliary_streams: int = -1
    cuda_use_tf32: bool = True
    # --- I/O + runtime speed controls ---
    skip_yolo_when_faceonly: bool = True   # if any face is visible, skip YOLO in face_only
    # --- curator (MMR over crops) ---
    curate_enable: bool = True
    curate_max_images: int = 200
    curate_fd_gate: float = 0.45           # keep only if fd_anchor ≤ this
    curate_cos_face_dedup: float = 0.985   # drop if ≥ vs any selected
    curate_phash_dedup: float = 0.92       # drop if ≥ vs any selected
    curate_lambda: float = 0.70            # MMR λ in [0,1]
    curate_weights: Tuple[float, float, float] = (0.60, 0.35, 0.05)  # (face, clip/reid, pHash)
    curate_bucket_quota: Tuple[float, float, float] = (0.50, 0.25, 0.25)  # (frontal, left, right)
    curate_use_yaw_quota: bool = True

    def to_json(self, include_paths: bool = False) -> str:
        d = asdict(self)
        if not include_paths:
            # do not publish local paths to preset files
            for k in ("video", "ref", "out_dir"):
                d.pop(k, None)
        return json.dumps(d, indent=2)

    @staticmethod
    def from_json(s: str, ignore_paths_in_json: bool = True) -> "SessionConfig":
        d = json.loads(s)
        c = SessionConfig()
        for k, v in d.items():
            # ignore any paths that might be present in old presets
            if ignore_paths_in_json and k in ("video", "ref", "out_dir"):
                continue
            if hasattr(c, k):
                setattr(c, k, v)
        return c


def _apply_ffmpeg_hwaccel_env(cfg: SessionConfig) -> None:
    """Configure FFmpeg/libplacebo hardware decode env vars from the session config."""

    mode = str(getattr(cfg, "ff_hwaccel", "off") or "off").strip().lower()
    if mode == "cuda":
        os.environ["PC_HWACCEL"] = "cuda"
        os.environ["PC_HWACCEL_OUT_FMT"] = "cuda"
    else:
        os.environ.pop("PC_HWACCEL", None)
        os.environ.pop("PC_HWACCEL_OUT_FMT", None)

# ---------------------- Worker Thread ----------------------

class Processor(QtCore.QObject):
    def _ema(self, prev, new, m):
        if prev is None:
            return new
        return (m * prev + (1.0 - m) * new) if isinstance(prev, np.ndarray) else new

    @staticmethod
    def _fd_min(feat, ref_bank):
        if feat is None or ref_bank is None:
            return 9.0
        vec = np.asarray(feat, dtype=np.float32).reshape(-1)
        vec = vec / max(float(np.linalg.norm(vec)), 1e-6)
        bank = np.asarray(ref_bank, dtype=np.float32)
        if bank.ndim == 1:
            return 1.0 - float(np.dot(vec, bank))
        if bank.size == 0:
            return 9.0
        sims = bank @ vec
        if sims.size == 0:
            return 9.0
        return 1.0 - float(np.max(sims))

    @staticmethod
    def _prescan_weights(cfg) -> Tuple[float, float, float]:
        weights = getattr(cfg, "prescan_weights", (0.70, 0.25, 0.05))
        if not isinstance(weights, (list, tuple)) or len(weights) < 3:
            return 0.70, 0.25, 0.05
        try:
            wa, wd, wq = (float(weights[0]), float(weights[1]), float(weights[2]))
        except Exception:
            return 0.70, 0.25, 0.05
        return wa, wd, wq

    @staticmethod
    def _cache_file_identity(path: str) -> dict:
        p = str(path or "").strip()
        if not p:
            return {"path": "", "missing": True}
        try:
            ap = os.path.abspath(p)
        except Exception:
            ap = p
        try:
            st = os.stat(ap)
            return {
                "path": ap,
                "size": int(getattr(st, "st_size", 0) or 0),
                "mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000))),
            }
        except Exception:
            return {"path": ap, "missing": True}

    @staticmethod
    def _jsonable_cfg_value(v):
        if isinstance(v, tuple):
            return [Processor._jsonable_cfg_value(x) for x in v]
        if isinstance(v, list):
            return [Processor._jsonable_cfg_value(x) for x in v]
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        return v

    def _prescan_cache_root(self, cfg: SessionConfig) -> Path:
        raw = str(getattr(cfg, "prescan_cache_dir", "prescan_cache") or "prescan_cache").strip()
        root = Path(raw)
        if not root.is_absolute():
            root = _REPO_ROOT / root
        return root

    def _prescan_cache_meta(self, cfg: SessionConfig, fps: float, total_frames: int) -> dict:
        # Only include knobs that can change pre-scan spans/ref-bank output. Do not
        # include HDR export/preview settings, crop scoring, curation, or output paths.
        prescan_keys = (
            "prescan_stride",
            "prescan_max_width",
            "prescan_decode_max_w",
            "prescan_face_conf",
            "prescan_fd_enter",
            "prescan_fd_add",
            "prescan_fd_exit",
            "prescan_add_cooldown_samples",
            "prescan_rot_probe_period",
            "prescan_probe_imgsz",
            "prescan_probe_conf",
            "prescan_heavy_90",
            "prescan_heavy_180",
            "prescan_min_segment_sec",
            "prescan_pad_sec",
            "prescan_bridge_gap_sec",
            "prescan_exit_cooldown_sec",
            "prescan_boundary_refine_sec",
            "prescan_refine_stride_min",
            "prescan_trim_pad",
            "prescan_skip_trailing_refine",
            "prescan_refine_budget_sec",
            "prescan_bank_max",
            "prescan_diversity_dedup_cos",
            "prescan_replace_margin",
            "prescan_fd9_skip",
            "prescan_fd9_grace",
            "prescan_fd9_probe_period",
            "prescan_weights",
            "face_model",
            "clip_face_backbone",
            "clip_face_pretrained",
            "use_arcface",
        )
        settings = {
            k: self._jsonable_cfg_value(getattr(cfg, k, None))
            for k in prescan_keys
        }
        refs = [part.strip() for part in str(getattr(cfg, "ref", "") or "").split(";") if part.strip()]
        meta = {
            "version": 1,
            "video": self._cache_file_identity(getattr(cfg, "video", "")),
            "refs": [self._cache_file_identity(p) for p in refs],
            "fps": round(float(fps or 0.0), 6),
            "total_frames": int(total_frames or 0),
            "settings": settings,
        }
        key_json = json.dumps(meta, sort_keys=True, separators=(",", ":"))
        meta["key"] = hashlib.sha256(key_json.encode("utf-8")).hexdigest()
        return meta

    def _prescan_cache_path(self, cfg: SessionConfig, meta: dict) -> Path:
        key = str(meta.get("key") or "")
        return self._prescan_cache_root(cfg) / f"{key}.npz"

    def _load_prescan_cache(
        self,
        cfg: SessionConfig,
        fps: float,
        total_frames: int,
    ) -> tuple[bool, list[tuple[int, int]], Optional[np.ndarray], Optional[dict]]:
        mode = str(getattr(cfg, "prescan_cache_mode", "auto") or "auto").lower()
        if mode not in ("auto", "reuse"):
            return False, [], None, None
        meta = self._prescan_cache_meta(cfg, fps, total_frames)
        path = self._prescan_cache_path(cfg, meta)
        if not path.is_file():
            return False, [], None, meta
        try:
            with np.load(str(path), allow_pickle=False) as data:
                stored_meta = json.loads(str(data["meta"].item()))
                if stored_meta.get("key") != meta.get("key") or stored_meta.get("version") != meta.get("version"):
                    return False, [], None, meta
                spans_arr = np.asarray(data["spans"], dtype=np.int64).reshape(-1, 2)
                spans = [(int(s), int(e)) for s, e in spans_arr.tolist() if int(e) >= int(s)]
                has_ref_arr = data["has_ref"] if "has_ref" in data.files else np.array([0], dtype=np.uint8)
                has_ref = bool(int(np.asarray(has_ref_arr).reshape(-1)[0]))
                ref_feat = None
                if has_ref and "ref_face_feat" in data:
                    arr = np.asarray(data["ref_face_feat"], dtype=np.float32)
                    if arr.size > 0:
                        ref_feat = arr.reshape(arr.shape[0], -1) if arr.ndim >= 2 else arr.reshape(1, -1)
                self._status(
                    f"Pre-scan cache hit • segments={len(spans)}",
                    key="prescan_cache",
                    interval=0.0,
                )
                return True, spans, ref_feat, meta
        except Exception as exc:
            self._status(f"Pre-scan cache ignored: {exc}", key="prescan_cache", interval=5.0)
            return False, [], None, meta

    def _save_prescan_cache(
        self,
        cfg: SessionConfig,
        fps: float,
        total_frames: int,
        spans: list[tuple[int, int]],
        ref_face_feat: Optional[np.ndarray],
    ) -> None:
        mode = str(getattr(cfg, "prescan_cache_mode", "auto") or "auto").lower()
        if mode not in ("auto", "refresh", "reuse") or self._abort:
            return
        try:
            meta = self._prescan_cache_meta(cfg, fps, total_frames)
            path = self._prescan_cache_path(cfg, meta)
            path.parent.mkdir(parents=True, exist_ok=True)
            spans_arr = np.asarray(spans or [], dtype=np.int64).reshape(-1, 2)
            if ref_face_feat is None:
                feat_arr = np.zeros((0, 0), dtype=np.float32)
                has_ref = np.array([0], dtype=np.uint8)
            else:
                feat_arr = np.asarray(ref_face_feat, dtype=np.float32)
                if feat_arr.ndim == 1:
                    feat_arr = feat_arr.reshape(1, -1)
                has_ref = np.array([1], dtype=np.uint8)
            tmp = path.with_suffix(path.suffix + ".tmp")
            with open(tmp, "wb") as f:
                np.savez_compressed(
                    f,
                    meta=np.array(json.dumps(meta, sort_keys=True), dtype=np.str_),
                    spans=spans_arr,
                    ref_face_feat=feat_arr,
                    has_ref=has_ref,
                )
            os.replace(tmp, path)
            self._status(f"Pre-scan cache saved • segments={len(spans_arr)}", key="prescan_cache", interval=0.0)
        except Exception as exc:
            self._status(f"Pre-scan cache save failed: {exc}", key="prescan_cache", interval=5.0)

    def _stream_ref_bank_update(
        self,
        ref_bank_list: List[np.ndarray],
        ref_face_feat: Optional[np.ndarray],
        vec_new: Optional[np.ndarray],
        quality_val: float,
        cfg: SessionConfig,
    ) -> Tuple[Optional[np.ndarray], str, Optional[int]]:
        if vec_new is None:
            return ref_face_feat, "skip", None
        try:
            cap = max(1, int(getattr(cfg, "prescan_bank_max", 64)))
        except Exception:
            cap = 64
        try:
            dedup_cos = float(getattr(cfg, "prescan_diversity_dedup_cos", 0.968))
        except Exception:
            dedup_cos = 0.968
        try:
            rep_margin = float(getattr(cfg, "prescan_replace_margin", 0.010))
        except Exception:
            rep_margin = 0.010
        w_anchor, w_div, w_q = self._prescan_weights(cfg)
        v = np.asarray(vec_new, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(v))
        if norm <= 1e-6:
            return ref_face_feat, "skip", None
        v = v / norm
        if ref_face_feat is not None:
            bank = np.asarray(ref_face_feat, dtype=np.float32)
        else:
            bank = np.asarray(ref_bank_list, dtype=np.float32)
        if bank.ndim == 1:
            bank = bank.reshape(1, -1)
        if bank.size == 0:
            ref_bank_list.append(v)
            ref_face_feat = np.vstack(ref_bank_list).astype(np.float32)
            return ref_face_feat, "added", None
        sims = bank @ v
        if sims.size > 0 and float(sims.max()) >= dedup_cos:
            return ref_face_feat, "dup", None
        anchor = bank[0]
        cos_anchor = float(np.dot(anchor, v))
        cos_anchor = max(-1.0, min(1.0, cos_anchor))
        fd_anchor = float(np.sqrt(max(0.0, 2.0 - 2.0 * cos_anchor)))
        nn_sim = float(sims.max()) if sims.size else 0.0
        q_term = float(min(max(quality_val or 0.0, 0.0), 1000.0) / 300.0)
        s_new = w_anchor * (1.0 - fd_anchor) + w_div * (1.0 - nn_sim) + w_q * q_term
        if len(ref_bank_list) < cap:
            ref_bank_list.append(v)
            ref_face_feat = np.vstack(ref_bank_list).astype(np.float32)
            return ref_face_feat, "added", None
        bank_sims = bank @ bank.T
        np.fill_diagonal(bank_sims, -1.0)
        nn_sim_each = bank_sims.max(axis=1)
        cos_anchor_each = bank @ anchor
        cos_anchor_each = np.clip(cos_anchor_each, -1.0, 1.0)
        fd_anchor_each = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * cos_anchor_each))
        s_bank = w_anchor * (1.0 - fd_anchor_each) + w_div * (1.0 - nn_sim_each)
        worst_idx = int(np.argmin(s_bank))
        if s_new > float(s_bank[worst_idx]) + rep_margin:
            ref_bank_list[worst_idx] = v
            ref_face_feat = np.vstack(ref_bank_list).astype(np.float32)
            return ref_face_feat, "replaced", worst_idx
        return ref_face_feat, "skip", None

    def _span_index_for(self, f, spans):
        # return index of span containing f, else next span index
        for i, (s, e) in enumerate(spans):
            if s <= f <= e:
                return i
            if f < s:
                return i

    def _prescan_skip_forward(self, cap, count: int) -> int:
        """Discard up to ``count`` decoded frames without restarting the reader."""

        def _at_known_eof() -> bool:
            reader_eof = getattr(cap, "_at_known_eof", None)
            if callable(reader_eof):
                try:
                    return bool(reader_eof())
                except Exception:
                    pass
            try:
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            except Exception:
                total = 0
            if total <= 0:
                return False
            try:
                pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
            except Exception:
                pos = 0
            if bool(getattr(cap, "_is_hdr_pipe", False)):
                if not bool(getattr(cap, "_total_is_exact", False)):
                    return False
                # FfmpegPipeReader reports the last emitted frame index.
                return pos >= total - 1
            # OpenCV usually reports the next frame index.
            return pos >= total

        advanced = 0
        remaining = max(0, int(count))
        for _ in range(remaining):
            if self._abort or _at_known_eof():
                break
            try:
                grabbed = cap.grab()
            except Exception as exc:
                if _at_known_eof():
                    break
                self._status(
                    f"Pre-scan skip failed: {exc}",
                    key="prescan_skip_error",
                    interval=0.5,
                )
                raise
            if not grabbed:
                startup_exc = getattr(cap, "_last_startup_error", None)
                if startup_exc is not None:
                    self._status(
                        f"Pre-scan skip failed: {startup_exc}",
                        key="prescan_skip_error",
                        interval=0.5,
                    )
                    raise RuntimeError("Pre-scan reader failed during grab") from startup_exc
                break
            advanced += 1
        return advanced

    def _prescan(self, cap, fps, total_frames, face: "FaceEmbedder", ref_feat, cfg):
        """
        Fast pass to find keep-spans. Now:
          - obeys play/pause/seek/step/speed from _cmd_q
          - grows ref face bank on confident matches (dedup + cap)
        Returns: (spans, updated_ref_feat)
        """
        import numpy as _np
        cap_main = cap
        cap_override = None
        cap_ps = cap
        _restore_env: dict[str, str] = {}
        try:
            ps_maxw = int(getattr(cfg, "prescan_decode_max_w", 0))
        except Exception:
            ps_maxw = 0
        if ps_maxw > 0:
            try:
                _restore_env["PC_DECODE_MAX_W"] = os.environ.get("PC_DECODE_MAX_W", "")
                _restore_env["PC_FORCE_TONEMAP"] = os.environ.get("PC_FORCE_TONEMAP", "")
                _restore_env["PC_TONEMAP_MIN_W"] = os.environ.get("PC_TONEMAP_MIN_W", "")
                os.environ["PC_DECODE_MAX_W"] = str(ps_maxw)
                # For prescan we allow low-res tonemapping; keep tonemap floor in sync
                os.environ["PC_TONEMAP_MIN_W"] = str(ps_maxw)
                os.environ.setdefault("PC_FORCE_TONEMAP", "scale")
                try:
                    from .video_io import FfmpegPipeReader as _Pipe, _ffmpeg_path as _ffp
                except Exception:
                    from video_io import FfmpegPipeReader as _Pipe, _ffmpeg_path as _ffp  # type: ignore
                _fmpeg = _ffp()
                if _fmpeg:
                    cap_override = _Pipe(cfg.video, _fmpeg)
                    cap_ps = cap_override
            except Exception:
                cap_override = None
        cap = cap_ps
        pos0 = int(cap_main.get(cv2.CAP_PROP_POS_FRAMES) or 0)

        def _run():
            nonlocal cap
            # seed bank
            if ref_feat is None:
                ref_bank_list = []
            else:
                arr = _np.asarray(ref_feat, dtype=_np.float32)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                norms = _np.linalg.norm(arr, axis=1, keepdims=True)
                arr = arr / _np.maximum(norms, 1e-6)
                ref_bank_list = [row.copy() for row in arr]
            initial_bank_len = len(ref_bank_list)
            ref_feat_local = _np.vstack(ref_bank_list).astype(_np.float32) if ref_bank_list else None
            added_vecs = 0
            stride = max(1, int(cfg.prescan_stride))
            pad = int(round(cfg.prescan_pad_sec * fps))
            min_len = int(round(cfg.prescan_min_segment_sec * fps))
            Wmax = int(getattr(cfg, "prescan_max_width", 0))
            enter, exit_ = float(cfg.prescan_fd_enter), float(cfg.prescan_fd_exit)
            fd_add = float(getattr(cfg, "prescan_fd_add", enter))
            old_face_conf = getattr(face, "conf", 0.5)
            face.conf = float(cfg.prescan_face_conf)
            old_rot_adapt = getattr(face, "rot_adaptive", True)
            try:
                # freeze legacy rotation gating; rely on pre-scan throttle
                try:
                    face.configure_rotation_strategy(adaptive=False)
                except Exception:
                    pass
                try:
                    face.set_prescan_fast(True, mode="rr")
                    face.set_prescan_hint(escalate=False)
                    face._probe_conf = float(getattr(cfg, "prescan_probe_conf", 0.03))
                    face._prescan_period = int(getattr(cfg, "prescan_rot_probe_period", 3))
                    face._prescan_probe_imgsz = int(getattr(cfg, "prescan_probe_imgsz", 512))
                    face._high_90  = int(getattr(cfg, "prescan_heavy_90", 1536))
                    face._high_180 = int(getattr(cfg, "prescan_heavy_180", 1280))
                except Exception:
                    pass
                add_cooldown_samples = int(
                    getattr(
                        cfg,
                        "prescan_add_cooldown_samples",
                        getattr(cfg, "prescan_add_cooldown_frames", 5),
                    )
                )
                last_add_sample = -10**9
                spans = []
                active = False
                start = 0
                neg_run = 0
                total_samples = max(1, (total_frames + stride - 1) // stride)
                progress_step = max(1, total_samples // 50)
                next_progress_sample = 0
                preview_step = max(stride, int(getattr(cfg, "preview_every", 3)))
                last_preview_idx = -preview_step
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._hdr_preview_seek(0)
                processed_samples = 0
                fd9_streak = 0
                fd9_skip_samples = 0
                fd9_probe_samples = 0
                fd9_gate_period = 1
                i = 0
                # fast pass
                while i < total_frames:
                    # drain + coalesce controls
                    last_seek: Optional[int] = None
                    step_accum: int = 0
                    pending_speed: Optional[float] = None
                    pending_cfg: dict = {}
                    try:
                        while True:
                            cmd, arg = self._cmd_q.get_nowait()
                            if cmd == "pause":
                                self._paused = True
                            elif cmd == "play":
                                self._paused = False
                            elif cmd == "seek":
                                self._prescan_cache_dirty = True
                                try:
                                    last_seek = int(arg)
                                except Exception:
                                    pass
                            elif cmd == "step":
                                self._prescan_cache_dirty = True
                                try:
                                    step_accum += int(arg) if arg is not None else 1
                                except Exception:
                                    step_accum += 1
                            elif cmd == "speed":
                                try:
                                    pending_speed = float(arg)
                                except Exception:
                                    pass
                            elif cmd == "cfg":
                                # allow live edits to prescan_* fields if sent
                                try:
                                    ch = dict(arg or {})
                                    for k, v in ch.items():
                                        if k.startswith("prescan_"):
                                            self._prescan_cache_dirty = True
                                            setattr(cfg, k, v)
                                    stride = max(1, int(cfg.prescan_stride))
                                    enter, exit_ = float(cfg.prescan_fd_enter), float(cfg.prescan_fd_exit)
                                    fd_add = float(getattr(cfg, "prescan_fd_add", enter))
                                    Wmax = int(cfg.prescan_max_width)
                                    face.conf = float(cfg.prescan_face_conf)
                                    add_cooldown_samples = int(
                                        getattr(
                                            cfg,
                                            "prescan_add_cooldown_samples",
                                            getattr(cfg, "prescan_add_cooldown_frames", 5),
                                        )
                                    )
                                except Exception:
                                    pass
                    except queue.Empty:
                        pass

                    if pending_speed is not None:
                        self._speed = max(0.1, min(4.0, float(pending_speed)))
                    if last_seek is not None or step_accum:
                        tgt = int(last_seek) if last_seek is not None else int(i + step_accum)
                        tgt = max(0, min(total_frames - 1, tgt))
                        new_pos = self._seek_to(
                            cap,
                            i,
                            tgt,
                            fast=bool(getattr(cfg, "seek_fast", True)),
                            max_grabs=int(getattr(cfg, "seek_max_grabs", 12)),
                            allow_partial=True,
                            hdr_reader=self._hdr_preview_reader,
                        )
                        # ensure forward progress even if seek advanced < stride
                        i_floor = (new_pos // stride) * stride
                        if i_floor <= i:
                            i = i + stride
                        else:
                            i = i_floor
                        processed_samples = i // stride
                        active = False
                        neg_run = 0
                        fd9_streak = 0
                        fd9_skip_samples = 0
                        fd9_probe_samples = 0
                        fd9_gate_period = 1
                        try:
                            self.progress.emit(i)
                        except Exception:
                            pass

                    if self._abort:
                        break
                    if self._paused:
                        time.sleep(0.02)
                        continue
                    if i % stride != 0:
                        next_i = ((i // stride) + 1) * stride
                        if next_i >= total_frames:
                            break
                        skipped = self._prescan_skip_forward(cap, next_i - i)
                        if skipped < (next_i - i):
                            break
                        i = next_i
                        continue
                    # Preempt before IO to honor newly queued seeks/steps
                    try:
                        while True:
                            cmd, arg = self._cmd_q.get_nowait()
                            if cmd == "seek":
                                self._prescan_cache_dirty = True
                                try:
                                    last_seek = int(arg)
                                except Exception:
                                    pass
                            elif cmd == "step":
                                self._prescan_cache_dirty = True
                                try:
                                    step_accum += int(arg) if arg is not None else 1
                                except Exception:
                                    step_accum += 1
                            elif cmd == "speed":
                                try:
                                    pending_speed = float(arg)
                                except Exception:
                                    pass
                            elif cmd == "cfg":
                                try:
                                    ch = dict(arg or {})
                                    for k, v in ch.items():
                                        if k.startswith("prescan_"):
                                            self._prescan_cache_dirty = True
                                            setattr(cfg, k, v)
                                    stride = max(1, int(cfg.prescan_stride))
                                    enter, exit_ = float(cfg.prescan_fd_enter), float(cfg.prescan_fd_exit)
                                    fd_add = float(getattr(cfg, "prescan_fd_add", enter))
                                    Wmax = int(cfg.prescan_max_width)
                                    face.conf = float(cfg.prescan_face_conf)
                                    add_cooldown_samples = int(
                                        getattr(
                                            cfg,
                                            "prescan_add_cooldown_samples",
                                            getattr(cfg, "prescan_add_cooldown_frames", 5),
                                        )
                                    )
                                except Exception:
                                    pass
                            elif cmd == "pause":
                                self._paused = True
                            elif cmd == "play":
                                self._paused = False
                    except queue.Empty:
                        pass

                    if pending_speed is not None:
                        self._speed = max(0.1, min(4.0, float(pending_speed)))
                        pending_speed = None
                    if last_seek is not None or step_accum:
                        tgt = int(last_seek) if last_seek is not None else int(i + step_accum)
                        tgt = max(0, min(total_frames - 1, tgt))
                        new_pos = self._seek_to(
                            cap,
                            i,
                            tgt,
                            fast=bool(getattr(cfg, "seek_fast", True)),
                            max_grabs=int(getattr(cfg, "seek_max_grabs", 12)),
                            allow_partial=True,
                            hdr_reader=self._hdr_preview_reader,
                        )
                        i_floor = (new_pos // stride) * stride
                        if i_floor <= i:
                            i = i + stride
                        else:
                            i = i_floor
                        processed_samples = i // stride
                        active = False
                        neg_run = 0
                        fd9_streak = 0
                        fd9_skip_samples = 0
                        fd9_probe_samples = 0
                        fd9_gate_period = 1
                        try:
                            self.progress.emit(i)
                        except Exception:
                            pass
                        continue
                    try:
                        grabbed = cap.grab()
                    except Exception as exc:
                        at_known_eof = False
                        reader_eof = getattr(cap, "_at_known_eof", None)
                        if callable(reader_eof):
                            try:
                                at_known_eof = bool(reader_eof())
                            except Exception:
                                at_known_eof = False
                        if at_known_eof:
                            break
                        startup_exc = getattr(cap, "_last_startup_error", None)
                        detail = startup_exc if startup_exc is not None else exc
                        self._status(
                            f"Pre-scan read failed: {detail}",
                            key="prescan_skip_error",
                            interval=0.5,
                        )
                        raise RuntimeError("Pre-scan reader failed during grab") from exc
                    if not grabbed:
                        startup_exc = getattr(cap, "_last_startup_error", None)
                        if startup_exc is not None:
                            self._status(
                                f"Pre-scan read failed: {startup_exc}",
                                key="prescan_skip_error",
                                interval=0.5,
                            )
                            raise RuntimeError("Pre-scan reader failed during grab") from startup_exc
                        break
                    ok, frame = cap.retrieve()
                    if not ok or frame is None:
                        startup_exc = getattr(cap, "_last_startup_error", None)
                        if startup_exc is not None:
                            self._status(
                                f"Pre-scan read failed: {startup_exc}",
                                key="prescan_skip_error",
                                interval=0.5,
                            )
                            raise RuntimeError("Pre-scan reader failed during retrieve") from startup_exc
                        target_skip = min(
                            max(0, stride - 1),
                            max(0, total_frames - i - 1),
                        )
                        skipped = self._prescan_skip_forward(cap, target_skip)
                        if skipped < target_skip:
                            break
                        i = i + 1 + skipped
                        continue
                    if self._hdr_preview_enabled():
                        self._hdr_preview_seek(i)
                        self._pump_hdr_preview()
                    idx = i
                    sample_idx = processed_samples
                    processed_samples += 1
                    h, w = frame.shape[:2]
                    # widen angles while active; keep probe-throttled when idle
                    try:
                        face._prescan_rr_mode = "full" if active else "rr"
                        face.set_prescan_hint(escalate=active)
                    except Exception:
                        pass
                    best = 9.0  # defensive default
                    # ---- fd9 skip-gate ----
                    skip_extract = False
                    fd9_gate_active = False
                    try:
                        if (not active) and bool(getattr(cfg, "prescan_fd9_skip", True)):
                            grace = max(0, int(getattr(cfg, "prescan_fd9_grace", 1)))
                            period = max(1, int(getattr(cfg, "prescan_fd9_probe_period", 3)))
                            if fd9_streak >= grace:
                                fd9_gate_active = True
                                fd9_gate_period = period
                                if (fd9_streak % period) != 0:
                                    skip_extract = True
                    except Exception:
                        pass

                    if fd9_gate_active:
                        if skip_extract:
                            fd9_skip_samples += 1
                        else:
                            fd9_probe_samples += 1
                    else:
                        fd9_skip_samples = 0
                        fd9_probe_samples = 0

                    if not skip_extract:
                        # defer resize until we actually extract
                        if w > Wmax:
                            nh = int(round(h * (Wmax / float(w))))
                            frame = cv2.resize(frame, (Wmax, nh), interpolation=cv2.INTER_AREA)
                        try:
                            faces = face.extract(frame)
                        except Exception:
                            faces = ()
                        for f in faces:
                            feat = f.get("feat")
                            if feat is None:
                                continue
                            # current best vs live bank
                            fd = self._fd_min(feat, ref_feat_local)
                            best = min(best, fd)
                            if (
                                fd <= fd_add
                                and (sample_idx - last_add_sample) >= add_cooldown_samples
                                and f.get("quality", 1e9) >= cfg.face_quality_min
                            ):
                                try:
                                    quality_val = float(f.get("quality", 0.0))
                                except Exception:
                                    quality_val = 0.0
                                ref_feat_local, action, idx_info = self._stream_ref_bank_update(
                                    ref_bank_list,
                                    ref_feat_local,
                                    feat,
                                    quality_val,
                                    cfg,
                                )
                                if action in {"added", "replaced"}:
                                    last_add_sample = sample_idx
                                    if action == "added":
                                        added_vecs += 1
                                        self._status(
                                            f"Pre-scan ref bank +1 (size={len(ref_bank_list)}) fd={fd:.3f}",
                                            key="prescan_bank",
                                            interval=2.0,
                                        )
                                    elif action == "replaced":
                                        self._status(
                                            f"Pre-scan replaced #{idx_info} with better ref (score↑)",
                                            key="prescan_bank",
                                            interval=2.0,
                                        )
                    else:
                        best = 9.0
                        self._status(
                            "Pre-scan: fd9-skip gate active",
                            key="prescan_skip",
                            interval=5.0,
                        )
                        self._status(
                            f"Pre-scan fd9 cadence skip={fd9_skip_samples} probe={fd9_probe_samples} period={fd9_gate_period}",
                            key="prescan_skip_cadence",
                            interval=5.0,
                        )

                    if fd9_gate_active and not skip_extract:
                        self._status(
                            f"Pre-scan fd9 cadence probe skip={fd9_skip_samples} probe={fd9_probe_samples} period={fd9_gate_period}",
                            key="prescan_skip_probe",
                            interval=5.0,
                        )

                    if best >= 8.99:
                        fd9_streak += 1
                    else:
                        fd9_streak = 0
                    if sample_idx >= next_progress_sample:
                        pct = min(100.0, (idx + stride) / max(1.0, float(total_frames)) * 100.0)
                        self._status(
                            f"Pre-scan {pct:.1f}% ({idx}/{total_frames})",
                            key="prescan_progress",
                            interval=0.25,
                        )
                        try:
                            self.progress.emit(int(min(idx, max(total_frames - 1, 0))))
                        except Exception:
                            pass
                        next_progress_sample = sample_idx + progress_step
                    emit_preview = False
                    if idx - last_preview_idx >= preview_step:
                        emit_preview = True
                        last_preview_idx = idx
                    if best <= enter:
                        if not active:
                            active = True
                            fd9_streak = 0
                            fd9_skip_samples = 0
                            fd9_probe_samples = 0
                            fd9_gate_period = 1
                            start = idx
                        neg_run = 0
                        emit_preview = True
                    else:
                        if active:
                            neg_run += 1
                            # configurable exit cooldown instead of fixed 0.5s
                            exit_cool = int(
                                round(
                                    max(0.0, float(getattr(cfg, "prescan_exit_cooldown_sec", 0.5)))
                                    * fps
                                )
                            )
                            if neg_run * stride >= exit_cool or best >= exit_:
                                end = idx
                                # pad, clamp, merge
                                s = max(0, start - pad)
                                e = min(total_frames - 1, end + pad)
                                if e - s + 1 >= min_len:
                                    if spans and s <= spans[-1][1] + 1:
                                        spans[-1] = (spans[-1][0], max(spans[-1][1], e))
                                    else:
                                        spans.append((s, e))
                                active = False
                                neg_run = 0
                                fd9_streak = 0
                                fd9_skip_samples = 0
                                fd9_probe_samples = 0
                                fd9_gate_period = 1
                    if emit_preview:
                        try:
                            vis = frame.copy()
                            color = (0, 200, 0) if best <= enter else (0, 0, 200)
                            cv2.putText(
                                vis,
                                f"Pre-scan fd={best:.2f} f={idx}",
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.75,
                                color,
                                2,
                                cv2.LINE_AA,
                            )
                            self._emit_preview_bgr(vis)
                        except Exception:
                            pass
                    target_skip = min(
                        max(0, stride - 1),
                        max(0, total_frames - idx - 1),
                    )
                    skipped = self._prescan_skip_forward(cap, target_skip)
                    if skipped < target_skip:
                        break
                    i = idx + 1 + skipped
                if active:
                    s = max(0, start - pad)
                    e = total_frames - 1
                    if e - s + 1 >= min_len:
                        if spans and s <= spans[-1][1] + 1:
                            spans[-1] = (spans[-1][0], max(spans[-1][1], e))
                        else:
                            spans.append((s, e))
                # bridge tiny gaps
                if spans and getattr(cfg, "prescan_bridge_gap_sec", 0) > 0:
                    bridged = []
                    gap = int(round(cfg.prescan_bridge_gap_sec * fps))
                    cs, ce = spans[0]
                    for s, e in spans[1:]:
                        if s - ce <= gap:
                            ce = max(ce, e)
                        else:
                            bridged.append((cs, ce))
                            cs, ce = s, e
                    bridged.append((cs, ce))
                    spans = bridged

                # --- boundary refinement: tighten edges to cut dead time, but bounded ---
                def _refine_edges(sp_list):
                    if not sp_list:
                        return sp_list
                    import time
                    # smaller stride and stronger probe around edges
                    stride_ref = max(
                        1,
                        min(
                            int(max(1, cfg.prescan_stride) // 4),
                            int(getattr(cfg, "prescan_refine_stride_min", 3)),
                        ),
                    )
                    win = int(
                        round(
                            max(0.0, float(getattr(cfg, "prescan_boundary_refine_sec", 0.75)))
                            * fps
                        )
                    )
                    pad_frames = int(round(max(0.0, float(cfg.prescan_pad_sec)) * fps))
                    search = max(pad_frames, win)
                    refined = []
                    budget_s = float(getattr(cfg, "prescan_refine_budget_sec", 0.0))
                    t0 = time.perf_counter()

                    def over_budget():
                        return budget_s > 1e-3 and (time.perf_counter() - t0) > budget_s

                    # temporarily crank prescan hints
                    rr_old = getattr(face, "_prescan_rr_mode", "rr")
                    try:
                        face._prescan_rr_mode = "full"
                    except Exception:
                        pass
                    try:
                        face.set_prescan_hint(escalate=True)
                    except Exception:
                        pass
                    timeout = False
                    for idx, (s, e) in enumerate(sp_list):
                        ls = s
                        le = e
                        skip_right = bool(getattr(cfg, "prescan_skip_trailing_refine", True)) and (
                            e >= total_frames - 1
                        )
                        # LEFT edge: scan forward from s to s+search
                        left_stop = min(e, s + search)
                        j = s
                        best_left = None
                        while j <= left_stop:
                            if over_budget():
                                timeout = True
                                break
                            # keyframe-aware seek for responsiveness
                            self._seek_to(
                                cap,
                                j,
                                j,
                                fast=bool(getattr(cfg, "seek_fast", True)),
                                max_grabs=int(getattr(cfg, "seek_max_grabs", 12)),
                                allow_partial=False,
                                hdr_reader=self._hdr_preview_reader,
                            )
                            ret, frame = cap.read()
                            if not ret or frame is None:
                                j += stride_ref
                                continue
                            self._pump_hdr_preview()
                            h, w = frame.shape[:2]
                            if Wmax > 0 and w > Wmax:
                                scale = float(Wmax) / float(w)
                                frame = cv2.resize(
                                    frame,
                                    (int(round(w * scale)), int(round(h * scale))),
                                    interpolation=cv2.INTER_AREA,
                                )
                            faces = face.extract(frame)
                            if faces:
                                bank = ref_feat_local if ref_feat_local is not None else ref_feat
                                for f in faces:
                                    feat = f.get("feat")
                                    if feat is not None and self._fd_min(feat, bank) <= enter:
                                        best_left = j
                                        break
                            if best_left is not None:
                                break
                            j += stride_ref
                        if timeout:
                            refined.append((ls, le))
                            refined.extend(sp_list[idx + 1 :])
                            break
                        if best_left is not None and bool(getattr(cfg, "prescan_trim_pad", True)):
                            ls = max(s, best_left)
                        # RIGHT edge: scan backward region [e-search, e] to last good
                        last_good = None
                        if not skip_right:
                            right_start = max(ls, e - search)
                            j = right_start
                            while j <= e:
                                if over_budget():
                                    timeout = True
                                    break
                                self._seek_to(
                                    cap,
                                    j,
                                    j,
                                    fast=bool(getattr(cfg, "seek_fast", True)),
                                    max_grabs=int(getattr(cfg, "seek_max_grabs", 12)),
                                    allow_partial=False,
                                    hdr_reader=self._hdr_preview_reader,
                                )
                                ret, frame = cap.read()
                                if not ret or frame is None:
                                    j += stride_ref
                                    continue
                                self._pump_hdr_preview()
                                h, w = frame.shape[:2]
                                if Wmax > 0 and w > Wmax:
                                    scale = float(Wmax) / float(w)
                                    frame = cv2.resize(
                                        frame,
                                        (int(round(w * scale)), int(round(h * scale))),
                                        interpolation=cv2.INTER_AREA,
                                    )
                                faces = face.extract(frame)
                                if faces:
                                    bank = ref_feat_local if ref_feat_local is not None else ref_feat
                                    for f in faces:
                                        feat = f.get("feat")
                                        if feat is not None and self._fd_min(feat, bank) <= enter:
                                            last_good = j
                                j += stride_ref
                            if timeout:
                                refined.append((ls, le))
                                refined.extend(sp_list[idx + 1 :])
                                break
                        if last_good is not None and bool(getattr(cfg, "prescan_trim_pad", True)):
                            le = min(e, last_good)
                        # keep only if span remains big enough
                        if le >= ls and (le - ls + 1) >= min_len:
                            refined.append((ls, le))
                    # restore hints
                    try:
                        face.set_prescan_hint(escalate=False)
                    except Exception:
                        pass
                    try:
                        face._prescan_rr_mode = rr_old
                    except Exception:
                        pass
                    return refined

                spans = _refine_edges(spans)
                if spans and getattr(cfg, "prescan_bridge_gap_sec", 0) > 0:
                    # bridge tiny gaps (repeat post-refine)
                    bridged = []
                    gap = int(round(cfg.prescan_bridge_gap_sec * fps))
                    cs, ce = spans[0]
                    for s, e in spans[1:]:
                        if s - ce <= gap:
                            ce = max(ce, e)
                        else:
                            bridged.append((cs, ce))
                            cs, ce = s, e
                    bridged.append((cs, ce))
                    spans = bridged
                try:
                    fps_f = float(fps)
                except Exception:
                    fps_f = 0.0
                if fps_f <= 0:
                    fps_f = 1.0
                total_sec = sum((e - s + 1) / fps_f for s, e in spans)
                self._status(
                    f"Pre-scan refined spans={len(spans)} • total_keep≈{total_sec:.1f}s",
                    key="prescan_refine",
                    interval=1.0,
                )
            finally:
                try:
                    face.configure_rotation_strategy(adaptive=bool(old_rot_adapt))
                except Exception:
                    pass
                try:
                    face.set_prescan_fast(False)
                    face.set_prescan_hint(escalate=False)
                except Exception:
                    pass
                face.conf = old_face_conf
            cap_main.set(cv2.CAP_PROP_POS_FRAMES, pos0)
            try:
                # Restore progress to the position we were at before pre-scan so the UI
                # doesn't jump to EOF and queue a seek there.
                self.progress.emit(max(0, pos0))
            except Exception:
                pass
            # Final prescan progress: do not claim 100% coverage if we bailed early.
            try:
                samples = int(processed_samples)
            except Exception:
                samples = 0
            msg = (
                f"Pre-scan done • samples≈{samples} • segments={len(spans)}"
                if samples > 0
                else f"Pre-scan done • segments={len(spans)}"
            )
            self._status(msg, key="prescan_progress", interval=0.1)
            self._status(
                f"Pre-scan ref bank added {added_vecs} vector(s); size={len(ref_bank_list)} (start={initial_bank_len})",
                key="prescan_bank_summary",
                interval=0.5,
            )
            return spans, (ref_feat_local if ref_feat_local is not None else ref_feat)

        try:
            return _run()
        finally:
            if cap_override is not None:
                try:
                    cap_override.release()
                except Exception:
                    pass
            for k, v in _restore_env.items():
                if v:
                    os.environ[k] = v
                elif k in os.environ:
                    del os.environ[k]

    @staticmethod
    def _clip_to_frame(x1: float, y1: float, x2: float, y2: float, frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
        """Shift and clamp coordinates so the crop stays inside the frame."""

        dx1 = -x1 if x1 < 0 else 0.0
        dx2 = frame_w - x2 if x2 > frame_w else 0.0
        dy1 = -y1 if y1 < 0 else 0.0
        dy2 = frame_h - y2 if y2 > frame_h else 0.0

        shift_x = dx1 if dx1 != 0.0 else (dx2 if dx2 != 0.0 else 0.0)
        shift_y = dy1 if dy1 != 0.0 else (dy2 if dy2 != 0.0 else 0.0)

        x1 += shift_x
        x2 += shift_x
        y1 += shift_y
        y2 += shift_y

        ix1 = max(0, min(frame_w - 1, int(round(x1))))
        ix2 = max(ix1 + 1, min(frame_w, int(round(x2))))
        iy1 = max(0, min(frame_h - 1, int(round(y1))))
        iy2 = max(iy1 + 1, min(frame_h, int(round(y2))))
        return ix1, iy1, ix2, iy2

    def _face_head_proxy_box(
        self,
        face_box: Optional[Tuple[float, float, float, float]],
        frame_w: int,
        frame_h: int,
    ) -> Optional[Tuple[float, float, float, float]]:
        """Conservative head/hair protection box derived from a face detector box.

        SCRFD-style face boxes usually cover the facial region, not hair, forehead,
        ears, or jaw/neck context.  Using the raw face box as the crop safety
        invariant lets the cropper legally cut the visible head while still
        keeping the detected face rectangle inside the crop.
        """
        if face_box is None:
            return None
        try:
            fx1, fy1, fx2, fy2 = [float(v) for v in face_box]
        except Exception:
            return None
        fw = max(1.0, fx2 - fx1)
        fh = max(1.0, fy2 - fy1)
        cfg = self.cfg
        side = max(0.0, float(getattr(cfg, "crop_head_side_pad_frac", 0.70))) * fw
        top = max(0.0, float(getattr(cfg, "crop_head_top_pad_frac", 0.85))) * fh
        bottom = max(0.0, float(getattr(cfg, "crop_head_bottom_pad_frac", 0.30))) * fh
        hx1 = max(0.0, fx1 - side)
        hy1 = max(0.0, fy1 - top)
        hx2 = min(float(frame_w), fx2 + side)
        hy2 = min(float(frame_h), fy2 + bottom)
        if hx2 <= hx1 + 1.0 or hy2 <= hy1 + 1.0:
            return None
        return hx1, hy1, hx2, hy2

    @staticmethod
    def _shift_crop_to_include_box(
        crop_xyxy: Tuple[float, float, float, float],
        protect_xyxy: Optional[Tuple[float, float, float, float]],
        bounds_xyxy: Tuple[int, int, int, int],
        margin_px: float = 0.0,
    ) -> Tuple[int, int, int, int]:
        """Shift a fixed-size crop so the protected box remains visible.

        This preserves the selected aspect ratio and crop size.  It is a final
        correction pass, not a crop rescorer.
        """
        cx1, cy1, cx2, cy2 = [float(v) for v in crop_xyxy]
        bx1, by1, bx2, by2 = [int(v) for v in bounds_xyxy]
        if protect_xyxy is None:
            return int(round(cx1)), int(round(cy1)), int(round(cx2)), int(round(cy2))
        try:
            px1, py1, px2, py2 = [float(v) for v in protect_xyxy]
        except Exception:
            return int(round(cx1)), int(round(cy1)), int(round(cx2)), int(round(cy2))
        m = max(0.0, float(margin_px))
        w = max(1.0, cx2 - cx1)
        h = max(1.0, cy2 - cy1)

        # Horizontal correction. Positive dx moves the crop right.
        dx = 0.0
        if px1 - m < cx1:
            dx = (px1 - m) - cx1
        if px2 + m > cx2 + dx:
            dx = (px2 + m) - cx2
        nx1 = max(float(bx1), min(float(bx2) - w, cx1 + dx))
        nx2 = nx1 + w

        # Vertical correction. Positive dy moves the crop down.
        dy = 0.0
        if py1 - m < cy1:
            dy = (py1 - m) - cy1
        if py2 + m > cy2 + dy:
            dy = (py2 + m) - cy2
        ny1 = max(float(by1), min(float(by2) - h, cy1 + dy))
        ny2 = ny1 + h

        ix1 = max(bx1, min(bx2 - 1, int(round(nx1))))
        iy1 = max(by1, min(by2 - 1, int(round(ny1))))
        ix2 = max(ix1 + 1, min(bx2, int(round(nx2))))
        iy2 = max(iy1 + 1, min(by2, int(round(ny2))))
        return ix1, iy1, ix2, iy2

    @staticmethod
    def _coerce_box_xyxy(
        box: Optional[Tuple[float, float, float, float]],
        bounds_xyxy: Tuple[int, int, int, int],
    ) -> Optional[Tuple[float, float, float, float]]:
        if box is None:
            return None
        try:
            x1, y1, x2, y2 = [float(v) for v in box]
            bx1, by1, bx2, by2 = [float(v) for v in bounds_xyxy]
        except Exception:
            return None
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2, bx1, by1, bx2, by2)):
            return None
        x1 = max(bx1, min(bx2, x1))
        y1 = max(by1, min(by2, y1))
        x2 = max(bx1, min(bx2, x2))
        y2 = max(by1, min(by2, y2))
        if x2 <= x1 + 1.0 or y2 <= y1 + 1.0:
            return None
        return x1, y1, x2, y2

    @staticmethod
    def _union_boxes_xyxy(*boxes: Optional[Tuple[float, float, float, float]]) -> Optional[Tuple[float, float, float, float]]:
        valid = []
        for box in boxes:
            if box is None:
                continue
            try:
                x1, y1, x2, y2 = [float(v) for v in box]
            except Exception:
                continue
            if all(math.isfinite(v) for v in (x1, y1, x2, y2)) and x2 > x1 + 1.0 and y2 > y1 + 1.0:
                valid.append((x1, y1, x2, y2))
        if not valid:
            return None
        return (
            min(b[0] for b in valid),
            min(b[1] for b in valid),
            max(b[2] for b in valid),
            max(b[3] for b in valid),
        )

    @staticmethod
    def _pad_box_xyxy(
        box: Optional[Tuple[float, float, float, float]],
        pad_x: float,
        pad_y_top: float,
        pad_y_bottom: Optional[float],
        bounds_xyxy: Tuple[int, int, int, int],
    ) -> Optional[Tuple[float, float, float, float]]:
        if box is None:
            return None
        try:
            x1, y1, x2, y2 = [float(v) for v in box]
            bx1, by1, bx2, by2 = [float(v) for v in bounds_xyxy]
        except Exception:
            return None
        pxb = max(0.0, float(pad_x))
        pyt = max(0.0, float(pad_y_top))
        pyb = pyt if pad_y_bottom is None else max(0.0, float(pad_y_bottom))
        x1 = max(bx1, x1 - pxb)
        y1 = max(by1, y1 - pyt)
        x2 = min(bx2, x2 + pxb)
        y2 = min(by2, y2 + pyb)
        if x2 <= x1 + 1.0 or y2 <= y1 + 1.0:
            return None
        return x1, y1, x2, y2

    @staticmethod
    def _containment_deficit_xyxy(
        crop_xyxy: Tuple[float, float, float, float],
        protect_xyxy: Optional[Tuple[float, float, float, float]],
        margin_px: float = 0.0,
    ) -> float:
        if protect_xyxy is None:
            return 0.0
        cx1, cy1, cx2, cy2 = [float(v) for v in crop_xyxy]
        px1, py1, px2, py2 = [float(v) for v in protect_xyxy]
        pw = max(1.0, px2 - px1)
        ph = max(1.0, py2 - py1)
        m = max(0.0, float(margin_px))
        dx = max(0.0, (cx1 + m) - px1) + max(0.0, px2 - (cx2 - m))
        dy = max(0.0, (cy1 + m) - py1) + max(0.0, py2 - (cy2 - m))
        return (dx / pw) + (dy / ph)

    def _ratio_crop_containing_box(
        self,
        protect_xyxy: Tuple[float, float, float, float],
        ratio_str: str,
        bounds_xyxy: Tuple[int, int, int, int],
        anchor: Optional[Tuple[float, float]] = None,
        min_size_xy: Optional[Tuple[float, float]] = None,
    ) -> Tuple[int, int, int, int]:
        """Return the smallest in-bounds ratio crop that tries to contain protect_xyxy.

        Unlike expand_box_to_ratio(), this is allowed to grow after clamping.
        It therefore preserves the invariant needed by dataset crops: the identity
        boxes are inputs to composition, and the final crop must not cut the
        protected face/head/person region just because the requested ratio hits a
        frame edge.
        """
        bx1, by1, bx2, by2 = [float(v) for v in bounds_xyxy]
        bounds_w = max(1.0, bx2 - bx1)
        bounds_h = max(1.0, by2 - by1)
        px1, py1, px2, py2 = [float(v) for v in protect_xyxy]
        px1 = max(bx1, min(bx2, px1))
        py1 = max(by1, min(by2, py1))
        px2 = max(px1 + 1.0, min(bx2, px2))
        py2 = max(py1 + 1.0, min(by2, py2))
        try:
            rw, rh = parse_ratio(str(ratio_str))
            target = max(1e-6, float(rw) / float(rh))
        except Exception:
            target = 1.0

        need_w = max(1.0, px2 - px1)
        need_h = max(1.0, py2 - py1)
        if min_size_xy is not None:
            try:
                need_w = max(need_w, float(min_size_xy[0]))
                need_h = max(need_h, float(min_size_xy[1]))
            except Exception:
                pass

        crop_w = max(need_w, need_h * target)
        crop_h = crop_w / target
        if crop_h < need_h:
            crop_h = need_h
            crop_w = crop_h * target

        # Largest legal crop at this ratio inside the content bounds.
        if (bounds_w / bounds_h) >= target:
            max_h = bounds_h
            max_w = bounds_h * target
        else:
            max_w = bounds_w
            max_h = bounds_w / target
        crop_w = min(crop_w, max_w)
        crop_h = min(crop_h, max_h)

        if anchor is not None:
            try:
                ax, ay = float(anchor[0]), float(anchor[1])
            except Exception:
                ax, ay = (px1 + px2) * 0.5, (py1 + py2) * 0.5
        else:
            ax, ay = (px1 + px2) * 0.5, (py1 + py2) * 0.5
        ax = max(bx1, min(bx2, ax))
        ay = max(by1, min(by2, ay))

        x1 = ax - crop_w * 0.5
        y1 = ay - crop_h * 0.5

        # Shift to contain the protected box first, then clamp to content bounds.
        if px1 < x1:
            x1 = px1
        if px2 > x1 + crop_w:
            x1 = px2 - crop_w
        if py1 < y1:
            y1 = py1
        if py2 > y1 + crop_h:
            y1 = py2 - crop_h
        x1 = max(bx1, min(bx2 - crop_w, x1))
        y1 = max(by1, min(by2 - crop_h, y1))
        x2 = x1 + crop_w
        y2 = y1 + crop_h

        # Quantize content bounds first, then clip in local coordinates so
        # rounding cannot re-enter trimmed regions when bounds are offset.
        ibx1 = int(math.ceil(bx1))
        iby1 = int(math.ceil(by1))
        ibx2 = int(math.floor(bx2))
        iby2 = int(math.floor(by2))
        if ibx2 <= ibx1:
            ibx1 = int(round(bx1))
            ibx2 = max(ibx1 + 1, int(round(bx2)))
        if iby2 <= iby1:
            iby1 = int(round(by1))
            iby2 = max(iby1 + 1, int(round(by2)))

        lx1, ly1, lx2, ly2 = self._clip_to_frame(
            x1 - float(ibx1),
            y1 - float(iby1),
            x2 - float(ibx1),
            y2 - float(iby1),
            ibx2 - ibx1,
            iby2 - iby1,
        )
        return ibx1 + lx1, iby1 + ly1, ibx1 + lx2, iby1 + ly2

    @staticmethod
    def _find_person_box_for_face(
        face_xyxy: Tuple[float, float, float, float],
        persons: list,
        frame_w: int,
        frame_h: int,
    ) -> Optional[Tuple[float, float, float, float]]:
        """Find the detected person box most likely to own a matched face box."""
        try:
            fx1, fy1, fx2, fy2 = [float(v) for v in face_xyxy]
        except Exception:
            return None
        fcx = 0.5 * (fx1 + fx2)
        fcy = 0.5 * (fy1 + fy2)
        fw = max(1.0, fx2 - fx1)
        fh = max(1.0, fy2 - fy1)
        best = None
        best_score = 1.0e18
        for p in persons or []:
            try:
                px1, py1, px2, py2 = [float(v) for v in p.get("xyxy", p)]
            except Exception:
                continue
            px1 = max(0.0, min(float(frame_w), px1))
            py1 = max(0.0, min(float(frame_h), py1))
            px2 = max(px1 + 1.0, min(float(frame_w), px2))
            py2 = max(py1 + 1.0, min(float(frame_h), py2))
            pw = max(1.0, px2 - px1)
            ph = max(1.0, py2 - py1)
            # A face belonging to a person should be near the top portion and
            # horizontally inside the person box.  Give containment priority, then
            # use normalized distance as a tiebreaker.
            contains_center = (px1 <= fcx <= px2) and (py1 <= fcy <= py2)
            face_inside = (px1 <= fx1 + 0.2 * fw and fx2 - 0.2 * fw <= px2 and py1 <= fy1 + 0.2 * fh and fy2 - 0.2 * fh <= py2)
            top_band_y = py1 + 0.42 * ph
            top_bias = max(0.0, (fcy - top_band_y) / ph)
            dx = 0.0 if px1 <= fcx <= px2 else min(abs(fcx - px1), abs(fcx - px2)) / pw
            dy = 0.0 if py1 <= fcy <= py2 else min(abs(fcy - py1), abs(fcy - py2)) / ph
            area_penalty = 0.02 * ((pw * ph) / max(1.0, float(frame_w * frame_h)))
            score = (0.0 if contains_center else 4.0) + (0.0 if face_inside else 1.5) + dx + dy + top_bias + area_penalty
            if score < best_score:
                best_score = score
                best = (px1, py1, px2, py2)
        if best is None or best_score >= 5.0:
            return None
        return best

    def _compose_dataset_crop(
        self,
        base_crop_xyxy: Tuple[float, float, float, float],
        ratio_candidates: list[str],
        bounds_xyxy: Tuple[int, int, int, int],
        subject_box: Optional[Tuple[float, float, float, float]] = None,
        face_box: Optional[Tuple[float, float, float, float]] = None,
        frame_idx: Optional[int] = None,
    ) -> Tuple[Tuple[int, int, int, int], str, str]:
        """Compose the final dataset crop after identity has been decided.

        The candidate detector boxes are identity evidence.  They are not allowed
        to become arbitrary crop anchors.  Composition is driven by the matched
        face/head and, only when it is a real associated person detection, the
        subject/body box.
        """
        cfg = self.cfg
        bx1, by1, bx2, by2 = [int(v) for v in bounds_xyxy]
        bounds = (bx1, by1, bx2, by2)
        bound_w = max(1.0, float(bx2 - bx1))
        bound_h = max(1.0, float(by2 - by1))
        bound_area = max(1.0, bound_w * bound_h)

        validated_user_ratios: list[str] = []
        for rs in [str(r).strip() for r in (ratio_candidates or []) if str(r).strip()]:
            try:
                parse_ratio(rs)
            except Exception:
                continue
            if rs not in validated_user_ratios:
                validated_user_ratios.append(rs)

        def _ratio_list_for_profile(profile: str) -> list[str]:
            preferred = {
                "close": ["1:1", "2:3", "3:4"],
                "upper": ["2:3", "3:4", "1:1"],
                "body": ["2:3", "3:4", "1:1", "3:2"],
                "base": ["1:1", "2:3"],
            }.get(profile, ["1:1", "2:3"])

            # User ratios are an availability list, not a command to use the same
            # geometry for every crop profile. Close/upper portraits must never
            # become landscape crops merely because 3:2 exists in the UI list.
            # Keep 3:2 available only for body/context profiles after the scorer
            # confirms the current frame is actually suited to it.
            allow_landscape = profile == "body"
            available = validated_user_ratios if validated_user_ratios else preferred
            out: list[str] = []

            def _add_ratio(rs: str) -> None:
                try:
                    rw, rh = parse_ratio(rs)
                    aspect = float(rw) / max(1e-6, float(rh))
                except Exception:
                    return
                if aspect > 1.05 and not allow_landscape:
                    return
                if rs not in out:
                    out.append(rs)

            for rs in preferred:
                if rs in available:
                    _add_ratio(rs)
            for rs in available:
                _add_ratio(rs)
            if out:
                return out
            return [] if validated_user_ratios else ["1:1", "2:3"]

        base = self._coerce_box_xyxy(base_crop_xyxy, bounds)
        subj = self._coerce_box_xyxy(subject_box, bounds)
        face = self._coerce_box_xyxy(face_box, bounds)
        if base is None:
            base = face or subj or (bx1, by1, bx2, by2)

        head = self._face_head_proxy_box(face, bx2, by2) if face is not None else None
        head = self._coerce_box_xyxy(head, bounds)
        face_protect = self._union_boxes_xyxy(head, face) or face

        profiles: list[tuple[str, Tuple[float, float, float, float], float, Tuple[float, float], Tuple[float, float]]] = []
        face_h = 0.0
        face_frame_frac = 0.0
        subj_h_frac = ((subj[3] - subj[1]) / bound_h) if subj is not None else 0.0
        body_period = max(0, int(getattr(cfg, "compose_body_every_n", 6)))
        body_cadence = body_period > 0 and frame_idx is not None and (int(frame_idx) % body_period == 0)

        if face is not None:
            fx1, fy1, fx2, fy2 = face
            fw = max(1.0, fx2 - fx1)
            face_h = max(1.0, fy2 - fy1)
            fcx = 0.5 * (fx1 + fx2)
            fcy = 0.5 * (fy1 + fy2)
            face_frame_frac = face_h / bound_h
            hx1, hy1, hx2, hy2 = face_protect or face

            close_target = max(0.20, min(0.46, float(getattr(cfg, "compose_close_face_h_frac", 0.34))))
            upper_target = max(0.12, min(0.34, float(getattr(cfg, "compose_upper_face_h_frac", 0.22))))
            body_target = max(0.035, min(0.16, float(getattr(cfg, "compose_body_face_h_frac", 0.085))))

            close_protect = self._pad_box_xyxy(
                (hx1, hy1, hx2, max(hy2, fy2 + 0.85 * face_h)),
                pad_x=0.12 * fw,
                pad_y_top=0.00,
                pad_y_bottom=0.45 * face_h,
                bounds_xyxy=bounds,
            ) or (hx1, hy1, hx2, max(hy2, fy2 + 0.85 * face_h))
            profiles.append(("close", close_protect, close_target, (fcx, fcy + 0.70 * face_h), (fw * 2.0, face_h / close_target)))

            if subj is not None:
                sx1, sy1, sx2, sy2 = subj
                sw = max(1.0, sx2 - sx1)
                sh = max(1.0, sy2 - sy1)
                upper_bottom = min(float(by2), max(fy2 + 3.6 * face_h, sy1 + 0.58 * sh))
                upper_half_w = max(1.15 * fw, 0.48 * sw)
                upper_protect = (
                    max(float(bx1), min(hx1, fcx - upper_half_w)),
                    max(float(by1), min(hy1, sy1)),
                    min(float(bx2), max(hx2, fcx + upper_half_w)),
                    upper_bottom,
                )
            else:
                upper_protect = self._pad_box_xyxy(
                    (hx1, hy1, hx2, max(hy2, fy2 + 2.6 * face_h)),
                    pad_x=0.35 * fw,
                    pad_y_top=0.00,
                    pad_y_bottom=0.55 * face_h,
                    bounds_xyxy=bounds,
                ) or (hx1, hy1, hx2, max(hy2, fy2 + 2.6 * face_h))
            profiles.append(("upper", upper_protect, upper_target, (fcx, fcy + 1.45 * face_h), (fw * 2.8, face_h / upper_target)))

            if subj is not None:
                sx1, sy1, sx2, sy2 = subj
                sw = max(1.0, sx2 - sx1)
                sh = max(1.0, sy2 - sy1)
                body_box = self._pad_box_xyxy(
                    subj,
                    pad_x=max(0.07 * sw, 0.35 * fw),
                    pad_y_top=max(0.025 * sh, 0.25 * face_h),
                    pad_y_bottom=max(0.035 * sh, 0.35 * face_h),
                    bounds_xyxy=bounds,
                ) or subj
                profiles.append(("body", body_box, body_target, ((sx1 + sx2) * 0.5, (sy1 + sy2) * 0.5), (sw, sh)))
        elif subj is not None:
            sx1, sy1, sx2, sy2 = subj
            sw = max(1.0, sx2 - sx1)
            sh = max(1.0, sy2 - sy1)
            profiles.append(("body", subj, float(getattr(cfg, "compose_body_face_h_frac", 0.085)), ((sx1 + sx2) * 0.5, (sy1 + sy2) * 0.5), (sw, sh)))
        else:
            b = base or (bx1, by1, bx2, by2)
            profiles.append(("base", b, 0.20, ((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5), (b[2] - b[0], b[3] - b[1])))

        best: Optional[tuple[float, Tuple[int, int, int, int], str, str]] = None
        for profile, protect_raw, target_face_h_frac, anchor, min_size in profiles:
            protect = self._coerce_box_xyxy(protect_raw, bounds)
            if protect is None:
                continue
            px1, py1, px2, py2 = protect
            protect_w = max(1.0, px2 - px1)
            protect_h = max(1.0, py2 - py1)
            min_w = max(float(min_size[0]), protect_w)
            min_h = max(float(min_size[1]), protect_h)

            for rs in _ratio_list_for_profile(profile):
                try:
                    rw, rh = parse_ratio(rs)
                    aspect = float(rw) / float(rh)
                except Exception:
                    continue
                is_landscape = aspect > 1.05
                if profile in {"close", "upper", "base"} and is_landscape:
                    continue
                if profile == "body" and is_landscape:
                    # Landscape is a rare context/body sample, not a normal face
                    # framing choice. Eligibility requires an associated subject
                    # plus small-face/tall-subject gates; body_cadence influences
                    # scoring later and is not an eligibility gate.
                    if subj is None:
                        continue
                    if face is not None and face_frame_frac >= 0.12:
                        continue
                    if subj_h_frac < 0.60:
                        continue

                crop = self._ratio_crop_containing_box(
                    protect,
                    rs,
                    bounds,
                    anchor=anchor,
                    min_size_xy=(min_w, min_h),
                )
                cx1, cy1, cx2, cy2 = crop
                crop_w = max(1.0, float(cx2 - cx1))
                crop_h = max(1.0, float(cy2 - cy1))
                crop_area = crop_w * crop_h

                face_deficit = self._containment_deficit_xyxy(crop, face_protect, margin_px=1.0) if face_protect is not None else 0.0
                body_deficit = self._containment_deficit_xyxy(crop, subj, margin_px=1.0) if (profile == "body" and subj is not None) else 0.0
                protect_deficit = self._containment_deficit_xyxy(crop, protect, margin_px=1.0)
                # A crop candidate that cuts the identified face/head is invalid,
                # not merely lower quality. Penalizing it was too weak and let
                # wide/body candidates win after dark-scene repairs.
                if face_deficit > 0.01:
                    continue
                if body_deficit > 0.02:
                    continue

                containment = 120.0 * face_deficit + 120.0 * body_deficit + 20.0 * protect_deficit

                ratio_prior = 0.0
                if profile == "close":
                    profile_prior = 0.00
                    ratio_prior += 0.00 if rs == "1:1" else 0.08
                elif profile == "upper":
                    profile_prior = 0.12
                    ratio_prior += 0.00 if rs == "2:3" else 0.06
                elif profile == "body":
                    landscape_penalty = max(0.0, min(20.0, float(getattr(cfg, "compose_landscape_face_penalty", 5.0))))
                    profile_prior = 0.78
                    if body_cadence and face_frame_frac < 0.10 and subj_h_frac > 0.62:
                        profile_prior -= 0.076 * landscape_penalty
                    if face is not None and face_frame_frac >= 0.10:
                        profile_prior += 0.70
                    if is_landscape:
                        profile_prior += 0.70
                    if rs == "2:3":
                        ratio_prior += 0.00
                    elif rs == "3:4":
                        ratio_prior += 0.08
                    elif rs == "1:1":
                        ratio_prior += 0.12
                    else:
                        ratio_prior += 0.30
                    if is_landscape and subj is not None:
                        subj_aspect = (subj[2] - subj[0]) / max(1.0, subj[3] - subj[1])
                        if subj_aspect < 0.72:
                            ratio_prior += 0.12 * landscape_penalty
                else:
                    profile_prior = 0.35

                if face is not None:
                    actual_face_h_frac = face_h / crop_h
                    face_loss = abs(actual_face_h_frac - max(1e-6, target_face_h_frac))
                    if profile == "close" and face_frame_frac < 0.085:
                        profile_prior += 0.65
                    if profile == "upper" and face_frame_frac < 0.085:
                        profile_prior -= 0.12
                else:
                    face_loss = 0.0

                area_penalty = 0.08 * (crop_area / bound_area)
                if profile != "body" and crop_area / bound_area > 0.72:
                    area_penalty += 0.35

                placement_penalty = 0.0
                if face is not None and profile in {"close", "upper"}:
                    fcx = 0.5 * (face[0] + face[2])
                    fcy = 0.5 * (face[1] + face[3])
                    rel_x = (fcx - cx1) / crop_w
                    rel_y = (fcy - cy1) / crop_h
                    placement_penalty += 0.25 * abs(rel_x - 0.50)
                    target_y = 0.36 if profile == "close" else 0.28
                    placement_penalty += 0.35 * abs(rel_y - target_y)

                score = containment + profile_prior + ratio_prior + 2.2 * face_loss + area_penalty + placement_penalty
                if best is None or score < best[0]:
                    best = (score, crop, rs, profile)

        if best is not None:
            _, crop, rs, profile = best
            return crop, rs, profile

        fallback_protect = face_protect or subj or base or (bx1, by1, bx2, by2)
        fallback_ratio = None
        fallback_profile = "fallback"
        for rs in validated_user_ratios:
            try:
                rw, rh = parse_ratio(rs)
                aspect = float(rw) / max(1e-6, float(rh))
            except Exception:
                continue
            is_landscape = aspect > 1.05
            if is_landscape:
                if subj is None:
                    continue
                if face is not None and face_frame_frac >= 0.12:
                    continue
                if subj_h_frac < 0.60:
                    continue
                fallback_profile = "body"
            fallback_ratio = rs
            break
        if fallback_ratio is None:
            fallback_ratio = "1:1" if face_protect is not None else "2:3"
            fallback_profile = "fallback"
        crop = self._ratio_crop_containing_box(fallback_protect, fallback_ratio, bounds)
        return crop, fallback_ratio, fallback_profile

    def _enforce_scale_and_margins(
        self,
        crop_xyxy: Tuple[int, int, int, int],
        ratio_str: str,
        frame_w: int,
        frame_h: int,
        face_box: Optional[Tuple[int, int, int, int]] = None,
        anchor: Optional[Tuple[float, float]] = None,
    ) -> Tuple[int, int, int, int]:
        """Enforce bounds for face fraction, side margins and min height.
        Expands when needed (max face frac / min height / side margins), and
        also *shrinks* when the face is smaller than face_min_frac_in_crop."""

        cx1, cy1, cx2, cy2 = map(int, crop_xyxy)
        current_w = float(cx2 - cx1)
        current_h = float(cy2 - cy1)
        try:
            rw, rh = parse_ratio(ratio_str)
            target_aspect = float(rw) / float(rh)
        except Exception:
            current_aspect = current_w / current_h if current_h > 0 else 1.0
            target_aspect = current_aspect if current_aspect > 0 else 1.0
        cfg = self.cfg

        # lower bound (things that force crop to be at least this tall)
        min_required_h = current_h
        # upper bound (things that cap crop height to keep face sufficiently large)
        max_allowed_h = float("inf")

        if face_box is not None:
            fx1, fy1, fx2, fy2 = face_box
            face_w = float(fx2 - fx1)
            face_h = float(fy2 - fy1)
            if face_h > 0:
                # keep face <= max fraction
                min_required_h = max(min_required_h, face_h / max(cfg.face_max_frac_in_crop, 1e-6))
                want_side = float(cfg.crop_face_side_margin_frac) * face_w
                required_w = face_w + 2.0 * want_side
                min_required_h = max(min_required_h, required_w / max(target_aspect, 1e-6))
                if cfg.face_min_frac_in_crop > 0:
                    # keep face >= min fraction  => crop_h <= face_h / min_frac
                    max_allowed_h = min(max_allowed_h, face_h / max(cfg.face_min_frac_in_crop, 1e-6))

        min_required_h = max(min_required_h, float(cfg.crop_min_height_frac) * float(frame_h))

        # If both bounds clash, prefer feasibility
        if max_allowed_h < min_required_h:
            max_allowed_h = min_required_h

        # Expand when too small
        if current_h + 0.5 < min_required_h:
            new_h = min_required_h
        # Shrink when too large (face too small)
        elif current_h > max_allowed_h + 0.5:
            new_h = max_allowed_h
        else:
            return cx1, cy1, cx2, cy2

        new_w = new_h * target_aspect
        if anchor is not None:
            anchor_x, anchor_y = anchor
        else:
            anchor_x = (cx1 + cx2) / 2.0
            anchor_y = (cy1 + cy2) / 2.0

        new_x1 = anchor_x - new_w / 2.0
        new_x2 = anchor_x + new_w / 2.0
        new_y1 = anchor_y - new_h / 2.0
        new_y2 = anchor_y + new_h / 2.0
        return self._clip_to_frame(new_x1, new_y1, new_x2, new_y2, frame_w, frame_h)

    def _choose_best_ratio(self, det_box, ratios, frame_w, frame_h, anchor=None, face_box=None):
        """
        Expand det_box to each candidate ratio, then score:
          score = area_factor + λ * placement_penalty
        placement_penalty discourages: side-cut faces, excess headroom, missing lower torso.
        """
        (x1, y1, x2, y2) = det_box
        det_area = max(1, (x2 - x1) * (y2 - y1))
        best = None
        best_ratio = None
        best_score = 1e9
        best_template_loss = 0.0
        cfg = self.cfg
        head_box = self._face_head_proxy_box(face_box, frame_w, frame_h)

        def _penalty(crop_xyxy, face_xyxy):
            if face_xyxy is None:
                return 0.0
            cx1, cy1, cx2, cy2 = crop_xyxy
            fx1, fy1, fx2, fy2 = face_xyxy
            cw, ch = max(1.0, cx2 - cx1), max(1.0, cy2 - cy1)
            fw, fh = max(1.0, fx2 - fx1), max(1.0, fy2 - fy1)
            # margins around face inside crop
            L = max(0.0, (fx1 - cx1))
            R = max(0.0, (cx2 - fx2))
            T = max(0.0, (fy1 - cy1))
            B = max(0.0, (cy2 - fy2))
            # 1) side margin deficit -> penalize half-face crops
            want_side = float(cfg.crop_face_side_margin_frac) * fw
            side_def = max(0.0, want_side - min(L, R)) / fw
            # 2) headroom cap -> penalize too much space above head
            headroom = T / ch
            headroom_def = max(0.0, headroom - float(cfg.crop_top_headroom_max_frac))
            # 3) bottom margin minimum in face-heights -> encourage torso inclusion
            tight = (fh / ch) >= float(getattr(cfg, "tight_face_relax_thresh", 0.48))
            relax = float(getattr(cfg, "tight_face_relax_scale", 0.5)) if tight else 1.0
            want_bottom = float(cfg.crop_bottom_min_face_heights) * fh * relax
            bottom_def = max(0.0, want_bottom - B) / fh
            # 4) face centrality
            cx = 0.5 * (cx1 + cx2)
            cy = 0.5 * (cy1 + cy2)
            fcx = 0.5 * (fx1 + fx2)
            fcy = 0.5 * (fy1 + fy2)
            center_def = math.hypot((fcx - cx) / cw, (fcy - cy) / ch)
            return side_def + headroom_def + bottom_def + float(cfg.crop_center_weight) * center_def

        def _huber(x, delta):
            ax = abs(x)
            return 0.5 * ax * ax if ax <= delta else delta * (ax - 0.5 * delta)

        def _containment_deficit(crop_xyxy, protect_xyxy, margin_px=0.0):
            if protect_xyxy is None:
                return 0.0
            cx1, cy1, cx2, cy2 = crop_xyxy
            px1, py1, px2, py2 = protect_xyxy
            pw = max(1.0, px2 - px1)
            ph = max(1.0, py2 - py1)
            m = max(0.0, float(margin_px))
            dx = max(0.0, (cx1 + m) - px1) + max(0.0, px2 - (cx2 - m))
            dy = max(0.0, (cy1 + m) - py1) + max(0.0, py2 - (cy2 - m))
            return (dx / pw) + (dy / ph)

        for rs in ratios:
            try:
                rw, rh = parse_ratio(rs)
            except Exception:
                continue
            # dynamic head_bias: push framing downward to include torso
            hb = 0.0
            if face_box is not None:
                fbw = max(1.0, face_box[2] - face_box[0])
                fbh = max(1.0, face_box[3] - face_box[1])
                bh = max(1.0, y2 - y1)
                # move center DOWN by ~face_anchor_down_frac * face_h => negative head_bias
                hb = -float(cfg.face_anchor_down_frac) * (fbh / bh)
            ex1, ey1, ex2, ey2 = expand_box_to_ratio(
                x1, y1, x2, y2, rw, rh, frame_w, frame_h, anchor=anchor, head_bias=hb
            )
            area = max(1, (ex2 - ex1) * (ey2 - ey1))
            area_raw = float(area) / float(det_area)
            # soften pure-area dominance
            area_term = pow(area_raw, float(cfg.area_gamma))

            crop_xyxy = (ex1, ey1, ex2, ey2)
            pen = _penalty(crop_xyxy, face_box)
            total = area_term + float(cfg.crop_penalty_weight) * pen
            if head_box is not None:
                # This is the invariant the previous scorer violated: the crop may
                # not cut the visible head/hair merely because the detector's face
                # rectangle still fits.  Use a very large but graded penalty so if
                # no candidate can fully satisfy it, the least-bad candidate wins.
                total += 1.0e6 * _containment_deficit(crop_xyxy, head_box, margin_px=1.0)
            tmpl_loss = 0.0

            if face_box is not None:
                fx1, fy1, fx2, fy2 = face_box
                farea = max(1.0, (fx2 - fx1) * (fy2 - fy1))
                carea = float(area)
                face_frac = farea / max(1.0, carea)
                fw = max(1.0, fx2 - fx1)
                fh = max(1.0, fy2 - fy1)

                # HARD SIDE GUARD: discard ratios that would cut the face
                L = max(0.0, (fx1 - ex1))
                R = max(0.0, (ex2 - fx2))
                want_side = float(cfg.crop_face_side_margin_frac) * fw
                if min(L, R) < want_side:
                    total += 1e9  # effectively skip this ratio

                # reduce the influence of area when the face is large in-frame
                face_scale = max(fw / max(1.0, frame_w), fh / max(1.0, frame_h))  # 0..1
                area_scale = max(0.30, 1.0 - float(cfg.area_face_scale_weight) * face_scale)
                total += (area_scale - 1.0) * area_term  # scales area term down only
                allow_close = max(
                    fw / max(1.0, frame_w),
                    fh / max(1.0, frame_h),
                ) >= float(getattr(cfg, "face_target_close_min_frac", 0.10))
                targ = [
                    (float(cfg.face_target_upper), float(cfg.w_upper)),
                    (float(cfg.face_target_cowboy), float(cfg.w_cowboy)),
                    (float(cfg.face_target_body), float(cfg.w_body)),
                ]
                if allow_close:
                    targ.append((float(cfg.face_target_close), float(cfg.w_close)))
                delta = float(cfg.face_target_tolerance)
                best_tloss = min(w * _huber(face_frac - t, delta) for t, w in targ)
                tmpl_loss = best_tloss
                total += float(cfg.lambda_facefrac) * best_tloss

                # “square pull” only when face is large: penalize non-square aspect
                asp = float(rw) / float(rh)
                if (fh / max(1.0, frame_h)) > float(cfg.square_pull_face_min):
                    pull = (fh / float(frame_h)) - float(cfg.square_pull_face_min)
                    total += float(cfg.square_pull_weight) * pull * abs(asp - 1.0)

                # Landscape crops are usually wrong for a prominent face unless the
                # subject/body box truly requires them.  Previously the area term
                # could choose 3:2 because it was the smallest crop, even when that
                # produced a face/head cut.
                face_scale = max(fw / max(1.0, frame_w), fh / max(1.0, frame_h))
                wide_min = max(1e-6, float(getattr(cfg, "wide_face_min_frame_frac", 0.12)))
                wide_limit = max(1.0, float(getattr(cfg, "wide_face_aspect_limit", 1.05)))
                if face_scale >= wide_min and asp > wide_limit:
                    strength = min(4.0, face_scale / wide_min)
                    total += float(getattr(cfg, "wide_face_aspect_penalty_weight", 10.0)) * strength * (asp - wide_limit)

            if total < best_score:
                best_score = total
                best = (
                    int(round(ex1)),
                    int(round(ey1)),
                    int(round(ex2)),
                    int(round(ey2)),
                )
                best_ratio = rs
                best_template_loss = tmpl_loss

        # Fallback if no candidate selected
        if best is None:
            try:
                rw, rh = parse_ratio(str(ratios[0]))
                ex1, ey1, ex2, ey2 = expand_box_to_ratio(
                    x1, y1, x2, y2, rw, rh, frame_w, frame_h, anchor=anchor, head_bias=0.0
                )
                best = (
                    int(round(ex1)),
                    int(round(ey1)),
                    int(round(ex2)),
                    int(round(ey2)),
                )
                best_ratio = str(ratios[0])
                best_template_loss = 0.0
            except Exception:
                best = (
                    int(round(x1)),
                    int(round(y1)),
                    int(round(x2)),
                    int(round(y2)),
                )
                best_ratio = None
                best_template_loss = 0.0
        return best, best_ratio, best_template_loss

    def _calc_sharpness(self, bgr):
        if bgr is None or bgr.size == 0:
            return 0.0
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h, w = g.shape[:2]
        max_dim = max(h, w)
        if max_dim > 256:
            scale = 256.0 / float(max_dim)
            g = cv2.resize(
                g,
                (int(round(w * scale)), int(round(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        lap = cv2.Laplacian(g, cv2.CV_32F)
        variance = float(np.var(lap))
        mean_intensity = float(np.mean(g))
        return variance / (mean_intensity * mean_intensity + 1e-6)

    def _calc_saved_file_sharpness(self, path: str):
        try:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        except Exception:
            return None
        if img is None or getattr(img, "size", 0) == 0:
            return None
        try:
            return self._calc_sharpness(img)
        except Exception:
            return None

    def _autocrop_borders(self, frame, thr):
        # Package-safe import (supports both module and flat execution)
        try:
            from .utils import detect_black_borders  # type: ignore
        except Exception:
            from utils import detect_black_borders  # type: ignore
        # deeper, configurable scan so thick bars get removed
        h, w = frame.shape[:2]
        scan_frac = float(getattr(self.cfg, "border_scan_frac", 0.25))
        max_scan = int(round(min(h, w) * max(0.0, scan_frac))) if scan_frac > 0 else None
        x1,y1,x2,y2 = detect_black_borders(frame, thr=int(thr), max_scan=max_scan)
        return frame[y1:y2, x1:x2], (x1,y1)

    def _iou(self, a, b):
        ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
        inter_x1, inter_y1 = max(ax1,bx1), max(ay1,by1)
        inter_x2, inter_y2 = min(ax2,bx2), min(ay2,by2)
        iw, ih = max(0, inter_x2-inter_x1), max(0, inter_y2-inter_y1)
        inter = iw*ih
        a_area = max(0, (ax2-ax1)*(ay2-ay1))
        b_area = max(0, (bx2-bx1)*(by2-by1))
        union = a_area + b_area - inter + 1e-9
        return inter/union

    def _expand_to_ratio(self, box, ratio_w, ratio_h, frame_w, frame_h, anchor=None):
        try:
            from .utils import expand_box_to_ratio  # type: ignore
        except Exception:
            from utils import expand_box_to_ratio  # type: ignore
        x1,y1,x2,y2 = box
        ex1,ey1,ex2,ey2 = expand_box_to_ratio(x1,y1,x2,y2, ratio_w, ratio_h, frame_w, frame_h, anchor=anchor, head_bias=0.12)
        return ex1,ey1,ex2,ey2

    def _shrink_to_ratio_inside(self, box, ratio_w, ratio_h, bounds, anchor=None):
        """Shrink `box` to target ratio inside `bounds`, biased to keep `anchor` centered."""
        x1,y1,x2,y2 = map(int, box)
        bx1,by1,bx2,by2 = map(int, bounds)
        w = max(1, x2-x1); h = max(1, y2-y1)
        tgt = float(ratio_w) / float(ratio_h)
        cur = w / float(h)
        if anchor is None:
            ax = (x1 + x2) * 0.5; ay = (y1 + y2) * 0.5
        else:
            ax, ay = anchor
        ax = max(bx1, min(bx2, ax)); ay = max(by1, min(by2, ay))
        if abs(cur - tgt) <= 1e-3:
            return x1,y1,x2,y2
        if cur < tgt:
            new_h = min(h, max(1, int(round(w / max(1e-6, tgt)))))
            ny1 = int(round(ay - new_h * 0.5))
            ny1 = max(by1, min(by2 - new_h, ny1))
            ny2 = ny1 + new_h
            return x1, ny1, x2, ny2
        else:
            new_w = min(w, max(1, int(round(h * tgt))))
            nx1 = int(round(ax - new_w * 0.5))
            nx1 = max(bx1, min(bx2 - new_w, nx1))
            nx2 = nx1 + new_w
            return nx1, y1, nx2, y2

    # Signals for UI
    setup = QtCore.Signal(int, float)                # total_frames, fps
    progress = QtCore.Signal(int)                    # current frame idx
    status = QtCore.Signal(str)                      # status text
    preview = QtCore.Signal(QtGui.QImage)            # annotated frame
    preview_hdr_p010 = QtCore.Signal(object)         # single (w, h, y, uv, stride_y, stride_uv)
    hit = QtCore.Signal(str)                         # crop path
    finished = QtCore.Signal(bool, str)              # success, message
    keyframes = QtCore.Signal(object)                # list[int]

    # --- Player control slots (queued, thread-safe) ---
    @QtCore.Slot(dict)
    def update_cfg(self, changes: dict):
        try:
            self._cmd_q.put_nowait(("cfg", dict(changes)))
        except Exception:
            pass

    @QtCore.Slot(int)
    def seek_frame(self, frame_idx: int):
        try:
            self._cmd_q.put_nowait(("seek", int(frame_idx)))
        except Exception:
            pass

    @QtCore.Slot(float)
    def seek_time(self, secs: float):
        try:
            if getattr(self, "_fps", None):
                tgt = int(max(0, round(secs * float(self._fps))))
                self._cmd_q.put_nowait(("seek", tgt))
        except Exception:
            pass

    @QtCore.Slot()
    def play(self):
        try:
            self._cmd_q.put_nowait(("play", None))
        except Exception:
            pass

    @QtCore.Slot()
    def pause(self):
        try:
            self._cmd_q.put_nowait(("pause", None))
        except Exception:
            pass

    @QtCore.Slot(int)
    def step(self, n: int = 1):
        try:
            self._cmd_q.put_nowait(("step", int(n)))
        except Exception:
            pass

    @QtCore.Slot(float)
    def set_speed(self, s: float):
        try:
            self._cmd_q.put_nowait(("speed", float(s)))
        except Exception:
            pass

    # -------- keyframe utils inside worker (MP4/MOV/MKV native; no externals required) --------
    def _read_keyframes_worker(self, video_path: str, fps: Optional[float], total: Optional[int]) -> List[int]:
        if not video_path or not os.path.exists(video_path):
            return []
        if not fps or fps <= 0:
            fps = float(self._fps or 30.0) or 30.0
        ext = os.path.splitext(video_path)[1].lower()
        ks: List[int] = []
        if ext in (".mkv", ".webm"):
            try:
                mkv_ks = self._mkv_read_cues(video_path, float(fps), int(total or 0))
                if mkv_ks:
                    self.status.emit(f"Keyframes (MKV cues): {len(mkv_ks)}")
                    ks = mkv_ks
            except Exception:
                pass
        if not ks and ext in (".mp4", ".m4v", ".mov"):
            try:
                mp4_ks = self._mp4_read_stss(video_path, int(total or 0))
                if mp4_ks:
                    self.status.emit(f"Keyframes (MP4 stss): {len(mp4_ks)}")
                    ks = mp4_ks
            except Exception:
                pass
        if not ks:
            f = float(fps or 30.0)
            tf = int(total or 0)
            step = max(1, int(round(f)))
            if tf > 0:
                ks = list(range(0, tf, step))
            else:
                ks = [0]
            if ks:
                self.status.emit(f"Keyframes (grid): {len(ks)} at ~1 Hz")
        ks = sorted(set(ks))
        if isinstance(total, int) and total > 0:
            last = total - 1
            ks = [k for k in ks if 0 <= k <= last]
        if not ks:
            self.status.emit(
                f"KF stats: ext={ext} total={total} fps={float(fps):.3f} ks={len(ks)}"
            )
        return ks

    def _mp4_read_stss(self, path: str, total_frames: int) -> List[int]:
        def _u32(b: bytes) -> int:
            return struct.unpack(">I", b)[0]

        def _u64(b: bytes) -> int:
            return struct.unpack(">Q", b)[0]

        def _read_box(f, end_pos: int):
            pos = f.tell()
            if pos + 8 > end_pos:
                return None
            hdr = f.read(8)
            if len(hdr) < 8:
                return None
            sz = _u32(hdr[:4])
            typ = hdr[4:8].decode("ascii", "ignore")
            header = 8
            if sz == 1:
                ext = f.read(8)
                if len(ext) < 8:
                    return None
                sz = _u64(ext)
                header = 16
            if sz == 0:
                sz = end_pos - pos
            return pos, typ, sz, header

        def _find_child(f, parent_start: int, parent_size: int, name: str):
            end = parent_start + parent_size
            f.seek(parent_start + 8)
            while f.tell() + 8 <= end:
                box = _read_box(f, end)
                if not box:
                    break
                pos, typ, sz, header = box
                if typ == name:
                    return (pos, sz)
                f.seek(pos + sz)
            return None

        size = os.path.getsize(path)
        with open(path, "rb") as f:
            end = size
            f.seek(0)
            moov = None
            while f.tell() + 8 <= end:
                box = _read_box(f, end)
                if not box:
                    break
                pos, typ, sz, header = box
                if typ == "moov":
                    moov = (pos, sz)
                    break
                f.seek(pos + sz)
            if not moov:
                return []

            moov_pos, moov_sz = moov
            f.seek(moov_pos + 8)
            moov_end = moov_pos + moov_sz
            video_trak = None
            while f.tell() + 8 <= moov_end:
                box = _read_box(f, moov_end)
                if not box:
                    break
                pos, typ, sz, header = box
                if typ == "trak":
                    mdia = _find_child(f, pos, sz, "mdia")
                    if mdia:
                        hdlr = _find_child(f, mdia[0], mdia[1], "hdlr")
                        if hdlr:
                            f.seek(hdlr[0] + 16)
                            handler = f.read(4).decode("ascii", "ignore")
                            if handler == "vide":
                                video_trak = (pos, sz)
                                break
                f.seek(pos + sz)
            if not video_trak:
                return []

            mdia = _find_child(f, video_trak[0], video_trak[1], "mdia")
            if not mdia:
                return []
            minf = _find_child(f, mdia[0], mdia[1], "minf")
            if not minf:
                return []
            stbl = _find_child(f, minf[0], minf[1], "stbl")
            if not stbl:
                return []
            stss = _find_child(f, stbl[0], stbl[1], "stss")
            if not stss:
                return []

            f.seek(stss[0] + 8)
            _ = f.read(4)
            cnt = f.read(4)
            if len(cnt) < 4:
                return []
            n = _u32(cnt)
            out: List[int] = []
            max_sample = 0
            for _ in range(n):
                data = f.read(4)
                if len(data) < 4:
                    break
                sample_num = _u32(data)
                max_sample = max(max_sample, sample_num)
                out.append(max(sample_num - 1, 0))
            out.sort()
            if not out:
                return []
            if total_frames and max_sample and max_sample - 1 != total_frames:
                scale = float(total_frames) / float(max_sample)
                out = [
                    max(0, min(total_frames - 1, int(round((s + 1) * scale) - 1)))
                    for s in out
                ]
                out = sorted(set(out))
            elif total_frames:
                out = [min(total_frames - 1, s) for s in out]
            return out

    def _mkv_read_cues(self, path: str, fps: float, total_frames: int) -> List[int]:
        ID_SEGMENT = 0x18538067
        ID_INFO = 0x1549A966
        ID_TIMECODESCALE = 0x2AD7B1
        ID_TRACKS = 0x1654AE6B
        ID_TRACKENTRY = 0xAE
        ID_TRACKNUMBER = 0xD7
        ID_TRACKTYPE = 0x83
        ID_CUES = 0x1C53BB6B
        ID_CUEPOINT = 0xBB
        ID_CUETIME = 0xB3
        ID_CUETRACKPOS = 0xB7
        ID_CUETRACK = 0xF7

        def read_vint(f):
            first = f.read(1)
            if not first:
                return None
            b0 = first[0]
            mask = 0x80
            length = 1
            while length <= 8 and (b0 & mask) == 0:
                mask >>= 1
                length += 1
            if length > 8:
                return None
            value = b0 & (mask - 1)
            for _ in range(length - 1):
                nxt = f.read(1)
                if not nxt:
                    return None
                value = (value << 8) | nxt[0]
            return value, length

        def read_id(f):
            first = f.read(1)
            if not first:
                return None
            b0 = first[0]
            mask = 0x80
            length = 1
            while length <= 4 and (b0 & mask) == 0:
                mask >>= 1
                length += 1
            if length > 4:
                return None
            raw = bytes([b0]) + f.read(length - 1)
            val = 0
            for b in raw:
                val = (val << 8) | b
            return val, length

        def read_elem_header(f):
            rid = read_id(f)
            if not rid:
                return None
            size = read_vint(f)
            if not size:
                return None
            eid, _ = rid
            esz, _ = size
            return eid, esz

        size = os.path.getsize(path)
        with open(path, "rb") as f:
            seg_start = None
            seg_size = None
            end = size
            f.seek(0)
            while f.tell() < end:
                header = read_elem_header(f)
                if not header:
                    break
                eid, esz = header
                body = f.tell()
                if eid == ID_SEGMENT:
                    seg_start = body
                    # esz of all 1 bits means "unknown"; treat as None
                    max_unknown = (1 << (7 * 4)) - 1
                    seg_size = esz if esz < max_unknown else None
                    break
                f.seek(body + esz)
            if seg_start is None:
                return []

            timecode_scale = 1_000_000
            video_track = None
            cues_pos = None
            cues_size = None

            seg_end = end if seg_size is None else seg_start + seg_size
            f.seek(seg_start)
            while f.tell() < seg_end:
                header = read_elem_header(f)
                if not header:
                    break
                eid, esz = header
                body = f.tell()
                if eid == ID_INFO:
                    info_end = body + esz
                    while f.tell() < info_end:
                        sub = read_elem_header(f)
                        if not sub:
                            break
                        sid, ssz = sub
                        sub_body = f.tell()
                        if sid == ID_TIMECODESCALE:
                            raw = f.read(ssz)
                            val = 0
                            for b in raw:
                                val = (val << 8) | b
                            if val > 0:
                                timecode_scale = int(val)
                        f.seek(sub_body + ssz)
                elif eid == ID_TRACKS:
                    tracks_end = body + esz
                    while f.tell() < tracks_end:
                        sub = read_elem_header(f)
                        if not sub:
                            break
                        sid, ssz = sub
                        sub_body = f.tell()
                        if sid == ID_TRACKENTRY:
                            te_end = sub_body + ssz
                            track_num = None
                            track_type = None
                            while f.tell() < te_end:
                                entry = read_elem_header(f)
                                if not entry:
                                    break
                                eid3, ssz3 = entry
                                entry_body = f.tell()
                                if eid3 == ID_TRACKNUMBER:
                                    raw = f.read(ssz3)
                                    val = 0
                                    for b in raw:
                                        val = (val << 8) | b
                                    track_num = val
                                elif eid3 == ID_TRACKTYPE:
                                    raw = f.read(ssz3)
                                    val = 0
                                    for b in raw:
                                        val = (val << 8) | b
                                    track_type = val
                                f.seek(entry_body + ssz3)
                            if track_type == 1 and track_num is not None and video_track is None:
                                video_track = track_num
                        f.seek(sub_body + ssz)
                elif eid == ID_CUES:
                    cues_pos = body
                    cues_size = esz
                    f.seek(body + esz)
                    break
                f.seek(body + esz)

            if cues_pos is None or cues_size is None:
                return []

            f.seek(cues_pos)
            cues_end = cues_pos + cues_size
            keyframes: List[int] = []
            while f.tell() < cues_end:
                cp = read_elem_header(f)
                if not cp:
                    break
                eid, esz = cp
                body = f.tell()
                if eid != ID_CUEPOINT:
                    f.seek(body + esz)
                    continue
                cue_end = body + esz
                cue_time = None
                track_ok = video_track is None
                while f.tell() < cue_end:
                    sub = read_elem_header(f)
                    if not sub:
                        break
                    sid, ssz = sub
                    sub_body = f.tell()
                    if sid == ID_CUETIME:
                        raw = f.read(ssz)
                        val = 0
                        for b in raw:
                            val = (val << 8) | b
                        cue_time = val
                    elif sid == ID_CUETRACKPOS:
                        sub_end = sub_body + ssz
                        while f.tell() < sub_end:
                            entry = read_elem_header(f)
                            if not entry:
                                break
                            eid3, ssz3 = entry
                            entry_body = f.tell()
                            if eid3 == ID_CUETRACK:
                                raw = f.read(ssz3)
                                val = 0
                                for b in raw:
                                    val = (val << 8) | b
                                if video_track is None or val == video_track:
                                    track_ok = True
                            f.seek(entry_body + ssz3)
                    f.seek(sub_body + ssz)
                if cue_time is not None and track_ok:
                    seconds = cue_time * (timecode_scale / 1_000_000_000.0)
                    idx = int(round(seconds * float(fps)))
                    if total_frames:
                        idx = max(0, min(total_frames - 1, idx))
                    keyframes.append(idx)
                f.seek(body + esz)

        keyframes = sorted(set(keyframes))
        return keyframes

    def _seek_to(
        self,
        cap: cv2.VideoCapture,
        cur_idx: int,
        tgt_idx: int,
        *,
        fast: bool = False,
        max_grabs: int = 0,
        peek_preview: bool = False,
        allow_partial: bool = False,
        hdr_reader=None,
    ) -> int:
        """Keyframe-aware seek. Fast UI seeks may return before target; internal jumps must not."""
        if self._total_frames is not None:
            tgt_idx = max(0, min(self._total_frames - 1, int(tgt_idx)))

        # The external HDR ffmpeg pipe already seeks by timestamp and lets ffmpeg do
        # accurate decode/drop internally. Rewinding it to the previous container
        # keyframe and then capped-grabbing in Python is both slower and can livelock
        # when a segment jump is repeatedly time-capped before reaching tgt_idx.
        direct_pipe_seek = bool(getattr(cap, "_is_hdr_pipe", False))
        if direct_pipe_seek:
            base = tgt_idx
            ks = []
        else:
            base = tgt_idx
            ks = self._keyframes
            if ks:
                i = bisect.bisect_right(ks, tgt_idx) - 1
                if i >= 0:
                    base = int(ks[i])

            # If a previous partial fast seek already decoded past the selected
            # keyframe, continue forward from the current position instead of
            # seeking backward to the same base again. Otherwise repeated
            # time-capped seeks can make zero net progress.
            try:
                cur_i = int(cur_idx)
            except Exception:
                cur_i = -1
            if fast and base < cur_i <= tgt_idx:
                base = cur_i

            # If we have no KF info or landed exactly on tgt, choose a near target
            # base only for partial/UI seeks. Internal segment jumps must reach
            # their target in a single call.
            if fast and allow_partial and (not ks or base == tgt_idx):
                mg = max_grabs
                if mg <= 0:
                    f = float(self._fps or 30.0)
                    mg = max(15, min(240, int(round(f))))
                base = max(0, tgt_idx - int(mg))
        # For OpenCV readers, cur_idx is the caller's processing cursor. For
        # HDR ffmpeg pipes, some pre-scan callers pass the desired index as
        # cur_idx even after the pipe has advanced to EOF. Compare against the
        # pipe's real next index so a requested seek cannot be skipped.
        cur_for_seek = cur_idx
        if direct_pipe_seek:
            try:
                cur_for_seek = int(cap._next_frame_index())
            except Exception:
                try:
                    cur_for_seek = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                except Exception:
                    cur_for_seek = -1
        if base != cur_for_seek:
            # time-based reposition is often faster on some backends; fall back if it fails.
            # HDR ffmpeg pipes implement only frame-based seeking and add their own small
            # preroll internally, so do not route those seeks through POS_MSEC.
            if fast and self._fps and not direct_pipe_seek:
                ok = cap.set(cv2.CAP_PROP_POS_MSEC, (base / float(self._fps)) * 1000.0)
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, base)
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, base)
            self._hdr_preview_seek(base, reader=hdr_reader)
        idx = base
        limit = max(0, tgt_idx - base)
        # Cap forward grabs only for partial/UI fast seeks. Segment jumps in the
        # processing pass must not return before the requested keep-span start.
        if fast and allow_partial:
            if max_grabs <= 0:
                f = float(self._fps or 30.0)
                max_grabs = max(15, min(240, int(round(f))))
            capped = (tgt_idx - base) > max_grabs
            limit = min(limit, int(max_grabs))
            if capped:
                try:
                    self._status(
                        f"Fast-seek capped forward decode to {limit} frames",
                        key="seek_cap",
                        interval=1.0,
                    )
                except Exception:
                    pass
        # Time-budgeted forward grabs to avoid long decode stalls during UI seeks.
        t0 = time.perf_counter() if (fast and allow_partial) else None
        budget = 0.15  # ~150 ms max spent per partial/UI seek
        for i in range(limit):
            if not cap.grab():
                break
            idx += 1
            # Light preview peek only when explicitly enabled (UI scrubs)
            peek_n = max(1, int(getattr(self.cfg, "seek_preview_peek_every", 16)))
            do_peek = peek_preview and fast and (i % peek_n) == 0
            if do_peek:
                ok, frame = cap.retrieve()
                if ok:
                    try:
                        self._pump_hdr_preview(reader=hdr_reader)
                        self._emit_preview_bgr(frame)
                    except Exception:
                        pass
                else:
                    self._hdr_preview_skip(1, reader=hdr_reader)
            else:
                self._hdr_preview_skip(1, reader=hdr_reader)
            if fast and allow_partial and t0 is not None and (time.perf_counter() - t0) > budget:
                try:
                    self._status(
                        f"Fast-seek time-capped at {idx - base} grabs",
                        key="seek_cap_t",
                        interval=1.0,
                    )
                except Exception:
                    pass
                break
        return idx

    def __init__(self, cfg: SessionConfig, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self._abort = False
        self._pause = False
        # Player state
        self._cmd_q: queue.Queue = queue.Queue()
        self._paused = False
        self._speed = 1.0
        self._fps: Optional[float] = None
        self._total_frames: Optional[int] = None
        self._keyframes: List[int] = []   # worker-side keyframe index
        self._seek_cooldown_frames: int = 0  # disable lock/IoU for a few frames after seek
        self._runtime_last_bank_add_idx: int = -10**9  # cooldown anchor for runtime bank adds
        # Target lock state for faceless fallback
        self._lock_active: bool = False
        self._lock_last_bbox: Optional[Tuple[int, int, int, int]] = None  # x,y,w,h in frame coords
        self._lock_last_seen_idx: int = -10**9
        self._locked_reid_feat: Optional[np.ndarray] = None
        self._prev_gray: Optional[np.ndarray] = None
        self._last_preview_t: float = 0.0
        self.ui_hdr_passthrough_enabled: bool = False
        self._hdr_passthrough_active: bool = False
        self._hdr_preview_reader: Optional[object] = None
        self._hdr_preview_latest: Optional[
            tuple[int, int, np.ndarray, np.ndarray, int, int]
        ] = None

    def _emit_hit(self, path: str) -> None:
        emit = getattr(self.hit, "emit", None)
        if emit is None:
            return
        try:
            emit(path)
        except Exception:
            logger.exception("Failed to emit hit for %s", path)

    @QtCore.Slot()
    def request_abort(self):
        self._abort = True

    @QtCore.Slot(bool)
    def request_pause(self, p: bool):
        self._pause = bool(p)
        self._paused = bool(p)

    # --- Lock helpers ---
    def _set_lock(self, bbox_xywh: Tuple[int, int, int, int], frame_idx: int):
        x, y, w, h = [int(v) for v in bbox_xywh]
        self._lock_active = True
        self._lock_last_bbox = (x, y, w, h)
        self._lock_last_seen_idx = int(frame_idx)

    def _clear_lock(self):
        self._lock_active = False
        self._lock_last_bbox = None
        self._lock_last_seen_idx = -10**9
        self._locked_reid_feat = None

    def _iou_xywh(self, a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        return self._iou((ax, ay, ax + aw, ay + ah), (bx, by, bx + bw, by + bh))

    def _faceless_pick_from_persons(self, persons_xywh, reid_feats, locked_feat):
        """Return (idx, reason) of best faceless fallback candidate or (-1, None)."""
        best_idx, best_score, reason = -1, 1e9, None
        if locked_feat is not None and reid_feats:
            try:
                lock_vec = np.asarray(locked_feat, dtype=np.float32)
                lock_vec = lock_vec / max(float(np.linalg.norm(lock_vec)), 1e-9)
            except Exception:
                lock_vec = None
            if lock_vec is not None:
                for i, feat in enumerate(reid_feats):
                    if feat is None:
                        continue
                    try:
                        vec = np.asarray(feat, dtype=np.float32)
                        vec = vec / max(float(np.linalg.norm(vec)), 1e-9)
                        d = 1.0 - float(np.dot(lock_vec, vec))
                    except Exception:
                        continue
                    if d < best_score:
                        best_score, best_idx, reason = d, i, "reid"
                if best_idx >= 0 and best_score <= float(self.cfg.faceless_reid_thresh):
                    return best_idx, reason
        if self._lock_last_bbox and persons_xywh:
            iou_best, j_best = 0.0, -1
            for j, b in enumerate(persons_xywh):
                try:
                    cur_iou = float(self._iou_xywh(self._lock_last_bbox, tuple(map(int, b))))
                except Exception:
                    cur_iou = 0.0
                if cur_iou > iou_best:
                    iou_best, j_best = cur_iou, j
            if j_best >= 0 and iou_best >= float(self.cfg.faceless_iou_min):
                return j_best, "iou"
        return -1, None

    def _faceless_validate(self, box_xyxy, frame_shape, gray_frame):
        cfg = self.cfg
        if gray_frame is None or frame_shape is None:
            return True, None
        if len(frame_shape) < 2:
            return True, None
        H = int(frame_shape[0])
        W = int(frame_shape[1])
        x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]
        x1 = max(0, min(W - 1, x1))
        y1 = max(0, min(H - 1, y1))
        x2 = max(x1 + 1, min(W, x2))
        y2 = max(y1 + 1, min(H, y2))
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        area = float(w * h)
        frame_area = float(max(1, W * H))
        frac = area / frame_area
        if frac < float(cfg.faceless_min_area_frac):
            return False, "area_min"
        if frac > float(cfg.faceless_max_area_frac):
            return False, "area_max"
        if self._lock_last_bbox is not None:
            lx, ly, lw, lh = self._lock_last_bbox
            lock_cx = float(lx) + float(lw) / 2.0
            lock_cy = float(ly) + float(lh) / 2.0
            cx = x1 + w / 2.0
            cy = y1 + h / 2.0
            diag = math.hypot(W, H)
            if diag > 0:
                drift = math.hypot(cx - lock_cx, cy - lock_cy) / diag
                if drift > float(cfg.faceless_center_max_frac):
                    return False, "center_drift"
        prev = self._prev_gray
        if prev is not None and prev.shape == gray_frame.shape:
            roi_prev = prev[y1:y2, x1:x2]
            roi_cur = gray_frame[y1:y2, x1:x2]
            if roi_prev.size > 0 and roi_cur.size > 0:
                diff = cv2.absdiff(roi_cur, roi_prev)
                _, mask = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
                motion = cv2.countNonZero(mask)
                motion_frac = motion / area if area > 0 else 0.0
                if motion_frac < float(cfg.faceless_min_motion_frac):
                    return False, "motion"
        return True, None

    @QtCore.Slot()
    def run_curator(self) -> None:
        """Delegate to the shared dataset curator for scoring and export."""

        cfg = self.cfg
        if not bool(getattr(cfg, "curate_enable", True)):
            self._status("Curator disabled.", key="curate", interval=5.0)
            return
        out_root = Path(cfg.out_dir or "output")
        pool = out_root / "crops"
        if not pool.exists():
            self._status("Curator: no crops/ directory found.", key="curate", interval=5.0)
            return
        ref_candidates = [p.strip() for p in str(cfg.ref or "").split(";") if p.strip()]
        ref_path = next((str(Path(p)) for p in ref_candidates if Path(p).exists()), "")
        assume_identity = bool(not ref_path)
        if assume_identity:
            self._status("Curator: identity gate disabled (no reference).", key="curate", interval=5.0)
        try:
            from .dataset_curator import Curator as _Curator  # type: ignore
        except Exception:
            try:
                from dataset_curator import Curator as _Curator  # type: ignore
            except Exception as exc:  # pragma: no cover
                self._status(f"Curator import failed: {exc}", key="curate", interval=5.0)
                return
        max_imgs = int(getattr(cfg, "curate_max_images", 200))
        out_dir = out_root / "dataset_out"

        def _progress(phase: str, done: int, total: int) -> None:
            # be robust to numpy scalars etc.
            try:
                di, ti = int(done), int(total)
            except Exception:
                di, ti = 0, 0
            self._status(
                f"Curator {phase}: {di}/{ti}",
                key="curate_prog",
                interval=0.2,
            )

        # show *something* instantly in the log/status
        _progress("init: starting", 0, 0)

        try:
            curator = _Curator(
                ref_image=(ref_path or None),
                device=str(getattr(cfg, "device", "cuda")),
                trt_lib_dir=(getattr(cfg, "trt_lib_dir", "") or None),
                face_model=str(getattr(cfg, "face_model", "scrfd_10g_bnkps")),
                face_det_conf=float(getattr(cfg, "face_det_conf", 0.50)),
                assume_identity=assume_identity,
                progress=_progress,
            )
            if not getattr(curator, "id_already_passed", False) and getattr(curator, "ref_feat", None) is None:
                self._status("Curator: no face in the reference image.", key="curate", interval=5.0)
                return
            built = curator.run(str(pool), str(out_dir), max_images=max_imgs)
        except Exception as exc:  # pragma: no cover
            self._status(f"Curator failed: {exc}", key="curate", interval=5.0)
            return

        self._status(f"Curator: done → {built}", key="curate_done", interval=1.0)

    def _combine_scores(self, face_dist, reid_dist, mode='min'):
        vals = []
        if face_dist is not None:
            vals.append(face_dist)
        if reid_dist is not None:
            vals.append(reid_dist)
        if not vals:
            return None
        if mode == 'min':
            return min(vals)
        if mode == 'avg':
            return sum(vals)/len(vals)
        if mode == 'face_priority':
            if face_dist is not None:
                return 0.7*face_dist + 0.3*(reid_dist if reid_dist is not None else 0.5)
            else:
                return reid_dist
        return min(vals)

    @QtCore.Slot()
    def run(self):
        cap = save_q = saver_thread = archive_q = archive_thread = csv_f = dbg_f = hit_q = None
        finished_ok = False
        finished_msg = "Unknown termination"
        try:
            # Apply TRT/ORT env from cfg early
            def _env_set(k, v):
                if v is None:
                    return
                os.environ[k] = str(v)

            # Resolve caches against REPO ROOT, not package dir

            def _abs_repo(p: Optional[str]) -> Optional[str]:
                if not p:
                    return None
                q = Path(p)
                return str(q if q.is_absolute() else (_REPO_ROOT / q))

            # Allow null/empty in cfg, then fall back to default under repo root
            _cfg_trt = getattr(self.cfg, "trt_cache_root", None)
            trt_cache_root = _abs_repo(_cfg_trt) or str(_REPO_ROOT / "trt_cache")
            _env_set("PERSON_CAPTURE_TRT_CACHE_ROOT", trt_cache_root)

            logger.info("Repo root=%s", _REPO_ROOT)
            logger.info("TRT cache root=%s", trt_cache_root)
            logger.info("ULTRALYTICS_HOME=%s", os.environ.get("ULTRALYTICS_HOME"))
            _env_set(
                "PERSON_CAPTURE_TRT_FP16",
                "True" if getattr(self.cfg, "trt_fp16_enable", True) else "False",
            )
            _env_set(
                "PERSON_CAPTURE_TRT_TIMING_CACHE_ENABLE",
                "True" if getattr(self.cfg, "trt_timing_cache_enable", True) else "False",
            )
            _env_set(
                "PERSON_CAPTURE_TRT_ENGINE_CACHE_ENABLE",
                "True" if getattr(self.cfg, "trt_engine_cache_enable", True) else "False",
            )
            _env_set("PERSON_CAPTURE_TRT_BUILDER_OPT_LEVEL", int(getattr(self.cfg, "trt_builder_optimization_level", 5)))
            _env_set(
                "PERSON_CAPTURE_TRT_CUDA_GRAPH_ENABLE",
                "True" if getattr(self.cfg, "trt_cuda_graph_enable", True) else "False",
            )
            _env_set(
                "PERSON_CAPTURE_TRT_CONTEXT_MEMORY_SHARING_ENABLE",
                "True" if getattr(self.cfg, "trt_context_memory_sharing_enable", True) else "False",
            )
            _env_set("PERSON_CAPTURE_TRT_AUX_STREAMS", int(getattr(self.cfg, "trt_auxiliary_streams", -1)))
            _env_set("PERSON_CAPTURE_CUDA_USE_TF32", "1" if getattr(self.cfg, "cuda_use_tf32", True) else "0")

            # Honor TF32 after env is set
            try:
                import torch as _torch_tf32
                _tf32 = os.getenv("PERSON_CAPTURE_CUDA_USE_TF32", "1").lower() not in ("0", "", "false")
                _torch_tf32.backends.cuda.matmul.allow_tf32 = _tf32
                _torch_tf32.backends.cudnn.allow_tf32 = _tf32
            except Exception:
                pass

            self._paused = False
            self._init_status()
            cfg = self.cfg
            _apply_ffmpeg_hwaccel_env(cfg)
            self._clear_lock()
            self._prev_gray = None
            if not os.path.isfile(cfg.video):
                raise FileNotFoundError(f"Video not found: {cfg.video}")
            ref_paths = [p.strip() for p in str(cfg.ref).split(';') if p.strip()]
            if not ref_paths:
                raise FileNotFoundError(f"Reference image not found: {cfg.ref}")
            ref_paths_existing = [rp for rp in ref_paths if os.path.isfile(rp)]
            if not ref_paths_existing:
                raise FileNotFoundError(f"Reference image not found: {cfg.ref}")

            ensure_dir(cfg.out_dir)
            crops_dir = os.path.join(cfg.out_dir, "crops")
            hdr_crops_dir = os.path.join(cfg.out_dir, "hdr_crops")
            ann_dir = os.path.join(cfg.out_dir, "annot") if cfg.save_annot else None
            ensure_dir(crops_dir)
            if ann_dir:
                ensure_dir(ann_dir)
            # Debug I/O
            if getattr(cfg, "debug_dump", False):
                dbg_dir = os.path.join(cfg.out_dir, getattr(cfg, "debug_dir", "debug"))
                ensure_dir(dbg_dir)
                dbg_f = open(os.path.join(dbg_dir, "debug.jsonl"), "w", encoding="utf-8")

            def _dump_debug(obj):
                if dbg_f:
                    try:
                        import json

                        json.dump(obj, dbg_f, ensure_ascii=False)
                        dbg_f.write("\n")
                        dbg_f.flush()
                    except Exception:
                        pass

            def _frame_time(idx: int) -> float:
                fps_val = float(self._fps or 30.0)
                if fps_val <= 0:
                    return 0.0
                return float(idx) / fps_val

            def _frame_stride() -> int:
                try:
                    stride_val = int(getattr(cfg, "frame_stride", 1))
                except Exception:
                    stride_val = 1
                return max(1, stride_val)

            def _cooldown_frames() -> int:
                stride = _frame_stride()
                return max(3, min(8, int(5 / stride)))

            self._status("Loading models...", key="phase", interval=5.0)
            det = PersonDetector(model_name=cfg.yolo_model, device=cfg.device, progress=self.status.emit)
            face = FaceEmbedder(
                ctx=cfg.device,
                yolo_model=cfg.face_model,
                conf=float(cfg.face_det_conf),
                use_arcface=cfg.use_arcface,
                clip_model_name=cfg.clip_face_backbone,
                clip_pretrained=cfg.clip_face_pretrained,
                progress=self.status.emit,
                trt_lib_dir=(cfg.trt_lib_dir or None),
            )
            face.configure_rotation_strategy(
                adaptive=getattr(cfg, "rot_adaptive", True),
                every_n=getattr(cfg, "rot_every_n", 12),
                after_hit_frames=getattr(cfg, "rot_after_hit_frames", 8),
                fast_no_face_imgsz=getattr(cfg, "fast_no_face_imgsz", 512),
            )
            reid = None
            if not (getattr(cfg, "disable_reid", False) or getattr(cfg, "match_mode", "") == "face_only"):
                reid = ReIDEmbedder(
                    device=cfg.device,
                    model_name=cfg.reid_backbone,
                    pretrained=cfg.reid_pretrained,
                    progress=self.status.emit,
                )
            self.reid = reid

            # Reference
            self._status("Preparing reference features...", key="phase", interval=5.0)
            ref_bank_list: List[np.ndarray] = []
            ref_img_primary = None
            ref_face_feat_initial = np.vstack(ref_bank_list).astype(np.float32) if ref_bank_list else None
            for rp in ref_paths_existing:
                img = cv2.imread(rp, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                if ref_img_primary is None:
                    ref_img_primary = img
                # Rotation augments removed: keep only the original image.
                aug_images: List[np.ndarray] = [img]
                aug_with_flips: List[np.ndarray] = []
                for aug in aug_images:
                    aug_with_flips.append(aug)
                    try:
                        aug_with_flips.append(cv2.flip(aug, 1))
                    except Exception:
                        pass
                for aug in aug_with_flips:
                    faces = face.extract(aug)
                    bf = FaceEmbedder.best_face(faces)
                    if bf and bf.get("feat") is not None:
                        try:
                            quality_val = float(bf.get("quality", 0.0))
                        except Exception:
                            quality_val = 0.0
                        ref_face_feat_initial, _, _ = self._stream_ref_bank_update(
                            ref_bank_list,
                            ref_face_feat_initial,
                            bf["feat"],
                            quality_val,
                            cfg,
                        )

            if ref_img_primary is None:
                raise RuntimeError("Cannot read reference image(s)")

            ref_face_feat = np.vstack(ref_bank_list).astype(np.float32) if ref_bank_list else None
            if ref_face_feat is not None:
                bank_arr = np.asarray(ref_face_feat, dtype=np.float32)
                if bank_arr.ndim == 1:
                    bank_arr = bank_arr.reshape(1, -1)
                norms = [float(np.linalg.norm(row)) for row in bank_arr]
                preview = ", ".join(f"{n:.3f}" for n in norms[:3])
                if len(norms) > 3:
                    preview += ", …"
                self._status(
                    f"Ref face bank: {len(norms)} feats from {len(ref_paths_existing)} refs | norms={preview} backend={getattr(face, 'backend', None)}",
                    key="ref",
                    interval=60.0,
                )
            else:
                self._status(
                    f"Ref face: missing (checked {len(ref_paths_existing)} refs)",
                    key="ref",
                    interval=60.0,
                )

            try:
                bank_fd_thresh = float(getattr(cfg, "face_thresh", 0.38))
            except Exception:
                bank_fd_thresh = 0.38
            if bank_fd_thresh <= 0:
                bank_fd_thresh = 0.38
            else:
                bank_fd_thresh = min(0.38, bank_fd_thresh)
            ref_reid_feat = None
            if reid is not None:
                ref_persons = det.detect(ref_img_primary, conf=0.1)
                if ref_persons:
                    ref_persons.sort(key=lambda d: (d['xyxy'][2]-d['xyxy'][0])*(d['xyxy'][3]-d['xyxy'][1]), reverse=True)
                    rx1, ry1, rx2, ry2 = [int(v) for v in ref_persons[0]['xyxy']]
                    ref_crop = ref_img_primary[ry1:ry2, rx1:rx2]
                    ref_reid_feat = reid.extract([ref_crop])[0]
                else:
                    ref_reid_feat = reid.extract([ref_img_primary])[0]
            self._status(
                f"Ref ReID: {'ok' if ref_reid_feat is not None else 'skipped'}",
                key="ref_reid",
                interval=60.0,
            )

            # Effective matching mode after refs known
            base_match_mode = getattr(cfg, "match_mode", "face_only")
            if ref_face_feat is None and base_match_mode in ("both", "face_only"):
                base_match_mode = "reid_only"
            if ref_reid_feat is None and base_match_mode in ("both", "reid_only"):
                base_match_mode = "face_only"
            face_only_pipeline = (base_match_mode == "face_only")

            # Video
            hdr_active = False
            # --- Video / HDR setup ---
            # Apply HDR tonemap env overrides from cfg (GUI always supplies defaults)
            try:
                os.environ["PC_SDR_NITS"] = str(float(getattr(cfg, "sdr_nits", SessionConfig.sdr_nits)))
                os.environ["PC_TM_DESAT"] = str(float(getattr(cfg, "tm_desat", 0.25)))
                os.environ["PC_TM_PARAM"] = str(float(getattr(cfg, "tm_param", 0.40)))
                for key in ("PC_FORCE_ZSCALE", "PC_FORCE_SCALE", "PC_FORCE_TONEMAP"):
                    os.environ.pop(key, None)
                pref = str(getattr(cfg, "hdr_tonemap_pref", "auto") or "auto").lower()
                if pref in {"libplacebo", "zscale", "scale"}:
                    os.environ["PC_FORCE_TONEMAP"] = pref
            except Exception:
                pass
            try:
                s = QtCore.QSettings(APP_ORG, APP_NAME)
                ffdir = s.value(_SETTINGS_KEY_FFMPEG_DIR, "", type=str) or ""
                if ffdir:
                    set_ffmpeg_env(ffdir)
            except Exception:
                pass
            try:
                hdr_reason = hdr_detect_reason(cfg.video)
            except Exception:
                hdr_reason = "unknown"

            # Ensure any previous preview reader is torn down before opening a new one.
            self._hdr_preview_close()

            # Processing/cropping path: always HDR→SDR tone-mapped BGR (or OpenCV fallback).
            cap = None

            def _open_opencv_reader(video_path: str):
                def _probe_and_rewind(candidate) -> bool:
                    if hasattr(candidate, "isOpened") and not candidate.isOpened():
                        return False
                    pos0 = int(candidate.get(cv2.CAP_PROP_POS_FRAMES) or 0)
                    ok = bool(candidate.grab())
                    if not ok:
                        return False
                    for _ in range(2):
                        try:
                            candidate.set(cv2.CAP_PROP_POS_FRAMES, pos0)
                            cur = int(candidate.get(cv2.CAP_PROP_POS_FRAMES) or -1)
                            if cur == pos0:
                                return True
                        except Exception:
                            break
                    return False

                hwmode = str(getattr(cfg, "ff_hwaccel", "off") or "off").strip().lower()
                prev_ffmpeg_opts = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")
                try:
                    if hwmode != "off":
                        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"hwaccel;{hwmode}"
                    else:
                        os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
                    reader = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
                finally:
                    if prev_ffmpeg_opts is None:
                        os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
                    else:
                        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = prev_ffmpeg_opts
                if hasattr(reader, "isOpened") and not reader.isOpened():
                    reader.release()
                    fallback = cv2.VideoCapture(video_path)
                    if _probe_and_rewind(fallback):
                        return fallback
                    fallback.release()
                    raise RuntimeError(f"OpenCV reader produced no frames: {video_path}")
                if _probe_and_rewind(reader):
                    return reader
                reader.release()
                fallback = cv2.VideoCapture(video_path)
                if _probe_and_rewind(fallback):
                    return fallback
                fallback.release()
                raise RuntimeError(f"OpenCV reader produced no frames: {video_path}")

            try:
                cap = open_video_with_tonemap(cfg.video)
                if cap is not None:
                    # Log which HDR reader implementation and mode we actually ended up using.
                    reader_type = type(cap).__name__
                    mode = getattr(cap, "mode", None)
                    _mode = getattr(cap, "_mode", None)
                    logger.info(
                        "Video open: HDR tone-map reader selected (reader=%s mode=%s _mode=%s reason=%s)",
                        reader_type,
                        mode,
                        _mode,
                        hdr_reason,
                    )
                    self._status(
                        f"HDR: active ({hdr_reason}) [{reader_type}/{mode or _mode or 'n/a'}]",
                        key="hdr_state",
                        interval=60.0,
                    )
                    hdr_active = True
                else:
                    logger.info(
                        "Video open: OpenCV/FFmpeg reader selected (reason=%s)", hdr_reason
                    )
                    self._status(
                        f"HDR: inactive ({hdr_reason}) [OpenCV/FFmpeg]",
                        key="hdr_state",
                        interval=60.0,
                    )
                    hdr_active = False
                    cap = _open_opencv_reader(cfg.video)
            except Exception:
                cap = cv2.VideoCapture(cfg.video)

            # Do not early-exit on “not opened” if this is a custom pipe or duck-types like one.
            def _pipe_like(obj) -> bool:
                return hasattr(obj, "try_fallback_chain") or (hasattr(obj, "grab") and hasattr(obj, "retrieve"))

            if hasattr(cap, "isOpened") and not cap.isOpened() and not _pipe_like(cap):
                raise RuntimeError(f"Cannot open video: {cfg.video}")
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            except Exception:
                pass

            # Default: HDR passthrough ON when available unless explicitly disabled in cfg.
            cfg_hdr_passthrough = bool(getattr(cfg, "hdr_passthrough", True))
            want_hdr_passthrough = (
                hdr_active
                and cfg_hdr_passthrough
                and bool(self.ui_hdr_passthrough_enabled)
                and _hdr_passthrough_available()
            )
            self._status(
                f"HDR passthrough gate: hdr={hdr_active} cfg={cfg_hdr_passthrough} "
                f"ui={self.ui_hdr_passthrough_enabled} dll={_hdr_passthrough_available()}",
                key="hdr_gate",
                interval=10.0,
            )
            if want_hdr_passthrough:
                if self._hdr_preview_reader is None:
                    try:
                        hwmode = (getattr(cfg, "ff_hwaccel", "off") or "off").strip().lower()
                        if hwmode != "off":
                            os.environ["PC_HWACCEL"] = hwmode
                            os.environ["PC_HWACCEL_OUT_FMT"] = "cuda" if hwmode == "cuda" else hwmode
                        else:
                            os.environ.pop("PC_HWACCEL", None)
                            os.environ.pop("PC_HWACCEL_OUT_FMT", None)
                            os.environ.pop("PCHWACCELOUTFMT", None)

                        logger.info(
                            "HDR passthrough preview requested; opening P010 reader for preview"
                        )
                        preview_reader = open_hdr_passthrough_reader(cfg.video)
                        if preview_reader is not None and getattr(
                            preview_reader, "isOpened", lambda: True
                        )():
                            self._hdr_preview_reader = preview_reader
                        else:
                            logger.warning(
                                "HDR passthrough preview requested but P010 reader failed"
                            )
                    except Exception:
                        logger.exception("Failed to open HDR passthrough preview reader")

                if self._hdr_preview_reader is not None:
                    # Drive preview from a dedicated P010 reader.
                    self._hdr_passthrough_active = True
                    self._hdr_preview_latest = None
                    try:
                        pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
                        self._hdr_preview_seek(pos)
                    except Exception:
                        pass
                    self._status(
                        "HDR passthrough preview: enabled (P010 preview reader)",
                        key="hdr_passthrough",
                        interval=30.0,
                    )
                else:
                    self._hdr_passthrough_active = False
            else:
                self._hdr_passthrough_active = False

            # --- Sanity probe: ensure the selected reader actually delivers frames.
            # If HDR tone-map reader stalls, fall back to OpenCV/FFmpeg so the preview and pre-scan advance.
            def _probe_reader(c, fps_hint: float) -> bool:
                try:
                    tries = max(8, min(32, int(round(fps_hint or 24.0))))
                except Exception:
                    tries = 12
                ok_any = False
                warmup_ms = int(os.getenv("PC_HDR_PIPE_WARMUP_MS", "3000"))
                warmup_deadline = time.time() + (warmup_ms / 1000.0)
                # some wrappers may not expose get/set; fall back to 0
                get = getattr(c, "get", None)
                set_ = getattr(c, "set", None)
                pos0 = int(get(cv2.CAP_PROP_POS_FRAMES) or 0) if callable(get) else 0
                for _ in range(tries):
                    ok, fr = c.read()
                    if not ok:
                        # Warm-up by grab/retrieve if available.
                        grab = getattr(c, "grab", None)
                        retrieve = getattr(c, "retrieve", None)
                        if callable(grab) and callable(retrieve):
                            while time.time() < warmup_deadline and not ok:
                                try:
                                    if grab():
                                        ok, fr = retrieve()
                                        if ok:
                                            break
                                    self._hdr_preview_skip(1)
                                except Exception:
                                    break
                                time.sleep(0.05)

                    if ok and fr is not None:
                        ok_any = True
                        try:
                            mode = getattr(c, "mode", "")
                            if mode == "p010_passthrough":
                                # HDR passthrough: fr is a P010 payload; drive Vulkan widget via passthrough reader.
                                self._pump_hdr_preview(reader=c)
                            else:
                                # Tonemapped path: fr is BGR; show first SDR frame to prove the pipe is alive.
                                self._emit_preview_bgr(fr)
                        except Exception:
                            pass
                        break
                # rewind for normal processing
                if callable(set_):
                    try:
                        set_(cv2.CAP_PROP_POS_FRAMES, pos0)
                    except Exception:
                        pass
                self._hdr_preview_seek(pos0)
                return ok_any

            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

            defer_reader_probe = (
                hdr_active
                and bool(getattr(cfg, "prescan_enable", True))
                and int(getattr(cfg, "prescan_decode_max_w", 0) or 0) > 0
                and bool(getattr(cap, "_is_hdr_pipe", False))
            )
            if defer_reader_probe:
                ok = True
                self._status(
                    "HDR reader first-frame probe deferred until after pre-scan",
                    key="hdr_probe_deferred",
                    interval=60.0,
                )
            else:
                ok = _probe_reader(cap, float(fps or 24.0))
            # No CPU/zscale fallbacks when strict LP is active.
            strict = bool(getattr(cap, "_strict_lp", False))
            if (not ok) and hasattr(cap, "try_fallback_chain") and not strict:
                try:
                    # If HDR pipe failed on libplacebo (e.g., Vulkan), ask it to fall back to zscale+tonemap.
                    if cap.try_fallback_chain():
                        # Log which chain we fell back to, and (if available) a small
                        # slice of the stderr tail from the previous ffmpeg run.
                        mode = getattr(cap, "mode", None)
                        _mode = getattr(cap, "_mode", None)
                        tail = ""
                        try:
                            tail_list = getattr(cap, "_stderr_tail", []) or []
                            tail = " | ".join(tail_list[-5:])
                        except Exception:
                            tail = ""
                        logger.warning(
                            "HDR reader fallback: now mode=%s _mode=%s (stderr_tail=%s)",
                            mode,
                            _mode,
                            tail,
                        )
                        self._status(
                            "HDR: libplacebo emitted no frames; retrying with alternate filter chain…"
                        )
                        ok = _probe_reader(cap, float(fps or 24.0))
                except Exception:
                    pass
            if (not ok) and strict:
                raise RuntimeError("libplacebo(Vulkan) produced no frames in strict mode")

            if not ok:
                msg = (
                    "HDR reader delivered no frames; falling back to OpenCV reader"
                    if hdr_active
                    else "Selected reader delivered no frames; reopening with OpenCV/FFmpeg"
                )
                self._status(msg, key="hdr_probe_fail", interval=60.0)
                try:
                    cap.release()
                except Exception:
                    pass
                self._hdr_preview_close()
                cap = _open_opencv_reader(cfg.video)
                if not cap.isOpened():
                    raise RuntimeError(f"Cannot open video after HDR fallback: {cfg.video}")
                try:
                    logger.warning("HDR reader produced no frames; reopened with OpenCV/FFmpeg")
                    self._status("HDR: inactive (fallback)", key="hdr_state", interval=60.0)
                    hdr_active = False
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                    fps = float(cap.get(cv2.CAP_PROP_FPS) or fps or 30.0)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or total_frames or 0)
                except Exception:
                    pass

            # If ffprobe is missing, OpenCV may report total=0 and/or fps=0 → fix BEFORE keyframes/setup.
            try:
                if (not total_frames or total_frames <= 0) or (not fps or fps <= 1e-3 or fps != fps):
                    try:
                        from .video_io import probe_fps_total as _pc_probe_fps_total  # type: ignore
                    except Exception:
                        from video_io import probe_fps_total as _pc_probe_fps_total  # type: ignore
                    fps_fixed, total_fixed = _pc_probe_fps_total(cfg.video, fps_hint=fps)
                    if (total_fixed and total_fixed > 0) or (fps_fixed and fps_fixed > 1e-3):
                        if fps_fixed and fps_fixed > 1e-3:
                            fps = float(fps_fixed)
                        if total_fixed and total_fixed > 0:
                            total_frames = int(total_fixed)
                        self.status.emit(
                            f"Recovered fps/total via PyAV/OpenCV → fps={fps:.3f} total={total_frames}"
                        )
            except Exception:
                pass

            # Persist corrected values (some UI/paths use these attrs later)
            self._fps = fps
            self._total_frames = total_frames
            keep_spans = []
            span_i = 0
            # Build keyframe index in worker to enable fast seeking
            try:
                self._keyframes = self._read_keyframes_worker(cfg.video, fps, total_frames)
            except Exception:
                self._keyframes = []
            try:
                # publish to UI before setup so UI doesn't fall back
                self.keyframes.emit(list(self._keyframes))
            except Exception:
                pass
            # Now that fps/total are final, set up the UI/processor with correct values
            self.setup.emit(total_frames, fps)
            try:
                self.status.emit(
                    f"KFs={len(self._keyframes)} fps={float(fps):.3f} total={total_frames}"
                )
            except Exception:
                pass
            try:
                self.progress.emit(0)
            except Exception:
                pass

            def _run_deferred_probe_after_prescan(cap_obj, fps_val, total_val, hdr_is_active):
                self._status(
                    "HDR reader first-frame probe running after pre-scan",
                    key="hdr_probe_deferred",
                    interval=60.0,
                )
                ok = _probe_reader(cap_obj, float(fps_val or 24.0))
                strict = bool(getattr(cap_obj, "_strict_lp", False))
                if (not ok) and hasattr(cap_obj, "try_fallback_chain") and not strict:
                    try:
                        if cap_obj.try_fallback_chain():
                            logger.warning("HDR reader fallback after pre-scan")
                            ok = _probe_reader(cap_obj, float(fps_val or 24.0))
                    except Exception:
                        pass
                if (not ok) and strict:
                    raise RuntimeError("libplacebo(Vulkan) produced no frames in strict mode")
                if not ok:
                    self._status(
                        "HDR reader delivered no frames after pre-scan; falling back to OpenCV reader",
                        key="hdr_probe_fail",
                        interval=60.0,
                    )
                    try:
                        cap_obj.release()
                    except Exception:
                        pass
                    self._hdr_preview_close()
                    cap_obj = _open_opencv_reader(cfg.video)
                    if not cap_obj.isOpened():
                        raise RuntimeError(
                            f"Cannot open video after deferred HDR fallback: {cfg.video}"
                        )
                    hdr_is_active = False
                    cap_obj.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                    fps_val = float(cap_obj.get(cv2.CAP_PROP_FPS) or fps_val or 30.0)
                    total_val = int(cap_obj.get(cv2.CAP_PROP_FRAME_COUNT) or total_val or 0)
                    self._fps = fps_val
                    self._total_frames = total_val
                    try:
                        self._keyframes = self._read_keyframes_worker(cfg.video, fps_val, total_val)
                    except Exception:
                        self._keyframes = []
                    try:
                        self.keyframes.emit(list(self._keyframes))
                    except Exception:
                        pass
                    self.setup.emit(total_val, fps_val)
                    try:
                        self.status.emit(
                            f"KFs={len(self._keyframes)} fps={float(fps_val):.3f} total={total_val}"
                        )
                    except Exception:
                        pass
                return cap_obj, hdr_is_active, fps_val, total_val

            if bool(getattr(cfg, "prescan_enable", True)) and total_frames > 0:
                prescan_loaded = False
                keep_spans = []
                cached_ref_face_feat = None
                if str(getattr(cfg, "prescan_cache_mode", "auto") or "auto").lower() != "refresh":
                    prescan_loaded, keep_spans, cached_ref_face_feat, _cache_meta = self._load_prescan_cache(
                        cfg, float(fps), int(total_frames)
                    )
                    if prescan_loaded:
                        ref_face_feat = cached_ref_face_feat
                if not prescan_loaded:
                    mode = str(getattr(cfg, "prescan_cache_mode", "auto") or "auto").lower()
                    suffix = " (refresh cache)" if mode == "refresh" else ""
                    self._status(f"Pre-scan.{suffix}", key="phase", interval=2.0)
                    self._prescan_cache_dirty = False
                    keep_spans, ref_face_feat = self._prescan(
                        cap, int(round(fps)), total_frames, face, ref_face_feat, cfg
                    )
                    if not getattr(self, "_prescan_cache_dirty", False):
                        self._save_prescan_cache(cfg, float(fps), int(total_frames), keep_spans, ref_face_feat)
                    else:
                        self._status(
                            "Pre-scan cache not saved: run was interactively modified",
                            key="prescan_cache",
                            interval=0.0,
                        )
                if defer_reader_probe:
                    cap, hdr_active, fps, total_frames = _run_deferred_probe_after_prescan(
                        cap, fps, total_frames, hdr_active
                    )
                if keep_spans:
                    s0 = keep_spans[0][0]
                    #
                    # Purge only *stale* navigation noise created by pre-scan progress updates.
                    # Do this *before* the jump so user-issued controls during the jump survive.
                    #
                    _buf: list[tuple[str, object]] = []
                    # Drain only what was present *before* we started purging.
                    # This avoids consuming fresh user input that arrives mid-purge.
                    _to_drain = getattr(self._cmd_q, "qsize", lambda: 0)()
                    for _ in range(max(0, int(_to_drain))):
                        try:
                            item = self._cmd_q.get_nowait()
                        except queue.Empty:
                            break
                        # Normalize to (cmd, arg)
                        if isinstance(item, tuple):
                            if len(item) == 0:
                                continue
                            cmd = item[0]
                            arg = item[1] if len(item) > 1 else None
                        else:
                            cmd, arg = item, None
                        # Drop only nav spam likely enqueued by UI progress updates.
                        if cmd in ("seek", "step", "scrub", "seek_abs", "seek_rel"):
                            continue
                        _buf.append((cmd, arg))
                    # Requeue non-navigation commands in original order.
                    for cmd, arg in _buf:
                        try:
                            self._cmd_q.put_nowait((cmd, arg))
                        except Exception:
                            pass

                    frame_idx = self._seek_to(
                        cap,
                        0,
                        s0,
                        fast=bool(getattr(cfg, "seek_fast", True)),
                        max_grabs=int(getattr(cfg, "seek_max_grabs", 12)),
                        peek_preview=False,  # segment jump: no peeks
                        allow_partial=False,
                        hdr_reader=self._hdr_preview_reader,
                    )
                    # Neutralize any stray cooldown/coalesced state from the jump *before* progress emit.
                    if hasattr(self, "_seek_cooldown_frames"):
                        self._seek_cooldown_frames = 0
                    if hasattr(self, "_pending_seek"):
                        self._pending_seek = None
                    # Sync the UI to the true starting frame of the main pass.
                    try:
                        self.progress.emit(frame_idx)
                    except Exception:
                        pass
                    # Paint something immediately so the UI isn't black after the jump.
                    try:
                        ok = cap.grab()
                        if ok:
                            ok, frame = cap.retrieve()
                            if ok and frame is not None:
                                self._pump_hdr_preview()
                                self._emit_preview_bgr(frame)
                        # Do not advance past s0 for processing—restore read head.
                        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                        self._hdr_preview_seek(int(frame_idx))
                    except Exception:
                        pass
                    self._status(f"Pre-scan segments: {len(keep_spans)}", key="prescan", interval=30.0)
                else:
                    self._status("Pre-scan found no matches; full scan", key="prescan", interval=30.0)
            elif defer_reader_probe:
                cap, hdr_active, fps, total_frames = _run_deferred_probe_after_prescan(
                    cap, fps, total_frames, hdr_active
                )
            else:
                keep_spans = []
                span_i = 0
            ratios = [r.strip() for r in str(cfg.ratio).split(',') if r.strip()]
            if not ratios:
                ratios = ["2:3"]

            # CSV
            csv_path = os.path.join(cfg.out_dir, "index.csv")
            csv_f = open(csv_path, "w", newline="")
            writer = csv.writer(csv_f)
            writer.writerow(["frame","time_secs","score","face_dist","reid_dist","x1","y1","x2","y2","crop_path","sharpness","ratio"])
            # If async saver is enabled, close this header handle now to prevent Windows write locks.
            header_only_csv_handle = csv_f
            flush_every = max(1, int(getattr(cfg, "csv_flush_every", 25)))
            sync_wrote = 0

            # Async saver
            save_q: Optional[queue.Queue] = None
            archive_q: Optional[queue.Queue] = None
            hit_q: Optional[queue.Queue] = None
            jpg_q = int(getattr(cfg, "jpg_quality", 85))

            def _atomic_jpeg_write(img: np.ndarray, out_path: str, q: int) -> tuple[bool, str]:
                try:
                    tmp = out_path + ".tmp"
                    params: list[int] = []
                    try:
                        qi = int(q)
                        if qi > 0:
                            params = [int(cv2.IMWRITE_JPEG_QUALITY), qi]
                    except Exception:
                        params = []
                    ok, buf = cv2.imencode(".jpg", img, params)
                    if not ok or buf is None:
                        return False, "imencode_failed"
                    with open(tmp, "wb") as fh:
                        fh.write(buf.tobytes())
                        fh.flush()
                        try:
                            os.fsync(fh.fileno())
                        except Exception:
                            pass
                    os.replace(tmp, out_path)
                    if not os.path.exists(out_path) or os.path.getsize(out_path) < 1024:
                        return False, "file_too_small"
                    return True, ""
                except Exception as e:
                    try:
                        tmp_path = out_path + ".tmp"
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                    except Exception:
                        pass
                    return False, f"{type(e).__name__}: {e}"

            if bool(getattr(cfg, "async_save", True)):
                try:
                    header_only_csv_handle.close()
                except Exception:
                    pass
                writer = None
                save_q = queue.Queue(maxsize=512)
                archive_q = queue.Queue(maxsize=256)
                hit_q = queue.Queue(maxsize=512)

                def _saver():
                    f = open(csv_path, "a", newline="")
                    w = csv.writer(f)
                    wrote = 0
                    while True:
                        item = save_q.get()
                        if item is None:
                            save_q.task_done()
                            break

                        ack_q = None
                        ok = False
                        why = "save_not_attempted"
                        img_path = ""
                        row = None
                        kind = ""
                        if isinstance(item, dict):
                            ack_q = item.get("ack_q")
                            kind = str(item.get("type") or "")
                            if kind == "hdr_sdr":
                                img_path = str(item.get("path") or "")
                                row = item.get("row")
                                frame_pts_sec = item.get("frame_pts_sec")
                                if frame_pts_sec is not None:
                                    try:
                                        frame_pts_sec = float(frame_pts_sec)
                                    except Exception:
                                        frame_pts_sec = None
                                ok, why = self._save_hdr_sdr_screencap(
                                    int(item.get("frame_idx", 0)),
                                    frame_pts_sec,
                                    tuple(item.get("crop_xyxy") or (0, 0, 1, 1)),
                                    img_path,
                                )
                            elif kind == "jpeg":
                                img_path = str(item.get("path") or "")
                                row = item.get("row")
                                img = item.get("img")
                                if isinstance(img, np.ndarray):
                                    ok, why = _atomic_jpeg_write(img, img_path, jpg_q)
                                else:
                                    ok, why = False, "invalid_jpeg_payload"
                            else:
                                ok, why = False, f"unknown_save_item_type:{kind or 'none'}"
                        else:
                            img_path, img, row = item
                            ok, why = _atomic_jpeg_write(img, img_path, jpg_q)

                        if ok:
                            if kind == "hdr_sdr" and isinstance(row, list) and len(row) > 10:
                                saved_sharp = self._calc_saved_file_sharpness(img_path)
                                if saved_sharp is not None:
                                    row[10] = float(saved_sharp)
                        if ack_q is not None:
                            try:
                                ack_q.put_nowait((bool(ok), str(why or "")))
                            except Exception:
                                pass
                        if ok:
                            # hand off to worker thread for emitting
                            try:
                                hit_q.put_nowait(img_path)
                            except queue.Full:
                                pass
                            # CSV is best-effort and must not block the UI update
                            try:
                                w.writerow(row)
                                wrote += 1
                                if wrote % flush_every == 0:
                                    f.flush()
                            except Exception:
                                logger.exception("CSV write failed for %s", img_path)

                        else:
                            logger.error("Failed to save crop %s (%s)", img_path, why)
                            self._status(
                                f"Save failed ({why}): {img_path}",
                                key="save_err_async",
                                interval=0.5,
                            )

                        save_q.task_done()

                    f.close()

                def _archive_saver():
                    while True:
                        item = archive_q.get()
                        if item is None:
                            archive_q.task_done()
                            break
                        try:
                            frame_pts_sec = item.get("frame_pts_sec")
                            if frame_pts_sec is not None:
                                try:
                                    frame_pts_sec = float(frame_pts_sec)
                                except (TypeError, ValueError):
                                    frame_pts_sec = None
                            self._save_hdr_crop_p010(
                                int(item.get("frame_idx", 0)),
                                frame_pts_sec,
                                tuple(item.get("crop_xyxy") or (0, 0, 2, 2)),
                                str(item.get("path") or ""),
                            )
                        finally:
                            archive_q.task_done()

                saver_thread = threading.Thread(target=_saver, name="pc.saver", daemon=True)
                saver_thread.start()
                archive_thread = threading.Thread(target=_archive_saver, name="pc.archive_saver", daemon=True)
                archive_thread.start()

            hit_count = 0
            lock_hits = 0
            locked_face = None
            locked_reid = None
            prev_box = None
            source_size_cached: Optional[tuple[int, int]] = None
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
            if keep_spans:
                span_i = self._span_index_for(frame_idx, keep_spans)
            last_preview_emit = -999999
            self._seek_cooldown_frames = 0

            self._status("Processing...", key="phase", interval=5.0)
            next_seek: Optional[int] = None
            next_steps: int = 0
            next_speed: Optional[float] = None
            next_cfg: dict = {}
            while True:
                if self._abort:
                    self._status("Aborting...", key="phase")
                    break
                force_process = False
                # --------- drain and coalesce controls ---------
                last_seek: Optional[int] = next_seek
                step_accum: int = next_steps
                pending_speed: Optional[float] = next_speed
                pending_cfg: dict = dict(next_cfg)
                next_seek = None
                next_steps = 0
                next_speed = None
                next_cfg = {}

                def _apply_live_updates() -> None:
                    nonlocal pending_speed, pending_cfg
                    if pending_speed is not None:
                        self._speed = max(0.1, min(4.0, float(pending_speed)))
                        pending_speed = None
                    if pending_cfg:
                        LIVE = {
                            "frame_stride",
                            "min_det_conf",
                            "face_thresh",
                            "reid_thresh",
                            "combine",
                            "match_mode",
                            "only_best",
                            "min_sharpness",
                            "min_gap_sec",
                            "min_box_pixels",
                            "auto_crop_borders",
                            "border_threshold",
                            "score_margin",
                            "iou_gate",
                            "preview_every",
                            "preview_max_dim",
                            "preview_fps_cap",
                            "require_face_if_visible",
                            "prefer_face_when_available",
                            "suppress_negatives",
                            "neg_tolerance",
                            "max_negatives",
                            "log_interval_sec",
                            "lock_after_hits",
                            "lock_face_thresh",
                            "lock_reid_thresh",
                            "lock_momentum",
                            "allow_faceless_when_locked",
                            "learn_bank_runtime",
                            "drop_reid_if_any_face_match",
                            "faceless_reid_thresh",
                            "faceless_iou_min",
                            "faceless_persist_frames",
                            "faceless_min_area_frac",
                            "faceless_max_area_frac",
                            "faceless_center_max_frac",
                            "faceless_min_motion_frac",
                            "seek_fast",
                            "seek_max_grabs",
                            "seek_preview_peek_every",
                            "crop_face_side_margin_frac",
                            "crop_top_headroom_max_frac",
                            "crop_bottom_min_face_heights",
                            "crop_penalty_weight",
                            "crop_head_side_pad_frac",
                            "crop_head_top_pad_frac",
                            "crop_head_bottom_pad_frac",
                            "wide_face_aspect_penalty_weight",
                            "wide_face_min_frame_frac",
                            "wide_face_aspect_limit",
                            "face_anchor_down_frac",
                            "face_max_frac_in_crop",
                            "face_min_frac_in_crop",
                            "crop_min_height_frac",
                            "face_visible_uses_quality",
                            "face_quality_min",
                            "face_det_conf",
                            "face_det_pad",
                            "face_fullframe_imgsz",
                            "rot_adaptive",
                            "rot_every_n",
                            "rot_after_hit_frames",
                            "fast_no_face_imgsz",
                            # new per-frame crop scorer knobs
                            "lambda_facefrac",
                            "crop_center_weight",
                            "area_gamma",
                            "area_face_scale_weight",
                            "square_pull_face_min",
                            "square_pull_weight",
                            "face_target_close",
                            "face_target_upper",
                            "face_target_cowboy",
                            "face_target_body",
                            "face_target_tolerance",
                            "face_target_close_min_frac",
                            "w_close",
                            "w_upper",
                            "w_cowboy",
                            "w_body",
                            "overlay_scores",
                            "hdr_screencap_fullres",
                            "hdr_archive_crops",
                            "hdr_crop_format",
                            "hdr_sdr_quality",
                            "hdr_sdr_tonemap",
                            "hdr_sdr_gamut_mapping",
                            "hdr_sdr_contrast_recovery",
                            "hdr_sdr_peak_detect",
                            "hdr_sdr_allow_inaccurate_fallback",
                            "compose_crop_enable",
                            "compose_detect_person_for_face",
                            "compose_close_face_h_frac",
                            "compose_upper_face_h_frac",
                            "compose_body_face_h_frac",
                            "compose_landscape_face_penalty",
                            "compose_body_every_n",
                            "compose_person_detect_cadence",
                        }
                        rot_keys = {
                            "rot_adaptive",
                            "rot_every_n",
                            "rot_after_hit_frames",
                            "fast_no_face_imgsz",
                        }
                        rot_changed = False
                        for k, v in pending_cfg.items():
                            if k in LIVE and hasattr(self.cfg, k):
                                setattr(self.cfg, k, v)
                                if k in rot_keys:
                                    rot_changed = True
                        if rot_changed:
                            try:
                                face.configure_rotation_strategy(
                                    adaptive=getattr(self.cfg, "rot_adaptive", True),
                                    every_n=getattr(self.cfg, "rot_every_n", 12),
                                    after_hit_frames=getattr(self.cfg, "rot_after_hit_frames", 8),
                                    fast_no_face_imgsz=getattr(self.cfg, "fast_no_face_imgsz", 512),
                                )
                            except Exception:
                                pass
                        pending_cfg = {}
                try:
                    while True:
                        cmd, arg = self._cmd_q.get_nowait()
                        if cmd == "seek":
                            try:
                                last_seek = int(arg)
                            except Exception:
                                pass
                        elif cmd == "step":
                            try:
                                step_accum += int(arg) if arg is not None else 1
                            except Exception:
                                step_accum += 1
                        elif cmd == "pause":
                            self._paused = True
                        elif cmd == "play":
                            self._paused = False
                        elif cmd == "speed":
                            try:
                                pending_speed = float(arg)
                            except Exception:
                                pass
                        elif cmd == "cfg":
                            try:
                                pending_cfg.update(dict(arg or {}))
                            except Exception:
                                pass
                        else:
                            # ignore unknowns
                            pass
                except queue.Empty:
                    pass
                _apply_live_updates()

                # apply coalesced seek/step
                if last_seek is not None or step_accum:
                    if last_seek is not None:
                        tgt = int(last_seek)
                    else:
                        tgt = frame_idx + int(step_accum)
                    if self._total_frames is not None:
                        tgt = max(0, min(self._total_frames - 1, tgt))
                    frame_idx = self._seek_to(
                        cap,
                        frame_idx,
                        tgt,
                        fast=bool(getattr(self.cfg, "seek_fast", True)),
                        max_grabs=int(getattr(self.cfg, "seek_max_grabs", 12)),
                        peek_preview=True,  # UI scrub: allow peeks
                        allow_partial=True,
                        hdr_reader=self._hdr_preview_reader,
                    )
                    self.progress.emit(frame_idx)
                    # reset temporal gates so back/forward scrubs work
                    lock_hits = 0
                    locked_face = None
                    locked_reid = None
                    prev_box = None
                    self._clear_lock()
                    self._prev_gray = None
                    # allow immediate capture after seek by backdating last_hit
                    now_t = _frame_time(frame_idx)
                    self._last_hit_t = now_t - float(self.cfg.min_gap_sec)
                    self._seek_cooldown_frames = _cooldown_frames()
                    self._status(
                        f"Seek@{frame_idx} cooldown={self._seek_cooldown_frames}",
                        key="seek_reset",
                        interval=0.5,
                    )
                    if keep_spans:
                        span_i = self._span_index_for(frame_idx, keep_spans)
                    force_process = True
                if hit_q is not None:
                    try:
                        while True:
                            _p = hit_q.get_nowait()
                            self._emit_hit(_p)
                            hit_q.task_done()
                    except queue.Empty:
                        pass

                # segment gate (auto-skip regions with no target)
                if keep_spans:
                    if span_i >= len(keep_spans):
                        break
                    s, e = keep_spans[span_i]
                    if frame_idx < s:
                        frame_idx = self._seek_to(
                            cap,
                            frame_idx,
                            s,
                            fast=bool(getattr(self.cfg, "seek_fast", True)),
                            max_grabs=int(getattr(self.cfg, "seek_max_grabs", 12)),
                            peek_preview=False,  # segment jump: no peeks
                            allow_partial=False,
                            hdr_reader=self._hdr_preview_reader,
                        )
                        self._seek_cooldown_frames = int(max(2, (self._fps or 30) * 0.25))
                        continue
                    if frame_idx > e:
                        span_i += 1
                        if span_i >= len(keep_spans):
                            break
                        s2, _ = keep_spans[span_i]
                        frame_idx = self._seek_to(
                            cap,
                            frame_idx,
                            s2,
                            fast=bool(getattr(self.cfg, "seek_fast", True)),
                            max_grabs=int(getattr(self.cfg, "seek_max_grabs", 12)),
                            peek_preview=False,  # segment jump: no peeks
                            allow_partial=False,
                            hdr_reader=self._hdr_preview_reader,
                        )
                        self._seek_cooldown_frames = int(max(2, (self._fps or 30) * 0.25))
                        continue

                if self._paused:
                    self._paused = True
                if self._paused and not force_process:
                    time.sleep(0.01)
                    continue

                # --- preempt before any IO/inference ---
                try:
                    while True:
                        cmd, arg = self._cmd_q.get_nowait()
                        if cmd == "seek":
                            try:
                                next_seek = int(arg)
                            except Exception:
                                pass
                        elif cmd == "step":
                            try:
                                next_steps += int(arg) if arg is not None else 1
                            except Exception:
                                next_steps += 1
                        elif cmd == "speed":
                            try:
                                next_speed = float(arg)
                            except Exception:
                                pass
                        elif cmd == "cfg":
                            try:
                                next_cfg.update(dict(arg or {}))
                            except Exception:
                                pass
                        elif cmd == "pause":
                            self._paused = True
                        elif cmd == "play":
                            self._paused = False
                        else:
                            pass
                except queue.Empty:
                    pass

                if next_speed is not None:
                    pending_speed = next_speed
                    next_speed = None
                if next_cfg:
                    if pending_cfg:
                        pending_cfg.update(next_cfg)
                    else:
                        pending_cfg = dict(next_cfg)
                    next_cfg = {}

                _apply_live_updates()

                if next_seek is not None or next_steps:
                    continue

                ret = cap.grab()
                if not ret:
                    break
                current_idx = frame_idx
                idx = current_idx
                if not force_process and current_idx % max(1, int(cfg.frame_stride)) != 0:
                    frame_idx = current_idx + 1
                    self.progress.emit(current_idx)
                    continue
                ret, frame = cap.retrieve()
                if not ret:
                    break
                self._pump_hdr_preview()

                H, W = frame.shape[:2]
                frame_pts_sec = self._capture_frame_pts_sec(cap, current_idx, fps)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


                # Optional black border crop
                frame_for_det = frame
                off_x, off_y = 0, 0
                if cfg.auto_crop_borders:
                    frame_for_det, (off_x, off_y) = self._autocrop_borders(frame, cfg.border_threshold)
                    H2, W2 = frame_for_det.shape[:2]
                else:
                    H2, W2 = H, W
                # Keep the original border-aware content ROI for save/repair bounds
                # even if person detection later falls back to full-frame inference.
                base_det_off_x, base_det_off_y = off_x, off_y
                base_det_w, base_det_h = W2, H2

                candidates = []
                faces_local = {}
                reid_feats = []
                face_debug_boxes = []
                diag_persons = 0
                faces_detected = 0
                faces_passing_quality = 0
                face_dists_all = []
                face_dists_quality = []
                min_fd_all = None
                best_face_dist = None
                any_face_match = False
                any_face_visible = False
                any_face_detected = False
                short_circuit_candidate = None
                persons = []
                fullframe_imgsz = getattr(cfg, "face_fullframe_imgsz", None)
                if fullframe_imgsz is not None:
                    try:
                        fullframe_imgsz = int(fullframe_imgsz)
                    except (TypeError, ValueError):
                        fullframe_imgsz = None

                face_short_ok = (
                    ref_face_feat is not None
                    and face is not None
                    and getattr(cfg, "face_fullframe_when_missed", True)
                    and (face_only_pipeline or self._lock_last_bbox is not None)
                )
                # Run full-frame face detect at a coarse cadence only.
                gfaces = []
                if face_short_ok:
                    try:
                        ff_cad = int(getattr(cfg, "face_fullframe_cadence", 12))
                    except Exception:
                        ff_cad = 12
                    if ff_cad <= 0 or (idx % max(1, ff_cad) == 0):
                        try:
                            gfaces = face.extract(frame_for_det, imgsz=fullframe_imgsz)
                        except Exception:
                            gfaces = []
                if gfaces:
                        quality_min = float(cfg.face_quality_min)
                        use_quality_vis = bool(cfg.face_visible_uses_quality)
                        tmp_faces_detected = len(gfaces)
                        tmp_faces_passing_quality = sum(
                            1 for gf in gfaces if float(gf.get("quality", 0.0)) >= quality_min
                        )
                        tmp_dists_all = []
                        tmp_dists_quality = []
                        faces_with_feat = [gf for gf in gfaces if gf.get("feat") is not None]
                        if ref_face_feat is not None and faces_with_feat:
                            cand = faces_with_feat
                            if use_quality_vis:
                                cand = [
                                    gf
                                    for gf in cand
                                    if float(gf.get("quality", 0.0)) >= quality_min
                                ]
                            gbest = min(
                                cand or faces_with_feat,
                                key=lambda f: self._fd_min(f["feat"], ref_face_feat),
                            )
                        else:
                            gbest = FaceEmbedder.best_face(gfaces)
                        for gf in gfaces:
                            bbox = gf.get("bbox")
                            if bbox is None or len(bbox) != 4:
                                continue
                            fx1, fy1, fx2, fy2 = [int(round(v)) for v in bbox]
                            fx1 = max(0, min(W2 - 1, fx1))
                            fy1 = max(0, min(H2 - 1, fy1))
                            fx2 = max(fx1 + 1, min(W2, int(round(fx2))))
                            fy2 = max(fy1 + 1, min(H2, int(round(fy2))))
                            q = float(gf.get("quality", 0.0))
                            feat_vec = gf.get("feat")
                            fd_val = None
                            if feat_vec is not None and ref_face_feat is not None:
                                fd_val = self._fd_min(feat_vec, ref_face_feat)
                                tmp_dists_all.append(fd_val)
                                if (not use_quality_vis) or q >= quality_min:
                                    tmp_dists_quality.append(fd_val)
                            face_debug_boxes.append(
                                (
                                    max(0, min(W - 1, fx1 + off_x)),
                                    max(0, min(H - 1, fy1 + off_y)),
                                    max(0, min(W - 1, fx2 + off_x)),
                                    max(0, min(H - 1, fy2 + off_y)),
                                    q,
                                    fd_val,
                                )
                            )

                        if (
                            gbest
                            and gbest.get("feat") is not None
                            and gbest.get("bbox") is not None
                            and len(gbest.get("bbox")) == 4
                        ):
                            gbbox = gbest.get("bbox")
                            fx1, fy1, fx2, fy2 = gbbox
                            fx1 = max(0.0, min(float(W2), fx1))
                            fy1 = max(0.0, min(float(H2), fy1))
                            fx2 = max(fx1 + 1.0, min(float(W2), fx2))
                            fy2 = max(fy1 + 1.0, min(float(H2), fy2))
                            face_box_abs = (fx1, fy1, fx2, fy2)
                            acx = (fx1 + fx2) / 2.0
                            acy = (fy1 + fy2) / 2.0
                            fd_val = self._fd_min(gbest["feat"], ref_face_feat)
                            if fd_val <= float(cfg.face_thresh):
                                subject_box_abs = None
                                assoc_persons = []
                                if bool(getattr(cfg, "compose_detect_person_for_face", True)):
                                    try:
                                        body_period = max(1, int(getattr(cfg, "compose_person_detect_cadence", getattr(cfg, "compose_body_every_n", 6)) or 1))
                                    except Exception:
                                        body_period = 6
                                    face_h_frac = (fy2 - fy1) / max(1.0, float(H2))
                                    need_person_assoc = (face_h_frac < 0.11) or (body_period > 0 and int(idx) % body_period == 0)
                                    if need_person_assoc:
                                        try:
                                            assoc_persons = det.detect(frame_for_det, conf=float(cfg.min_det_conf))
                                            subject_box_abs = self._find_person_box_for_face(face_box_abs, assoc_persons, W2, H2)
                                        except Exception:
                                            assoc_persons = []
                                            subject_box_abs = None
                                compose_seed_box = subject_box_abs or face_box_abs
                                (ex1, ey1, ex2, ey2), chosen_ratio, chosen_tloss = self._choose_best_ratio(
                                    compose_seed_box, ratios, W2, H2, anchor=(acx, acy), face_box=face_box_abs
                                )
                                ex1, ey1, ex2, ey2 = self._enforce_scale_and_margins(
                                    (ex1, ey1, ex2, ey2),
                                    chosen_ratio,
                                    W2,
                                    H2,
                                    face_box=face_box_abs,
                                    anchor=(acx, acy),
                                )
                                ox1, oy1, ox2, oy2 = ex1 + off_x, ey1 + off_y, ex2 + off_x, ey2 + off_y
                                ox1 = max(0, min(W - 1, int(round(ox1))))
                                oy1 = max(0, min(H - 1, int(round(oy1))))
                                ox2 = max(ox1 + 1, min(W, int(round(ox2))))
                                oy2 = max(oy1 + 1, min(H, int(round(oy2))))
                                if ox2 > ox1 + 1 and oy2 > oy1 + 1:
                                    crop_img = frame[oy1:oy2, ox1:ox2]
                                    sharp = self._calc_sharpness(crop_img)
                                    carea = max(1.0, float((ex2 - ex1) * (ey2 - ey1)))
                                    farea = max(1.0, (fx2 - fx1) * (fy2 - fy1))
                                    face_frac = float(farea) / carea
                                    fx1i = max(0, min(W - 1, int(round(fx1 + off_x))))
                                    fy1i = max(0, min(H - 1, int(round(fy1 + off_y))))
                                    fx2i = max(fx1i + 1, min(W, int(round(fx2 + off_x))))
                                    fy2i = max(fy1i + 1, min(H, int(round(fy2 + off_y))))
                                    face_box_global = (fx1i, fy1i, fx2i, fy2i)
                                    subject_box_global = None
                                    if subject_box_abs is not None:
                                        sx1, sy1, sx2, sy2 = subject_box_abs
                                        sgx1 = max(0, min(W - 1, int(round(sx1 + off_x))))
                                        sgy1 = max(0, min(H - 1, int(round(sy1 + off_y))))
                                        sgx2 = max(sgx1 + 1, min(W, int(round(sx2 + off_x))))
                                        sgy2 = max(sgy1 + 1, min(H, int(round(sy2 + off_y))))
                                        subject_box_global = (sgx1, sgy1, sgx2, sgy2)
                                    head_box_global = None
                                    head_box_roi = self._face_head_proxy_box(face_box_abs, W2, H2)
                                    if head_box_roi is not None:
                                        hx1, hy1, hx2, hy2 = head_box_roi
                                        head_box_global = (
                                            max(0, min(W, hx1 + off_x)),
                                            max(0, min(H, hy1 + off_y)),
                                            max(0, min(W, hx2 + off_x)),
                                            max(0, min(H, hy2 + off_y)),
                                        )
                                    short_circuit_candidate = dict(
                                        score=fd_val,
                                        fd=fd_val,
                                        rd=None,
                                        sharp=sharp,
                                        box=(ox1, oy1, ox2, oy2),
                                        area=(ox2 - ox1) * (oy2 - oy1),
                                        show_box=subject_box_global or (ox1, oy1, ox2, oy2),
                                        subject_box=subject_box_global,
                                        face_box=face_box_global,
                                        head_box=head_box_global,
                                        face_feat=gbest["feat"],
                                        reid_feat=None,
                                        ratio=chosen_ratio,
                                        face_frac=face_frac,
                                        tloss=float(chosen_tloss),
                                        reasons=["face_short_circuit"],
                                        face_quality=float(gbest.get("quality", 0.0)),
                                        accept_pre=True,
                                    )
                                    candidates = [short_circuit_candidate]
                                    diag_persons = len(assoc_persons)
                                    reid_feats = []
                                    faces_local = {}
                                    faces_detected = tmp_faces_detected
                                    faces_passing_quality = tmp_faces_passing_quality
                                    face_dists_all = tmp_dists_all
                                    face_dists_quality = tmp_dists_quality
                                    any_face_detected = tmp_faces_detected > 0
                                    any_face_visible = (
                                        tmp_faces_passing_quality > 0 if use_quality_vis else any_face_detected
                                    )
                                    any_face_match = True
                                    min_fd_all = min(tmp_dists_all) if tmp_dists_all else None
                                    best_face_dist = min(tmp_dists_quality) if tmp_dists_quality else None
                if short_circuit_candidate is None:
                    # Ensure defined even if no prior face pass ran this iteration
                    any_face_detected = bool(locals().get("any_face_detected", False))
                    run_persons = True
                    # Face-only fast path: if any face is visible, skip YOLO
                    if face_only_pipeline and bool(getattr(cfg, "skip_yolo_when_faceonly", True)):
                        run_persons = not any_face_detected
                    persons = det.detect(frame_for_det, conf=float(cfg.min_det_conf)) if run_persons else []
                    if not persons and cfg.auto_crop_borders:
                        # Fallback to full frame if border-cropped frame yields nothing
                        persons = det.detect(frame, conf=float(cfg.min_det_conf))
                        frame_for_det = frame
                        off_x, off_y = 0, 0
                        H2, W2 = H, W
                        self._status(
                            "Border-crop yielded no detections. Fallback to full frame.",
                            key="fallback",
                            interval=2.0,
                        )
                    candidates = []
                    diag_persons = len(persons)
                    crops = []
                    boxes = []
                    faces_local = {}

                    # Build candidate list
                    for p in persons:
                        x1,y1,x2,y2 = [int(v) for v in p["xyxy"]]
                        x1 = max(0, x1); y1 = max(0, y1); x2 = min(W2-1, x2); y2 = min(H2-1, y2)
                        if x2 <= x1+2 or y2 <= y1+2:
                            continue
                        area = (x2-x1)*(y2-y1)
                        if area < int(cfg.min_box_pixels):
                            continue
                        ar = (y2-y1) / max(1, (x2-x1))
                        if ar < 0.7 or ar > 4.0:  # extreme aspect -> likely false
                            continue
                        crop = frame_for_det[y1:y2, x1:x2]
                        crops.append(crop); boxes.append((x1,y1,x2,y2))

                    reid_feats = (
                        reid.extract(crops) if (reid is not None and crops) else [None] * len(crops)
                    )

                    face_debug_boxes = []  # [(x1,y1,x2,y2,q,fd)]
                    faces_detected = 0
                    faces_passing_quality = 0
                    min_fd_all = None

                    # Face features per person
                    for i, crop in enumerate(crops):
                        x1, y1, x2, y2 = boxes[i]
                        ffaces = face.extract(crop)
                        if not ffaces and float(cfg.face_det_pad) > 0.0:
                            pw = int(round((x2 - x1) * float(cfg.face_det_pad)))
                            ph = int(round((y2 - y1) * float(cfg.face_det_pad)))
                            if pw > 0 or ph > 0:
                                px1 = max(0, x1 - pw)
                                py1 = max(0, y1 - ph)
                                px2 = min(W2, x2 + pw)
                                py2 = min(H2, y2 + ph)
                                if px2 > px1 and py2 > py1:
                                    big = frame_for_det[py1:py2, px1:px2]
                                    ff2 = face.extract(big)
                                    remap = []
                                    dx, dy = (x1 - px1), (y1 - py1)
                                    for f in ff2:
                                        bb = f["bbox"].copy()
                                        bb[0] -= dx
                                        bb[2] -= dx
                                        bb[1] -= dy
                                        bb[3] -= dy
                                        remap.append({"bbox": bb, "feat": f.get("feat"), "quality": f.get("quality", 0.0)})
                                    ffaces = remap
                        faces_detected += len(ffaces)
                        for f in ffaces:
                            bx1, by1, bx2, by2 = [int(round(v)) for v in f["bbox"]]
                            fd_all = None
                            feat_vec = f.get("feat")
                            if ref_face_feat is not None and feat_vec is not None:
                                fd_all = self._fd_min(feat_vec, ref_face_feat)
                                min_fd_all = fd_all if min_fd_all is None else min(min_fd_all, fd_all)
                            face_debug_boxes.append(
                                (
                                    max(0, x1 + bx1 + off_x),
                                    max(0, y1 + by1 + off_y),
                                    min(W - 1, x1 + bx2 + off_x),
                                    min(H - 1, y1 + by2 + off_y),
                                    float(f.get("quality", 0.0)),
                                    fd_all,
                                )
                            )
                        if ref_face_feat is not None and ffaces:
                            faces_with_feat = [f for f in ffaces if f.get("feat") is not None]
                            if faces_with_feat:
                                bestf = min(
                                    faces_with_feat,
                                    key=lambda f: self._fd_min(f["feat"], ref_face_feat),
                                )
                            else:
                                bestf = FaceEmbedder.best_face(ffaces)
                        else:
                            bestf = FaceEmbedder.best_face(ffaces)
                        faces_local[i] = bestf
                        if bestf is not None and bestf.get("quality", 0.0) >= float(cfg.face_quality_min):
                            faces_passing_quality += 1

                    any_face_detected = faces_detected > 0
                    any_face_match = False
                    any_face_visible = (
                        faces_passing_quality > 0 if bool(cfg.face_visible_uses_quality) else any_face_detected
                    )

                    face_dists_all = []
                    face_dists_quality = []
                    for bf in faces_local.values():
                        if bf is None or bf.get("feat") is None or ref_face_feat is None:
                            continue
                        fd_tmp = self._fd_min(bf["feat"], ref_face_feat)
                        face_dists_all.append(fd_tmp)
                        if bf.get("quality", 0.0) >= float(cfg.face_quality_min):
                            face_dists_quality.append(fd_tmp)
                        if fd_tmp <= float(cfg.face_thresh):
                            any_face_match = True
                    best_face_dist = min(face_dists_quality) if face_dists_quality else None
                    if min_fd_all is None and face_dists_all:
                        min_fd_all = min(face_dists_all)

                # Evaluate candidates
                last_reject_reasons: list[str] = []

                for i, feat in enumerate(reid_feats):
                    cand_reason = []
                    rd = None
                    if ref_reid_feat is not None and feat is not None:
                        f = feat / max(np.linalg.norm(feat), 1e-9)
                        r = ref_reid_feat / max(np.linalg.norm(ref_reid_feat), 1e-9)
                        rd = 1.0 - float(np.dot(f, r))
                    if rd is None:
                        cand_reason.append("no_reid_ref_or_feat")
                    else:
                        cand_reason.append(f"rd={rd:.3f} thr={float(cfg.reid_thresh):.3f}")

                    bf = faces_local.get(i, None)
                    fd = None
                    if bf is not None and bf.get("feat") is not None and ref_face_feat is not None:
                        fd = self._fd_min(bf["feat"], ref_face_feat)
                    if bf is None:
                        cand_reason.append("no_face_in_crop")
                    if ref_face_feat is None:
                        cand_reason.append("no_ref_face_feat")
                    if fd is not None:
                        cand_reason.append(
                            f"fd={fd:.3f} thr={float(cfg.face_thresh):.3f} q={bf.get('quality', 0):.0f}"
                        )

                    # Match decision with dynamic fallback
                    face_ok = (fd is not None and fd <= float(cfg.face_thresh))
                    reid_ok = (rd is not None and rd <= float(cfg.reid_thresh))

                    eff_mode = base_match_mode
                    # If reference features are missing, degrade mode
                    if ref_face_feat is None and eff_mode in ("both", "face_only"):
                        eff_mode = "reid_only"
                    if ref_reid_feat is None and eff_mode in ("both", "reid_only"):
                        eff_mode = "face_only"
                    # If this candidate lacks a face or reid feature, allow the other branch
                    if eff_mode == "both":
                        if fd is None and rd is not None:
                            accept = reid_ok
                        elif rd is None and fd is not None:
                            accept = face_ok
                        else:
                            accept = face_ok and reid_ok
                    elif eff_mode == "face_only":
                        accept = face_ok
                    elif eff_mode == "reid_only":
                        accept = reid_ok
                    else:
                        accept = face_ok or reid_ok

                    # Respect match_mode: allow ReID when faces are present in 'either'/'reid_only'
                    allow_reid_when_faces_present = eff_mode in ("either", "reid_only")
                    if (
                        getattr(cfg, "drop_reid_if_any_face_match", True)
                        and not allow_reid_when_faces_present
                        and any_face_match
                        and not face_ok
                        and accept
                    ):
                        accept = False
                        cand_reason.append("drop_reid_due_to_face_match_present")

                    accept_before_face_policy = accept

                    # Face-first policy (revised): do not overrule a solid identity match with
                    # a quality gate triggered by another face elsewhere in the frame.
                    if (
                        cfg.require_face_if_visible
                        and any_face_visible
                        and (ref_face_feat is not None)
                        and eff_mode in ("both", "face_only")
                    ):
                        # Only hard-drop when the candidate has no detectable face or the
                        # detected face fails identity; a low quality score alone is insufficient
                        # once face_ok is True.
                        qfail = bf is None
                        if (
                            bf is not None
                            and bf.get("quality", 0.0)
                            < float(getattr(cfg, "face_quality_floor_absurd", 15))
                        ):
                            qfail = True
                        if (bf is not None) and (not face_ok):
                            qfail = True
                        if qfail:
                            accept = False
                            cand_reason.append("hard_gate_face_required")
                    elif (
                        cfg.prefer_face_when_available
                        and any_face_visible
                        and (bf is None)
                        and eff_mode in ("both", "face_only")
                    ):
                        cand_reason.append("soft_pref_face_missing")

                    if not accept:
                        cand_reason.append("reject")
                        last_reject_reasons = list(cand_reason)
                        continue

                    score = self._combine_scores(fd, rd, mode=cfg.combine)
                    x1,y1,x2,y2 = boxes[i]
                    # map anchor to frame_for_det coords
                    anchor = None
                    face_box_abs = None
                    if bf is not None:
                        fb = bf["bbox"]
                        acx = x1 + (fb[0]+fb[2])/2.0
                        acy = y1 + (fb[1]+fb[3])/2.0
                        anchor = (acx, acy)
                        fx1 = x1 + fb[0]
                        fy1 = y1 + fb[1]
                        fx2 = x1 + fb[2]
                        fy2 = y1 + fb[3]
                        fx1 = max(0.0, min(float(W2), fx1))
                        fy1 = max(0.0, min(float(H2), fy1))
                        fx2 = max(fx1 + 1.0, min(float(W2), fx2))
                        fy2 = max(fy1 + 1.0, min(float(H2), fy2))
                        face_box_abs = (fx1, fy1, fx2, fy2)

                    # expand to ratio within the DET frame using placement heuristics
                    (ex1, ey1, ex2, ey2), chosen_ratio, chosen_tloss = self._choose_best_ratio(
                        (x1,y1,x2,y2), ratios, W2, H2, anchor=anchor, face_box=face_box_abs
                    )
                    ex1, ey1, ex2, ey2 = self._enforce_scale_and_margins(
                        (ex1, ey1, ex2, ey2),
                        chosen_ratio,
                        W2,
                        H2,
                        face_box=face_box_abs,
                        anchor=anchor,
                    )

                    # Compute sharpness on final crop
                    crop_img = frame_for_det[ey1:ey2, ex1:ex2]
                    sharp = self._calc_sharpness(crop_img)
                    if float(cfg.min_sharpness) > 0 and sharp < float(cfg.min_sharpness):
                        cand_reason.append(
                            f"sharp={sharp:.1f} min={float(cfg.min_sharpness):.1f}"
                        )
                        last_reject_reasons = list(cand_reason)
                        continue

                    # Map to original frame coords for annotation
                    ox1, oy1, ox2, oy2 = ex1+off_x, ey1+off_y, ex2+off_x, ey2+off_y
                    area = (ox2-ox1)*(oy2-oy1)
                    face_box_global = None
                    head_box_global = None
                    face_frac = 0.0
                    if face_box_abs is not None:
                        fx1, fy1, fx2, fy2 = face_box_abs
                        farea = max(1.0, (fx2 - fx1) * (fy2 - fy1))
                        carea = max(1.0, float((ex2 - ex1) * (ey2 - ey1)))
                        face_frac = float(farea) / carea
                        face_box_global = (
                            max(0, min(W, fx1 + off_x)),
                            max(0, min(H, fy1 + off_y)),
                            max(0, min(W, fx2 + off_x)),
                            max(0, min(H, fy2 + off_y)),
                        )
                        head_box_roi = self._face_head_proxy_box(face_box_abs, W2, H2)
                        if head_box_roi is not None:
                            hx1, hy1, hx2, hy2 = head_box_roi
                            head_box_global = (
                                max(0, min(W, hx1 + off_x)),
                                max(0, min(H, hy1 + off_y)),
                                max(0, min(W, hx2 + off_x)),
                                max(0, min(H, hy2 + off_y)),
                            )
                    candidates.append(
                        dict(
                            score=score,
                            fd=fd,
                            rd=rd,
                            sharp=sharp,
                            box=(ox1, oy1, ox2, oy2),
                            area=area,
                            show_box=(
                                int(x1 + off_x),
                                int(y1 + off_y),
                                int(x2 + off_x),
                                int(y2 + off_y),
                            ),
                            subject_box=(
                                int(x1 + off_x),
                                int(y1 + off_y),
                                int(x2 + off_x),
                                int(y2 + off_y),
                            ),
                            face_box=face_box_global,
                            head_box=head_box_global,
                            face_feat=(bf["feat"] if bf is not None else None),
                            reid_feat=(reid_feats[i] if i < len(reid_feats) else None),
                            ratio=chosen_ratio,
                            face_frac=face_frac,
                            tloss=float(chosen_tloss),
                            reasons=cand_reason,
                            face_quality=(bf.get('quality', 0.0) if bf is not None else None),
                            accept_pre=accept_before_face_policy,
                        )
                    )

                # Choose best and save with cadence + lock + margin + IoU gate
                def save_hit(
                    c,
                    idx,
                    *,
                    frame_w,
                    frame_h,
                    det_off_x,
                    det_off_y,
                    det_w,
                    det_h,
                    ratio_list,
                    repair_bounds_xyxy=None,
                ):
                    nonlocal hit_count, lock_hits, locked_face, locked_reid, prev_box, ref_face_feat, ref_bank_list, source_size_cached
                    crop_img_path = os.path.join(crops_dir, f"f{idx:08d}.jpg")
                    hdr_primary_fullres = bool(
                        hdr_active
                        and bool(getattr(self.cfg, "hdr_screencap_fullres", True))
                    )
                    hdr_out_path = None
                    if hdr_active and bool(getattr(self.cfg, "hdr_archive_crops", False)):
                        try:
                            ensure_dir(hdr_crops_dir)
                        except Exception:
                            pass
                        hdr_fmt = str(getattr(self.cfg, "hdr_crop_format", "mkv") or "mkv").lower()
                        hdr_ext = ".avif" if hdr_fmt == "avif" else ".mkv"
                        hdr_out_path = os.path.join(hdr_crops_dir, f"f{idx:08d}{hdr_ext}")
                    # Start from candidate box in GLOBAL coords
                    gx1, gy1, gx2, gy2 = c["box"]
                    ratio_str = str(
                        c.get("ratio")
                        or (
                            ratio_list[0]
                            if ratio_list
                            else (self.cfg.ratio.split(',')[0] if self.cfg.ratio else '2:3')
                        )
                    )

                    # Compose the saved dataset crop from identity evidence.
                    # The detector/person/face boxes identify the target; they do not
                    # directly dictate the final crop.  The composition stage chooses
                    # close/portrait/body framing while preserving the invariant that
                    # the target face/head or person is not cut off when a feasible
                    # in-bounds ratio crop exists.
                    if bool(getattr(cfg, "auto_crop_borders", False)):
                        bx1 = int(det_off_x)
                        by1 = int(det_off_y)
                        bx2 = int(bx1 + det_w)
                        by2 = int(by1 + det_h)
                    else:
                        bx1, by1, bx2, by2 = 0, 0, frame_w, frame_h
                    bx1 = max(0, min(frame_w - 1, bx1))
                    by1 = max(0, min(frame_h - 1, by1))
                    bx2 = max(bx1 + 1, min(frame_w, bx2))
                    by2 = max(by1 + 1, min(frame_h, by2))

                    ratio_candidates = list(ratio_list) if ratio_list else [ratio_str]
                    if bool(getattr(cfg, "compose_crop_enable", True)):
                        (cx1, cy1, cx2, cy2), ratio_str, crop_profile = self._compose_dataset_crop(
                            (gx1, gy1, gx2, gy2),
                            ratio_candidates,
                            (bx1, by1, bx2, by2),
                            subject_box=c.get("subject_box"),
                            face_box=c.get("face_box"),
                            frame_idx=idx,
                        )
                        c["crop_profile"] = crop_profile
                        c["ratio"] = ratio_str
                    else:
                        cx1, cy1, cx2, cy2 = gx1, gy1, gx2, gy2
                        try:
                            rw, rh = parse_ratio(ratio_str)
                            cx1, cy1, cx2, cy2 = expand_box_to_ratio(
                                cx1, cy1, cx2, cy2, rw, rh, W, H, anchor=None, head_bias=0.0
                            )
                        except Exception:
                            pass

                    if repair_bounds_xyxy is not None and len(repair_bounds_xyxy) == 4:
                        bx1, by1, bx2, by2 = [int(v) for v in repair_bounds_xyxy]
                    elif bool(getattr(cfg, "auto_crop_borders", False)):
                        bx1 = int(det_off_x)
                        by1 = int(det_off_y)
                        bx2 = int(bx1 + det_w)
                        by2 = int(by1 + det_h)
                    else:
                        bx1, by1, bx2, by2 = 0, 0, frame_w, frame_h
                    repair_bx1, repair_by1, repair_bx2, repair_by2 = bx1, by1, bx2, by2

                    # Do not run border detection on the already-composed crop.
                    # In candle-lit/dark scenes the subject's hair, clothing, or a dark
                    # room edge can look like a black border.  Trimming inside the final
                    # crop destroys the composition after identity/crop selection and can
                    # leave only a partial face or bright background object.  Letterbox
                    # removal is handled once at frame level through auto_crop_borders and
                    # the resulting content bounds above.

                    cx1 = max(0, min(frame_w - 1, int(round(cx1))))
                    cy1 = max(0, min(frame_h - 1, int(round(cy1))))
                    cx2 = max(cx1 + 1, min(frame_w, int(round(cx2))))
                    cy2 = max(cy1 + 1, min(frame_h, int(round(cy2))))

                    # Ensure ratio terms exist before using them in corrections
                    try:
                        rw, rh = parse_ratio(ratio_str)
                    except Exception:
                        rw, rh = 1, 1  # ultra-safe fallback

                    try:
                        w = cx2 - cx1
                        h = cy2 - cy1
                        target_w = max(1, int(round(h * float(rw) / float(rh))))
                        # Only correct if materially off (avoid jitter from rounding)
                        if abs(w - target_w) > 1:
                            # Center inside content window, not full frame
                            cx1 = max(repair_bx1, min(repair_bx2 - target_w, cx1 + (w - target_w) // 2))
                            cx2 = cx1 + target_w
                        # Height correction stays inside the same content window
                        target_h = max(1, int(round((cx2 - cx1) * float(rh) / float(rw))))
                        if abs((cy2 - cy1) - target_h) > 1:
                            cy1 = max(repair_by1, min(repair_by2 - target_h, cy1 + ((cy2 - cy1) - target_h) // 2))
                            cy2 = cy1 + target_h
                    except Exception:
                        pass

                    crop_profile_for_guard = str(c.get("crop_profile") or "").lower()
                    if crop_profile_for_guard == "body":
                        protect_box = self._union_boxes_xyxy(
                            c.get("subject_box"),
                            c.get("head_box"),
                            c.get("face_box"),
                        )
                    else:
                        # Close/upper/portrait crops protect the detected head and
                        # face plus the active associated person box. show_box is
                        # a fallback only when subject_box is missing, because
                        # show_box can be stale composed geometry.
                        subject_or_show = c.get("subject_box") or c.get("show_box")
                        protect_box = self._union_boxes_xyxy(
                            subject_or_show,
                            c.get("head_box"),
                            c.get("face_box"),
                        )
                    if protect_box is not None:
                        try:
                            cur_w = max(1.0, float(cx2 - cx1))
                            cur_h = max(1.0, float(cy2 - cy1))
                            cx1, cy1, cx2, cy2 = self._ratio_crop_containing_box(
                                protect_box,
                                ratio_str,
                                (repair_bx1, repair_by1, repair_bx2, repair_by2),
                                anchor=((cx1 + cx2) * 0.5, (cy1 + cy2) * 0.5),
                                min_size_xy=(cur_w, cur_h),
                            )
                        except Exception:
                            cx1, cy1, cx2, cy2 = self._shift_crop_to_include_box(
                                (cx1, cy1, cx2, cy2),
                                protect_box,
                                (repair_bx1, repair_by1, repair_bx2, repair_by2),
                                margin_px=1.0,
                            )

                    # Edge-aware face-margin repair.  This must repair the crop, not
                    # reject the frame.  A rejected frame here means the cropper failed
                    # after identity had already succeeded.
                    fb = c.get("face_box")
                    if fb is not None and bool(getattr(cfg, "side_guard_drop_enable", True)):
                        try:
                            fw = max(1.0, float(fb[2]) - float(fb[0]))
                            desired = float(cfg.crop_face_side_margin_frac) * fw
                            fd_val = float(c.get("fd")) if c.get("fd") is not None else 9.0
                            reasons = set(c.get("reasons", []))
                            is_rescue = ("face_short_circuit" in reasons) or ("global_face" in reasons)
                            relax_fd = float(getattr(cfg, "side_guard_relax_fd", 0.22))
                            relax_factor = float(getattr(cfg, "side_guard_relax_factor", 0.50))
                            required = float(getattr(cfg, "side_guard_drop_factor", 0.66)) * desired
                            if (fd_val <= relax_fd) or is_rescue:
                                required *= relax_factor
                            padded_face = self._pad_box_xyxy(
                                fb,
                                pad_x=required,
                                pad_y_top=float(getattr(cfg, "face_edge_inner_px", 1.0)),
                                pad_y_bottom=float(getattr(cfg, "face_edge_inner_px", 1.0)),
                                bounds_xyxy=(repair_bx1, repair_by1, repair_bx2, repair_by2),
                            ) or fb
                            cur_w = max(1.0, float(cx2 - cx1))
                            cur_h = max(1.0, float(cy2 - cy1))
                            side_guard_box = self._union_boxes_xyxy(protect_box, padded_face) or padded_face
                            # Do not preserve a stale landscape crop size for non-body
                            # profiles. Side repair must keep the face/head visible,
                            # not lock in a bad earlier composition.
                            min_size_for_side = (cur_w, cur_h) if crop_profile_for_guard == "body" else None
                            cx1, cy1, cx2, cy2 = self._ratio_crop_containing_box(
                                side_guard_box,
                                ratio_str,
                                (repair_bx1, repair_by1, repair_bx2, repair_by2),
                                anchor=((cx1 + cx2) * 0.5, (cy1 + cy2) * 0.5),
                                min_size_xy=min_size_for_side,
                            )
                            left_margin = max(0.0, float(fb[0]) - float(cx1))
                            right_margin = max(0.0, float(cx2) - float(fb[2]))
                            self._status(
                                f"side_guard repair L={left_margin:.1f} R={right_margin:.1f} req={required:.1f} fw={fw:.1f} fd={fd_val:.3f}",
                                key="side_guard_dbg",
                                interval=0.8,
                            )
                        except Exception:
                            pass

                    # Final face containment / shape repair. Identity boxes identify the
                    # target, but the final dataset crop must still satisfy the hard
                    # invariant that the detected face is inside the saved crop. Also
                    # prefer square/portrait crops when the face is already prominent;
                    # landscape is only acceptable for body/context shots.
                    hard_face_box = c.get("face_box")
                    if hard_face_box is not None:
                        try:
                            hf = self._coerce_box_xyxy(hard_face_box, (repair_bx1, repair_by1, repair_bx2, repair_by2))
                            if hf is not None:
                                hfx1, hfy1, hfx2, hfy2 = hf
                                hfw = max(1.0, hfx2 - hfx1)
                                hfh = max(1.0, hfy2 - hfy1)
                                hard_head_box = self._coerce_box_xyxy(c.get("head_box"), (repair_bx1, repair_by1, repair_bx2, repair_by2))
                                hard_face_padded = self._union_boxes_xyxy(
                                    hard_head_box,
                                    self._pad_box_xyxy(
                                        hf,
                                        pad_x=0.12 * hfw,
                                        pad_y_top=0.18 * hfh,
                                        pad_y_bottom=0.18 * hfh,
                                        bounds_xyxy=(repair_bx1, repair_by1, repair_bx2, repair_by2),
                                    ) or hf,
                                ) or hf
                                cur_crop = (float(cx1), float(cy1), float(cx2), float(cy2))
                                cur_w = max(1.0, float(cx2 - cx1))
                                cur_h = max(1.0, float(cy2 - cy1))
                                cur_face_h_frac = hfh / cur_h
                                try:
                                    rrw, rrh = parse_ratio(ratio_str)
                                    cur_aspect = float(rrw) / max(1e-6, float(rrh))
                                except Exception:
                                    cur_aspect = cur_w / cur_h
                                was_landscape = cur_aspect > 1.05
                                hard_def = self._containment_deficit_xyxy(cur_crop, hard_face_padded, margin_px=1.0)
                                frame_face_h_frac = hfh / max(1.0, float(repair_by2 - repair_by1))
                                prominent_face = (
                                    cur_face_h_frac >= 0.10
                                    or frame_face_h_frac >= 0.075
                                    or float(c.get("face_frac") or 0.0) >= 0.035
                                )
                                force_portrait = was_landscape and (crop_profile_for_guard != "body" or prominent_face)
                                if hard_def > 0.01 or force_portrait:
                                    if crop_profile_for_guard == "body" and not force_portrait:
                                        identity_guard = self._coerce_box_xyxy(
                                            self._union_boxes_xyxy(
                                                c.get("subject_box"),
                                                c.get("head_box"),
                                                c.get("face_box"),
                                            ),
                                            (repair_bx1, repair_by1, repair_bx2, repair_by2),
                                        )
                                    else:
                                        subject_or_show = c.get("subject_box") or c.get("show_box")
                                        identity_guard = self._coerce_box_xyxy(
                                            self._union_boxes_xyxy(
                                                subject_or_show,
                                                c.get("head_box"),
                                                c.get("face_box"),
                                            ),
                                            (repair_bx1, repair_by1, repair_bx2, repair_by2),
                                        )
                                    protect_box_clamped = (
                                        self._coerce_box_xyxy(protect_box, (repair_bx1, repair_by1, repair_bx2, repair_by2))
                                        if (protect_box is not None and crop_profile_for_guard == "body" and not force_portrait)
                                        else None
                                    )
                                    # No show_box fallback here. show_box may be the stale
                                    # candidate/preview box; using it as a guard resurrects
                                    # bad wide crops and can still cut the head.
                                    full_guard_box = self._union_boxes_xyxy(
                                        hard_face_padded,
                                        identity_guard,
                                        protect_box_clamped,
                                    ) or hard_face_padded
                                    best_fix = None
                                    for fix_ratio in ("1:1", "2:3", "3:4"):
                                        try:
                                            fixed = self._ratio_crop_containing_box(
                                                full_guard_box,
                                                fix_ratio,
                                                (repair_bx1, repair_by1, repair_bx2, repair_by2),
                                                anchor=((hfx1 + hfx2) * 0.5, (hfy1 + hfy2) * 0.5 + 0.18 * hfh),
                                                min_size_xy=(max(hfw * 1.45, 2.0), max(hfh * 1.55, 2.0)),
                                            )
                                            guard_def = self._containment_deficit_xyxy(fixed, full_guard_box, margin_px=1.0)
                                            if guard_def > 0.01:
                                                continue
                                            fw2 = max(1.0, float(fixed[2] - fixed[0]))
                                            fh2 = max(1.0, float(fixed[3] - fixed[1]))
                                            face_h_frac2 = hfh / fh2
                                            target_frac = 0.34 if fix_ratio == "1:1" else 0.24
                                            score = abs(face_h_frac2 - target_frac)
                                            score += 0.02 if fix_ratio == "2:3" else (0.04 if fix_ratio == "3:4" else 0.0)
                                            score += 0.04 * ((fw2 * fh2) / max(1.0, float((repair_bx2 - repair_bx1) * (repair_by2 - repair_by1))))
                                            if best_fix is None or score < best_fix[0]:
                                                best_fix = (score, fixed, fix_ratio)
                                        except Exception:
                                            continue
                                    if best_fix is not None:
                                        _, fixed, fixed_ratio = best_fix
                                        cx1, cy1, cx2, cy2 = fixed
                                        ratio_str = fixed_ratio
                                        c["ratio"] = fixed_ratio
                                        if crop_profile_for_guard == "body" and was_landscape and fixed_ratio in {"1:1", "2:3", "3:4"}:
                                            c["crop_profile"] = "upper"
                                            crop_profile_for_guard = "upper"
                                    elif hard_def > 0.01 or force_portrait:
                                        fallback_ratio = "2:3" if force_portrait else ratio_str
                                        fallback_done = False
                                        try:
                                            fixed = self._ratio_crop_containing_box(
                                                full_guard_box,
                                                fallback_ratio,
                                                (repair_bx1, repair_by1, repair_bx2, repair_by2),
                                                anchor=((hfx1 + hfx2) * 0.5, (hfy1 + hfy2) * 0.5 + 0.18 * hfh),
                                                min_size_xy=(
                                                    (max(cur_w, hfw * 1.45) if not force_portrait else max(hfw * 1.45, 2.0)),
                                                    (max(cur_h, hfh * 1.55) if not force_portrait else max(hfh * 1.55, 2.0)),
                                                ),
                                            )
                                            guard_def = self._containment_deficit_xyxy(fixed, full_guard_box, margin_px=1.0)
                                            if guard_def <= 0.01:
                                                cx1, cy1, cx2, cy2 = fixed
                                                ratio_str = fallback_ratio
                                                c["ratio"] = fallback_ratio
                                                fallback_done = True
                                        except Exception:
                                            fallback_done = False
                                        if not fallback_done:
                                            try:
                                                fixed = self._ratio_crop_containing_box(
                                                    hard_face_padded,
                                                    fallback_ratio,
                                                    (repair_bx1, repair_by1, repair_bx2, repair_by2),
                                                    anchor=((hfx1 + hfx2) * 0.5, (hfy1 + hfy2) * 0.5 + 0.18 * hfh),
                                                    min_size_xy=(
                                                        (max(cur_w, hfw * 1.45) if not force_portrait else max(hfw * 1.45, 2.0)),
                                                        (max(cur_h, hfh * 1.55) if not force_portrait else max(hfh * 1.55, 2.0)),
                                                    ),
                                                )
                                                cx1, cy1, cx2, cy2 = fixed
                                                ratio_str = fallback_ratio
                                                c["ratio"] = fallback_ratio
                                                fallback_done = True
                                            except Exception:
                                                pass
                                        if (not fallback_done) and (hard_def > 0.01 or force_portrait):
                                            if force_portrait:
                                                ratio_str = fallback_ratio
                                                c["ratio"] = fallback_ratio
                                            try:
                                                shift_seed = cur_crop
                                                if force_portrait:
                                                    shift_seed = self._ratio_crop_containing_box(
                                                        hard_face_padded,
                                                        fallback_ratio,
                                                        (repair_bx1, repair_by1, repair_bx2, repair_by2),
                                                        anchor=((hfx1 + hfx2) * 0.5, (hfy1 + hfy2) * 0.5 + 0.18 * hfh),
                                                        min_size_xy=(
                                                            (max(cur_w, hfw * 1.45) if not force_portrait else max(hfw * 1.45, 2.0)),
                                                            (max(cur_h, hfh * 1.55) if not force_portrait else max(hfh * 1.55, 2.0)),
                                                        ),
                                                    )
                                                cx1, cy1, cx2, cy2 = self._shift_crop_to_include_box(
                                                    shift_seed,
                                                    hard_face_padded,
                                                    (repair_bx1, repair_by1, repair_bx2, repair_by2),
                                                    margin_px=1.0,
                                                )
                                            except Exception:
                                                if force_portrait:
                                                    try:
                                                        cx1, cy1, cx2, cy2 = self._ratio_crop_containing_box(
                                                            hard_face_padded,
                                                            fallback_ratio,
                                                            (repair_bx1, repair_by1, repair_bx2, repair_by2),
                                                            anchor=((hfx1 + hfx2) * 0.5, (hfy1 + hfy2) * 0.5 + 0.18 * hfh),
                                                            min_size_xy=(max(hfw * 1.45, 2.0), max(hfh * 1.55, 2.0)),
                                                        )
                                                    except Exception:
                                                        pass
                                        if crop_profile_for_guard == "body" and was_landscape and c.get("ratio") in {"1:1", "2:3", "3:4"}:
                                            c["crop_profile"] = "upper"
                                            crop_profile_for_guard = "upper"
                        except Exception:
                            logger.exception(
                                "Final face containment repair failed idx=%s face=%s bounds=%s crop=%s",
                                idx,
                                hard_face_box,
                                (repair_bx1, repair_by1, repair_bx2, repair_by2),
                                (cx1, cy1, cx2, cy2),
                            )

                    # Final clamp inside de-barred content window (prevents 1px bar re-entry)
                    cx1 = max(repair_bx1, min(repair_bx2 - 1, cx1))
                    cy1 = max(repair_by1, min(repair_by2 - 1, cy1))
                    cx2 = max(cx1 + 1, min(repair_bx2, cx2))
                    cy2 = max(cy1 + 1, min(repair_by2, cy2))
                    try:
                        rw, rh = parse_ratio(ratio_str)
                        asp = (cx2 - cx1) / float(max(1, cy2 - cy1))
                        targ = float(rw) / float(rh)
                        self._status(
                            f"final_aspect={asp:.5f} target={targ:.5f} ratio={ratio_str}",
                            key="aspect",
                            interval=2.0,
                        )
                    except Exception:
                        pass

                    self._status(
                        f"crop@{idx}: face_box={c.get('face_box')}", key="cropface", interval=2.0
                    )

                    processed_crop_xyxy = (int(cx1), int(cy1), int(cx2), int(cy2))
                    if not hdr_primary_fullres:
                        try:
                            final_crop_for_sharp = frame[int(cy1):int(cy2), int(cx1):int(cx2)]
                            if final_crop_for_sharp.size > 0:
                                c["sharp"] = self._calc_sharpness(final_crop_for_sharp)
                        except Exception:
                            pass
                    # Keep the annotated preview in sync with the actual saved crop.
                    # Earlier preview boxes were drawn from the pre-final candidate box,
                    # while save_hit later changed the crop via smart-crop, border trim,
                    # ratio correction, and face/head guards.  That made the UI show a
                    # different box from the file that was actually written.
                    c["box"] = processed_crop_xyxy
                    c["saved_box"] = processed_crop_xyxy
                    c["selected"] = True
                    source_size = (int(W), int(H))
                    source_crop_xyxy = processed_crop_xyxy
                    needs_source_space = bool(hdr_primary_fullres or hdr_out_path)
                    if needs_source_space:
                        if source_size_cached is None:
                            source_size_cached = self._capture_source_size(cap, frame.shape[:2])
                        source_size = source_size_cached
                        source_crop_xyxy = self._scale_crop_xyxy_to_source(
                            processed_crop_xyxy,
                            (int(W), int(H)),
                            source_size,
                        )
                    primary_row_crop = source_crop_xyxy if hdr_primary_fullres else processed_crop_xyxy
                    crop_img2 = frame[cy1:cy2, cx1:cx2]
                    row = [
                        idx,
                        frame_pts_sec if frame_pts_sec is not None else (idx / float(fps) if fps > 0 else 0.0),
                        c.get("score"),
                        c.get("fd"),
                        c.get("rd"),
                        primary_row_crop[0],
                        primary_row_crop[1],
                        primary_row_crop[2],
                        primary_row_crop[3],
                        crop_img_path,
                        c.get("sharp"),
                        str(ratio_str),
                    ]
                    primary_saved_or_enqueued = False
                    if save_q is not None:
                        ack_q: queue.Queue = queue.Queue(maxsize=1)
                        try:
                            if hdr_primary_fullres:
                                save_q.put_nowait({
                                    "type": "hdr_sdr",
                                    "path": crop_img_path,
                                    "row": row,
                                    "frame_idx": int(idx),
                                    "frame_pts_sec": frame_pts_sec,
                                    "crop_xyxy": source_crop_xyxy,
                                    "ack_q": ack_q,
                                })
                            else:
                                # enqueue a contiguous copy; slices are views into `frame`
                                buf = np.ascontiguousarray(crop_img2)
                                if not buf.flags.owndata:
                                    buf = buf.copy()
                                save_q.put_nowait({
                                    "type": "jpeg",
                                    "path": crop_img_path,
                                    "row": row,
                                    "img": buf,
                                    "ack_q": ack_q,
                                })
                        except queue.Full:
                            self._status(
                                f"Save queue full: {crop_img_path}",
                                key="save_backpressure",
                                interval=0.5,
                            )
                            return False
                        save_timeout_sec = max(5, int(getattr(self.cfg, "hdr_export_timeout_sec", 300) or 300))
                        try:
                            ok, why = ack_q.get(timeout=float(save_timeout_sec))
                        except queue.Empty:
                            self._status(
                                f"Save ack timeout after {save_timeout_sec}s: {crop_img_path}",
                                key="save_timeout",
                                interval=0.5,
                            )
                            return False
                        ack_why = str(why or "")
                        if not ok:
                            self._status(
                                f"Save failed ({why}): {crop_img_path}",
                                key="save_err",
                                interval=0.5,
                            )
                            return False
                        if hdr_primary_fullres and isinstance(row, list) and len(row) > 10:
                            try:
                                c["sharp"] = float(row[10])
                            except Exception:
                                pass
                        primary_saved_or_enqueued = True
                    else:
                        if hdr_primary_fullres:
                            ok, why = self._save_hdr_sdr_screencap(
                                int(idx),
                                frame_pts_sec,
                                source_crop_xyxy,
                                crop_img_path,
                            )
                        else:
                            ok, why = _atomic_jpeg_write(crop_img2, crop_img_path, jpg_q)
                        if ok:
                            if hdr_primary_fullres:
                                saved_sharp = self._calc_saved_file_sharpness(crop_img_path)
                                if saved_sharp is not None:
                                    row[10] = float(saved_sharp)
                                    c["sharp"] = float(saved_sharp)
                            # emit on worker thread directly
                            self.hit.emit(crop_img_path)
                            # CSV best-effort
                            try:
                                writer.writerow(row)
                                sync_wrote += 1
                                if sync_wrote % flush_every == 0:
                                    csv_f.flush()
                            except Exception:
                                logger.exception("CSV write failed for %s", crop_img_path)
                        else:
                            self._status(
                                f"Save failed ({why}): {crop_img_path}",
                                key="save_err",
                                interval=0.5,
                            )
                            return False
                        primary_saved_or_enqueued = True
                    if primary_saved_or_enqueued and hdr_out_path:
                        hdr_crop_xyxy = self._even_hdr_crop_xyxy(source_crop_xyxy, source_size)
                        if archive_q is not None:
                            try:
                                archive_q.put_nowait({
                                    "type": "hdr_archive",
                                    "path": hdr_out_path,
                                    "frame_idx": int(idx),
                                    "frame_pts_sec": frame_pts_sec,
                                    "crop_xyxy": hdr_crop_xyxy,
                                })
                            except queue.Full:
                                self._status(
                                    f"HDR archive queue full: {hdr_out_path}",
                                    key="save_backpressure",
                                    interval=0.5,
                                )
                        else:
                            self._save_hdr_crop_p010(idx, frame_pts_sec, hdr_crop_xyxy, hdr_out_path)
                    reasons_list = c.get("reasons") or []
                    reasons = "|".join(reasons_list)
                    bx1, by1, bx2, by2 = c["box"]
                    area = (bx2 - bx1) * (by2 - by1)
                    self._status(
                        (
                            f"CAPTURE idx={idx} t={idx/float(fps):.2f} "
                            f"fd={c.get('fd')} rd={c.get('rd')} score={c.get('score')} "
                            f"area={area} ratio={ratio_str} face_frac={c.get('face_frac'):.3f} tloss={c.get('tloss', 0.0):.4f} reasons={reasons}"
                        ),
                        key="cap",
                    )
                    # hit is emitted by the saver (async) or right after atomic save (sync)
                    # update lock source
                    face_feat_curr = c.get("face_feat")
                    if face_feat_curr is not None:
                        locked_face = face_feat_curr
                        fd_val = c.get("fd")
                        quality_val = c.get("face_quality")
                        quality_threshold = float(cfg.face_quality_min)
                        try:
                            fd_val_f = float(fd_val) if fd_val is not None else None
                        except Exception:
                            fd_val_f = None
                        quality_ok = (
                            quality_val is None
                            or quality_threshold <= 0
                            or float(quality_val) >= quality_threshold
                        )
                        if getattr(cfg, "learn_bank_runtime", False):
                            rt_fd_thresh = float(getattr(cfg, "prescan_fd_add", 0.22))
                            cooldown_frames = int(getattr(cfg, "prescan_add_cooldown_samples", 5)) * max(1, int(getattr(cfg, "frame_stride", 1)))
                            cooldown_ok = (idx - getattr(self, "_runtime_last_bank_add_idx", -10**9)) >= cooldown_frames
                            if (
                                fd_val_f is not None
                                and fd_val_f <= rt_fd_thresh
                                and quality_ok
                                and cooldown_ok
                            ):
                                try:
                                    quality_val_f = float(quality_val) if quality_val is not None else 0.0
                                except Exception:
                                    quality_val_f = 0.0
                                ref_face_feat, action, idx_info = self._stream_ref_bank_update(
                                    ref_bank_list,
                                    ref_face_feat,
                                    face_feat_curr,
                                    quality_val_f,
                                    cfg,
                                )
                                if action in {"added", "replaced"}:
                                    self._runtime_last_bank_add_idx = idx
                                    if action == "added":
                                        self._status(
                                            f"Ref bank +1 (size={len(ref_bank_list)}) fd={fd_val_f:.3f}",
                                            key="ref_bank",
                                            interval=5.0,
                                        )
                                    else:
                                        self._status(
                                            f"Ref bank replaced #{idx_info} with better ref (score↑)",
                                            key="ref_bank",
                                            interval=5.0,
                                        )
                    if c.get("reid_feat") is not None:
                        lr_prev = locked_reid
                        locked_reid = self._ema(lr_prev, c["reid_feat"], float(cfg.lock_momentum))
                    person_box = c.get("show_box") or (cx1, cy1, cx2, cy2)
                    px1, py1, px2, py2 = [int(round(v)) for v in person_box]
                    self._set_lock((px1, py1, px2 - px1, py2 - py1), idx)
                    if locked_reid is not None:
                        self._locked_reid_feat = locked_reid
                    elif c.get("reid_feat") is not None:
                        try:
                            self._locked_reid_feat = np.asarray(c["reid_feat"], dtype=np.float32)
                        except Exception:
                            self._locked_reid_feat = None
                    lock_hits += 1
                    prev_box = (cx1,cy1,cx2,cy2)
                    return True

                # initialize cadence tracking
                if not hasattr(self, "_last_hit_t"):
                    self._last_hit_t = -1e9

                fallback_candidate = None
                if not candidates:
                    # Face-only global fallback: per-person search yielded no faces.
                    if (
                        face_only_pipeline
                        and getattr(cfg, "face_fullframe_when_missed", True)
                        and faces_detected == 0
                        and ref_face_feat is not None
                    ):
                        gfaces = face.extract(frame_for_det, imgsz=fullframe_imgsz)
                        quality_min = float(cfg.face_quality_min)
                        use_quality_vis = bool(cfg.face_visible_uses_quality)
                        faces_with_feat = [gf for gf in gfaces if gf.get("feat") is not None]
                        if ref_face_feat is not None and faces_with_feat:
                            cand = faces_with_feat
                            if use_quality_vis:
                                cand = [
                                    gf
                                    for gf in cand
                                    if float(gf.get("quality", 0.0)) >= quality_min
                                ]
                            gbest = min(
                                cand or faces_with_feat,
                                key=lambda f: self._fd_min(f["feat"], ref_face_feat),
                            )
                        else:
                            gbest = FaceEmbedder.best_face(gfaces)
                        if gbest is not None and gbest.get("feat") is not None:
                            gfd = self._fd_min(gbest["feat"], ref_face_feat)
                            if gfd <= float(cfg.face_thresh):
                                fx1, fy1, fx2, fy2 = gbest["bbox"]
                                fx1 = max(0.0, min(float(W2), fx1))
                                fy1 = max(0.0, min(float(H2), fy1))
                                fx2 = max(fx1 + 1.0, min(float(W2), fx2))
                                fy2 = max(fy1 + 1.0, min(float(H2), fy2))
                                acx = (fx1 + fx2) / 2.0
                                acy = (fy1 + fy2) / 2.0
                                face_box_abs = (fx1, fy1, fx2, fy2)
                                subject_box_abs = None
                                if bool(getattr(cfg, "compose_detect_person_for_face", True)):
                                    try:
                                        body_period = max(1, int(getattr(cfg, "compose_person_detect_cadence", getattr(cfg, "compose_body_every_n", 6)) or 1))
                                    except Exception:
                                        body_period = 6
                                    face_h_frac = (fy2 - fy1) / max(1.0, float(H2))
                                    need_person_assoc = (face_h_frac < 0.11) or (body_period > 0 and int(idx) % body_period == 0)
                                    if need_person_assoc:
                                        try:
                                            assoc_persons = persons or det.detect(frame_for_det, conf=float(cfg.min_det_conf))
                                            subject_box_abs = self._find_person_box_for_face(face_box_abs, assoc_persons, W2, H2)
                                        except Exception:
                                            subject_box_abs = None
                                compose_seed_box = subject_box_abs or face_box_abs
                                (ex1, ey1, ex2, ey2), chosen_ratio, chosen_tloss = self._choose_best_ratio(
                                    compose_seed_box, ratios, W2, H2, anchor=(acx, acy), face_box=face_box_abs
                                )
                                ex1, ey1, ex2, ey2 = self._enforce_scale_and_margins(
                                    (ex1, ey1, ex2, ey2),
                                    chosen_ratio,
                                    W2,
                                    H2,
                                    face_box=face_box_abs,
                                    anchor=(acx, acy),
                                )
                                ox1, oy1, ox2, oy2 = ex1 + off_x, ey1 + off_y, ex2 + off_x, ey2 + off_y
                                ox1 = max(0, min(W - 1, int(round(ox1))))
                                oy1 = max(0, min(H - 1, int(round(oy1))))
                                ox2 = max(ox1 + 1, min(W, int(round(ox2))))
                                oy2 = max(oy1 + 1, min(H, int(round(oy2))))
                                if ox2 > ox1 + 2 and oy2 > oy1 + 2:
                                    crop_img = frame[oy1:oy2, ox1:ox2]
                                    sharp = self._calc_sharpness(crop_img)
                                    sfx1 = max(0, min(W - 1, int(round(fx1 + off_x))))
                                    sfy1 = max(0, min(H - 1, int(round(fy1 + off_y))))
                                    sfx2 = max(sfx1 + 1, min(W, int(round(fx2 + off_x))))
                                    sfy2 = max(sfy1 + 1, min(H, int(round(fy2 + off_y))))
                                    subject_box_global = None
                                    if subject_box_abs is not None:
                                        sx1, sy1, sx2, sy2 = subject_box_abs
                                        sgx1 = max(0, min(W - 1, int(round(sx1 + off_x))))
                                        sgy1 = max(0, min(H - 1, int(round(sy1 + off_y))))
                                        sgx2 = max(sgx1 + 1, min(W, int(round(sx2 + off_x))))
                                        sgy2 = max(sgy1 + 1, min(H, int(round(sy2 + off_y))))
                                        subject_box_global = (sgx1, sgy1, sgx2, sgy2)
                                    head_box_global = None
                                    head_box_roi = self._face_head_proxy_box(face_box_abs, W2, H2)
                                    if head_box_roi is not None:
                                        hx1, hy1, hx2, hy2 = head_box_roi
                                        head_box_global = (
                                            max(0, min(W, hx1 + off_x)),
                                            max(0, min(H, hy1 + off_y)),
                                            max(0, min(W, hx2 + off_x)),
                                            max(0, min(H, hy2 + off_y)),
                                        )
                                    carea = max(1.0, float((ex2 - ex1) * (ey2 - ey1)))
                                    farea = max(1.0, (fx2 - fx1) * (fy2 - fy1))
                                    face_frac = float(farea) / carea
                                    fallback_candidate = dict(
                                        score=gfd,
                                        fd=gfd,
                                        rd=None,
                                        sharp=sharp,
                                        box=(ox1, oy1, ox2, oy2),
                                        area=(ox2 - ox1) * (oy2 - oy1),
                                        show_box=subject_box_global or (ox1, oy1, ox2, oy2),
                                        subject_box=subject_box_global,
                                        face_box=(sfx1, sfy1, sfx2, sfy2),
                                        head_box=head_box_global,
                                        face_feat=gbest["feat"],
                                        reid_feat=None,
                                        ratio=chosen_ratio,
                                        face_frac=face_frac,
                                        tloss=float(chosen_tloss),
                                        reasons=["global_face"],
                                        face_quality=float(gbest.get("quality", 0.0)),
                                    )

                    if (
                        fallback_candidate is None
                        and (not face_only_pipeline)
                        and cfg.allow_faceless_when_locked
                        and self._lock_active
                        and lock_hits >= int(cfg.lock_after_hits)
                        and self._seek_cooldown_frames <= 0
                    ):
                        persons_xywh = []
                        for bx1, by1, bx2, by2 in boxes:
                            ox1, oy1 = bx1 + off_x, by1 + off_y
                            ow, oh = (bx2 - bx1), (by2 - by1)
                            if ow <= 2 or oh <= 2:
                                continue
                            persons_xywh.append((ox1, oy1, ow, oh))
                        locked_feat = self._locked_reid_feat
                        reid_feats_fb = []
                        try:
                            reid_inst = getattr(self, "reid", None)
                            if reid_inst is not None and persons_xywh:
                                reid_feats_fb = [reid_inst.embed(frame, b) for b in persons_xywh]
                        except Exception:
                            reid_feats_fb = []
                        pick, why = self._faceless_pick_from_persons(persons_xywh, reid_feats_fb, locked_feat)
                        if pick >= 0 and pick < len(boxes):
                            if why != "reid" or pick >= len(reid_feats_fb) or reid_feats_fb[pick] is None:
                                pick, why = -1, None
                            if pick >= 0:
                                bx1, by1, bx2, by2 = boxes[pick]
                                sx1 = max(0, min(W - 1, int(round(bx1 + off_x))))
                                sy1 = max(0, min(H - 1, int(round(by1 + off_y))))
                                sx2 = max(sx1 + 1, min(W, int(round(bx2 + off_x))))
                                sy2 = max(sy1 + 1, min(H, int(round(by2 + off_y))))
                                (ex1, ey1, ex2, ey2), chosen_ratio, chosen_tloss = self._choose_best_ratio(
                                    (bx1, by1, bx2, by2), ratios, W2, H2, anchor=None
                                )
                                ex1, ey1, ex2, ey2 = self._enforce_scale_and_margins(
                                    (ex1, ey1, ex2, ey2),
                                    chosen_ratio,
                                    W2,
                                    H2,
                                    face_box=None,
                                    anchor=None,
                                )
                                ox1, oy1, ox2, oy2 = ex1 + off_x, ey1 + off_y, ex2 + off_x, ey2 + off_y
                                ox1 = max(0, min(W - 1, int(round(ox1))))
                                oy1 = max(0, min(H - 1, int(round(oy1))))
                                ox2 = max(ox1 + 1, min(W, int(round(ox2))))
                                oy2 = max(oy1 + 1, min(H, int(round(oy2))))
                                if ox2 > ox1 + 2 and oy2 > oy1 + 2:
                                    valid, _why = self._faceless_validate((ox1, oy1, ox2, oy2), frame.shape, gray)
                                    if valid:
                                        crop_img = frame[oy1:oy2, ox1:ox2]
                                        sharp = self._calc_sharpness(crop_img)
                                        if float(cfg.min_sharpness) <= 0 or sharp >= float(cfg.min_sharpness):
                                            reasons = [f"faceless_{why}" if why else "faceless"]
                                            fallback_candidate = dict(
                                                score=None,
                                                fd=None,
                                                rd=None,
                                                sharp=sharp,
                                                box=(ox1, oy1, ox2, oy2),
                                                area=(ox2 - ox1) * (oy2 - oy1),
                                                show_box=(sx1, sy1, sx2, sy2),
                                                face_feat=None,
                                                reid_feat=(reid_feats_fb[pick] if pick < len(reid_feats_fb) else None),
                                                ratio=chosen_ratio,
                                                face_frac=0.0,
                                                tloss=float(chosen_tloss),
                                                reasons=reasons,
                                                face_quality=None,
                                                accept_pre=True,
                                            )

                    if (
                        fallback_candidate is None
                        and (not face_only_pipeline)
                        and self._lock_last_bbox is not None
                        and (current_idx - self._lock_last_seen_idx) <= int(cfg.faceless_persist_frames)
                    ):
                        lx, ly, lw, lh = self._lock_last_bbox
                        x1 = max(0, min(W - 1, int(round(lx))))
                        y1 = max(0, min(H - 1, int(round(ly))))
                        x2 = max(x1 + 1, min(W, int(round(lx + lw))))
                        y2 = max(y1 + 1, min(H, int(round(ly + lh))))
                        if x2 > x1 + 2 and y2 > y1 + 2:
                            (ex1, ey1, ex2, ey2), chosen_ratio, chosen_tloss = self._choose_best_ratio(
                                (x1, y1, x2, y2), ratios, W, H, anchor=None
                            )
                            ex1, ey1, ex2, ey2 = self._enforce_scale_and_margins(
                                (ex1, ey1, ex2, ey2),
                                chosen_ratio,
                                W,
                                H,
                                face_box=None,
                                anchor=None,
                            )
                            ex1 = max(0, min(W - 1, int(round(ex1))))
                            ey1 = max(0, min(H - 1, int(round(ey1))))
                            ex2 = max(ex1 + 1, min(W, int(round(ex2))))
                            ey2 = max(ey1 + 1, min(H, int(round(ey2))))
                            if ex2 > ex1 + 2 and ey2 > ey1 + 2:
                                valid, _why = self._faceless_validate((ex1, ey1, ex2, ey2), frame.shape, gray)
                                if valid:
                                    crop_img = frame[ey1:ey2, ex1:ex2]
                                    sharp = self._calc_sharpness(crop_img)
                                    if float(cfg.min_sharpness) <= 0 or sharp >= float(cfg.min_sharpness):
                                        fallback_candidate = dict(
                                            score=None,
                                            fd=None,
                                            rd=None,
                                            sharp=sharp,
                                            box=(ex1, ey1, ex2, ey2),
                                            area=(ex2 - ex1) * (ey2 - ey1),
                                            show_box=(x1, y1, x2, y2),
                                            subject_box=(x1, y1, x2, y2),
                                            face_feat=None,
                                            reid_feat=None,
                                            ratio=chosen_ratio,
                                            face_frac=0.0,
                                            tloss=float(chosen_tloss),
                                            reasons=["faceless_carry"],
                                            face_quality=None,
                                            accept_pre=True,
                                        )

                if fallback_candidate is not None:
                    candidates = [fallback_candidate]
                else:
                    if not candidates:
                        reasons_to_log = (last_reject_reasons or [])[:6]
                        self._status(
                            f"reject_reasons={reasons_to_log}",
                            key="rej_reasons",
                            interval=1.0,
                        )
                    extra_note = " (face-only mode; faceless fallback disabled)" if face_only_pipeline else ""
                    self._status(
                        f"No match. persons={diag_persons} faces={faces_detected} pass_q={faces_passing_quality} "
                        f"visible={any_face_visible} thr={float(cfg.face_thresh):.3f} qmin={float(cfg.face_quality_min):.0f} "
                        f"best_fd_qonly={best_face_dist if best_face_dist is not None else 'n/a'} "
                        f"min_fd_all={min_fd_all if min_fd_all is not None else 'n/a'}{extra_note}",
                        key="no_match",
                        interval=1.0,
                    )
                chosen = None
                if candidates:
                    # Lock-aware scoring
                    # face margin check: chosen must be best face by a margin if faces are present
                    if cfg.prefer_face_when_available and any_face_visible:
                        # filter to face-bearing candidates
                        face_cands = [c for c in candidates if c.get('fd') is not None]
                        if len(face_cands) >= 2:
                            face_cands.sort(key=lambda d: d['fd'])
                            if (face_cands[1]['fd'] - face_cands[0]['fd']) < float(cfg.face_margin_min):
                                # ambiguous faces -> drop frame
                                last_reject_reasons = [
                                    "ambiguous_face_margin",
                                    *list(face_cands[0].get("reasons", [])),
                                ]
                                self._status(
                                    f"reject_reasons={last_reject_reasons[:6]}",
                                    key="rej_reasons",
                                    interval=1.0,
                                )
                                candidates = []
                    def eff_score(c):
                        s = c["score"] if c["score"] is not None else 1e9
                        # prefer higher sharpness slightly
                        return (s, -c["area"], -c["sharp"]) 

                    candidates.sort(key=lambda d: eff_score(d))

                    # Disambiguation margin vs #2
                    if len(candidates) >= 2 and candidates[0]["score"] is not None and candidates[1]["score"] is not None:
                        if abs(candidates[0]["score"] - candidates[1]["score"]) < float(cfg.score_margin):
                            candidates = [c for c in candidates if c is candidates[0]]  # keep best only

                    now_t = current_idx/float(fps)

                    # Lock logic: after N hits tighten thresholds and require IoU gate
                    use_lock = (
                        self._seek_cooldown_frames <= 0
                        and lock_hits >= int(cfg.lock_after_hits)
                        and (locked_face is not None or locked_reid is not None)
                    )
                    chosen = None
                    for c in candidates:
                        if use_lock:
                            ok = True
                            if locked_face is not None and c.get("fd") is not None:
                                ok = ok and (c["fd"] <= float(cfg.lock_face_thresh))
                            if locked_reid is not None and c.get("rd") is not None:
                                ok = ok and (c["rd"] <= float(cfg.lock_reid_thresh))
                            if prev_box is not None:
                                if self._iou(prev_box, c["box"]) < float(cfg.iou_gate):
                                    ok = False
                            if not ok:
                                continue
                        chosen = c
                        break

                    if chosen is None and candidates:
                        chosen = candidates[0]

                    if chosen is not None:
                        show_box = chosen.get("show_box") or chosen.get("box")
                        if show_box is not None and len(show_box) == 4:
                            px1, py1, px2, py2 = [int(round(v)) for v in show_box]
                            self._set_lock((px1, py1, px2 - px1, py2 - py1), current_idx)
                        if chosen.get("reid_feat") is not None:
                            try:
                                self._locked_reid_feat = np.asarray(chosen["reid_feat"], dtype=np.float32)
                            except Exception:
                                pass
                    if (
                        chosen is not None
                        and now_t - self._last_hit_t >= float(cfg.min_gap_sec)
                    ):
                        if save_hit(
                            chosen,
                            current_idx,
                            frame_w=W,
                            frame_h=H,
                            det_off_x=off_x,
                            det_off_y=off_y,
                            det_w=W2,
                            det_h=H2,
                            repair_bounds_xyxy=(
                                int(base_det_off_x),
                                int(base_det_off_y),
                                int(base_det_off_x + base_det_w),
                                int(base_det_off_y + base_det_h),
                            ),
                            ratio_list=ratios,
                        ):
                            hit_count += 1
                            self._last_hit_t = now_t

                preview_due = bool(force_process)
                if cfg.preview_every > 0:
                    if force_process or (current_idx - last_preview_emit) >= int(cfg.preview_every):
                        preview_due = True
                        last_preview_emit = current_idx
                elif preview_due:
                    last_preview_emit = current_idx

                # Annotated preview
                if cfg.save_annot or preview_due:
                    show = frame.copy()
                    for fx1, fy1, fx2, fy2, q, fd_box in face_debug_boxes:
                        color = (0, 255, 0) if q >= float(cfg.face_quality_min) else (0, 165, 255)
                        cv2.rectangle(show, (int(fx1), int(fy1)), (int(fx2), int(fy2)), color, 1)
                        label = f"q {q:.0f}"
                        if getattr(cfg, "overlay_face_fd", True) and fd_box is not None:
                            label = f"{label} fd {fd_box:.2f}"
                        cv2.putText(
                            show,
                            label,
                            (int(fx1), max(0, int(fy1) - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            color,
                            1,
                            cv2.LINE_AA,
                        )
                    H, W = show.shape[:2]
                    # draw person boxes
                    for c in candidates:
                        sb = c.get("show_box") or c.get("box")
                        if not sb or len(sb) != 4:
                            continue
                        vals = np.asarray(sb, dtype=float)
                        if not np.isfinite(vals).all():
                            continue
                        x1, y1, x2, y2 = [int(round(v)) for v in vals]
                        x1 = max(0, min(W - 1, x1))
                        y1 = max(0, min(H - 1, y1))
                        x2 = max(0, min(W - 1, x2))
                        y2 = max(0, min(H - 1, y2))
                        if x2 <= x1 or y2 <= y1:
                            continue
                        cv2.rectangle(show, (x1, y1), (x2, y2), (0,255,0), 1)
                        if getattr(cfg, "overlay_scores", True):
                            txt = []
                            if c.get("fd") is not None:
                                txt.append(f"fd {c['fd']:.2f}")
                            if c.get("rd") is not None:
                                txt.append(f"rd {c['rd']:.2f}")
                            if c.get("face_quality") is not None:
                                txt.append(f"q {c['face_quality']:.0f}")
                            if txt:
                                cv2.putText(
                                    show,
                                    " | ".join(txt),
                                    (int(x1), max(0, int(y1) - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.45,
                                    (0, 255, 0),
                                    1,
                                    cv2.LINE_AA,
                                )
                    if candidates:
                        selected_preview = next((cc for cc in candidates if cc.get("selected")), candidates[0])
                        bvals = np.asarray(selected_preview.get("saved_box") or selected_preview.get("box"), dtype=float)
                        if np.isfinite(bvals).all():
                            bx1, by1, bx2, by2 = [int(round(v)) for v in bvals]
                            bx1 = max(0, min(W - 1, bx1))
                            by1 = max(0, min(H - 1, by1))
                            bx2 = max(0, min(W - 1, bx2))
                            by2 = max(0, min(H - 1, by2))
                            if bx2 > bx1 and by2 > by1:
                                cv2.rectangle(show, (bx1, by1), (bx2, by2), (255,0,0), 2)
                                rs = selected_preview.get("reasons")
                                if rs:
                                    cv2.putText(
                                        show,
                                        str(rs[0]),
                                        (int(bx1), max(0, int(by1) - 5)),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.45,
                                        (255, 0, 0),
                                        1,
                                        cv2.LINE_AA,
                                    )
                    if ann_dir and cfg.save_annot:
                        ann_path = os.path.join(ann_dir, f"f{current_idx:08d}.jpg")
                        ok, why = _atomic_jpeg_write(show, ann_path, jpg_q)
                        if not ok:
                            self._status(
                                f"Annot save failed ({why}): {ann_path}",
                                key="save_err_ann",
                                interval=1.0,
                            )
                    self._emit_preview_bgr(show)

                # drain any pending hit notifications (async saver)
                if hit_q is not None:
                    try:
                        while True:
                            _p = hit_q.get_nowait()
                            try:
                                self.hit.emit(_p)
                            finally:
                                hit_q.task_done()
                    except Exception:
                        pass
                # Always-on preview cadence
                if preview_due:
                    base = frame.copy()
                    self._emit_preview_bgr(base)

                # single progress update per loop
                self._prev_gray = gray.copy()
                self.progress.emit(current_idx)
                frame_idx = current_idx + 1
                if self._seek_cooldown_frames > 0:
                    if self._seek_cooldown_frames == _cooldown_frames():
                        self._status("Lock cooldown active", key="cooldown", interval=0.5)
                    self._seek_cooldown_frames -= 1
                if self._lock_active:
                    lose_after = max(int(cfg.faceless_persist_frames) * 2, _frame_stride() * 6)
                    if current_idx - self._lock_last_seen_idx > lose_after:
                        self._clear_lock()
                if self._speed and self._speed > 0:
                    time.sleep(max(0.0, (1.0 / float(fps)) / float(self._speed)))
                # Per-frame debug dump
                if dbg_f:
                    _dump_debug(
                        {
                            "frame": current_idx,
                            "persons": int(diag_persons),
                            "faces_detected": int(faces_detected),
                            "faces_pass_quality": int(faces_passing_quality),
                            "any_face_detected": bool(any_face_detected),
                            "any_face_visible": bool(any_face_visible),
                            "min_fd_all": (float(min_fd_all) if min_fd_all is not None else None),
                            "best_face_dist": (float(best_face_dist) if best_face_dist is not None else None),
                            "cfg": {
                                "face_det_conf": float(cfg.face_det_conf),
                                "face_det_pad": float(cfg.face_det_pad),
                                "face_thresh": float(cfg.face_thresh),
                                "reid_thresh": float(cfg.reid_thresh),
                                "face_quality_min": float(cfg.face_quality_min),
                                "face_visible_uses_quality": bool(cfg.face_visible_uses_quality),
                                "prefer_face_when_available": bool(cfg.prefer_face_when_available),
                                "require_face_if_visible": bool(cfg.require_face_if_visible),
                                "match_mode": str(cfg.match_mode),
                                "allow_faceless_when_locked": bool(cfg.allow_faceless_when_locked),
                                "faceless_reid_thresh": float(cfg.faceless_reid_thresh),
                                "faceless_iou_min": float(cfg.faceless_iou_min),
                                "faceless_persist_frames": int(cfg.faceless_persist_frames),
                                "faceless_min_area_frac": float(cfg.faceless_min_area_frac),
                                "faceless_max_area_frac": float(cfg.faceless_max_area_frac),
                                "faceless_center_max_frac": float(cfg.faceless_center_max_frac),
                                "faceless_min_motion_frac": float(cfg.faceless_min_motion_frac),
                                "learn_bank_runtime": bool(cfg.learn_bank_runtime),
                                "drop_reid_if_any_face_match": bool(cfg.drop_reid_if_any_face_match),
                            },
                            "candidates": [
                                {
                                    "fd": (float(c["fd"]) if c["fd"] is not None else None),
                                    "rd": (float(c["rd"]) if c["rd"] is not None else None),
                                    "sharp": float(c["sharp"]),
                                    "box": [int(v) for v in c["box"]],
                                    "reasons": c.get("reasons", []),
                                }
                                for c in candidates
                            ],
                        }
                    )
            if self._abort:
                finished_ok, finished_msg = False, "Aborted"
            else:
                finished_ok, finished_msg = True, f"Done. Hits: {hit_count}. Index: {csv_path}"
        except Exception as e:
            err = f"Error: {e}\n{traceback.format_exc()}"
            finished_ok, finished_msg = False, err
        finally:
            if hit_q is not None:
                try:
                    while True:
                        _p = hit_q.get_nowait()
                        self._emit_hit(_p)
                        hit_q.task_done()
                except queue.Empty:
                    pass
            if save_q is not None:
                save_q.put(None)
                save_q.join()
            if saver_thread is not None:
                saver_thread.join()
            # Saver acks can arrive before hit enqueue; drain once more after saver exit.
            if hit_q is not None:
                try:
                    while True:
                        _p = hit_q.get_nowait()
                        self._emit_hit(_p)
                        hit_q.task_done()
                except queue.Empty:
                    pass
            if archive_q is not None:
                archive_q.put(None)
                archive_q.join()
            if archive_thread is not None:
                archive_thread.join()
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            self._hdr_preview_close()
            if csv_f is not None:
                try:
                    csv_f.close()
                except Exception:
                    pass
            if dbg_f:
                try:
                    dbg_f.close()
                except Exception:
                    pass
            self.finished.emit(bool(finished_ok), str(finished_msg))

    def _init_status(self):
        # Per-key throttle timestamps and last texts
        self._status_last_time = {}
        self._status_last_text = {}

    # ---------- Smart crop helpers ----------
    def _smart_crop_box(self, frame, box, face_box, ratio_str, cfg):
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box]
        x1 = max(0, x1); y1 = max(0, y1); x2 = min(W-1, x2); y2 = min(H-1, y2)
        if x2 <= x1+2 or y2 <= y1+2:
            return x1, y1, x2, y2
        # parse ratio
        try:
            tw, th = parse_ratio(ratio_str)
            target = float(tw) / float(th)
        except Exception:
            target = 2.0 / 3.0
        w = x2 - x1; h = y2 - y1
        # ensure exact ratio by adjusting height
        h = int(round(max(2, w / max(1e-6, target))))
        cy = (y1 + y2) // 2
        y1 = max(0, min(H - h, cy - h // 2)); y2 = y1 + h

        skip_side_search = False

        # Build a downscaled gradient saliency once (cheap)
        if bool(getattr(cfg, "smart_crop_use_grad", True)):
            small_w = min(384, W)
            scale = (small_w / float(W)) if W > 0 else 1.0
            small_h = max(8, int(round(H * scale)))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            small = cv2.resize(gray, (small_w, small_h), interpolation=interp)
            gradx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
            grady = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
            sal = cv2.magnitude(gradx, grady)
        else:
            sal = None; scale = 1.0

        # Dynamic scale so face ≤ max_frac depending on profile-ness
        if face_box is not None:
            fx1, fy1, fx2, fy2 = [int(v) for v in face_box]
            fx1 = max(0, fx1); fy1 = max(0, fy1); fx2 = min(W-1, fx2); fy2 = min(H-1, fy2)
            fw = max(1, fx2 - fx1)
            # proxy for profile: face near crop side -> lower max_frac
            rel = None
            cw_left, cw_right = x1, x2
            fcx = (fx1 + fx2) * 0.5
            if cw_right > cw_left:
                rel = (fcx - cw_left) / float(cw_right - cw_left)  # 0..1
            profile = 2.0 * abs((rel if rel is not None else 0.5) - 0.5)  # 0=frontal, 1=profile
            max_frac = 0.42 - 0.12 * profile  # 0.42 frontal → 0.30 profile
            need_w = int(math.ceil(fw / max(1e-6, max_frac)))
            # also ensure requested side margins around face
            want_side = float(getattr(cfg, "crop_face_side_margin_frac", 0.30)) * fw
            need_w = max(need_w, int(round(fw + 2.0 * want_side)))
            if need_w > w:
                w = min(W, need_w)
                h = int(round(max(2, w / target)))
            # center on face horizontally by default
            cx = int(round(fcx))
            x1 = max(0, min(W - w, cx - w // 2)); x2 = x1 + w
            y1 = max(0, min(H - h, cy - h // 2)); y2 = y1 + h
            # when a face is known, skip lateral saliency drift to avoid off-centering
            sal = None
            skip_side_search = True

        # Lateral search to maximize saliency and keep face centered with margin
        steps_cfg = int(getattr(cfg, "smart_crop_steps", 6))
        sfrac = float(getattr(cfg, "smart_crop_side_search_frac", 0.35))
        if skip_side_search:
            steps = 0
            max_shift = 0
        else:
            steps = max(0, steps_cfg)
            max_shift = int(round(w * sfrac))
        best = (x1, y1, x2, y2); best_score = -1e9
        if steps <= 0 or max_shift <= 0:
            dx_vals = (0.0,)
        else:
            dx_vals = np.linspace(-max_shift, max_shift, 2 * steps + 1)
        for dx in dx_vals:
            nx1 = int(max(0, min(W - w, x1 + dx))); nx2 = nx1 + w
            # saliency score
            if sal is not None:
                sx1 = int(nx1 * scale); sx2 = int(nx2 * scale)
                sy1 = int(y1 * scale);  sy2 = int(y2 * scale)
                s_patch = sal[sy1:sy2, sx1:sx2]
                s_val = float(np.mean(s_patch)) if s_patch.size else 0.0
            else:
                s_val = 0.0
            # margin score: prefer balanced headroom around face if available
            m_val = 0.0
            if face_box is not None:
                fx1, _, fx2, _ = face_box
                fcx = 0.5 * (fx1 + fx2)
                left_m = max(0.0, fcx - nx1); right_m = max(0.0, nx2 - fcx)
                m_val = min(left_m, right_m) / max(1.0, w)
            score = s_val + 0.15 * m_val
            if score > best_score:
                best_score = score; best = (nx1, y1, nx2, y2)
        return best

    def _status(self, msg: str, key: str = None, interval: float = None):
        """
        Thread-safe-ish status throttle.
        key: logical channel ('phase','det','nomatch', etc.). None uses a global key.
        interval: seconds between repeats. Defaults to cfg.log_interval_sec.
        """
        k = key or "_global"
        now = time.time()
        iv = float(interval if interval is not None else getattr(self.cfg, 'log_interval_sec', 1.0))
        last_t = self._status_last_time.get(k, 0.0)
        last_txt = self._status_last_text.get(k, None)
        if (now - last_t) >= iv or msg != last_txt:
            self.status.emit(msg)
            self._status_last_time[k] = now
            self._status_last_text[k] = msg

    def _resolve_ffmpeg_bin(self) -> Optional[str]:
        ffmpeg_cached = getattr(self, "_ffmpeg_cached", None)
        if isinstance(ffmpeg_cached, str) and ffmpeg_cached:
            return ffmpeg_cached

        try:
            reader = getattr(self, "_hdr_preview_reader", None)
            ffmpeg_reader = getattr(reader, "_ffmpeg", None)
            if isinstance(ffmpeg_reader, str) and ffmpeg_reader:
                self._ffmpeg_cached = ffmpeg_reader
                return ffmpeg_reader
        except Exception:
            pass

        _ffp = None
        try:
            from .video_io import _ffmpeg_path as _ffp  # type: ignore
        except Exception:
            try:
                from video_io import _ffmpeg_path as _ffp  # type: ignore
            except Exception:
                _ffp = None
        if _ffp is not None:
            try:
                ffmpeg = _ffp()
                if ffmpeg:
                    self._ffmpeg_cached = ffmpeg
                    return ffmpeg
            except Exception:
                pass
        return None

    def _capture_source_size(self, cap, frame_shape: tuple[int, int]) -> tuple[int, int]:
        """Return original source dimensions, not the possibly downscaled reader size."""
        try:
            if cap is not None:
                source_size = getattr(cap, "_source_size", None)
                if callable(source_size):
                    sw, sh = source_size()
                    if int(sw) > 0 and int(sh) > 0:
                        return int(sw), int(sh)
                sw = int(getattr(cap, "_source_w", 0) or 0)
                sh = int(getattr(cap, "_source_h", 0) or 0)
                if sw > 0 and sh > 0:
                    return sw, sh
        except Exception:
            pass
        try:
            ffmpeg_probe = None
            try:
                from .video_io import _ffprobe_json as ffmpeg_probe  # type: ignore
            except Exception:
                from video_io import _ffprobe_json as ffmpeg_probe  # type: ignore
            meta = ffmpeg_probe(self.cfg.video) if ffmpeg_probe is not None else {}
            stream = (meta.get("streams") or [{}])[0]
            sw = int(stream.get("width") or 0)
            sh = int(stream.get("height") or 0)
            if sw > 0 and sh > 0:
                return sw, sh
        except Exception:
            pass
        fh, fw = int(frame_shape[0]), int(frame_shape[1])
        return max(1, fw), max(1, fh)

    @staticmethod
    def _capture_frame_pts_sec(cap, frame_idx: int, fps: float) -> Optional[float]:
        """Return frame timestamp seconds, preferring capture-reported time over frame_idx/fps."""
        try:
            if cap is not None and hasattr(cap, "get"):
                pos_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
                if math.isfinite(pos_msec) and (pos_msec > 0.0 or int(frame_idx) == 0):
                    return pos_msec / 1000.0
        except Exception:
            pass
        if fps > 0.0 and math.isfinite(fps):
            return max(0.0, float(frame_idx) / float(fps))
        return None

    @staticmethod
    def _scale_crop_xyxy_to_source(
        crop_xyxy: tuple[int, int, int, int],
        frame_size: tuple[int, int],
        source_size: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        """Map processed-frame crop coordinates back to original source pixels."""
        fw, fh = max(1, int(frame_size[0])), max(1, int(frame_size[1]))
        sw, sh = max(1, int(source_size[0])), max(1, int(source_size[1]))
        sx = float(sw) / float(fw)
        sy = float(sh) / float(fh)
        x1, y1, x2, y2 = crop_xyxy
        ox1 = int(round(float(x1) * sx))
        oy1 = int(round(float(y1) * sy))
        ox2 = int(round(float(x2) * sx))
        oy2 = int(round(float(y2) * sy))
        ox1 = max(0, min(sw - 1, ox1))
        oy1 = max(0, min(sh - 1, oy1))
        ox2 = max(ox1 + 1, min(sw, ox2))
        oy2 = max(oy1 + 1, min(sh, oy2))
        return ox1, oy1, ox2, oy2

    @staticmethod
    def _even_hdr_crop_xyxy(crop_xyxy: tuple[int, int, int, int], source_size: tuple[int, int]) -> tuple[int, int, int, int]:
        """Make 4:2:0 HDR still/video crops legal without moving far from the chosen box."""
        def _legalize_axis(a1: int, a2: int, limit: int) -> tuple[int, int]:
            # 4:2:0-safe crop: even origin, even extent, in-bounds, size >= 2.
            a1 = max(0, min(limit - 2, a1 & ~1))
            a2 = max(a1 + 2, min(limit, a2))
            if (a2 - a1) & 1:
                if a2 < limit:
                    a2 += 1
                elif a2 > a1 + 2:
                    a2 -= 1
                elif a1 >= 2:
                    a1 -= 2
                else:
                    a2 = min(limit, a1 + 2)
            if a1 & 1:
                if a1 + 1 <= limit - 2:
                    a1 += 1
                    a2 = max(a1 + 2, min(limit, a2 + 1))
                else:
                    a1 -= 1
            a1 = max(0, min(limit - 2, a1 & ~1))
            a2 = max(a1 + 2, min(limit, a2))
            if (a2 - a1) & 1:
                a2 = max(a1 + 2, min(limit, a2 - 1))
            return a1, a2

        sw, sh = max(2, int(source_size[0])), max(2, int(source_size[1]))
        x1, y1, x2, y2 = [int(v) for v in crop_xyxy]
        x1, x2 = _legalize_axis(x1, x2, sw)
        y1, y2 = _legalize_axis(y1, y2, sh)
        return x1, y1, x2, y2

    def _ffmpeg_libplacebo_options(self, ffmpeg_bin: str) -> set[str]:
        """Return supported ffmpeg libplacebo option names for this binary."""
        cache = getattr(self, "_ffmpeg_lp_opts_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            self._ffmpeg_lp_opts_cache = cache
        cached = cache.get(ffmpeg_bin)
        if isinstance(cached, set):
            return cached
        opts: set[str] = set()
        try:
            cp = subprocess.run(
                [ffmpeg_bin, "-hide_banner", "-h", "filter=libplacebo"],
                text=True,
                capture_output=True,
                check=False,
                timeout=10,
            )
            text = (cp.stdout or "") + "\n" + (cp.stderr or "")
            for line in text.splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                name = parts[0].strip()
                if name and all((ch.isalnum() or ch == "_") for ch in name):
                    opts.add(name)
        except Exception as exc:
            self._status(
                f"HDR libplacebo option probe failed: {exc}",
                key="hdr_lp_probe",
                interval=30.0,
            )
        cache[ffmpeg_bin] = opts
        return opts

    @staticmethod
    def _add_lp_opt(opts: list[str], supported: set[str], name: str, value: str) -> bool:
        if supported and name not in supported:
            return False
        opts.append(f"{name}={value}")
        return True

    def _ffmpeg_still_seek_args_and_filter(self, seek_sec: Optional[float]) -> tuple[list[str], str]:
        """Return FFmpeg input seek args and a filter-prefix for exact still export.

        Do not use a second output-side ``-ss`` for HDR still exports. With some
        HEVC/MKV streams, especially after bounded pre-roll, FFmpeg can legally
        complete with a tiny/empty image muxer output when no decoded frame
        survives the output seek at the requested timestamp. Decode from a short
        pre-roll and perform the exact selection inside the filter graph instead,
        before tone mapping/cropping.
        """
        if seek_sec is None:
            return [], ""
        try:
            seek_f = max(0.0, float(seek_sec))
        except Exception:
            return [], ""
        try:
            preroll_sec = max(0.0, float(os.getenv("PC_HDR_EXPORT_PREROLL_SEC", "2.0")))
        except Exception:
            preroll_sec = 2.0
        preroll_sec = min(seek_f, preroll_sec)
        if preroll_sec > 1e-6:
            start_offset = max(0.0, seek_f - preroll_sec)
            seek_args = ["-ss", f"{start_offset:.6f}"] if start_offset > 1e-6 else []
            return seek_args, f"trim=start={preroll_sec:.6f},setpts=PTS-STARTPTS,"
        return [], f"trim=start={seek_f:.6f},setpts=PTS-STARTPTS,"

    @staticmethod
    def _validated_jpeg_quality(cfg: object, default: int = 95) -> int:
        try:
            return max(1, min(100, int(getattr(cfg, "jpg_quality", default))))
        except Exception:
            return default

    @staticmethod
    def _validate_hdr_sdr_export_image(
        path: str,
        expected_size: Optional[tuple[int, int]] = None,
    ) -> tuple[bool, str]:
        """Reject missing/corrupt/black HDR still exports without a byte-size heuristic.

        A valid high-quality JPEG of a small, very dark crop can be below 1 KiB.
        Rejecting by file size caused successful FFmpeg still renders to be
        treated as failed exports. Decode first, then validate image shape and
        near-black failure characteristics.
        """
        try:
            if not path or not os.path.exists(path):
                return False, "missing_output"
            if os.path.getsize(path) <= 16:
                return False, "empty_output"
            data = np.fromfile(path, dtype=np.uint8)
            if data.size <= 16:
                return False, "empty_output"
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None or img.ndim != 3 or img.size == 0:
                return False, "decode_failed"
            ih, iw = int(img.shape[0]), int(img.shape[1])
            if expected_size is not None:
                ew, eh = int(expected_size[0]), int(expected_size[1])
                if ew > 0 and eh > 0 and (abs(iw - ew) > 2 or abs(ih - eh) > 2):
                    return False, f"wrong_size got={iw}x{ih} expected={ew}x{eh}"
            y = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mean = float(np.mean(y))
            p95 = float(np.percentile(y, 95.0))
            p99 = float(np.percentile(y, 99.0))
            # Real candle-lit scenes are often dark; failed seek/decode frames are
            # effectively all black. Keep the rejection threshold deliberately low.
            if mean < 1.0 and p95 < 3.0 and p99 < 8.0:
                return False, f"near_black_output mean={mean:.3f} p95={p95:.3f} p99={p99:.3f}"
            return True, ""
        except Exception as exc:
            return False, f"validate_failed:{exc}"

    def _hdr_tonemap_filter_cmds(
        self,
        ffmpeg_bin: str,
        frame_idx: int,
        frame_pts_sec: Optional[float],
        crop_xyxy: tuple[int, int, int, int],
        out_path: str,
    ) -> list[list[str]]:
        """Build preferred-to-fallback full-res HDR→SDR still export commands."""
        x1, y1, x2, y2 = [int(v) for v in crop_xyxy]
        w = max(1, int(x2 - x1))
        h = max(1, int(y2 - y1))
        seek_sec: Optional[float] = None
        try:
            if frame_pts_sec is not None and math.isfinite(float(frame_pts_sec)):
                seek_sec = max(0.0, float(frame_pts_sec))
        except Exception:
            seek_sec = None
        if seek_sec is None:
            fps = float(getattr(self, "_fps", 0.0) or 0.0)
            if fps > 0 and math.isfinite(fps):
                seek_sec = max(0.0, float(frame_idx) / fps)
        pre_seek_args, seek_filter = self._ffmpeg_still_seek_args_and_filter(seek_sec)
        base = [ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y"]
        base += pre_seek_args
        base += ["-i", self.cfg.video, "-map", "0:v:0"]
        out_once = ["-frames:v", "1"]
        crop = f"crop={w}:{h}:{x1}:{y1}"
        src = "setparams=color_trc=smpte2084:color_primaries=bt2020:colorspace=bt2020nc:range=limited"
        desat = float(getattr(self.cfg, "tm_desat", 0.25))
        param = float(getattr(self.cfg, "tm_param", 0.40))
        nits = float(getattr(self.cfg, "sdr_nits", SessionConfig.sdr_nits))
        pref = str(getattr(self.cfg, "hdr_tonemap_pref", "auto") or "auto").lower()
        quality = str(getattr(self.cfg, "hdr_sdr_quality", "madvr_like") or "madvr_like").lower()
        algo = str(getattr(self.cfg, "hdr_sdr_tonemap", "auto") or "auto").lower()
        if algo not in {"auto", "spline", "bt.2390", "st2094-40", "mobius", "hable", "reinhard", "clip"}:
            algo = "auto"
        gamut = str(getattr(self.cfg, "hdr_sdr_gamut_mapping", "clip") or "clip").lower()
        if gamut not in {"perceptual", "relative", "saturation", "clip"}:
            gamut = "clip"
        try:
            contrast_recovery = max(0.0, min(2.0, float(getattr(self.cfg, "hdr_sdr_contrast_recovery", 0.30))))
        except Exception:
            contrast_recovery = 0.30
        peak_detect = bool(getattr(self.cfg, "hdr_sdr_peak_detect", True))
        allow_inaccurate = bool(getattr(self.cfg, "hdr_sdr_allow_inaccurate_fallback", False))

        jpg_quality = self._validated_jpeg_quality(self.cfg, default=95)
        q = max(1, min(31, int(round((100 - jpg_quality) / 5.0 + 1))))
        if quality in ("resolve_like", "madvr_like"):
            # Full-res still export is the final dataset artifact, not a fast
            # preview cache. Keep it at the visually lossless end of FFmpeg's
            # MJPEG scale; q=2 is still visibly softer in very dark, low-texture
            # faces once the UI or training tooling magnifies the crop.
            q = 1
        enc = [
            "-an", "-sn", "-dn",
            "-c:v", "mjpeg",
            "-q:v", str(q),
            "-qmin", "1",
            "-qmax", str(q),
            "-pix_fmt", "yuvj444p",
            "-color_range", "pc",
            "-colorspace", "bt709",
            "-color_primaries", "bt709",
            "-color_trc", "bt709",
            "-update", "1",
            out_path,
        ]

        filters = ffmpeg_has_hdr_filters(ffmpeg_bin)
        cmds: list[list[str]] = []
        if filters.get("libplacebo") and pref in ("auto", "libplacebo"):
            supported = self._ffmpeg_libplacebo_options(ffmpeg_bin)

            def _add_first_lp_opt(opts: list[str], names: tuple[str, ...], value: str) -> bool:
                for name in names:
                    if self._add_lp_opt(opts, supported, name, value):
                        return True
                return False

            def _add_output_color_opts(opts: list[str]) -> None:
                self._add_lp_opt(opts, supported, "colorspace", "bt709")
                # Prefer target_* names when available: color_* is accepted by older
                # ffmpeg builds but can be interpreted as the working frame metadata.
                _add_first_lp_opt(opts, ("target_primaries", "color_primaries"), "bt709")
                _add_first_lp_opt(opts, ("target_trc", "color_trc"), "bt709")
                self._add_lp_opt(opts, supported, "range", "full")
                _add_first_lp_opt(opts, ("sdr_peak", "target_peak", "dst_peak", "peak"), f"{nits:.6g}")
                if not _add_first_lp_opt(opts, ("desaturation", "desat"), f"{desat:.6g}"):
                    sat = max(0.0, 1.0 - desat)
                    _add_first_lp_opt(opts, ("saturation", "sat"), f"{sat:.6g}")
                _add_first_lp_opt(opts, ("gamut_mode", "gamut_mapping"), gamut)

            # Match the live HDR preview reader's libplacebo algorithm resolution:
            # GUI/default "auto" does not mean pass literal auto to the still
            # renderer. The live reader starts with BT.2390, then tries aliases if
            # a build dislikes the dotted spelling. Passing literal auto here made
            # saved crops use a different curve than the preview.
            if algo == "auto":
                # Keep still-export fallback order aligned with live preview:
                # bt.2390 -> mobius -> hable -> clip, with bt2390 alias handling.
                lp_algos = ["bt.2390", "bt2390", "mobius", "hable", "clip"]
            else:
                lp_algos = [algo]
                if algo in {"bt.2390", "bt2390"}:
                    lp_algos.append("bt2390" if algo == "bt.2390" else "bt.2390")

            lp_variants: list[list[str]] = []
            for lp_algo in lp_algos:
                base_lp_opts = [f"tonemapping='{lp_algo}'"]
                _add_output_color_opts(base_lp_opts)
                _add_first_lp_opt(base_lp_opts, ("format", "out_format", "out_pfmt"), "bgra")
                self._add_lp_opt(base_lp_opts, supported, "peak_detect", "true" if peak_detect else "false")
                if lp_algo in {"mobius", "hable", "reinhard", "gamma"}:
                    self._add_lp_opt(base_lp_opts, supported, "tonemapping_param", f"{param:.6g}")
                if quality in ("resolve_like", "madvr_like"):
                    self._add_lp_opt(base_lp_opts, supported, "peak_detection_preset", "high_quality")
                    self._add_lp_opt(base_lp_opts, supported, "color_map_preset", "high_quality")
                    self._add_lp_opt(base_lp_opts, supported, "contrast_recovery", f"{contrast_recovery:.6g}")
                    self._add_lp_opt(base_lp_opts, supported, "contrast_smoothness", "3.5")
                    # Do not enable libplacebo debanding for dataset stills. It is
                    # useful for playback gradients, but on close face crops it
                    # smooths skin/eye detail and makes the saved crop look worse
                    # than the live preview.
                    if not self._add_lp_opt(base_lp_opts, supported, "tonemapping_lut_size", "1024"):
                        self._add_lp_opt(base_lp_opts, supported, "tone_lut_size", "1024")
                    # Match the live preview renderer path: ordered dithering, not a
                    # separate still-export blue-noise path that changes dark gradients.
                    if not self._add_lp_opt(base_lp_opts, supported, "dithering", "ordered"):
                        self._add_lp_opt(base_lp_opts, supported, "dither_method", "ordered")
                elif quality == "balanced":
                    self._add_lp_opt(base_lp_opts, supported, "contrast_recovery", f"{min(contrast_recovery, 0.20):.6g}")
                    self._add_lp_opt(base_lp_opts, supported, "tonemapping_lut_size", "512")
                    if not self._add_lp_opt(base_lp_opts, supported, "dithering", "ordered"):
                        self._add_lp_opt(base_lp_opts, supported, "dither_method", "ordered")
                lp_variants.append(base_lp_opts)

                minimal_lp_opts = [f"tonemapping='{lp_algo}'"]
                _add_output_color_opts(minimal_lp_opts)
                _add_first_lp_opt(minimal_lp_opts, ("format", "out_format", "out_pfmt"), "bgra")
                if not self._add_lp_opt(minimal_lp_opts, supported, "dithering", "ordered"):
                    self._add_lp_opt(minimal_lp_opts, supported, "dither_method", "ordered")
                lp_variants.append(minimal_lp_opts)

            seen_lp: set[str] = set()
            for opts in lp_variants:
                opt_str = ":".join(opts)
                if opt_str in seen_lp:
                    continue
                seen_lp.add(opt_str)
                # Renderer-equivalent order: tone-map the full source frame first,
                # then crop in SDR/RGB. Cropping before tone mapping changes the
                # peak/gamut context and can misalign 4:2:0 chroma at arbitrary crop
                # offsets, causing visible artifacts versus madVR/MPCVR-style output.
                # libplacebo still outputs hardware frames; always download before
                # the CPU-only format/crop/encoder path.  The libplacebo ``format``
                # option chooses the rendered pixel format, it is not a readback.
                lp = (
                    f"{seek_filter}format=p010le,{src},"
                    f"hwupload=extra_hw_frames=1,libplacebo={opt_str},"
                    f"hwdownload,format=bgra,format=bgr24,{crop}"
                )
                cmds.append(
                    base[:1]
                    + ["-init_hw_device", "vulkan=vk:0", "-filter_hw_device", "vk"]
                    + base[1:]
                    + ["-vf", lp]
                    + out_once
                    + enc
                )
        if filters.get("zscale") and filters.get("tonemap") and pref in ("auto", "zscale"):
            # CPU fallback. It is slower, but it only runs on accepted captures and
            # preserves the main invariant: full-resolution source crop, not preview crop.
            z_algo_map = {
                "auto": "mobius",
                "mobius": "mobius",
                "hable": "hable",
                "reinhard": "reinhard",
                "clip": "clip",
                "bt.2390": "reinhard",
                "spline": "mobius",
                "st2094-40": "hable",
            }
            z_algo = z_algo_map.get(algo, "mobius")
            z_peak = nits if peak_detect else max(100.0, nits)
            z_param = param
            z_desat = desat
            z_dither = "error_diffusion"
            if quality in ("resolve_like", "madvr_like"):
                z_dither = "error_diffusion"
                z_param = min(1.0, z_param + 0.30 * contrast_recovery)
            elif quality == "balanced":
                z_dither = "ordered"
                z_param = min(1.0, z_param + 0.15 * contrast_recovery)
                z_desat = min(1.0, z_desat + 0.05)
            elif quality == "fast":
                z_dither = "none"
                z_param = min(1.0, z_param + 0.05 * contrast_recovery)
                z_desat = min(1.0, z_desat + 0.10)
            if gamut == "clip":
                z_desat = min(1.0, z_desat + 0.15)
            elif gamut == "saturation":
                z_desat = max(0.0, z_desat - 0.10)
            elif gamut == "relative":
                z_desat = max(0.0, z_desat - 0.05)
            zf = (
                f"{seek_filter}{src},"
                "zscale=primaries=bt2020:transfer=smpte2084:matrix=bt2020nc,"
                "zscale=transfer=linear:npl=1000,format=gbrpf32le,"
                f"tonemap=tonemap={z_algo}:param={z_param:.6g}:desat={z_desat:.6g}:peak={z_peak:.6g},"
                f"zscale=transfer=bt709:primaries=bt709:matrix=bt709:dither={z_dither},"
                f"zscale=rangein=pc:range=pc,format=bgr24,{crop}"
            )
            cmds.append(base + ["-vf", zf] + out_once + enc)
        if allow_inaccurate and pref in ("auto", "scale"):
            # Last-resort full-res export. This is not correct HDR tone mapping,
            # but it is still original-resolution and only used if the HDR filters fail.
            # Disabled by default because the goal is faithful HDR->SDR rendering,
            # not silently saving a washed-out fallback frame.
            sf = f"{seek_filter}scale=in_range=limited:out_range=full,format=bgr24,{crop}"
            cmds.append(base + ["-vf", sf] + out_once + enc)
        return cmds

    def _save_hdr_sdr_screencap(
        self,
        frame_idx: int,
        frame_pts_sec: Optional[float],
        crop_xyxy: tuple[int, int, int, int],
        out_path: str,
    ) -> tuple[bool, str]:
        """Export the primary crop as full-resolution HDR→SDR from the original source."""
        ffmpeg_bin = self._resolve_ffmpeg_bin()
        if not ffmpeg_bin:
            return False, "ffmpeg_unavailable"
        tmp = out_path + ".tmp.jpg"
        try:
            ensure_dir(os.path.dirname(out_path))
        except Exception:
            pass
        timeout_sec = max(5, int(getattr(self.cfg, "hdr_export_timeout_sec", 300) or 300))
        for cmd in self._hdr_tonemap_filter_cmds(ffmpeg_bin, frame_idx, frame_pts_sec, crop_xyxy, tmp):
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass
            try:
                cp = subprocess.run(cmd, text=True, capture_output=True, check=False, timeout=timeout_sec)
                if cp.returncode == 0:
                    valid, invalid_why = self._validate_hdr_sdr_export_image(tmp, (int(crop_xyxy[2] - crop_xyxy[0]), int(crop_xyxy[3] - crop_xyxy[1])))
                    if valid:
                        os.replace(tmp, out_path)
                        return True, ""
                    try:
                        if os.path.exists(tmp):
                            os.remove(tmp)
                    except Exception:
                        pass
                    self._status(
                        f"HDR full-res export rejected: {invalid_why}",
                        key="hdr_sdr_export",
                        interval=10.0,
                    )
                    continue
                tail = (cp.stderr or cp.stdout or "").splitlines()[-4:]
                why = " | ".join(tail) if tail else f"ffmpeg_rc={cp.returncode}"
                self._status(f"HDR full-res export fallback: {why}", key="hdr_sdr_export", interval=10.0)
            except subprocess.TimeoutExpired as exc:
                self._status(
                    f"HDR full-res export timeout after {timeout_sec}s: {exc}",
                    key="hdr_sdr_export",
                    interval=10.0,
                )
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                except Exception:
                    pass
                return False, f"ffmpeg_timeout_{timeout_sec}s"
            except Exception as exc:
                self._status(f"HDR full-res export fallback: {exc}", key="hdr_sdr_export", interval=10.0)
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        return False, "all_hdr_export_filters_failed"

    def _save_hdr_crop_p010(
        self, frame_idx: int, frame_pts_sec: Optional[float], crop_xyxy: tuple[int, int, int, int], out_path: str
    ) -> None:
        """Use ffmpeg directly to export an HDR crop from the original source."""

        ffmpeg_bin = self._resolve_ffmpeg_bin()
        if not ffmpeg_bin:
            self._status(
                "HDR crop export skipped: ffmpeg unavailable",
                key="hdr_crop_export",
                interval=30.0,
            )
            return

        x1, y1, x2, y2 = crop_xyxy
        w = max(1, int(x2 - x1))
        h = max(1, int(y2 - y1))
        try:
            ensure_dir(os.path.dirname(out_path))
        except Exception:
            pass

        vf = f"crop={w}:{h}:{int(x1)}:{int(y1)}"
        seek_sec: Optional[float] = None
        try:
            if frame_pts_sec is not None and math.isfinite(float(frame_pts_sec)):
                seek_sec = max(0.0, float(frame_pts_sec))
        except Exception:
            seek_sec = None
        if seek_sec is None:
            fps = float(getattr(self, "_fps", 0.0) or 0.0)
            if fps > 0 and math.isfinite(fps):
                seek_sec = max(0.0, float(frame_idx) / fps)

        is_avif = out_path.lower().endswith(".avif")

        pre_seek_args, seek_filter = self._ffmpeg_still_seek_args_and_filter(seek_sec)
        cmd = [ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y"]
        cmd += pre_seek_args
        cmd += [
            "-i",
            self.cfg.video,
            "-map",
            "0:v:0",
            "-vf",
            f"{seek_filter}{vf}",
            "-frames:v",
            "1",
        ]
        if is_avif:
            # High-quality / effectively lossless AVIF still, 10-bit HDR (BT.2020 + PQ).
            cmd += [
                "-an",
                "-c:v",
                "libaom-av1",
                "-still-picture",
                "1",
                "-pix_fmt",
                "yuv420p10le",
                "-g",
                "1",
                "-tile-columns",
                "0",
                "-tile-rows",
                "0",
                "-row-mt",
                "1",
                "-cpu-used",
                "4",
                # Lossless-ish: CRF 0 + no bitrate cap.
                "-crf",
                "0",
                "-b:v",
                "0",
                # Preserve HDR10 signaling in the AVIF container.
                "-color_range",
                "1",
                "-colorspace",
                "bt2020nc",
                "-color_primaries",
                "bt2020",
                "-color_trc",
                "smpte2084",
            ]
        else:
            # Lossless 10-bit HDR in Matroska via FFV1.
            # Decoded pixels in the crop match the original decode bit-for-bit.
            if not out_path.lower().endswith(".mkv"):
                cmd += ["-f", "matroska"]
            cmd += [
                "-c:v", "ffv1",
                "-level", "3",
                "-g", "1",
                "-pix_fmt", "yuv420p10le",
                "-color_range", "1",
                "-colorspace", "bt2020nc",
                "-color_primaries", "bt2020",
                "-color_trc", "smpte2084",
            ]
        out_ext = Path(out_path).suffix or ".mkv"
        tmp_out = out_path + f".tmp{out_ext}"
        try:
            if os.path.exists(tmp_out):
                os.remove(tmp_out)
        except Exception:
            pass
        cmd.append(tmp_out)

        timeout_sec = max(5, int(getattr(self.cfg, "hdr_export_timeout_sec", 300) or 300))
        try:
            cp = subprocess.run(
                cmd,
                check=False,
                timeout=timeout_sec,
                capture_output=True,
                text=True,
            )
            if cp.returncode == 0 and os.path.exists(tmp_out) and os.path.getsize(tmp_out) > 0:
                os.replace(tmp_out, out_path)
                return
            try:
                if os.path.exists(tmp_out):
                    os.remove(tmp_out)
            except Exception:
                pass
            if cp.returncode != 0:
                tail = (cp.stderr or cp.stdout or "").splitlines()[-4:]
                why = " | ".join(tail) if tail else f"ffmpeg_rc={cp.returncode}"
                self._status(
                    f"HDR crop export failed: {why}",
                    key="hdr_crop_export",
                    interval=10.0,
                )
            else:
                self._status(
                    f"HDR crop export failed: invalid output file ({out_path})",
                    key="hdr_crop_export",
                    interval=10.0,
                )
        except subprocess.TimeoutExpired as exc:
            self._status(
                f"HDR crop export timeout after {timeout_sec}s: {exc}",
                key="hdr_crop_export",
                interval=10.0,
            )
            try:
                if os.path.exists(tmp_out):
                    os.remove(tmp_out)
            except Exception:
                pass
        except Exception as exc:
            try:
                if os.path.exists(tmp_out):
                    os.remove(tmp_out)
            except Exception:
                pass
            self._status(
                f"HDR crop export failed: {exc}",
                key="hdr_crop_export",
                interval=10.0,
            )
        
    def _hdr_preview_enabled(self, reader=None) -> bool:
        if reader is None:
            reader = self._hdr_preview_reader
        return bool(self._hdr_passthrough_active and reader is not None)

    def _hdr_preview_close(self) -> None:
        reader = getattr(self, "_hdr_preview_reader", None)
        if reader is not None:
            try:
                release = getattr(reader, "release", None)
                if callable(release):
                    release()
            except Exception:
                pass
        self._hdr_preview_reader = None
        self._hdr_preview_latest = None
        self._hdr_passthrough_active = False

    @QtCore.Slot()
    def disable_hdr_passthrough(self) -> None:
        """Disable HDR passthrough and clean up resources.
        Called from MainWindow when watchdog detects HDR preview is stale/dead.
        """
        self._hdr_preview_close()

    def _normalize_hdr_preview_payload(
        self,
        payload: object,
    ) -> Optional[tuple[int, int, np.ndarray, np.ndarray, int, int]]:
        if payload is None:
            return None
        if isinstance(payload, tuple):
            if len(payload) >= 6 and isinstance(payload[0], (int, np.integer)):
                try:
                    width = int(payload[0])
                    height = int(payload[1])
                    y_plane = payload[2]
                    uv_plane = payload[3]
                    stride_y = int(payload[4])
                    stride_uv = int(payload[5])
                except Exception:
                    return None
                if isinstance(y_plane, np.ndarray) and isinstance(uv_plane, np.ndarray):
                    return width, height, y_plane, uv_plane, stride_y, stride_uv
            if len(payload) >= 2:
                y_plane, uv_plane = payload[:2]
                if isinstance(y_plane, np.ndarray) and isinstance(uv_plane, np.ndarray):
                    h, w = y_plane.shape[:2]
                    stride_y = int(y_plane.strides[0])
                    stride_uv = int(uv_plane.strides[0])
                    return w, h, y_plane, uv_plane, stride_y, stride_uv
        return None

    def _hdr_preview_seek(self, frame_idx: int, reader=None) -> None:
        if reader is None:
            reader = getattr(self, "_hdr_preview_reader", None)
        if reader is None:
            return
        try:
            set_fn = getattr(reader, "set", None)
            if callable(set_fn):
                set_fn(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        except Exception:
            self._hdr_preview_close()

    def _hdr_preview_capture(self, reader=None) -> None:
        if reader is None:
            reader = self._hdr_preview_reader
        if not self._hdr_preview_enabled(reader):
            return
        try:
            ok, payload = reader.read() if reader is not None else (False, None)  # type: ignore[attr-defined]
        except Exception:
            self._hdr_preview_close()
            return
        if not ok or payload is None:
            self._hdr_preview_latest = None
            return
        formatted = self._normalize_hdr_preview_payload(payload)
        if formatted is None:
            self._hdr_preview_latest = None
            return
        self._hdr_preview_latest = formatted
        try:
            w, h, *_ = formatted
            logger.info("HDR passthrough preview: got P010 frame %dx%d", int(w), int(h))
        except Exception:
            pass

    def _pump_hdr_preview(self, reader=None) -> None:
        """Advance the HDR passthrough reader and emit the latest normalized frame."""
        self._hdr_preview_capture(reader=reader)
        self._emit_preview_hdr_passthrough()

    def _hdr_preview_skip(self, count: int, reader=None) -> None:
        if reader is None:
            reader = self._hdr_preview_reader
        if not self._hdr_preview_enabled(reader):
            return
        try:
            skip_fn = getattr(reader, "skip", None)
            if callable(skip_fn):
                skip_fn(count)
            else:
                for _ in range(max(0, int(count))):
                    ok, _payload = reader.read()  # type: ignore[attr-defined]
                    if not ok:
                        break
        except Exception:
            self._hdr_preview_close()

    def _emit_preview_hdr_passthrough(self) -> None:
        if not self._hdr_preview_enabled():
            return
        payload = self._hdr_preview_latest
        if not payload:
            return
        # one-shot: consume cached frame
        self._hdr_preview_latest = None

        # payload should be normalized already, but guard against malformed tuples
        try:
            w, h, y_plane, uv_plane, stride_y, stride_uv = payload
        except Exception:
            return

        if not isinstance(y_plane, np.ndarray) or not isinstance(uv_plane, np.ndarray):
            return
        if y_plane.ndim != 2 or uv_plane.ndim != 2:
            return

        normalized = (
            int(w),
            int(h),
            y_plane,
            uv_plane,
            int(stride_y),
            int(stride_uv),
        )
        try:
            self.preview_hdr_p010.emit(normalized)
        except Exception:
            return

    def _emit_preview_bgr(self, bgr) -> None:
        emit = getattr(self.preview, "emit", None)
        if emit is None or bgr is None:
            return
        # time-based throttle
        try:
            cap_fps = int(getattr(self.cfg, "preview_fps_cap", 20))
        except Exception:
            cap_fps = 20
        if cap_fps > 0:
            now = time.perf_counter()
            min_dt = 1.0 / float(cap_fps)
            if (now - self._last_preview_t) < min_dt:
                return
            self._last_preview_t = now
        try:
            arr = np.asarray(bgr)
        except Exception:
            return
        if arr.ndim != 3 or arr.shape[2] != 3:
            return
        h, w = arr.shape[:2]
        try:
            max_dim = int(getattr(self.cfg, "preview_max_dim", 0) or 0)
        except Exception:
            max_dim = 0
        if max_dim > 0:
            max_dim = max(1, max_dim)
            longest = max(h, w)
            if longest > max_dim:
                scale = float(max_dim) / float(longest)
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
                arr = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
                h, w = arr.shape[:2]
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(
            rgb.data,
            int(w),
            int(h),
            int(rgb.strides[0]),
            QtGui.QImage.Format.Format_RGB888,
        ).copy()
        try:
            emit(qimg)
        except Exception:
            logger.debug("Failed to emit preview frame", exc_info=True)

    def _cv_bgr_to_qimage(self, bgr) -> QtGui.QImage:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        return QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888).copy()

    # ---------------------- Main Window ----------------------

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        # Toolbar
        self.toolbar = self.addToolBar("Main")
        self.toolbar.setObjectName("toolbar_main")
        self.toolbar.setMovable(True)
        self.act_start = QtGui.QAction("Start", self); self.act_start.triggered.connect(self.on_start)
        self.act_pause = QtGui.QAction("Pause", self); self.act_pause.triggered.connect(self.on_pause)
        self.act_stop = QtGui.QAction("Stop", self); self.act_stop.triggered.connect(self.on_stop)
        self.act_compact = QtGui.QAction("Compact", self); self.act_compact.setCheckable(True); self.act_compact.toggled.connect(self.toggle_compact_mode)
        self.act_reset_layout = QtGui.QAction("Reset layout", self); self.act_reset_layout.triggered.connect(self.reset_layout)
        self.toolbar.addActions([self.act_start, self.act_pause, self.act_stop, self.act_compact, self.act_reset_layout])
        self._thread: Optional[QtCore.QThread] = None
        self._worker: Optional[Processor] = None

        # HDR preview UI state flag (mirrors cfg.hdr_passthrough_preview)
        self._hdr_passthrough_enabled: bool = False
        self._last_hdr_preview_t: float = 0.0
        self._hdr_preview_seen: bool = False
        self._curator_fallback: Optional[Processor] = None
        self._fps: Optional[float] = None
        self._total_frames: Optional[int] = None
        self._keyframes: List[int] = []
        self._current_idx: int = 0
        self._last_preview_qimage: Optional[QtGui.QImage] = None

        self.cfg = SessionConfig()
        self._updating_refs = False
        self._initial_ffmpeg_dir: str = ""
        try:
            s = QtCore.QSettings(APP_ORG, APP_NAME)
            self._initial_ffmpeg_dir = s.value(_SETTINGS_KEY_FFMPEG_DIR, "", type=str) or ""
            if self._initial_ffmpeg_dir:
                set_ffmpeg_env(self._initial_ffmpeg_dir)
        except Exception:
            self._initial_ffmpeg_dir = ""
        self._build_ui()
        self._load_qsettings()
        self.statusbar = self.statusBar()
        # --- Preview screenshot button ---
        try:
            toolbar = self.addToolBar("Preview")
            toolbar.setObjectName("previewToolbar")
            act_ss = QtGui.QAction("Save preview frame", self)
            act_ss.triggered.connect(self.on_save_preview_frame)
            toolbar.addAction(act_ss)
        except Exception:
            pass
        # --- Updater wiring ---
        self.updater = UpdateManager(APP_NAME) if "UpdateManager" in globals() and UpdateManager is not None else None
        self._last_update_compare_url: Optional[str] = None
        if self.updater:
            self._connect_updater()
            self._maybe_auto_check_updates()
        self.statusbar.showMessage("Ready")
        self.safe_fit_window()
        self._install_filter()
        try:
            s = QtCore.QSettings(APP_ORG, APP_NAME)
            st = s.value("dock_state", None)
            if st is not None:
                self.restoreState(st)
        except Exception:
            pass

    # UI Construction
    def _build_ui(self):
        # Top: file pickers
        file_group = QtWidgets.QGroupBox("Inputs / Outputs")
        file_layout = QtWidgets.QGridLayout(file_group)

        self.video_edit = QtWidgets.QLineEdit()
        self.ref_edit = QtWidgets.QLineEdit()
        self.ref_edit.setPlaceholderText("C:\\ref1.jpg; C:\\ref2.jpg")
        self.ref_edit.setToolTip("Reference image paths; separate multiple entries with ';'")
        self.out_edit = QtWidgets.QLineEdit("output")

        vid_btn = QtWidgets.QPushButton("Browse...")
        ref_btn = QtWidgets.QPushButton("Browse...")
        out_btn = QtWidgets.QPushButton("Browse...")

        file_layout.addWidget(QtWidgets.QLabel("Video"), 0, 0)
        file_layout.addWidget(self.video_edit, 0, 1)
        file_layout.addWidget(vid_btn, 0, 2)

        file_layout.addWidget(QtWidgets.QLabel("Reference image(s)"), 1, 0)
        file_layout.addWidget(self.ref_edit, 1, 1)
        file_layout.addWidget(ref_btn, 1, 2)

        # --- Reference list UI (bank) ---
        self.ref_list = FileList()
        self.ref_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.ref_list.setDragDropMode(QtWidgets.QAbstractItemView.DragDrop)
        self.ref_list.setDefaultDropAction(QtCore.Qt.CopyAction)
        self.ref_list.setDragDropOverwriteMode(False)
        self.ref_list.setDragEnabled(True)
        self.ref_list.setUniformItemSizes(True)
        self.ref_list.setAlternatingRowColors(True)
        self.ref_list.setMinimumHeight(90)
        self.ref_list.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)

        ref_toolbar = QtWidgets.QWidget()
        rt = QtWidgets.QHBoxLayout(ref_toolbar)
        rt.setContentsMargins(0, 0, 0, 0)
        self.btn_ref_add = QtWidgets.QPushButton("Add…")
        self.btn_ref_remove = QtWidgets.QPushButton("Remove selected")
        self.btn_ref_clear = QtWidgets.QPushButton("Clear")
        rt.addWidget(self.btn_ref_add)
        rt.addWidget(self.btn_ref_remove)
        rt.addStretch(1)
        rt.addWidget(self.btn_ref_clear)

        ref_v = QtWidgets.QVBoxLayout()
        ref_v.setContentsMargins(0, 0, 0, 0)
        ref_v.addWidget(self.ref_list)
        ref_v.addWidget(ref_toolbar)
        ref_cell = QtWidgets.QWidget()
        ref_cell.setLayout(ref_v)

        file_layout.addWidget(QtWidgets.QLabel("Reference bank"), 2, 0)
        file_layout.addWidget(ref_cell, 2, 1, 1, 2)

        file_layout.addWidget(QtWidgets.QLabel("Output directory"), 3, 0)
        file_layout.addWidget(self.out_edit, 3, 1)
        file_layout.addWidget(out_btn, 3, 2)

        ffmpeg_row = self._make_ffmpeg_row()
        file_layout.addWidget(QtWidgets.QLabel("FFmpeg folder"), 4, 0)
        file_layout.addWidget(ffmpeg_row, 4, 1, 1, 2)

        self.chk_hdr_passthrough = QtWidgets.QCheckBox("HDR passthrough (Vulkan)")
        self.chk_hdr_passthrough.setToolTip(
            "Use Vulkan HDR swapchain preview for HDR videos and disable the HDR tone-map reader."
        )
        file_layout.addWidget(self.chk_hdr_passthrough, 5, 0, 1, 3)

        self.chk_hdr_screencap_fullres = QtWidgets.QCheckBox("HDR source-res screencaps")
        self.chk_hdr_screencap_fullres.setChecked(bool(getattr(self.cfg, "hdr_screencap_fullres", True)))
        self.chk_hdr_screencap_fullres.setToolTip(
            "For HDR input, save the primary crops/f*.jpg by cropping the original source frame "
            "and tone-mapping it with FFmpeg/libplacebo/zscale. This is independent of Vulkan preview."
        )
        self.chk_hdr_screencap_fullres.toggled.connect(self._on_ui_change)
        file_layout.addWidget(self.chk_hdr_screencap_fullres, 6, 0, 1, 3)

        self.chk_hdr_archive_crops = QtWidgets.QCheckBox("Also save source HDR crops")
        self.chk_hdr_archive_crops.setChecked(bool(getattr(self.cfg, "hdr_archive_crops", False)))
        self.chk_hdr_archive_crops.setToolTip(
            "Additionally write original-source HDR crops to hdr_crops/ in the selected HDR crop format."
        )
        self.chk_hdr_archive_crops.toggled.connect(self._on_ui_change)
        file_layout.addWidget(self.chk_hdr_archive_crops, 7, 0, 1, 3)

        # Params
        param_group = QtWidgets.QGroupBox("Parameters")
        grid = QtWidgets.QGridLayout(param_group)

        self.ratio_edit = QtWidgets.QLineEdit("1:1,2:3,3:2")
        self.sdr_nits_spin = QtWidgets.QDoubleSpinBox()
        self.sdr_nits_spin.setRange(50.0, 400.0)
        self.sdr_nits_spin.setDecimals(0)
        self.sdr_nits_spin.setSingleStep(5.0)
        self.sdr_nits_spin.setValue(float(self.cfg.sdr_nits))
        self.sdr_nits_spin.setToolTip("float: sdr_nits (target SDR brightness in nits)")
        self.sdr_nits_spin.valueChanged.connect(self._on_ui_change)
        self.tm_desat_spin = QtWidgets.QDoubleSpinBox()
        self.tm_desat_spin.setRange(0.0, 1.0)
        self.tm_desat_spin.setDecimals(2)
        self.tm_desat_spin.setSingleStep(0.05)
        self.tm_desat_spin.setValue(float(self.cfg.tm_desat))
        self.tm_desat_spin.setToolTip("float: tm_desat (0..1 chroma desaturation)")
        self.tm_desat_spin.valueChanged.connect(self._on_ui_change)
        self.tm_param_spin = QtWidgets.QDoubleSpinBox()
        self.tm_param_spin.setRange(0.0, 1.0)
        self.tm_param_spin.setDecimals(2)
        self.tm_param_spin.setSingleStep(0.05)
        self.tm_param_spin.setValue(float(self.cfg.tm_param))
        self.tm_param_spin.setToolTip("float: tm_param (Mobius shoulder softness)")
        self.tm_param_spin.valueChanged.connect(self._on_ui_change)
        self.tonemap_pref_combo = QtWidgets.QComboBox()
        self.tonemap_pref_combo.addItem("Auto", "auto")
        self.tonemap_pref_combo.addItem("Force libplacebo", "libplacebo")
        self.tonemap_pref_combo.addItem("Force zscale+tonemap", "zscale")
        self.tonemap_pref_combo.addItem("Force scale only", "scale")
        pref_idx = self.tonemap_pref_combo.findData(self.cfg.hdr_tonemap_pref)
        self.tonemap_pref_combo.setCurrentIndex(pref_idx if pref_idx >= 0 else 0)
        self.tonemap_pref_combo.currentIndexChanged.connect(self._on_ui_change)

        self.hdr_sdr_quality_combo = QtWidgets.QComboBox()
        self.hdr_sdr_quality_combo.addItem("MadVR/MPCVR-style high quality", "madvr_like")
        self.hdr_sdr_quality_combo.addItem("Resolve-style high quality", "resolve_like")
        self.hdr_sdr_quality_combo.addItem("Balanced", "balanced")
        self.hdr_sdr_quality_combo.addItem("Fast", "fast")
        _hdrq = str(getattr(self.cfg, "hdr_sdr_quality", "madvr_like") or "madvr_like")
        _hdrq_idx = self.hdr_sdr_quality_combo.findData(_hdrq)
        self.hdr_sdr_quality_combo.setCurrentIndex(_hdrq_idx if _hdrq_idx >= 0 else 0)
        self.hdr_sdr_quality_combo.setToolTip(
            "Quality preset for full-res HDR->SDR screencap export. Does not affect pre-scan."
        )
        self.hdr_sdr_quality_combo.currentIndexChanged.connect(self._on_ui_change)

        self.hdr_sdr_tonemap_combo = QtWidgets.QComboBox()
        for _label, _data in (
            ("Auto / renderer default", "auto"),
            ("BT.2390", "bt.2390"),
            ("Spline", "spline"),
            ("ST 2094-40", "st2094-40"),
            ("Mobius", "mobius"),
            ("Hable", "hable"),
        ):
            self.hdr_sdr_tonemap_combo.addItem(_label, _data)
        _hdrtm = str(getattr(self.cfg, "hdr_sdr_tonemap", "auto") or "auto")
        _hdrtm_idx = self.hdr_sdr_tonemap_combo.findData(_hdrtm)
        self.hdr_sdr_tonemap_combo.setCurrentIndex(_hdrtm_idx if _hdrtm_idx >= 0 else 0)
        self.hdr_sdr_tonemap_combo.setToolTip("Tone mapping curve for full-res HDR->SDR screencap export.")
        self.hdr_sdr_tonemap_combo.currentIndexChanged.connect(self._on_ui_change)

        self.hdr_sdr_gamut_combo = QtWidgets.QComboBox()
        for _label, _data in (("Clip", "clip"), ("Perceptual", "perceptual"), ("Relative", "relative"), ("Saturation", "saturation")):
            self.hdr_sdr_gamut_combo.addItem(_label, _data)
        _hdrgm = str(getattr(self.cfg, "hdr_sdr_gamut_mapping", "clip") or "clip")
        _hdrgm_idx = self.hdr_sdr_gamut_combo.findData(_hdrgm)
        self.hdr_sdr_gamut_combo.setCurrentIndex(_hdrgm_idx if _hdrgm_idx >= 0 else 0)
        self.hdr_sdr_gamut_combo.setToolTip("Gamut mapping for full-res HDR->SDR screencap export.")
        self.hdr_sdr_gamut_combo.currentIndexChanged.connect(self._on_ui_change)

        def _mk_fspin(minv: float, maxv: float, step: float, decimals: int, tooltip: str) -> QtWidgets.QDoubleSpinBox:
            sb = QtWidgets.QDoubleSpinBox()
            sb.setRange(minv, maxv)
            sb.setSingleStep(step)
            sb.setDecimals(decimals)
            sb.setToolTip(tooltip)
            sb.setKeyboardTracking(False)
            return sb

        self.hdr_sdr_contrast_spin = _mk_fspin(0.0, 2.0, 0.05, 2, "float: hdr_sdr_contrast_recovery")
        self.hdr_sdr_contrast_spin.setValue(float(getattr(self.cfg, "hdr_sdr_contrast_recovery", 0.30)))
        self.hdr_sdr_contrast_spin.valueChanged.connect(self._on_ui_change)
        self.hdr_sdr_peak_check = QtWidgets.QCheckBox()
        self.hdr_sdr_peak_check.setChecked(bool(getattr(self.cfg, "hdr_sdr_peak_detect", True)))
        self.hdr_sdr_peak_check.setToolTip("bool: hdr_sdr_peak_detect (dynamic libplacebo peak detection)")
        self.hdr_sdr_peak_check.stateChanged.connect(self._on_ui_change)
        self.hdr_sdr_bad_fallback_check = QtWidgets.QCheckBox()
        self.hdr_sdr_bad_fallback_check.setChecked(bool(getattr(self.cfg, "hdr_sdr_allow_inaccurate_fallback", False)))
        self.hdr_sdr_bad_fallback_check.setToolTip(
            "Allow inaccurate full-res scale fallback if HDR tone-map filters fail. Off preserves fidelity by failing instead of saving washed-out crops."
        )
        self.hdr_sdr_bad_fallback_check.stateChanged.connect(self._on_ui_change)
        self.hwaccel_combo = QtWidgets.QComboBox()
        self.hwaccel_combo.addItem("CPU decode (no hwaccel)", "off")
        self.hwaccel_combo.addItem("CUDA / NVDEC (GPU decode)", "cuda")
        _hw_mode = (self.cfg.ff_hwaccel or "off").strip().lower()
        _hw_idx = self.hwaccel_combo.findData(_hw_mode)
        self.hwaccel_combo.setCurrentIndex(_hw_idx if _hw_idx >= 0 else 0)
        self.hwaccel_combo.currentIndexChanged.connect(self._on_ui_change)
        self.hdr_crop_format_combo = QtWidgets.QComboBox()
        self.hdr_crop_format_combo.addItems(["mkv", "avif"])
        _hdr_fmt = str(getattr(self.cfg, "hdr_crop_format", "mkv") or "mkv").lower()
        _hdr_fmt_idx = self.hdr_crop_format_combo.findText(_hdr_fmt)
        self.hdr_crop_format_combo.setCurrentIndex(_hdr_fmt_idx if _hdr_fmt_idx >= 0 else 0)
        self.hdr_crop_format_combo.currentIndexChanged.connect(self._on_ui_change)
        self.stride_spin = QtWidgets.QSpinBox(); self.stride_spin.setRange(1, 1000); self.stride_spin.setValue(2)
        self.det_conf_spin = QtWidgets.QDoubleSpinBox(); self.det_conf_spin.setDecimals(3); self.det_conf_spin.setRange(0.0, 1.0); self.det_conf_spin.setSingleStep(0.01); self.det_conf_spin.setValue(0.35)
        self.face_thr_spin = QtWidgets.QDoubleSpinBox(); self.face_thr_spin.setDecimals(3); self.face_thr_spin.setRange(0.0, 2.0); self.face_thr_spin.setSingleStep(0.01); self.face_thr_spin.setValue(0.45)
        self.face_det_conf_spin = QtWidgets.QDoubleSpinBox(); self.face_det_conf_spin.setDecimals(3); self.face_det_conf_spin.setRange(0.0, 1.0); self.face_det_conf_spin.setSingleStep(0.01); self.face_det_conf_spin.setValue(0.5)
        self.face_det_pad_spin = QtWidgets.QDoubleSpinBox(); self.face_det_pad_spin.setDecimals(3); self.face_det_pad_spin.setRange(0.0, 1.0); self.face_det_pad_spin.setSingleStep(0.01); self.face_det_pad_spin.setValue(0.08)
        self.face_quality_spin = QtWidgets.QDoubleSpinBox(); self.face_quality_spin.setRange(0.0, 1000.0); self.face_quality_spin.setSingleStep(1.0); self.face_quality_spin.setValue(70.0)
        self.face_vis_quality_check = QtWidgets.QCheckBox(); self.face_vis_quality_check.setChecked(True)
        self.reid_thr_spin = QtWidgets.QDoubleSpinBox(); self.reid_thr_spin.setDecimals(3); self.reid_thr_spin.setRange(0.0, 2.0); self.reid_thr_spin.setSingleStep(0.01); self.reid_thr_spin.setValue(0.38)
        self.combine_combo = QtWidgets.QComboBox(); self.combine_combo.addItems(["min","avg","face_priority"])
        self.match_mode_combo = QtWidgets.QComboBox(); self.match_mode_combo.addItems(["either","both","face_only","reid_only"])
        self.disable_reid_check = QtWidgets.QCheckBox("Disable ReID")
        self.disable_reid_check.setChecked(True)
        self.face_fullframe_check = QtWidgets.QCheckBox("Full-frame face fallback when missed")
        self.face_fullframe_check.setChecked(True)
        self.face_fullframe_imgsz_spin = QtWidgets.QSpinBox()
        self.face_fullframe_imgsz_spin.setRange(0, 4096)
        self.face_fullframe_imgsz_spin.setSingleStep(32)
        self.face_fullframe_imgsz_spin.setValue(int(self.cfg.face_fullframe_imgsz))
        self.face_fullframe_imgsz_spin.setToolTip("int: face_fullframe_imgsz (0 disables override)")
        self.rot_adaptive_check = QtWidgets.QCheckBox()
        self.rot_adaptive_check.setChecked(bool(self.cfg.rot_adaptive))
        self.rot_adaptive_check.setToolTip("bool: rot_adaptive")
        self.rot_every_spin = QtWidgets.QSpinBox()
        self.rot_every_spin.setRange(1, 360)
        self.rot_every_spin.setSingleStep(1)
        self.rot_every_spin.setValue(int(self.cfg.rot_every_n))
        self.rot_every_spin.setToolTip("int: rot_every_n")
        self.rot_after_hit_spin = QtWidgets.QSpinBox()
        self.rot_after_hit_spin.setRange(0, 120)
        self.rot_after_hit_spin.setSingleStep(1)
        self.rot_after_hit_spin.setValue(int(self.cfg.rot_after_hit_frames))
        self.rot_after_hit_spin.setToolTip("int: rot_after_hit_frames")
        self.fast_no_face_spin = QtWidgets.QSpinBox()
        self.fast_no_face_spin.setRange(320, 4096)
        self.fast_no_face_spin.setSingleStep(32)
        self.fast_no_face_spin.setValue(int(self.cfg.fast_no_face_imgsz))
        self.fast_no_face_spin.setToolTip("int: fast_no_face_imgsz")
        self.only_best_check = QtWidgets.QCheckBox("Only best per frame")
        self.only_best_check.setChecked(True)
        # Pre-scan controls
        self.chk_prescan = QtWidgets.QCheckBox("Enable pre-scan")
        self.chk_prescan.setChecked(bool(self.cfg.prescan_enable))
        self.chk_prescan.setToolTip("bool: prescan_enable")
        self.chk_prescan.toggled.connect(self._on_ui_change)
        self.spin_prescan_stride = QtWidgets.QSpinBox()
        self.spin_prescan_stride.setRange(1, 600)
        self.spin_prescan_stride.setValue(int(self.cfg.prescan_stride))
        self.spin_prescan_stride.setToolTip("int: prescan_stride")
        self.spin_prescan_stride.valueChanged.connect(self._on_ui_change)
        self.spin_prescan_fd_add = QtWidgets.QDoubleSpinBox()
        self.spin_prescan_fd_add.setDecimals(3)
        self.spin_prescan_fd_add.setRange(0.0, 2.0)
        self.spin_prescan_fd_add.setSingleStep(0.01)
        self.spin_prescan_fd_add.setValue(float(self.cfg.prescan_fd_add))
        self.spin_prescan_fd_add.setToolTip("float: prescan_fd_add")
        self.spin_prescan_fd_add.valueChanged.connect(self._on_ui_change)
        self.spin_prescan_max_width = QtWidgets.QSpinBox()
        self.spin_prescan_max_width.setRange(64, 8192)
        self.spin_prescan_max_width.setValue(int(self.cfg.prescan_max_width))
        self.spin_prescan_max_width.setToolTip("int: prescan_max_width")
        self.spin_prescan_max_width.valueChanged.connect(self._on_ui_change)
        self.spin_prescan_decode_max_w = QtWidgets.QSpinBox()
        self.spin_prescan_decode_max_w.setRange(0, 8192)
        self.spin_prescan_decode_max_w.setValue(int(getattr(self.cfg, "prescan_decode_max_w", 384)))
        self.spin_prescan_decode_max_w.setToolTip("int: prescan_decode_max_w (decoder downscale; 0=disable)")
        self.spin_prescan_decode_max_w.valueChanged.connect(self._on_ui_change)
        self.prescan_cache_combo = QtWidgets.QComboBox()
        self.prescan_cache_combo.addItem("Auto reuse matching cache", "auto")
        self.prescan_cache_combo.addItem("Refresh/rebuild cache", "refresh")
        self.prescan_cache_combo.addItem("Disabled", "off")
        _pc_mode = str(getattr(self.cfg, "prescan_cache_mode", "auto") or "auto")
        _pc_idx = self.prescan_cache_combo.findData(_pc_mode)
        self.prescan_cache_combo.setCurrentIndex(_pc_idx if _pc_idx >= 0 else 0)
        self.prescan_cache_combo.setToolTip(
            "Persistent pre-scan cache. Auto reuses matching video/ref/pre-scan settings; "
            "HDR/export-only setting changes do not invalidate it."
        )
        self.prescan_cache_combo.currentIndexChanged.connect(self._on_ui_change)
        self.prescan_cache_clear_btn = QtWidgets.QPushButton("Clear")
        self.prescan_cache_clear_btn.setToolTip("Delete all persisted pre-scan cache files.")
        self.prescan_cache_clear_btn.clicked.connect(self._clear_prescan_cache)
        self.prescan_cache_widget = QtWidgets.QWidget()
        _pc_lay = QtWidgets.QHBoxLayout(self.prescan_cache_widget)
        _pc_lay.setContentsMargins(0, 0, 0, 0)
        _pc_lay.setSpacing(6)
        _pc_lay.addWidget(self.prescan_cache_combo, 1)
        _pc_lay.addWidget(self.prescan_cache_clear_btn, 0)
        self.spin_prescan_face_conf = QtWidgets.QDoubleSpinBox()
        self.spin_prescan_face_conf.setDecimals(3)
        self.spin_prescan_face_conf.setRange(0.0, 1.0)
        self.spin_prescan_face_conf.setSingleStep(0.01)
        self.spin_prescan_face_conf.setValue(float(self.cfg.prescan_face_conf))
        self.spin_prescan_face_conf.setToolTip("float: prescan_face_conf")
        self.spin_prescan_face_conf.valueChanged.connect(self._on_ui_change)
        self.spin_prescan_fd_enter = QtWidgets.QDoubleSpinBox()
        self.spin_prescan_fd_enter.setDecimals(3)
        self.spin_prescan_fd_enter.setRange(0.0, 2.0)
        self.spin_prescan_fd_enter.setSingleStep(0.01)
        self.spin_prescan_fd_enter.setValue(float(self.cfg.prescan_fd_enter))
        self.spin_prescan_fd_enter.setToolTip("float: prescan_fd_enter")
        self.spin_prescan_fd_enter.valueChanged.connect(self._on_ui_change)
        self.spin_prescan_fd_exit = QtWidgets.QDoubleSpinBox()
        self.spin_prescan_fd_exit.setDecimals(3)
        self.spin_prescan_fd_exit.setRange(0.0, 2.0)
        self.spin_prescan_fd_exit.setSingleStep(0.01)
        self.spin_prescan_fd_exit.setValue(float(self.cfg.prescan_fd_exit))
        self.spin_prescan_fd_exit.setToolTip("float: prescan_fd_exit")
        self.spin_prescan_fd_exit.valueChanged.connect(self._on_ui_change)
        self.spin_prescan_add_cooldown = QtWidgets.QSpinBox()
        self.spin_prescan_add_cooldown.setRange(0, 100)
        self.spin_prescan_add_cooldown.setValue(int(self.cfg.prescan_add_cooldown_samples))
        self.spin_prescan_add_cooldown.setToolTip("int: prescan_add_cooldown_samples")
        self.spin_prescan_add_cooldown.valueChanged.connect(self._on_ui_change)
        self.spin_prescan_min_segment = QtWidgets.QDoubleSpinBox()
        self.spin_prescan_min_segment.setDecimals(2)
        self.spin_prescan_min_segment.setRange(0.0, 120.0)
        self.spin_prescan_min_segment.setSingleStep(0.1)
        self.spin_prescan_min_segment.setValue(float(self.cfg.prescan_min_segment_sec))
        self.spin_prescan_min_segment.setToolTip("float: prescan_min_segment_sec")
        self.spin_prescan_min_segment.valueChanged.connect(self._on_ui_change)
        self.spin_prescan_pad = QtWidgets.QDoubleSpinBox()
        self.spin_prescan_pad.setDecimals(2)
        self.spin_prescan_pad.setRange(0.0, 120.0)
        self.spin_prescan_pad.setSingleStep(0.1)
        self.spin_prescan_pad.setValue(float(self.cfg.prescan_pad_sec))
        self.spin_prescan_pad.setToolTip("float: prescan_pad_sec")
        self.spin_prescan_pad.valueChanged.connect(self._on_ui_change)
        self.spin_prescan_bridge = QtWidgets.QDoubleSpinBox()
        self.spin_prescan_bridge.setDecimals(2)
        self.spin_prescan_bridge.setRange(0.0, 120.0)
        self.spin_prescan_bridge.setSingleStep(0.1)
        self.spin_prescan_bridge.setValue(float(self.cfg.prescan_bridge_gap_sec))
        self.spin_prescan_bridge.setToolTip("float: prescan_bridge_gap_sec")
        self.spin_prescan_bridge.valueChanged.connect(self._on_ui_change)
        self.spin_prescan_exit_cooldown = QtWidgets.QDoubleSpinBox()
        self.spin_prescan_exit_cooldown.setDecimals(2)
        self.spin_prescan_exit_cooldown.setRange(0.0, 10.0)
        self.spin_prescan_exit_cooldown.setSingleStep(0.05)
        self.spin_prescan_exit_cooldown.setValue(float(self.cfg.prescan_exit_cooldown_sec))
        self.spin_prescan_exit_cooldown.setToolTip("float: prescan_exit_cooldown_sec")
        self.spin_prescan_exit_cooldown.valueChanged.connect(self._on_ui_change)
        self.spin_prescan_refine_window = QtWidgets.QDoubleSpinBox()
        self.spin_prescan_refine_window.setDecimals(2)
        self.spin_prescan_refine_window.setRange(0.0, 10.0)
        self.spin_prescan_refine_window.setSingleStep(0.05)
        self.spin_prescan_refine_window.setValue(float(self.cfg.prescan_boundary_refine_sec))
        self.spin_prescan_refine_window.setToolTip("float: prescan_boundary_refine_sec")
        self.spin_prescan_refine_window.valueChanged.connect(self._on_ui_change)
        self.spin_prescan_refine_stride = QtWidgets.QSpinBox()
        self.spin_prescan_refine_stride.setRange(1, 120)
        self.spin_prescan_refine_stride.setValue(int(self.cfg.prescan_refine_stride_min))
        self.spin_prescan_refine_stride.setToolTip("int: prescan_refine_stride_min")
        self.spin_prescan_refine_stride.valueChanged.connect(self._on_ui_change)
        self.chk_prescan_trim_pad = QtWidgets.QCheckBox()
        self.chk_prescan_trim_pad.setChecked(bool(self.cfg.prescan_trim_pad))
        self.chk_prescan_trim_pad.setToolTip("bool: prescan_trim_pad")
        self.chk_prescan_trim_pad.stateChanged.connect(self._on_ui_change)
        self.spin_prescan_bank_max = QtWidgets.QSpinBox()
        self.spin_prescan_bank_max.setRange(8, 4096)
        self.spin_prescan_bank_max.setValue(int(self.cfg.prescan_bank_max))
        self.spin_prescan_bank_max.setToolTip("int: prescan_bank_max")
        self.spin_prescan_bank_max.valueChanged.connect(self._on_ui_change)
        self.spin_prescan_dedup = QtWidgets.QDoubleSpinBox()
        self.spin_prescan_dedup.setDecimals(3)
        self.spin_prescan_dedup.setRange(0.5, 1.0)
        self.spin_prescan_dedup.setSingleStep(0.001)
        self.spin_prescan_dedup.setValue(float(self.cfg.prescan_diversity_dedup_cos))
        self.spin_prescan_dedup.setToolTip("float: prescan_diversity_dedup_cos")
        self.spin_prescan_dedup.valueChanged.connect(self._on_ui_change)
        self.spin_prescan_margin = QtWidgets.QDoubleSpinBox()
        self.spin_prescan_margin.setDecimals(3)
        self.spin_prescan_margin.setRange(0.0, 0.1)
        self.spin_prescan_margin.setSingleStep(0.001)
        self.spin_prescan_margin.setValue(float(self.cfg.prescan_replace_margin))
        self.spin_prescan_margin.setToolTip("float: prescan_replace_margin")
        self.spin_prescan_margin.valueChanged.connect(self._on_ui_change)
        # TensorRT path controls
        self.trt_edit = QtWidgets.QLineEdit(self.cfg.trt_lib_dir or r"D:\\tensorrt\\TensorRT-10.13.3.9")
        self.trt_btn = QtWidgets.QPushButton("Browse…")
        # Advanced TensorRT/ORT collapsible

        def _pick_trt():
            d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select TensorRT lib folder", self.trt_edit.text() or "")
            if d:
                self.trt_edit.setText(d)

        self.trt_btn.clicked.connect(_pick_trt)
        self._trt_row_widget = QtWidgets.QWidget()
        trt_row_layout = QtWidgets.QHBoxLayout(self._trt_row_widget)
        trt_row_layout.setContentsMargins(0, 0, 0, 0)
        trt_row_layout.addWidget(self.trt_edit, 1)
        trt_row_layout.addWidget(self.trt_btn)

        # --- Collapsible: TensorRT / ORT (Advanced) ---
        self.trt_box = CollapsibleSection("TensorRT / ORT (Advanced)")
        trt_adv = QtWidgets.QGridLayout()
        r = 0
        self.chk_trt_fp16 = QtWidgets.QCheckBox("TRT FP16")
        self.chk_trt_fp16.setChecked(bool(self.cfg.trt_fp16_enable))
        trt_adv.addWidget(self.chk_trt_fp16, r, 0)
        r += 1
        self.chk_trt_timing = QtWidgets.QCheckBox("Timing cache")
        self.chk_trt_timing.setChecked(bool(self.cfg.trt_timing_cache_enable))
        trt_adv.addWidget(self.chk_trt_timing, r, 0)
        self.chk_trt_engine = QtWidgets.QCheckBox("Engine cache")
        self.chk_trt_engine.setChecked(bool(self.cfg.trt_engine_cache_enable))
        trt_adv.addWidget(self.chk_trt_engine, r, 1)
        r += 1
        self.edit_trt_cache = QtWidgets.QLineEdit(self.cfg.trt_cache_root or "trt_cache")
        trt_adv.addWidget(QtWidgets.QLabel("Cache root"), r, 0)
        trt_adv.addWidget(self.edit_trt_cache, r, 1)
        r += 1
        self.spin_trt_level = QtWidgets.QSpinBox()
        self.spin_trt_level.setRange(0, 5)
        self.spin_trt_level.setValue(int(self.cfg.trt_builder_optimization_level))
        trt_adv.addWidget(QtWidgets.QLabel("Builder opt level"), r, 0)
        trt_adv.addWidget(self.spin_trt_level, r, 1)
        r += 1
        self.chk_trt_cuda_graph = QtWidgets.QCheckBox("CUDA Graphs")
        self.chk_trt_cuda_graph.setChecked(bool(self.cfg.trt_cuda_graph_enable))
        trt_adv.addWidget(self.chk_trt_cuda_graph, r, 0)
        self.chk_trt_ctx_share = QtWidgets.QCheckBox("Context memory sharing")
        self.chk_trt_ctx_share.setChecked(bool(self.cfg.trt_context_memory_sharing_enable))
        trt_adv.addWidget(self.chk_trt_ctx_share, r, 1)
        r += 1
        self.spin_trt_aux = QtWidgets.QSpinBox()
        self.spin_trt_aux.setRange(-1, 16)
        self.spin_trt_aux.setValue(int(self.cfg.trt_auxiliary_streams))
        trt_adv.addWidget(QtWidgets.QLabel("Aux streams"), r, 0)
        trt_adv.addWidget(self.spin_trt_aux, r, 1)
        r += 1
        self.chk_cuda_tf32 = QtWidgets.QCheckBox("CUDA TF32")
        self.chk_cuda_tf32.setChecked(bool(self.cfg.cuda_use_tf32))
        trt_adv.addWidget(self.chk_cuda_tf32, r, 0)
        r += 1
        self.trt_box.setContentLayout(trt_adv)
        self.chk_trt_fp16.stateChanged.connect(self._on_ui_change)
        self.chk_trt_timing.stateChanged.connect(self._on_ui_change)
        self.chk_trt_engine.stateChanged.connect(self._on_ui_change)
        self.edit_trt_cache.textChanged.connect(self._on_ui_change)
        self.spin_trt_level.valueChanged.connect(self._on_ui_change)
        self.chk_trt_cuda_graph.stateChanged.connect(self._on_ui_change)
        self.chk_trt_ctx_share.stateChanged.connect(self._on_ui_change)
        self.spin_trt_aux.valueChanged.connect(self._on_ui_change)
        self.chk_cuda_tf32.stateChanged.connect(self._on_ui_change)
        self.min_sharp_spin = QtWidgets.QDoubleSpinBox(); self.min_sharp_spin.setRange(0.0, 5000.0); self.min_sharp_spin.setValue(0.0)
        self.min_gap_spin = QtWidgets.QDoubleSpinBox(); self.min_gap_spin.setDecimals(2); self.min_gap_spin.setRange(0.0, 30.0); self.min_gap_spin.setValue(1.5)
        self.min_box_pix_spin = QtWidgets.QSpinBox(); self.min_box_pix_spin.setRange(0, 5000000); self.min_box_pix_spin.setValue(5000)
        self.auto_crop_check = QtWidgets.QCheckBox("Auto‑crop black borders")
        self.auto_crop_check.setChecked(True)
        self.border_thr_spin = QtWidgets.QSpinBox(); self.border_thr_spin.setRange(0, 50); self.border_thr_spin.setValue(22)
        self.require_face_check = QtWidgets.QCheckBox("Require face if visible"); self.require_face_check.setChecked(True)
        self.pref_face_check = QtWidgets.QCheckBox()
        self.pref_face_check.setChecked(bool(self.cfg.prefer_face_when_available))
        self.pref_face_check.setToolTip("bool: prefer_face_when_available")
        self.lock_mom_spin = QtWidgets.QDoubleSpinBox(); self.lock_mom_spin.setRange(0.0, 1.0); self.lock_mom_spin.setSingleStep(0.05); self.lock_mom_spin.setValue(0.7)
        self.suppress_neg_check = QtWidgets.QCheckBox("Suppress hard negatives"); self.suppress_neg_check.setChecked(False)
        self.neg_tol_spin = QtWidgets.QDoubleSpinBox(); self.neg_tol_spin.setDecimals(3); self.neg_tol_spin.setRange(0.0, 2.0); self.neg_tol_spin.setSingleStep(0.01); self.neg_tol_spin.setValue(0.35)
        self.max_neg_spin = QtWidgets.QSpinBox(); self.max_neg_spin.setRange(0, 20); self.max_neg_spin.setValue(5)
        self.log_every_spin = QtWidgets.QDoubleSpinBox(); self.log_every_spin.setDecimals(1); self.log_every_spin.setRange(0.1, 10.0); self.log_every_spin.setValue(1.0)
        self.lock_after_spin = QtWidgets.QSpinBox(); self.lock_after_spin.setRange(0, 10); self.lock_after_spin.setValue(1)
        self.lock_face_spin = QtWidgets.QDoubleSpinBox(); self.lock_face_spin.setDecimals(3); self.lock_face_spin.setRange(0.0, 2.0); self.lock_face_spin.setValue(0.28)
        self.lock_reid_spin = QtWidgets.QDoubleSpinBox(); self.lock_reid_spin.setDecimals(3); self.lock_reid_spin.setRange(0.0, 2.0); self.lock_reid_spin.setValue(0.30)
        self.margin_spin = QtWidgets.QDoubleSpinBox(); self.margin_spin.setDecimals(3); self.margin_spin.setRange(0.0, 1.0); self.margin_spin.setValue(0.03)
        self.iou_gate_spin = QtWidgets.QDoubleSpinBox(); self.iou_gate_spin.setDecimals(3); self.iou_gate_spin.setRange(0.0, 1.0); self.iou_gate_spin.setValue(0.05)
        self.use_arc_check = QtWidgets.QCheckBox("Use ArcFace for face ID")
        self.use_arc_check.setChecked(True)
        self.device_combo = QtWidgets.QComboBox(); self.device_combo.addItems(["cuda","cpu"])
        self.yolo_edit = QtWidgets.QLineEdit("yolov8n.pt")
        self.face_yolo_edit = QtWidgets.QLineEdit("scrfd_10g_bnkps")
        self.annot_check = QtWidgets.QCheckBox("Save annotated frames")
        self.preview_every_spin = QtWidgets.QSpinBox(); self.preview_every_spin.setRange(0, 5000); self.preview_every_spin.setValue(3)
        self.preview_every_spin.valueChanged.connect(self._on_ui_change)
        self.preview_max_dim_spin = QtWidgets.QSpinBox()
        self.preview_max_dim_spin.setRange(0, 7680)
        self.preview_max_dim_spin.setValue(int(self.cfg.preview_max_dim))
        self.preview_max_dim_spin.setToolTip("int: preview_max_dim (0 = no limit)")
        self.preview_max_dim_spin.valueChanged.connect(self._on_ui_change)
        self.preview_fps_cap_spin = QtWidgets.QSpinBox()
        self.preview_fps_cap_spin.setRange(0, 240)
        self.preview_fps_cap_spin.setValue(int(self.cfg.preview_fps_cap))
        self.preview_fps_cap_spin.setToolTip("int: preview_fps_cap (0 = unlimited)")
        self.preview_fps_cap_spin.valueChanged.connect(self._on_ui_change)
        self.seek_fast_check = QtWidgets.QCheckBox()
        self.seek_fast_check.setChecked(bool(self.cfg.seek_fast))
        self.seek_fast_check.setToolTip("bool: seek_fast")
        self.seek_fast_check.stateChanged.connect(self._on_ui_change)
        self.seek_max_grabs_spin = QtWidgets.QSpinBox()
        self.seek_max_grabs_spin.setRange(0, 480)
        self.seek_max_grabs_spin.setValue(int(self.cfg.seek_max_grabs))
        self.seek_max_grabs_spin.setToolTip("int: seek_max_grabs")
        self.seek_max_grabs_spin.valueChanged.connect(self._on_ui_change)
        self.seek_preview_peek_spin = QtWidgets.QSpinBox()
        self.seek_preview_peek_spin.setRange(1, 120)
        self.seek_preview_peek_spin.setValue(int(self.cfg.seek_preview_peek_every))
        self.seek_preview_peek_spin.setToolTip("int: seek_preview_peek_every")
        self.seek_preview_peek_spin.valueChanged.connect(self._on_ui_change)
        self.overlay_scores_check = QtWidgets.QCheckBox()
        self.overlay_scores_check.setChecked(bool(self.cfg.overlay_scores))
        self.overlay_scores_check.setToolTip("bool: overlay_scores")
        self.overlay_scores_check.stateChanged.connect(self._on_ui_change)

        # --- Speed / I/O controls ---
        self.skip_yolo_faceonly_check = QtWidgets.QCheckBox()
        self.skip_yolo_faceonly_check.setChecked(True)
        self.skip_yolo_faceonly_check.setToolTip("bool: skip_yolo_when_faceonly")
        self.skip_yolo_faceonly_check.stateChanged.connect(self._on_ui_change)

        self.async_save_check = QtWidgets.QCheckBox()
        self.async_save_check.setChecked(True)
        self.async_save_check.setToolTip("bool: async_save")
        self.async_save_check.stateChanged.connect(self._on_ui_change)

        self.jpg_quality_spin = QtWidgets.QSpinBox()
        self.jpg_quality_spin.setRange(1, 100)
        self.jpg_quality_spin.setValue(85)
        self.jpg_quality_spin.setToolTip("int: jpg_quality")
        self.jpg_quality_spin.valueChanged.connect(self._on_ui_change)

        self.faceless_allow_check = QtWidgets.QCheckBox()
        self.faceless_allow_check.setChecked(bool(self.cfg.allow_faceless_when_locked))
        self.faceless_allow_check.setToolTip("bool: allow_faceless_when_locked")
        self.learn_bank_runtime_check = QtWidgets.QCheckBox()
        self.learn_bank_runtime_check.setChecked(bool(self.cfg.learn_bank_runtime))
        self.learn_bank_runtime_check.setToolTip("bool: learn_bank_runtime")
        self.drop_reid_if_any_face_match_check = QtWidgets.QCheckBox()
        self.drop_reid_if_any_face_match_check.setChecked(bool(self.cfg.drop_reid_if_any_face_match))
        self.drop_reid_if_any_face_match_check.setToolTip("bool: drop_reid_if_any_face_match")
        self.faceless_reid_spin = _mk_fspin(0.0, 1.0, 0.01, 3, "float: faceless_reid_thresh")
        self.faceless_reid_spin.setValue(float(self.cfg.faceless_reid_thresh))
        self.faceless_iou_spin = _mk_fspin(0.0, 1.0, 0.01, 3, "float: faceless_iou_min")
        self.faceless_iou_spin.setValue(float(self.cfg.faceless_iou_min))
        self.faceless_persist_spin = QtWidgets.QSpinBox()
        self.faceless_persist_spin.setRange(0, 300)
        self.faceless_persist_spin.setToolTip("int: faceless_persist_frames")
        self.faceless_persist_spin.setValue(int(self.cfg.faceless_persist_frames))
        self.faceless_min_area_spin = _mk_fspin(0.0, 1.0, 0.01, 3, "float: faceless_min_area_frac")
        self.faceless_min_area_spin.setValue(float(self.cfg.faceless_min_area_frac))
        self.faceless_max_area_spin = _mk_fspin(0.0, 1.0, 0.01, 3, "float: faceless_max_area_frac")
        self.faceless_max_area_spin.setValue(float(self.cfg.faceless_max_area_frac))
        self.faceless_center_spin = _mk_fspin(0.0, 1.0, 0.01, 3, "float: faceless_center_max_frac")
        self.faceless_center_spin.setValue(float(self.cfg.faceless_center_max_frac))
        self.faceless_motion_spin = _mk_fspin(0.0, 1.0, 0.01, 3, "float: faceless_min_motion_frac")
        self.faceless_motion_spin.setValue(float(self.cfg.faceless_min_motion_frac))
        self.crop_face_side_margin_spin = _mk_fspin(0.0, 1.5, 0.01, 3, "float: crop_face_side_margin_frac")
        self.crop_face_side_margin_spin.setValue(float(self.cfg.crop_face_side_margin_frac))
        self.crop_top_headroom_spin = _mk_fspin(0.0, 1.0, 0.01, 3, "float: crop_top_headroom_max_frac")
        self.crop_top_headroom_spin.setValue(float(self.cfg.crop_top_headroom_max_frac))
        self.crop_bottom_min_face_spin = _mk_fspin(0.0, 5.0, 0.05, 2, "float: crop_bottom_min_face_heights")
        self.crop_bottom_min_face_spin.setValue(float(self.cfg.crop_bottom_min_face_heights))
        self.crop_penalty_weight_spin = _mk_fspin(0.0, 10.0, 0.1, 2, "float: crop_penalty_weight")
        self.crop_penalty_weight_spin.setValue(float(self.cfg.crop_penalty_weight))
        self.face_anchor_down_spin = _mk_fspin(0.0, 2.0, 0.05, 2, "float: face_anchor_down_frac")
        self.face_anchor_down_spin.setValue(float(self.cfg.face_anchor_down_frac))

        # --- Framing scorer knobs ---
        self.lambda_facefrac_spin = _mk_fspin(0.0, 10.0, 0.1, 2, "float: lambda_facefrac")
        self.crop_center_weight_spin = _mk_fspin(0.0, 5.0, 0.05, 2, "float: crop_center_weight")
        self.area_gamma_spin = _mk_fspin(0.1, 2.0, 0.05, 2, "float: area_gamma")
        self.area_face_scale_w_spin = _mk_fspin(0.0, 1.0, 0.05, 2, "float: area_face_scale_weight")
        self.square_pull_face_min_spin = _mk_fspin(0.0, 1.0, 0.01, 2, "float: square_pull_face_min")
        self.square_pull_weight_spin = _mk_fspin(0.0, 2.0, 0.05, 2, "float: square_pull_weight")

        self.face_target_close_spin = _mk_fspin(0.0, 1.0, 0.01, 3, "float: face_target_close")
        self.face_target_upper_spin = _mk_fspin(0.0, 1.0, 0.01, 3, "float: face_target_upper")
        self.face_target_cowboy_spin = _mk_fspin(0.0, 1.0, 0.01, 3, "float: face_target_cowboy")
        self.face_target_body_spin = _mk_fspin(0.0, 1.0, 0.01, 3, "float: face_target_body")
        self.face_target_tol_spin = _mk_fspin(0.0, 0.5, 0.01, 3, "float: face_target_tolerance")
        self.face_target_close_min_spin = _mk_fspin(0.0, 1.0, 0.01, 3, "float: face_target_close_min_frac")

        self.w_close_spin = _mk_fspin(0.0, 2.0, 0.05, 2, "float: w_close")
        self.w_upper_spin = _mk_fspin(0.0, 2.0, 0.05, 2, "float: w_upper")
        self.w_cowboy_spin = _mk_fspin(0.0, 2.0, 0.05, 2, "float: w_cowboy")
        self.w_body_spin = _mk_fspin(0.0, 2.0, 0.05, 2, "float: w_body")

        # initialize from cfg
        self.lambda_facefrac_spin.setValue(float(self.cfg.lambda_facefrac))
        self.lambda_facefrac_spin.valueChanged.connect(self._on_ui_change)
        self.crop_center_weight_spin.setValue(float(self.cfg.crop_center_weight))
        self.crop_center_weight_spin.valueChanged.connect(self._on_ui_change)
        self.area_gamma_spin.setValue(float(self.cfg.area_gamma))
        self.area_gamma_spin.valueChanged.connect(self._on_ui_change)
        self.area_face_scale_w_spin.setValue(float(self.cfg.area_face_scale_weight))
        self.area_face_scale_w_spin.valueChanged.connect(self._on_ui_change)
        self.square_pull_face_min_spin.setValue(float(self.cfg.square_pull_face_min))
        self.square_pull_face_min_spin.valueChanged.connect(self._on_ui_change)
        self.square_pull_weight_spin.setValue(float(self.cfg.square_pull_weight))
        self.square_pull_weight_spin.valueChanged.connect(self._on_ui_change)
        self.face_target_close_spin.setValue(float(self.cfg.face_target_close))
        self.face_target_close_spin.valueChanged.connect(self._on_ui_change)
        self.face_target_upper_spin.setValue(float(self.cfg.face_target_upper))
        self.face_target_upper_spin.valueChanged.connect(self._on_ui_change)
        self.face_target_cowboy_spin.setValue(float(self.cfg.face_target_cowboy))
        self.face_target_cowboy_spin.valueChanged.connect(self._on_ui_change)
        self.face_target_body_spin.setValue(float(self.cfg.face_target_body))
        self.face_target_body_spin.valueChanged.connect(self._on_ui_change)
        self.face_target_tol_spin.setValue(float(self.cfg.face_target_tolerance))
        self.face_target_tol_spin.valueChanged.connect(self._on_ui_change)
        self.face_target_close_min_spin.setValue(float(self.cfg.face_target_close_min_frac))
        self.face_target_close_min_spin.valueChanged.connect(self._on_ui_change)
        self.w_close_spin.setValue(float(self.cfg.w_close))
        self.w_close_spin.valueChanged.connect(self._on_ui_change)
        self.w_upper_spin.setValue(float(self.cfg.w_upper))
        self.w_upper_spin.valueChanged.connect(self._on_ui_change)
        self.w_cowboy_spin.setValue(float(self.cfg.w_cowboy))
        self.w_cowboy_spin.valueChanged.connect(self._on_ui_change)
        self.w_body_spin.setValue(float(self.cfg.w_body))
        self.w_body_spin.valueChanged.connect(self._on_ui_change)
        self.smart_crop_enable_check = QtWidgets.QCheckBox()
        self.smart_crop_enable_check.setChecked(bool(self.cfg.smart_crop_enable))
        self.smart_crop_enable_check.setToolTip("bool: smart_crop_enable")
        self.smart_crop_enable_check.stateChanged.connect(self._on_ui_change)
        self.smart_crop_steps_spin = QtWidgets.QSpinBox()
        self.smart_crop_steps_spin.setRange(0, 50)
        self.smart_crop_steps_spin.setValue(int(self.cfg.smart_crop_steps))
        self.smart_crop_steps_spin.setToolTip("int: smart_crop_steps")
        self.smart_crop_steps_spin.valueChanged.connect(self._on_ui_change)
        self.smart_crop_side_search_spin = _mk_fspin(0.0, 1.5, 0.01, 3, "float: smart_crop_side_search_frac")
        self.smart_crop_side_search_spin.setValue(float(self.cfg.smart_crop_side_search_frac))
        self.smart_crop_side_search_spin.valueChanged.connect(self._on_ui_change)
        self.smart_crop_use_grad_check = QtWidgets.QCheckBox()
        self.smart_crop_use_grad_check.setChecked(bool(self.cfg.smart_crop_use_grad))
        self.smart_crop_use_grad_check.setToolTip("bool: smart_crop_use_grad")
        self.smart_crop_use_grad_check.stateChanged.connect(self._on_ui_change)
        self.face_max_frac_spin = _mk_fspin(0.05, 1.0, 0.01, 3, "float: face_max_frac_in_crop")
        self.face_max_frac_spin.setValue(float(self.cfg.face_max_frac_in_crop))
        self.face_min_frac_spin = _mk_fspin(0.0, 1.0, 0.01, 3, "float: face_min_frac_in_crop")
        self.face_min_frac_spin.setValue(float(self.cfg.face_min_frac_in_crop))
        self.crop_min_height_frac_spin = _mk_fspin(0.0, 1.0, 0.01, 3, "float: crop_min_height_frac")
        self.crop_min_height_frac_spin.setValue(float(self.cfg.crop_min_height_frac))
        self.compose_crop_enable_check = QtWidgets.QCheckBox()
        self.compose_crop_enable_check.setChecked(bool(self.cfg.compose_crop_enable))
        self.compose_crop_enable_check.setToolTip("bool: compose_crop_enable")
        self.compose_crop_enable_check.stateChanged.connect(self._on_ui_change)
        self.compose_detect_person_for_face_check = QtWidgets.QCheckBox()
        self.compose_detect_person_for_face_check.setChecked(bool(self.cfg.compose_detect_person_for_face))
        self.compose_detect_person_for_face_check.setToolTip("bool: compose_detect_person_for_face")
        self.compose_detect_person_for_face_check.stateChanged.connect(self._on_ui_change)
        self.compose_close_face_h_frac_spin = _mk_fspin(0.01, 1.0, 0.01, 3, "float: compose_close_face_h_frac")
        self.compose_close_face_h_frac_spin.setValue(float(self.cfg.compose_close_face_h_frac))
        self.compose_close_face_h_frac_spin.valueChanged.connect(self._on_ui_change)
        self.compose_upper_face_h_frac_spin = _mk_fspin(0.01, 1.0, 0.01, 3, "float: compose_upper_face_h_frac")
        self.compose_upper_face_h_frac_spin.setValue(float(self.cfg.compose_upper_face_h_frac))
        self.compose_upper_face_h_frac_spin.valueChanged.connect(self._on_ui_change)
        self.compose_body_face_h_frac_spin = _mk_fspin(0.01, 1.0, 0.005, 3, "float: compose_body_face_h_frac")
        self.compose_body_face_h_frac_spin.setValue(float(self.cfg.compose_body_face_h_frac))
        self.compose_body_face_h_frac_spin.valueChanged.connect(self._on_ui_change)
        self.compose_landscape_face_penalty_spin = _mk_fspin(0.0, 20.0, 0.1, 2, "float: compose_landscape_face_penalty")
        self.compose_landscape_face_penalty_spin.setValue(float(self.cfg.compose_landscape_face_penalty))
        self.compose_landscape_face_penalty_spin.valueChanged.connect(self._on_ui_change)
        self.compose_body_every_n_spin = QtWidgets.QSpinBox()
        self.compose_body_every_n_spin.setRange(0, 300)
        self.compose_body_every_n_spin.setValue(int(self.cfg.compose_body_every_n))
        self.compose_body_every_n_spin.setToolTip("int: compose_body_every_n")
        self.compose_body_every_n_spin.valueChanged.connect(self._on_ui_change)
        self.compose_person_detect_cadence_spin = QtWidgets.QSpinBox()
        self.compose_person_detect_cadence_spin.setRange(1, 300)
        self.compose_person_detect_cadence_spin.setValue(int(getattr(self.cfg, "compose_person_detect_cadence", 6)))
        self.compose_person_detect_cadence_spin.setToolTip("int: compose_person_detect_cadence")
        self.compose_person_detect_cadence_spin.valueChanged.connect(self._on_ui_change)

        labels = [
            ("Aspect ratio W:H", self.ratio_edit),
            ("SDR tonemap target (nits)", self.sdr_nits_spin),
            ("Tonemap desat", self.tm_desat_spin),
            ("Tonemap Mobius param", self.tm_param_spin),
            ("Tonemap backend", self.tonemap_pref_combo),
            ("HDR SDR quality", self.hdr_sdr_quality_combo),
            ("HDR SDR tone curve", self.hdr_sdr_tonemap_combo),
            ("HDR SDR gamut", self.hdr_sdr_gamut_combo),
            ("HDR contrast recovery", self.hdr_sdr_contrast_spin),
            ("HDR peak detect", self.hdr_sdr_peak_check),
            ("Allow inaccurate HDR fallback", self.hdr_sdr_bad_fallback_check),
            ("FFmpeg hardware decode", self.hwaccel_combo),
            ("HDR archive format", self.hdr_crop_format_combo),
            ("Frame stride", self.stride_spin),
            ("YOLO min conf", self.det_conf_spin),
            ("Face max dist", self.face_thr_spin),
            ("Face detector confidence", self.face_det_conf_spin),
            ("Face detector pad", self.face_det_pad_spin),
            ("Face quality min", self.face_quality_spin),
            ("Face visible uses quality", self.face_vis_quality_check),
            ("ReID max dist", self.reid_thr_spin),
            ("Combine", self.combine_combo),
            ("Match mode", self.match_mode_combo),
            ("Disable ReID", self.disable_reid_check),
            ("Face fallback when missed", self.face_fullframe_check),
            ("Full-frame face size", self.face_fullframe_imgsz_spin),
            ("Rotate adaptively", self.rot_adaptive_check),
            ("Rotate every N frames (empty)", self.rot_every_spin),
            ("Rotation grace frames", self.rot_after_hit_spin),
            ("Fast no-face imgsz", self.fast_no_face_spin),
            ("Only best", self.only_best_check),
            ("Enable pre-scan", self.chk_prescan),
            ("Pre-scan stride (frames)", self.spin_prescan_stride),
            ("Pre-scan bank add dist ≤", self.spin_prescan_fd_add),
            ("Pre-scan max width (px)", self.spin_prescan_max_width),
            ("Pre-scan decode max width (px)", self.spin_prescan_decode_max_w),
            ("Pre-scan cache", self.prescan_cache_widget),
            ("Pre-scan face det conf", self.spin_prescan_face_conf),
            ("Pre-scan ENTER dist ≤", self.spin_prescan_fd_enter),
            ("Pre-scan EXIT dist ≥", self.spin_prescan_fd_exit),
            ("Pre-scan add cooldown (samples)", self.spin_prescan_add_cooldown),
            ("Pre-scan min segment sec", self.spin_prescan_min_segment),
            ("Pre-scan pad sec", self.spin_prescan_pad),
            ("Pre-scan bridge gap sec", self.spin_prescan_bridge),
            ("Pre-scan exit cooldown sec", self.spin_prescan_exit_cooldown),
            ("Pre-scan boundary refine sec", self.spin_prescan_refine_window),
            ("Pre-scan refine stride min", self.spin_prescan_refine_stride),
            ("Pre-scan trim pad", self.chk_prescan_trim_pad),
            ("Pre-scan bank max", self.spin_prescan_bank_max),
            ("Pre-scan dedup cos ≥", self.spin_prescan_dedup),
            ("Pre-scan replace margin", self.spin_prescan_margin),
            ("TensorRT lib dir", self._trt_row_widget),
            ("", self.trt_box),
            ("Min sharpness", self.min_sharp_spin),
            ("Min seconds between hits", self.min_gap_spin),
            ("Min box area (px)", self.min_box_pix_spin),
            ("Auto‑crop black borders", self.auto_crop_check),
            ("Border threshold", self.border_thr_spin),
            ("Require face if visible", self.require_face_check),
            ("prefer_face_when_available", self.pref_face_check),
            ("Lock momentum", self.lock_mom_spin),
            ("Suppress negatives", self.suppress_neg_check),
            ("Neg tolerance", self.neg_tol_spin),
            ("Max negatives", self.max_neg_spin),
            ("Log interval (s)", self.log_every_spin),
            ("Lock after N hits", self.lock_after_spin),
            ("Lock face dist ≤", self.lock_face_spin),
            ("Lock reid dist ≤", self.lock_reid_spin),
            ("Score margin vs #2", self.margin_spin),
            ("IoU gate", self.iou_gate_spin),
            ("Face ID backend", self.use_arc_check),
            ("Device", self.device_combo),
            ("YOLO model", self.yolo_edit),
            ("Face YOLO model", self.face_yolo_edit),
            ("Preview every N frames", self.preview_every_spin),
            ("Preview max dimension", self.preview_max_dim_spin),
            ("Preview FPS cap", self.preview_fps_cap_spin),
            ("Seek fast", self.seek_fast_check),
            ("Seek max grabs", self.seek_max_grabs_spin),
            ("Seek preview peek every", self.seek_preview_peek_spin),
            ("Overlay scores", self.overlay_scores_check),
            ("Skip YOLO when faces visible (face-only)", self.skip_yolo_faceonly_check),
            ("Async save crops/CSV", self.async_save_check),
            ("JPEG quality", self.jpg_quality_spin),
            ("", self.annot_check),
            ("allow_faceless_when_locked", self.faceless_allow_check),
            ("learn_bank_runtime", self.learn_bank_runtime_check),
            ("drop_reid_if_any_face_match", self.drop_reid_if_any_face_match_check),
            ("faceless_reid_thresh", self.faceless_reid_spin),
            ("faceless_iou_min", self.faceless_iou_spin),
            ("faceless_persist_frames", self.faceless_persist_spin),
            ("faceless_min_area_frac", self.faceless_min_area_spin),
            ("faceless_max_area_frac", self.faceless_max_area_spin),
            ("faceless_center_max_frac", self.faceless_center_spin),
            ("faceless_min_motion_frac", self.faceless_motion_spin),
            ("crop_face_side_margin_frac", self.crop_face_side_margin_spin),
            ("crop_top_headroom_max_frac", self.crop_top_headroom_spin),
            ("crop_bottom_min_face_heights", self.crop_bottom_min_face_spin),
            ("crop_penalty_weight", self.crop_penalty_weight_spin),
            ("face_anchor_down_frac", self.face_anchor_down_spin),
            ("λ facefrac", self.lambda_facefrac_spin),
            ("Face-center weight", self.crop_center_weight_spin),
            ("Area γ", self.area_gamma_spin),
            ("Area scale by face", self.area_face_scale_w_spin),
            ("Square pull min face", self.square_pull_face_min_spin),
            ("Square pull weight", self.square_pull_weight_spin),
            ("Target close", self.face_target_close_spin),
            ("Target upper", self.face_target_upper_spin),
            ("Target cowboy", self.face_target_cowboy_spin),
            ("Target body", self.face_target_body_spin),
            ("Target tolerance", self.face_target_tol_spin),
            ("Close-up min frac", self.face_target_close_min_spin),
            ("w_close", self.w_close_spin),
            ("w_upper", self.w_upper_spin),
            ("w_cowboy", self.w_cowboy_spin),
            ("w_body", self.w_body_spin),
            ("smart_crop_enable", self.smart_crop_enable_check),
            ("smart_crop_steps", self.smart_crop_steps_spin),
            ("smart_crop_side_search_frac", self.smart_crop_side_search_spin),
            ("smart_crop_use_grad", self.smart_crop_use_grad_check),
            ("face_max_frac_in_crop", self.face_max_frac_spin),
            ("face_min_frac_in_crop", self.face_min_frac_spin),
            ("crop_min_height_frac", self.crop_min_height_frac_spin),
            ("compose_crop_enable", self.compose_crop_enable_check),
            ("compose_detect_person_for_face", self.compose_detect_person_for_face_check),
            ("compose_close_face_h_frac", self.compose_close_face_h_frac_spin),
            ("compose_upper_face_h_frac", self.compose_upper_face_h_frac_spin),
            ("compose_body_face_h_frac", self.compose_body_face_h_frac_spin),
            ("compose_landscape_face_penalty", self.compose_landscape_face_penalty_spin),
            ("compose_body_every_n", self.compose_body_every_n_spin),
            ("compose_person_detect_cadence", self.compose_person_detect_cadence_spin),
        ]
        for row, (lab, w) in enumerate(labels):
            if lab:
                grid.addWidget(QtWidgets.QLabel(lab), row, 0)
                grid.addWidget(w, row, 1)
            else:
                grid.addWidget(w, row, 0, 1, 2)

        # Controls
        ctrl_layout = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Start")
        self.pause_btn = QtWidgets.QPushButton("Pause")
        self.resume_btn = QtWidgets.QPushButton("Resume")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.open_out_btn = QtWidgets.QPushButton("Open Output")
        self.apply_live_btn = QtWidgets.QPushButton("Apply live")
        for b in (self.start_btn, self.pause_btn, self.resume_btn, self.stop_btn, self.open_out_btn, self.apply_live_btn):
            ctrl_layout.addWidget(b)

        # Player
        player_group = QtWidgets.QGroupBox("Player")
        player_v = QtWidgets.QVBoxLayout(player_group)

        # Row 1: transport + speed
        transport_h = QtWidgets.QHBoxLayout()
        self.play_toggle = QtWidgets.QToolButton()
        self.play_toggle.setCheckable(True)
        _style = self.style()
        self._icon_play = _style.standardIcon(QtWidgets.QStyle.SP_MediaPlay)
        self._icon_pause = _style.standardIcon(QtWidgets.QStyle.SP_MediaPause)
        self._update_play_toggle_ui(paused=True)

        self.step_back_btn = QtWidgets.QToolButton()
        self.step_back_btn.setIcon(_style.standardIcon(QtWidgets.QStyle.SP_MediaSkipBackward))
        self.step_back_btn.setToolTip("Prev keyframe (←)")
        self.step_fwd_btn = QtWidgets.QToolButton()
        self.step_fwd_btn.setIcon(_style.standardIcon(QtWidgets.QStyle.SP_MediaSkipForward))
        self.step_fwd_btn.setToolTip("Next keyframe (→)")

        self.speed_combo = QtWidgets.QComboBox()
        self.speed_combo.addItems(["0.25x", "0.5x", "1.0x", "1.5x", "2.0x"])
        self.speed_combo.setCurrentText("1.0x")
        self.speed_combo.setToolTip("Playback speed")

        transport_h.addWidget(self.play_toggle)
        transport_h.addWidget(self.step_back_btn)
        transport_h.addWidget(self.step_fwd_btn)
        transport_h.addWidget(self.speed_combo)
        transport_h.addStretch(1)

        # Row 2: full-width seek bar
        seek_h = QtWidgets.QHBoxLayout()
        self.seek_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.seek_slider.setRange(0, 0)
        self.seek_slider.setSingleStep(1)
        self.seek_slider.setTracking(False)  # only seek on release for performance
        self.seek_slider.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.time_lbl = QtWidgets.QLabel("00:00 / 00:00")
        seek_h.addWidget(self.seek_slider, 1)
        seek_h.addWidget(self.time_lbl)

        player_v.addLayout(transport_h)
        player_v.addLayout(seek_h)

        # Wire up transport
        self.play_toggle.toggled.connect(self._toggle_play_pause)
        self.step_back_btn.clicked.connect(self._handle_step_back)
        self.step_fwd_btn.clicked.connect(self._handle_step_forward)
        self.seek_slider.sliderReleased.connect(lambda: self._handle_seek_slider(self.seek_slider.value()))
        self.seek_slider.sliderMoved.connect(lambda _v: None)
        self.speed_combo.currentTextChanged.connect(self._on_speed_combo_changed)

        # Keyboard shortcuts (active even when slider has focus)
        self._shortcut_space = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Space), self)
        self._shortcut_space.activated.connect(self._shortcut_play_pause)
        self._shortcut_left = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Left), self)
        self._shortcut_left.activated.connect(lambda: self._shortcut_step(False))
        self._shortcut_right = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Right), self)
        self._shortcut_right.activated.connect(lambda: self._shortcut_step(True))
        # Ensure shortcuts fire regardless of focused widget
        for sc in (self._shortcut_space, self._shortcut_left, self._shortcut_right):
            sc.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)

        # Progress + status
        prog_layout = QtWidgets.QHBoxLayout()
        self.progress = QtWidgets.QProgressBar()
        self.status_lbl = QtWidgets.QLabel("Idle")
        self.meta_lbl = QtWidgets.QLabel("")  # fps + total frames
        prog_layout.addWidget(self.progress, 3)
        prog_layout.addWidget(self.meta_lbl, 1)
        prog_layout.addWidget(self.status_lbl, 2)

        # Center: preview + player (below) + last hit
        mid_split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        # Preview
        try:
            from .hdr_preview import HDRPreviewWidget  # type: ignore
        except Exception:
            try:
                from hdr_preview import HDRPreviewWidget  # type: ignore
            except Exception:  # pragma: no cover - optional dependency
                HDRPreviewWidget = None  # type: ignore

        prev_group = QtWidgets.QGroupBox("Live preview")
        prev_layout = QtWidgets.QVBoxLayout(prev_group)

        # SDR preview (existing path)
        self.preview_label = QtWidgets.QLabel()
        self.preview_label.setAlignment(QtCore.Qt.AlignCenter)
        self.preview_label.setMinimumHeight(1)

        # HDR passthrough widget (no-op placeholder if DLL missing)
        if HDRPreviewWidget is not None:
            self.hdr_widget = HDRPreviewWidget()
        else:
            self.hdr_widget = QtWidgets.QWidget()
            self.hdr_widget.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.hdr_widget.setMinimumHeight(1)

        self.preview_stack = QtWidgets.QStackedWidget()
        self.preview_stack.addWidget(self.preview_label)
        self.preview_stack.addWidget(self.hdr_widget)
        self.preview_stack.setCurrentIndex(0)
        prev_layout.addWidget(self.preview_stack)

        self._hdr_passthrough_supported = (
            HDRPreviewWidget is not None and _hdr_passthrough_available()
        )
        _log.info(
            "HDR passthrough support: %s (HDRPreviewWidget=%s, available=%s)",
            self._hdr_passthrough_supported,
            HDRPreviewWidget is not None,
            _hdr_passthrough_available(),
        )
        if hasattr(self, "chk_hdr_passthrough"):
            if not self._hdr_passthrough_supported:
                self.chk_hdr_passthrough.setChecked(False)
                self.chk_hdr_passthrough.setEnabled(False)
                self.chk_hdr_passthrough.setToolTip(
                    "HDR passthrough unavailable (pc_hdr_vulkan.dll missing or Vulkan HDR not supported)."
                )
                self.chk_hdr_passthrough.setVisible(False)
        prev_container = QtWidgets.QWidget()
        prev_container_layout = QtWidgets.QVBoxLayout(prev_container)
        prev_container_layout.setContentsMargins(0, 0, 0, 0)
        prev_container_layout.addWidget(prev_group, 4)
        prev_container_layout.addWidget(player_group, 1)
        # Last hit
        hit_group = QtWidgets.QGroupBox("Last saved crop")
        hit_layout = QtWidgets.QVBoxLayout(hit_group)
        self.hit_label = QtWidgets.QLabel()
        self.hit_label.setAlignment(QtCore.Qt.AlignCenter)
        self.hit_label.setMinimumHeight(1)
        hit_layout.addWidget(self.hit_label)
        mid_split.addWidget(prev_container)
        mid_split.addWidget(hit_group)
        mid_split.setSizes([600, 600])

        # Bottom: log
        log_group = QtWidgets.QGroupBox("Log")
        log_layout = QtWidgets.QVBoxLayout(log_group)
        self.log_edit = QtWidgets.QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMinimumHeight(60)
        log_layout.addWidget(self.log_edit)

        # ---------- Dockable UI ----------
        # Left dock: controls (scrollable, tabs + search)
        controls_container = QtWidgets.QWidget()
        controls_layout = QtWidgets.QVBoxLayout(controls_container); controls_layout.setContentsMargins(6, 6, 6, 6)
        # show update banner (hidden by default) just above the search/filter
        self.update_banner = QtWidgets.QFrame()
        self.update_banner.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.update_banner.setVisible(False)
        banner_l = QtWidgets.QHBoxLayout(self.update_banner); banner_l.setContentsMargins(8,4,8,4)
        self.update_lbl = QtWidgets.QLabel("Update available")
        self.btn_update_now = QtWidgets.QPushButton("Update")
        self.btn_update_now.clicked.connect(self._on_update_now)
        self.btn_update_later = QtWidgets.QPushButton("Later")
        self.btn_update_later.clicked.connect(lambda: self.update_banner.setVisible(False))
        self.btn_release_notes = QtWidgets.QPushButton("Release notes")
        self.btn_release_notes.clicked.connect(self._open_release_notes)
        banner_l.addWidget(self.update_lbl, 1)
        banner_l.addWidget(self.btn_update_now, 0)
        banner_l.addWidget(self.btn_release_notes, 0)
        banner_l.addWidget(self.btn_update_later, 0)
        controls_layout.addWidget(self.update_banner)
        self.search_edit = QtWidgets.QLineEdit(); self.search_edit.setPlaceholderText("Filter settings…")
        controls_layout.addWidget(self.search_edit)
        self.tabs = QtWidgets.QTabWidget()
        tab_files = QtWidgets.QWidget(); tab_files_l = QtWidgets.QVBoxLayout(tab_files); tab_files_l.setContentsMargins(0, 0, 0, 0)
        tab_params = QtWidgets.QWidget(); tab_params_l = QtWidgets.QVBoxLayout(tab_params); tab_params_l.setContentsMargins(0, 0, 0, 0)
        tab_files_l.addWidget(file_group, 1); tab_files_l.addStretch(0)
        tab_params_l.addWidget(param_group, 1); tab_params_l.addStretch(0)
        self.tabs.addTab(tab_files, "Files")
        self.tabs.addTab(tab_params, "Parameters")
        # Curate tab
        if "CurateTab" in globals() and CurateTab is not None:
            try:
                default_pool = self.out_edit.text().strip() if hasattr(self, "out_edit") else getattr(self, "cfg", None).out_dir
            except Exception:
                default_pool = "output"
            try:
                default_ref = self.ref_edit.text().strip() if hasattr(self, "ref_edit") else getattr(self, "cfg", None).ref
            except Exception:
                default_ref = ""
            try:
                self.curate_tab = CurateTab(self.tabs, default_pool=default_pool, default_ref=default_ref)
                self.tabs.addTab(self.curate_tab, "Curate")
            except Exception as _e:
                # Non-fatal if curate tab cannot load
                pass
        controls_layout.addWidget(self.tabs, 1)
        controls_layout.addLayout(ctrl_layout)
        controls_layout.addLayout(prog_layout)
        controls_scroll = QtWidgets.QScrollArea(); controls_scroll.setWidget(controls_container); controls_scroll.setWidgetResizable(True)
        self.dock_controls = QDockWidget("Controls", self); self.dock_controls.setObjectName("dock_controls")
        self.dock_controls.setWidget(controls_scroll)
        self.dock_controls.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        self.dock_controls.setFeatures(
            QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
        )
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.dock_controls)

        # Center: previews
        center = QtWidgets.QWidget(); center_l = QtWidgets.QVBoxLayout(center); center_l.setContentsMargins(0, 0, 0, 0)
        center_l.addWidget(mid_split, 1)
        self.setCentralWidget(center)

        # Bottom dock: log (resizable)
        self.dock_log = QDockWidget("Log", self); self.dock_log.setObjectName("dock_log")
        self.dock_log.setWidget(log_group)
        self.dock_log.setAllowedAreas(QtCore.Qt.BottomDockWidgetArea | QtCore.Qt.TopDockWidgetArea)
        self.dock_log.setFeatures(
            QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
        )
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.dock_log)

        # Initial sizes
        self.resize(1280, 800)
        self.dock_controls.resize(360, 800)
        self.dock_log.resize(1280, 180)

        # Menu
        self._build_menu()

        # Wire
        vid_btn.clicked.connect(lambda: self._pick_file(self.video_edit, "Video (*.mp4 *.mov *.mkv *.avi *.webm);;All files (*)"))
        ref_btn.clicked.connect(self._browse_refs_from_line)
        out_btn.clicked.connect(lambda: self._pick_dir(self.out_edit))
        self.btn_ref_add.clicked.connect(self._browse_refs_multi)
        self.btn_ref_remove.clicked.connect(self._remove_selected_refs)
        self.btn_ref_clear.clicked.connect(self._clear_refs)
        self.ref_edit.editingFinished.connect(self._sync_from_line)
        self.ref_edit.returnPressed.connect(self._sync_from_line)
        self.ref_list.model().rowsMoved.connect(self._on_ref_rows_moved)
        self.ref_list.filesDropped.connect(self._handle_ref_files_dropped)
        self.ref_list.customContextMenuRequested.connect(self._show_ref_context_menu)
        self.ref_list.itemDoubleClicked.connect(lambda _: self._open_selected_refs())
        QtGui.QShortcut(QtGui.QKeySequence.Delete, self.ref_list, activated=self._remove_selected_refs)
        self.start_btn.clicked.connect(self.on_start)
        self.stop_btn.clicked.connect(self.on_stop)
        self.pause_btn.clicked.connect(lambda: self._pause(True))
        self.resume_btn.clicked.connect(lambda: self._pause(False))
        self.open_out_btn.clicked.connect(self._open_out_dir)
        self.apply_live_btn.clicked.connect(self._apply_live_cfg)
        if hasattr(self, "btn_curate_run"):
            try:
                self.btn_curate_run.clicked.connect(self._invoke_curator)
            except Exception:
                pass

        self._update_buttons(state="idle")

    def _make_ffmpeg_row(self) -> QtWidgets.QWidget:
        row_widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(row_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        self._ffmpeg_edit = QtWidgets.QLineEdit()
        self._ffmpeg_edit.setPlaceholderText("e.g. D:\\ffmpeg or /usr/local/ffmpeg/bin")
        initial = getattr(self, "_initial_ffmpeg_dir", "")
        if initial:
            self._ffmpeg_edit.setText(initial)
        else:
            try:
                s = QtCore.QSettings(APP_ORG, APP_NAME)
                self._ffmpeg_edit.setText(s.value(_SETTINGS_KEY_FFMPEG_DIR, "", type=str) or "")
            except Exception:
                pass
        browse_btn = QtWidgets.QPushButton("Browse…")
        browse_btn.setAutoDefault(False)
        browse_btn.clicked.connect(self._browse_ffmpeg_dir)
        apply_btn = QtWidgets.QPushButton("Apply")
        apply_btn.setAutoDefault(False)
        apply_btn.clicked.connect(self._apply_ffmpeg_dir_from_field)
        layout.addWidget(self._ffmpeg_edit, 1)
        layout.addWidget(browse_btn)
        layout.addWidget(apply_btn)
        return row_widget

    @QtCore.Slot()
    def _browse_ffmpeg_dir(self):
        start = ""
        try:
            start = self._ffmpeg_edit.text().strip()
        except Exception:
            start = ""
        if not start:
            start = os.getcwd()
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Select FFmpeg folder", start)
        if directory:
            self._ffmpeg_edit.setText(directory)

    @QtCore.Slot()
    def _apply_ffmpeg_dir_from_field(self):
        path = ""
        try:
            path = self._ffmpeg_edit.text().strip()
        except Exception:
            path = ""
        if not path:
            QtWidgets.QMessageBox.warning(self, "FFmpeg", "Please select a folder.")
            return
        try:
            s = QtCore.QSettings(APP_ORG, APP_NAME)
            s.setValue(_SETTINGS_KEY_FFMPEG_DIR, path)
            s.sync()
        except Exception:
            pass
        try:
            set_ffmpeg_env(path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "FFmpeg", f"Failed to apply environment:\n{exc}")
            return
        self._initial_ffmpeg_dir = path
        ffmpeg_path, ffprobe_path = resolve_ffmpeg_bins(path)
        flags = (
            ffmpeg_has_hdr_filters(ffmpeg_path)
            if ffmpeg_path
            else {"libplacebo": False, "zscale": False, "tonemap": False}
        )
        message = [
            f"FFmpeg: {ffmpeg_path or 'not found'}",
            f"ffprobe: {ffprobe_path or 'not found'}",
            "Filters → libplacebo={libplacebo} zscale={zscale} tonemap={tonemap}".format(**flags),
        ]
        if self._worker is not None:
            message.append(
                "Note: if processing is currently running, the new FFmpeg will be used next time you open a video."
            )
        QtWidgets.QMessageBox.information(self, "FFmpeg configured", "\n".join(message))

    @property
    def processor(self) -> Processor:
        """Return the active processor or a fallback instance for curator runs."""

        if self._worker is not None:
            return self._worker
        if self._curator_fallback is None:
            self._curator_fallback = Processor(self.cfg)
            try:
                self._curator_fallback.status.connect(self._on_status)
            except Exception:
                pass
        else:
            self._curator_fallback.cfg = self.cfg
        return self._curator_fallback

    # ---- Ref list helpers ----
    def _filter_images(self, paths: list[str]) -> list[str]:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        collected: list[str] = []
        max_depth = 5
        for path in paths:
            candidate = path.strip() if isinstance(path, str) else path
            if not candidate:
                continue
            try:
                if os.path.isdir(candidate):
                    try:
                        base_depth = len(Path(candidate).resolve().parts)
                    except Exception:
                        base_depth = len(Path(candidate).parts)
                    for root, dirs, files in os.walk(candidate):
                        try:
                            root_depth = len(Path(root).resolve().parts)
                        except Exception:
                            root_depth = len(Path(root).parts)
                        depth = root_depth - base_depth
                        if max_depth >= 0 and depth >= max_depth:
                            dirs[:] = []
                        else:
                            dirs[:] = [d for d in dirs if not d.startswith('.')]
                        for name in files:
                            if name.startswith('.'):
                                continue
                            if os.path.splitext(name)[1].lower() in exts:
                                collected.append(os.path.join(root, name))
                elif os.path.isfile(candidate):
                    name = os.path.basename(candidate)
                    if name.startswith('.'):
                        continue
                    if os.path.splitext(candidate)[1].lower() in exts:
                        collected.append(candidate)
            except Exception:
                continue

        unique: list[str] = []
        seen: set[str] = set()
        for path in collected:
            try:
                resolved = os.path.normpath(os.path.abspath(path))
                key = os.path.normcase(resolved)
                display_path = resolved
            except Exception:
                display_path = path
                key = path
            if key in seen:
                continue
            seen.add(key)
            unique.append(display_path)
        return unique

    def _set_ref_paths(self, paths: list[str]) -> None:
        self._updating_refs = True
        self.ref_list.clear()
        seen: set[str] = set()
        for path in paths:
            raw = path.strip()
            if not raw:
                continue
            try:
                display_path = os.path.normpath(os.path.abspath(raw))
                key = os.path.normcase(display_path)
            except Exception:
                display_path = raw
                key = raw
            if key in seen:
                continue
            self.ref_list.addItem(display_path)
            item = self.ref_list.item(self.ref_list.count() - 1)
            if item is not None:
                item.setToolTip(display_path)
            seen.add(key)
        self._updating_refs = False
        self._sync_ref_edit_from_list()

    def _get_ref_paths(self) -> list[str]:
        return [self.ref_list.item(i).text().strip() for i in range(self.ref_list.count())]

    def _get_selected_ref_paths(self) -> list[str]:
        return [item.text().strip() for item in self.ref_list.selectedItems() if item.text().strip()]

    def _sync_ref_edit_from_list(self) -> None:
        if self._updating_refs:
            return
        self._updating_refs = True
        self.ref_edit.setText("; ".join(self._get_ref_paths()))
        self._updating_refs = False

    def _browse_refs_multi(self) -> None:
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select reference images",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp *.webp)",
        )
        if files:
            self._set_ref_paths(self._get_ref_paths() + self._filter_images(list(files)))
            self._save_qsettings()

    def _remove_selected_refs(self) -> None:
        for item in self.ref_list.selectedItems():
            self.ref_list.takeItem(self.ref_list.row(item))
        self._set_ref_paths(self._get_ref_paths())
        self._save_qsettings()

    def _clear_refs(self) -> None:
        self.ref_list.clear()
        self._set_ref_paths([])
        self._save_qsettings()

    def _browse_refs_from_line(self) -> None:
        self._browse_refs_multi()

    def _sync_from_line(self) -> None:
        if self._updating_refs:
            return
        text = self.ref_edit.text()
        paths = self._filter_images([part.strip() for part in text.split(';') if part.strip()])
        self._set_ref_paths(paths)
        self._save_qsettings()

    def _handle_ref_files_dropped(self, files: list[str]) -> None:
        filtered = self._filter_images(list(files))
        if not filtered:
            return
        self._set_ref_paths(self._get_ref_paths() + filtered)
        self._save_qsettings()

    def _on_ref_rows_moved(self, *args) -> None:
        self._sync_ref_edit_from_list()
        try:
            self._save_qsettings()
        except Exception:
            pass

    def _show_ref_context_menu(self, pos: QtCore.QPoint) -> None:
        menu = QtWidgets.QMenu(self)
        open_action = menu.addAction("Open")
        if sys.platform == "darwin":
            reveal_label = "Reveal in Finder"
        elif sys.platform.startswith("win"):
            reveal_label = "Reveal in Explorer"
        else:
            reveal_label = "Show in Folder"
        reveal_action = menu.addAction(reveal_label)
        menu.addSeparator()
        remove_action = menu.addAction("Remove selected")

        has_selection = bool(self.ref_list.selectedItems())
        open_action.setEnabled(has_selection)
        reveal_action.setEnabled(has_selection)
        remove_action.setEnabled(has_selection)

        action = menu.exec(self.ref_list.mapToGlobal(pos))
        if action == open_action:
            self._open_selected_refs()
        elif action == reveal_action:
            self._reveal_selected_refs()
        elif action == remove_action:
            self._remove_selected_refs()

    def _open_selected_refs(self) -> None:
        for path in self._get_selected_ref_paths():
            try:
                QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(path))
            except Exception:
                pass

    def _reveal_selected_refs(self) -> None:
        paths = self._get_selected_ref_paths()
        if not paths:
            return
        target = paths[0]
        if not os.path.exists(target):
            target = os.path.dirname(target)
        if not target:
            return
        try:
            if sys.platform.startswith("win"):
                path = os.path.normpath(target)
                if os.path.isdir(path):
                    subprocess.Popen(["explorer", path])
                else:
                    subprocess.Popen(["explorer", "/select,", path])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", "-R", target])
            else:
                folder = target if os.path.isdir(target) else os.path.dirname(target)
                subprocess.Popen(["xdg-open", folder or os.getcwd()])
        except Exception:
            pass

    def _build_menu(self):
        bar = self.menuBar()
        file_menu = bar.addMenu("&File")
        act_save = QtGui.QAction("Save preset...", self, triggered=self._save_preset)
        act_load = QtGui.QAction("Load preset...", self, triggered=self._load_preset)
        act_quit = QtGui.QAction("Quit", self, triggered=self.close)
        file_menu.addAction(act_save)
        file_menu.addAction(act_load)
        file_menu.addSeparator()
        file_menu.addAction(act_quit)

        help_menu = bar.addMenu("&Help")
        act_about = QtGui.QAction("About", self, triggered=self._about)
        help_menu.addAction(act_about)
        # --- Updates ---
        help_menu.addSeparator()
        self.act_check_updates = QtGui.QAction("Check for updates…", self, triggered=self._manual_check_updates)
        help_menu.addAction(self.act_check_updates)
        self.act_auto_check = QtGui.QAction("Auto-check at startup", self)
        self.act_auto_check.setCheckable(True)
        try:
            s = QtCore.QSettings(APP_ORG, APP_NAME)
            self.act_auto_check.setChecked(s.value("update_auto_check", True, type=bool))
        except Exception:
            self.act_auto_check.setChecked(True)
        self.act_auto_check.toggled.connect(self._toggle_auto_check)
        help_menu.addAction(self.act_auto_check)

    # ---------- Updater plumbing ----------
    def _connect_updater(self):
        if not self.updater:
            return
        try:
            self.updater.progress.connect(lambda m: self._log(f"[Update] {m}"))
            self.updater.info.connect(lambda m: self._log(f"[Update] {m}"))
            self.updater.updateAvailable.connect(self._on_update_available)
            self.updater.upToDate.connect(lambda: self.statusbar.showMessage("Up to date.", 3000))
            def _on_update_failed(message: str) -> None:
                try:
                    self._log(f"[Update] FAILED: {message}")
                except Exception:
                    pass
                try:
                    self.statusbar.showMessage(f"Update check failed: {message}", 6000)
                except Exception:
                    pass

            self.updater.updateFailed.connect(_on_update_failed)
            self.updater.updated.connect(self._on_updated)
        except Exception:
            pass

    def _maybe_auto_check_updates(self):
        try:
            if not self.updater:
                return
            s = QtCore.QSettings(APP_ORG, APP_NAME)
            if s.value("update_auto_check", True, type=bool):
                QtCore.QTimer.singleShot(
                    0,
                    lambda: self.updater.check_for_updates_async(
                        branch=None, force=True, throttle_sec=0
                    ),
                )
        except Exception:
            pass

    def safe_fit_window(self):
        try:
            screen = QtWidgets.QApplication.primaryScreen()
            if screen:
                avail = screen.availableGeometry()
                w = min(self.width() or 1200, max(800, avail.width() - 40))
                h = min(self.height() or 900, max(600, avail.height() - 80))
                self.resize(w, h)
        except Exception:
            pass

    def toggle_compact_mode(self, on: bool):
        m = 2 if on else 6
        try:
            for gb in self.findChildren(QtWidgets.QGroupBox):
                gb.setFlat(on)
            for lay in self.findChildren(QtWidgets.QLayout):
                if hasattr(lay, "setContentsMargins"):
                    lay.setContentsMargins(m, m, m, m)
        except Exception:
            pass
        if hasattr(self, "search_edit"):
            self.search_edit.setVisible(not on)

    def reset_layout(self):
        try:
            self.removeDockWidget(self.dock_controls)
            self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.dock_controls)
            self.removeDockWidget(self.dock_log)
            self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.dock_log)
            self.resize(1280, 800)
            self.dock_controls.resize(360, 800)
            self.dock_log.resize(1280, 180)
        except Exception:
            pass

    def eventFilter(self, watched, event):
        try:
            if (
                event.type() == QtCore.QEvent.Type.Wheel
                and bool(watched.property("_pc_no_wheel_value_edit"))
            ):
                self._scroll_parent_area(watched, event)
                return True
        except Exception:
            pass
        return super().eventFilter(watched, event)

    def _mark_no_wheel_value_edit(self, widget: QtWidgets.QWidget):
        watched = []
        value_types = (
            QtWidgets.QAbstractSpinBox,
            QtWidgets.QComboBox,
            QtWidgets.QAbstractSlider,
        )
        if isinstance(widget, value_types):
            watched.append(widget)
        for widget_type in value_types:
            watched.extend(widget.findChildren(widget_type))

        seen = set()
        for child in watched:
            ident = id(child)
            if ident in seen:
                continue
            seen.add(ident)
            child.setProperty("_pc_no_wheel_value_edit", True)
            child.installEventFilter(self)

    def _scroll_parent_area(self, widget: QtWidgets.QWidget, event: QtGui.QWheelEvent):
        parent = widget.parent()
        while parent is not None and not isinstance(parent, QtWidgets.QScrollArea):
            parent = parent.parent()
        if parent is None:
            return

        bar = parent.verticalScrollBar()
        pixel_delta = event.pixelDelta().y()
        if pixel_delta:
            delta = -pixel_delta
        else:
            delta = -int(event.angleDelta().y() / 120.0 * max(1, bar.singleStep() * 3))
        if delta:
            bar.setValue(bar.value() + delta)

    def _install_filter(self):
        # Build label->row index for settings filtering
        try:
            self._param_rows = []
            for gb in self.findChildren(QtWidgets.QGroupBox):
                if "Parameters" in gb.title():
                    layout = gb.layout()
                    if isinstance(layout, QtWidgets.QFormLayout):
                        for i in range(layout.rowCount()):
                            li = layout.itemAt(i, QtWidgets.QFormLayout.LabelRole)
                            fi = layout.itemAt(i, QtWidgets.QFormLayout.FieldRole)
                            if li and fi and li.widget() and fi.widget():
                                self._param_rows.append((li.widget(), fi.widget()))
                    elif isinstance(layout, QtWidgets.QGridLayout):
                        for i in range(layout.rowCount()):
                            li = layout.itemAtPosition(i, 0)
                            fi = layout.itemAtPosition(i, 1)
                            if li and fi and li.widget() and fi.widget():
                                self._param_rows.append((li.widget(), fi.widget()))
            for _, field in getattr(self, "_param_rows", []):
                self._mark_no_wheel_value_edit(field)
            if hasattr(self, "search_edit"):
                self.search_edit.textChanged.connect(self._apply_filter)
        except Exception:
            pass

    def _apply_filter(self, text: str):
        q = (text or "").strip().lower()
        try:
            for lbl, field in getattr(self, "_param_rows", []):
                show = True if not q else (q in lbl.text().lower())
                lbl.setVisible(show)
                field.setVisible(show)
        except Exception:
            pass

    # ---- Update actions ----------------------------------------------------
    def _toggle_auto_check(self, on: bool):
        try:
            s = QtCore.QSettings(APP_ORG, APP_NAME)
            s.setValue("update_auto_check", bool(on)); s.sync()
        except Exception:
            pass

    def _manual_check_updates(self):
        if not self.updater:
            QtWidgets.QMessageBox.information(self, "Updates", "Updater not available in this build.")
            return
        self.statusbar.showMessage("Checking for updates…", 2000)
        self.updater.check_for_updates_async(branch=None, force=True, throttle_sec=0)

    def _on_update_available(self, msg: str):
        self.update_lbl.setText(msg)
        self.update_banner.setVisible(True)
        self.statusbar.showMessage(msg, 7000)
        self._last_update_compare_url = self._compute_compare_url()

    def _on_update_now(self):
        if not self.updater:
            return
        btn = QtWidgets.QMessageBox.question(
            self,
            "Apply update",
            "Update will be downloaded/applied. Continue?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if btn != QtWidgets.QMessageBox.Yes:
            return
        self.updater.perform_update_async(prefer_git=True, branch=None)

    def _on_updated(self, msg: str):
        self._log("[Update] " + msg)
        self.update_banner.setVisible(False)
        self._last_update_compare_url = None
        # Offer restart
        btn = QtWidgets.QMessageBox.question(
            self,
            "Restart required",
            "Update installed. Restart now to finish?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if btn == QtWidgets.QMessageBox.Yes:
            try:
                # Save UI state before restart
                s = QtCore.QSettings(APP_ORG, APP_NAME)
                s.setValue("dock_state", self.saveState()); s.sync()
            except Exception:
                pass
            try:
                self._save_qsettings()
            except Exception:
                pass
            try:
                self.updater.restart_now()
            except Exception:
                # Best-effort fallback
                QtWidgets.QApplication.quit()

    def _compute_compare_url(self) -> str:
        try:
            if self.updater and getattr(self.updater, "repo", None) and (self.updater.repo / ".git").exists():
                local = subprocess.check_output([
                    "git",
                    "rev-parse",
                    "HEAD",
                ], cwd=self.updater.repo, text=True).strip()
                upstream_ref = subprocess.check_output([
                    "git",
                    "rev-parse",
                    "--abbrev-ref",
                    "--symbolic-full-name",
                    "@{u}",
                ], cwd=self.updater.repo, text=True).strip()
                remote = subprocess.check_output([
                    "git",
                    "rev-parse",
                    upstream_ref,
                ], cwd=self.updater.repo, text=True).strip()
                if local and remote:
                    return f"https://github.com/xmarre/person_capture/compare/{local}...{remote}"
        except Exception:
            pass
        return "https://github.com/xmarre/person_capture/commits"

    def _open_release_notes(self):
        import webbrowser

        url = self._last_update_compare_url or self._compute_compare_url()
        try:
            webbrowser.open(url)
        except Exception:
            pass

    def _on_ui_change(self, *_args) -> None:
        if getattr(self, "_in_ui_change", False):
            return
        self._in_ui_change = True
        try:
            if self._worker:
                self._apply_live_cfg()
            else:
                self._save_qsettings()
        except Exception:
            pass
        finally:
            self._in_ui_change = False

    def _clear_prescan_cache(self) -> None:
        root = _REPO_ROOT / "prescan_cache"
        try:
            # Respect a custom relative/absolute cache dir from the current config.
            raw = str(getattr(self.cfg, "prescan_cache_dir", "prescan_cache") or "prescan_cache").strip()
            root = Path(raw)
            if not root.is_absolute():
                root = _REPO_ROOT / root
        except Exception:
            root = _REPO_ROOT / "prescan_cache"
        try:
            root = root.resolve()
            repo_root = _REPO_ROOT.resolve()
            home = Path.home().resolve()
            fs_root = Path(root.anchor)
            if root in {fs_root, home, repo_root}:
                raise ValueError(f"Refusing to clear unsafe cache path: {root}")
            try:
                root.relative_to(repo_root)
            except ValueError as exc:
                raise ValueError(f"Refusing to clear non-repo cache path: {root}") from exc
            if root.exists():
                shutil.rmtree(root)
            root.mkdir(parents=True, exist_ok=True)
            self.statusBar().showMessage(f"Cleared pre-scan cache: {root}", 5000)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Clear pre-scan cache failed",
                f"Could not clear pre-scan cache:\n{root}\n\n{exc}",
            )

    # ------------- Actions -------------

    def _pick_file(self, line: QtWidgets.QLineEdit, filt: str):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select file", line.text() or "", filt)
        if p:
            line.setText(p)

    def _pick_dir(self, line: QtWidgets.QLineEdit):
        p = QtWidgets.QFileDialog.getExistingDirectory(self, "Select folder", line.text() or "")
        if p:
            line.setText(p)

    def _open_out_dir(self):
        p = self.out_edit.text().strip()
        if not p:
            return
        Path(p).mkdir(parents=True, exist_ok=True)
        if sys.platform.startswith("win"):
            os.startfile(p)
        elif sys.platform.startswith("darwin"):
            os.system(f'open "{p}"')
        else:
            os.system(f'xdg-open "{p}"')

    def _invoke_curator(self) -> None:
        """Trigger the processor's curator slot, falling back to an in-place run."""

        worker = getattr(self, "_worker", None)
        if not worker:
            try:
                proc = self.processor
                proc.cfg = self.cfg
                proc.run_curator()  # type: ignore[attr-defined]
            except Exception as exc:
                self._log(f"Curator failed: {exc}")
            return
        try:
            conn_type = (
                QtCore.Qt.ConnectionType.QueuedConnection
                if hasattr(QtCore.Qt, "ConnectionType")
                else QtCore.Qt.QueuedConnection
            )
            QtCore.QMetaObject.invokeMethod(
                worker,
                "run_curator",
                conn_type,
            )
        except Exception as exc:
            self._log(f"Curator dispatch failed: {exc}")

    def _queue_worker_disable_hdr_passthrough(self) -> None:
        worker = getattr(self, "_worker", None)
        if worker is None:
            return
        try:
            conn_type = (
                QtCore.Qt.ConnectionType.QueuedConnection
                if hasattr(QtCore.Qt, "ConnectionType")
                else QtCore.Qt.QueuedConnection
            )
            QtCore.QMetaObject.invokeMethod(
                worker,
                "disable_hdr_passthrough",
                conn_type,
            )
        except Exception:
            logging.getLogger(__name__).debug(
                "Failed to queue HDR passthrough shutdown",
                exc_info=True,
            )

    def _on_speed_combo_changed(self, txt: str):
        if not self._worker:
            return
        try:
            speed = float(txt.replace("x", ""))
        except Exception:
            speed = 1.0
        self._worker.set_speed(speed)

    # --- Play/Pause toggle ---
    def _update_play_toggle_ui(self, paused: bool):
        if not hasattr(self, "play_toggle"):
            return
        self.play_toggle.blockSignals(True)
        if paused:
            self.play_toggle.setChecked(False)
            self.play_toggle.setIcon(self._icon_play)
            self.play_toggle.setText("Play")
            self.play_toggle.setToolTip("Play (Space)")
        else:
            self.play_toggle.setChecked(True)
            self.play_toggle.setIcon(self._icon_pause)
            self.play_toggle.setText("Pause")
            self.play_toggle.setToolTip("Pause (Space)")
        self.play_toggle.blockSignals(False)

    def _toggle_play_pause(self, checked: bool):
        if not self._worker:
            return
        if checked:
            self._worker.play()
            self._update_play_toggle_ui(paused=False)
        else:
            self._worker.pause()
            self._update_play_toggle_ui(paused=True)

    def _handle_step_back(self, _checked: bool = False):
        self._jump_keyframe(False)

    def _handle_step_forward(self, _checked: bool = False):
        self._jump_keyframe(True)

    def _handle_seek_slider(self, value: int):
        if self._worker:
            self._worker.seek_frame(int(value))

    def _pause(self, flag: bool):
        if self._worker:
            self._worker.request_pause(flag)
            self._log(f"{'Paused' if flag else 'Resumed'}")
            self._update_play_toggle_ui(paused=bool(flag))
        else:
            self._update_play_toggle_ui(paused=True)

    def _save_preset(self):
        cfg = self._collect_cfg()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save preset", "preset.json", "JSON (*.json)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(cfg.to_json())
        self._log(f"Preset saved: {path}")

    def _load_preset(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load preset", "", "JSON (*.json)")
        if not path:
            return
        current_cfg = self._collect_cfg()
        with open(path, "r", encoding="utf-8") as f:
            cfg = SessionConfig.from_json(f.read())
        cfg.video = current_cfg.video
        cfg.ref = current_cfg.ref
        cfg.out_dir = current_cfg.out_dir
        self._apply_cfg(cfg)
        self._log(f"Preset loaded: {path}")

    def _about(self):
        QtWidgets.QMessageBox.information(self, "About", "PersonCapture GUI\nControls the video→crops pipeline with live preview, presets, and CSV index.")

    def _apply_live_cfg(self):
        if not self._worker:
            return
        new = self._collect_cfg()
        prescan_live = {
            "prescan_enable",
            "prescan_stride",
            "prescan_max_width",
            "prescan_decode_max_w",
            "prescan_face_conf",
            "prescan_fd_enter",
            "prescan_fd_add",
            "prescan_fd_exit",
            "prescan_add_cooldown_samples",
            "prescan_min_segment_sec",
            "prescan_pad_sec",
            "prescan_bridge_gap_sec",
            "prescan_exit_cooldown_sec",
            "prescan_boundary_refine_sec",
            "prescan_refine_stride_min",
            "prescan_trim_pad",
            "prescan_bank_max",
            "prescan_diversity_dedup_cos",
            "prescan_replace_margin",
            "prescan_fd9_skip",
            "prescan_fd9_grace",
            "prescan_fd9_probe_period",
        }
        live = {
            "frame_stride",
            "min_det_conf",
            "face_thresh",
            "reid_thresh",
            "combine",
            "match_mode",
            "only_best",
            "min_sharpness",
            "min_gap_sec",
            "min_box_pixels",
            "auto_crop_borders",
            "border_threshold",
            "score_margin",
            "iou_gate",
            "preview_every",
            "preview_max_dim",
            "preview_fps_cap",
            "require_face_if_visible",
            "prefer_face_when_available",
            "suppress_negatives",
            "neg_tolerance",
            "max_negatives",
            "log_interval_sec",
            "lock_after_hits",
            "lock_face_thresh",
            "lock_reid_thresh",
            "lock_momentum",
            "allow_faceless_when_locked",
            "learn_bank_runtime",
            "drop_reid_if_any_face_match",
            "faceless_reid_thresh",
            "faceless_iou_min",
            "faceless_persist_frames",
            "faceless_min_area_frac",
            "faceless_max_area_frac",
            "faceless_center_max_frac",
            "faceless_min_motion_frac",
            "seek_fast",
            "seek_max_grabs",
            "seek_preview_peek_every",
            "crop_face_side_margin_frac",
            "crop_top_headroom_max_frac",
            "crop_bottom_min_face_heights",
            "crop_penalty_weight",
            "crop_head_side_pad_frac",
            "crop_head_top_pad_frac",
            "crop_head_bottom_pad_frac",
            "wide_face_aspect_penalty_weight",
            "wide_face_min_frame_frac",
            "wide_face_aspect_limit",
            "face_anchor_down_frac",
            "smart_crop_enable",
            "smart_crop_steps",
            "smart_crop_side_search_frac",
            "smart_crop_use_grad",
            "face_max_frac_in_crop",
            "face_min_frac_in_crop",
            "crop_min_height_frac",
            "face_visible_uses_quality",
            "face_quality_min",
            "face_det_conf",
            "face_det_pad",
            "face_fullframe_imgsz",
            "rot_adaptive",
            "rot_every_n",
            "rot_after_hit_frames",
            "fast_no_face_imgsz",
            "overlay_scores",
            "hdr_screencap_fullres",
            "hdr_archive_crops",
            "hdr_crop_format",
            "hdr_sdr_quality",
            "hdr_sdr_tonemap",
            "hdr_sdr_gamut_mapping",
            "hdr_sdr_contrast_recovery",
            "hdr_sdr_peak_detect",
            "hdr_sdr_allow_inaccurate_fallback",
        }
        live |= {
            "lambda_facefrac",
            "crop_center_weight",
            "area_gamma",
            "area_face_scale_weight",
            "square_pull_face_min",
            "square_pull_weight",
            "face_target_close",
            "face_target_upper",
            "face_target_cowboy",
            "face_target_body",
            "face_target_tolerance",
            "face_target_close_min_frac",
            "w_close",
            "w_upper",
            "w_cowboy",
            "w_body",
            "compose_crop_enable",
            "compose_detect_person_for_face",
            "compose_close_face_h_frac",
            "compose_upper_face_h_frac",
            "compose_body_face_h_frac",
            "compose_landscape_face_penalty",
            "compose_body_every_n",
            "compose_person_detect_cadence",
        }
        delta = {}
        for k in prescan_live | live:
            if hasattr(self._worker.cfg, k) and hasattr(new, k):
                ov = getattr(self._worker.cfg, k)
                nv = getattr(new, k)
                if ov != nv:
                    delta[k] = nv
        if delta:
            self._worker.update_cfg(delta)
            self._log(f"Applied live: {sorted(delta.keys())}")
            self._save_qsettings()

    def _collect_cfg(self) -> SessionConfig:
        ref_join = "; ".join(self._get_ref_paths()) or self.ref_edit.text().strip()
        cfg = SessionConfig(
            video=self.video_edit.text().strip(),
            ref=ref_join,
            out_dir=self.out_edit.text().strip() or "output",
            ratio=self.ratio_edit.text().strip() or "1:1,2:3,3:2",
            sdr_nits=float(self.sdr_nits_spin.value()),
            tm_desat=float(self.tm_desat_spin.value()),
            tm_param=float(self.tm_param_spin.value()),
            hdr_tonemap_pref=self.tonemap_pref_combo.currentData() or "auto",
            ff_hwaccel=self.hwaccel_combo.currentData() or "off",
            seek_fast=bool(self.seek_fast_check.isChecked()) if hasattr(self, "seek_fast_check") else bool(getattr(self.cfg, "seek_fast", True)),
            seek_max_grabs=max(0, int(self.seek_max_grabs_spin.value())) if hasattr(self, "seek_max_grabs_spin") else max(0, int(getattr(self.cfg, "seek_max_grabs", 12))),
            frame_stride=int(self.stride_spin.value()),
            min_det_conf=float(self.det_conf_spin.value()),
            face_thresh=float(self.face_thr_spin.value()),
            reid_thresh=float(self.reid_thr_spin.value()),
            combine=self.combine_combo.currentText(),
            match_mode=self.match_mode_combo.currentText(),
            disable_reid=bool(self.disable_reid_check.isChecked()) if hasattr(self, "disable_reid_check") else True,
            face_fullframe_when_missed=bool(self.face_fullframe_check.isChecked()) if hasattr(self, "face_fullframe_check") else True,
            only_best=bool(self.only_best_check.isChecked()),
            prescan_enable=bool(self.chk_prescan.isChecked()) if hasattr(self, "chk_prescan") else True,
            prescan_stride=int(self.spin_prescan_stride.value()) if hasattr(self, "spin_prescan_stride") else 16,
            prescan_fd_add=float(self.spin_prescan_fd_add.value()) if hasattr(self, "spin_prescan_fd_add") else 0.22,
            prescan_max_width=int(self.spin_prescan_max_width.value()) if hasattr(self, "spin_prescan_max_width") else 416,
            prescan_decode_max_w=int(self.spin_prescan_decode_max_w.value()) if hasattr(self, "spin_prescan_decode_max_w") else int(getattr(self.cfg, "prescan_decode_max_w", 384)),
            prescan_cache_mode=(self.prescan_cache_combo.currentData() if hasattr(self, "prescan_cache_combo") else getattr(self.cfg, "prescan_cache_mode", "auto")) or "auto",
            prescan_cache_dir=(
                self.edit_prescan_cache_dir.text().strip()
                if hasattr(self, "edit_prescan_cache_dir")
                else str(getattr(self.cfg, "prescan_cache_dir", "prescan_cache") or "prescan_cache")
            ),
            prescan_face_conf=float(self.spin_prescan_face_conf.value()) if hasattr(self, "spin_prescan_face_conf") else 0.5,
            prescan_fd_enter=float(self.spin_prescan_fd_enter.value()) if hasattr(self, "spin_prescan_fd_enter") else 0.45,
            prescan_fd_exit=float(self.spin_prescan_fd_exit.value()) if hasattr(self, "spin_prescan_fd_exit") else 0.52,
            prescan_add_cooldown_samples=int(self.spin_prescan_add_cooldown.value()) if hasattr(self, "spin_prescan_add_cooldown") else 5,
            prescan_min_segment_sec=float(self.spin_prescan_min_segment.value()) if hasattr(self, "spin_prescan_min_segment") else 1.0,
            prescan_pad_sec=float(self.spin_prescan_pad.value()) if hasattr(self, "spin_prescan_pad") else 1.5,
            prescan_bridge_gap_sec=float(self.spin_prescan_bridge.value()) if hasattr(self, "spin_prescan_bridge") else 1.0,
            prescan_exit_cooldown_sec=float(self.spin_prescan_exit_cooldown.value()) if hasattr(self, "spin_prescan_exit_cooldown") else 0.5,
            prescan_boundary_refine_sec=float(self.spin_prescan_refine_window.value()) if hasattr(self, "spin_prescan_refine_window") else 0.75,
            prescan_refine_stride_min=int(self.spin_prescan_refine_stride.value()) if hasattr(self, "spin_prescan_refine_stride") else 3,
            prescan_trim_pad=bool(self.chk_prescan_trim_pad.isChecked()) if hasattr(self, "chk_prescan_trim_pad") else True,
            prescan_bank_max=int(self.spin_prescan_bank_max.value()) if hasattr(self, "spin_prescan_bank_max") else 64,
            prescan_diversity_dedup_cos=float(self.spin_prescan_dedup.value()) if hasattr(self, "spin_prescan_dedup") else 0.968,
            prescan_replace_margin=float(self.spin_prescan_margin.value()) if hasattr(self, "spin_prescan_margin") else 0.01,
            trt_lib_dir=self.trt_edit.text().strip() if hasattr(self, "trt_edit") else "",
            # TensorRT/ORT advanced
            trt_fp16_enable=bool(self.chk_trt_fp16.isChecked()) if hasattr(self, "chk_trt_fp16") else True,
            trt_timing_cache_enable=bool(self.chk_trt_timing.isChecked()) if hasattr(self, "chk_trt_timing") else True,
            trt_engine_cache_enable=bool(self.chk_trt_engine.isChecked()) if hasattr(self, "chk_trt_engine") else True,
            trt_cache_root=self.edit_trt_cache.text().strip() if hasattr(self, "edit_trt_cache") else "trt_cache",
            trt_builder_optimization_level=int(self.spin_trt_level.value()) if hasattr(self, "spin_trt_level") else 5,
            trt_cuda_graph_enable=bool(self.chk_trt_cuda_graph.isChecked()) if hasattr(self, "chk_trt_cuda_graph") else True,
            trt_context_memory_sharing_enable=bool(self.chk_trt_ctx_share.isChecked()) if hasattr(self, "chk_trt_ctx_share") else True,
            trt_auxiliary_streams=int(self.spin_trt_aux.value()) if hasattr(self, "spin_trt_aux") else -1,
            cuda_use_tf32=bool(self.chk_cuda_tf32.isChecked()) if hasattr(self, "chk_cuda_tf32") else True,
            min_sharpness=float(self.min_sharp_spin.value()),
            min_gap_sec=float(self.min_gap_spin.value()),
            min_box_pixels=int(self.min_box_pix_spin.value()),
            auto_crop_borders=bool(self.auto_crop_check.isChecked()),
            border_threshold=int(self.border_thr_spin.value()),
            require_face_if_visible=bool(self.require_face_check.isChecked()),
            lock_momentum=float(self.lock_mom_spin.value()),
            suppress_negatives=bool(self.suppress_neg_check.isChecked()),
            neg_tolerance=float(self.neg_tol_spin.value()),
            max_negatives=int(self.max_neg_spin.value()),
            log_interval_sec=float(self.log_every_spin.value()),
            lock_after_hits=int(self.lock_after_spin.value()),
            lock_face_thresh=float(self.lock_face_spin.value()),
            lock_reid_thresh=float(self.lock_reid_spin.value()),
            score_margin=float(self.margin_spin.value()),
            iou_gate=float(self.iou_gate_spin.value()),
            use_arcface=bool(self.use_arc_check.isChecked()),
            device=self.device_combo.currentText(),
            yolo_model=self.yolo_edit.text().strip() or "yolov8n.pt",
            face_model=self.face_yolo_edit.text().strip() or "scrfd_10g_bnkps",
            save_annot=bool(self.annot_check.isChecked()),
            preview_every=int(self.preview_every_spin.value()) if hasattr(self, "preview_every_spin") else int(getattr(self.cfg, "preview_every", 3)),
            preview_max_dim=int(self.preview_max_dim_spin.value()) if hasattr(self, "preview_max_dim_spin") else int(getattr(self.cfg, "preview_max_dim", 1280)),
            preview_fps_cap=int(self.preview_fps_cap_spin.value()) if hasattr(self, "preview_fps_cap_spin") else int(getattr(self.cfg, "preview_fps_cap", 20)),
            seek_preview_peek_every=int(self.seek_preview_peek_spin.value()) if hasattr(self, "seek_preview_peek_spin") else int(getattr(self.cfg, "seek_preview_peek_every", 16)),
            overlay_scores=bool(self.overlay_scores_check.isChecked()) if hasattr(self, "overlay_scores_check") else bool(getattr(self.cfg, "overlay_scores", False)),
            prefer_face_when_available=bool(self.pref_face_check.isChecked()) if hasattr(self, "pref_face_check") else True,
            face_quality_min=float(self.face_quality_spin.value()) if hasattr(self, "face_quality_spin") else 70.0,
            face_visible_uses_quality=bool(self.face_vis_quality_check.isChecked()) if hasattr(self, "face_vis_quality_check") else True,
            face_det_conf=float(self.face_det_conf_spin.value()) if hasattr(self, "face_det_conf_spin") else 0.5,
            face_det_pad=float(self.face_det_pad_spin.value()) if hasattr(self, "face_det_pad_spin") else 0.08,
            face_fullframe_imgsz=int(self.face_fullframe_imgsz_spin.value()) if hasattr(self, "face_fullframe_imgsz_spin") else 1408,
            # speed / I/O
            skip_yolo_when_faceonly=bool(self.skip_yolo_faceonly_check.isChecked()) if hasattr(self, "skip_yolo_faceonly_check") else True,
            async_save=bool(self.async_save_check.isChecked()) if hasattr(self, "async_save_check") else True,
            jpg_quality=int(self.jpg_quality_spin.value()) if hasattr(self, "jpg_quality_spin") else 85,
            rot_adaptive=bool(self.rot_adaptive_check.isChecked()) if hasattr(self, "rot_adaptive_check") else True,
            rot_every_n=int(self.rot_every_spin.value()) if hasattr(self, "rot_every_spin") else 12,
            rot_after_hit_frames=int(self.rot_after_hit_spin.value()) if hasattr(self, "rot_after_hit_spin") else 8,
            fast_no_face_imgsz=int(self.fast_no_face_spin.value()) if hasattr(self, "fast_no_face_spin") else 512,
            face_margin_min=float(getattr(self, 'margin_spin', QtWidgets.QDoubleSpinBox()).value()) if hasattr(self, 'margin_spin') else 0.05,
            allow_faceless_when_locked=bool(self.faceless_allow_check.isChecked()) if hasattr(self, "faceless_allow_check") else True,
            learn_bank_runtime=bool(self.learn_bank_runtime_check.isChecked()) if hasattr(self, "learn_bank_runtime_check") else False,
            drop_reid_if_any_face_match=bool(self.drop_reid_if_any_face_match_check.isChecked()) if hasattr(self, "drop_reid_if_any_face_match_check") else True,
            faceless_reid_thresh=float(self.faceless_reid_spin.value()) if hasattr(self, "faceless_reid_spin") else 0.40,
            faceless_iou_min=float(self.faceless_iou_spin.value()) if hasattr(self, "faceless_iou_spin") else 0.30,
            faceless_persist_frames=int(self.faceless_persist_spin.value()) if hasattr(self, "faceless_persist_spin") else 0,
            faceless_min_area_frac=float(self.faceless_min_area_spin.value()) if hasattr(self, "faceless_min_area_spin") else 0.03,
            faceless_max_area_frac=float(self.faceless_max_area_spin.value()) if hasattr(self, "faceless_max_area_spin") else 0.55,
            faceless_center_max_frac=float(self.faceless_center_spin.value()) if hasattr(self, "faceless_center_spin") else 0.12,
            faceless_min_motion_frac=float(self.faceless_motion_spin.value()) if hasattr(self, "faceless_motion_spin") else 0.02,
            crop_face_side_margin_frac=float(self.crop_face_side_margin_spin.value()) if hasattr(self, "crop_face_side_margin_spin") else 0.30,
            crop_top_headroom_max_frac=float(self.crop_top_headroom_spin.value()) if hasattr(self, "crop_top_headroom_spin") else 0.15,
            crop_bottom_min_face_heights=float(self.crop_bottom_min_face_spin.value()) if hasattr(self, "crop_bottom_min_face_spin") else 1.5,
            crop_penalty_weight=float(self.crop_penalty_weight_spin.value()) if hasattr(self, "crop_penalty_weight_spin") else 3.0,
            crop_head_side_pad_frac=float(getattr(self.cfg, "crop_head_side_pad_frac", 0.70)),
            crop_head_top_pad_frac=float(getattr(self.cfg, "crop_head_top_pad_frac", 0.85)),
            crop_head_bottom_pad_frac=float(getattr(self.cfg, "crop_head_bottom_pad_frac", 0.30)),
            wide_face_aspect_penalty_weight=float(getattr(self.cfg, "wide_face_aspect_penalty_weight", 10.0)),
            wide_face_min_frame_frac=float(getattr(self.cfg, "wide_face_min_frame_frac", 0.12)),
            wide_face_aspect_limit=float(getattr(self.cfg, "wide_face_aspect_limit", 1.05)),
            face_anchor_down_frac=float(self.face_anchor_down_spin.value()) if hasattr(self, "face_anchor_down_spin") else 1.1,
            lambda_facefrac=float(self.lambda_facefrac_spin.value()),
            crop_center_weight=float(self.crop_center_weight_spin.value()),
            area_gamma=float(self.area_gamma_spin.value()),
            area_face_scale_weight=float(self.area_face_scale_w_spin.value()),
            square_pull_face_min=float(self.square_pull_face_min_spin.value()),
            square_pull_weight=float(self.square_pull_weight_spin.value()),
            face_target_close=float(self.face_target_close_spin.value()),
            face_target_upper=float(self.face_target_upper_spin.value()),
            face_target_cowboy=float(self.face_target_cowboy_spin.value()),
            face_target_body=float(self.face_target_body_spin.value()),
            face_target_tolerance=float(self.face_target_tol_spin.value()),
            face_target_close_min_frac=float(self.face_target_close_min_spin.value()),
            w_close=float(self.w_close_spin.value()),
            w_upper=float(self.w_upper_spin.value()),
            w_cowboy=float(self.w_cowboy_spin.value()),
            w_body=float(self.w_body_spin.value()),
            smart_crop_enable=bool(self.smart_crop_enable_check.isChecked()) if hasattr(self, "smart_crop_enable_check") else True,
            smart_crop_steps=int(self.smart_crop_steps_spin.value()) if hasattr(self, "smart_crop_steps_spin") else 6,
            smart_crop_side_search_frac=float(self.smart_crop_side_search_spin.value()) if hasattr(self, "smart_crop_side_search_spin") else 0.35,
            smart_crop_use_grad=bool(self.smart_crop_use_grad_check.isChecked()) if hasattr(self, "smart_crop_use_grad_check") else True,
            face_max_frac_in_crop=float(self.face_max_frac_spin.value()) if hasattr(self, "face_max_frac_spin") else 0.42,
            face_min_frac_in_crop=float(self.face_min_frac_spin.value()) if hasattr(self, "face_min_frac_spin") else 0.18,
            crop_min_height_frac=float(self.crop_min_height_frac_spin.value()) if hasattr(self, "crop_min_height_frac_spin") else 0.28,
            compose_crop_enable=bool(self.compose_crop_enable_check.isChecked()) if hasattr(self, "compose_crop_enable_check") else bool(getattr(self.cfg, "compose_crop_enable", True)),
            compose_detect_person_for_face=bool(self.compose_detect_person_for_face_check.isChecked()) if hasattr(self, "compose_detect_person_for_face_check") else bool(getattr(self.cfg, "compose_detect_person_for_face", True)),
            compose_close_face_h_frac=float(self.compose_close_face_h_frac_spin.value()) if hasattr(self, "compose_close_face_h_frac_spin") else float(getattr(self.cfg, "compose_close_face_h_frac", 0.34)),
            compose_upper_face_h_frac=float(self.compose_upper_face_h_frac_spin.value()) if hasattr(self, "compose_upper_face_h_frac_spin") else float(getattr(self.cfg, "compose_upper_face_h_frac", 0.22)),
            compose_body_face_h_frac=float(self.compose_body_face_h_frac_spin.value()) if hasattr(self, "compose_body_face_h_frac_spin") else float(getattr(self.cfg, "compose_body_face_h_frac", 0.085)),
            compose_landscape_face_penalty=float(self.compose_landscape_face_penalty_spin.value()) if hasattr(self, "compose_landscape_face_penalty_spin") else float(getattr(self.cfg, "compose_landscape_face_penalty", 5.0)),
            compose_body_every_n=int(self.compose_body_every_n_spin.value()) if hasattr(self, "compose_body_every_n_spin") else int(getattr(self.cfg, "compose_body_every_n", 6)),
            compose_person_detect_cadence=int(self.compose_person_detect_cadence_spin.value()) if hasattr(self, "compose_person_detect_cadence_spin") else int(getattr(self.cfg, "compose_person_detect_cadence", 6)),
            hdr_export_timeout_sec=int(getattr(self.cfg, "hdr_export_timeout_sec", 300) or 300),
        )
        cfg.hdr_passthrough = (
            getattr(self, "chk_hdr_passthrough", None) is not None
            and self.chk_hdr_passthrough.isEnabled()
            and self.chk_hdr_passthrough.isChecked()
        )
        cfg.hdr_screencap_fullres = (
            getattr(self, "chk_hdr_screencap_fullres", None) is None
            or self.chk_hdr_screencap_fullres.isChecked()
        )
        cfg.hdr_archive_crops = (
            getattr(self, "chk_hdr_archive_crops", None) is not None
            and self.chk_hdr_archive_crops.isChecked()
        )
        try:
            cfg.hdr_crop_format = self.hdr_crop_format_combo.currentText().lower()
        except Exception:
            cfg.hdr_crop_format = "mkv"
        try:
            cfg.hdr_sdr_quality = self.hdr_sdr_quality_combo.currentData() or "madvr_like"
        except Exception:
            cfg.hdr_sdr_quality = "madvr_like"
        try:
            cfg.hdr_sdr_tonemap = self.hdr_sdr_tonemap_combo.currentData() or "auto"
        except Exception:
            cfg.hdr_sdr_tonemap = "auto"
        try:
            cfg.hdr_sdr_gamut_mapping = self.hdr_sdr_gamut_combo.currentData() or "clip"
        except Exception:
            cfg.hdr_sdr_gamut_mapping = "clip"
        cfg.hdr_sdr_contrast_recovery = float(self.hdr_sdr_contrast_spin.value()) if hasattr(self, "hdr_sdr_contrast_spin") else 0.30
        cfg.hdr_sdr_peak_detect = bool(self.hdr_sdr_peak_check.isChecked()) if hasattr(self, "hdr_sdr_peak_check") else True
        cfg.hdr_sdr_allow_inaccurate_fallback = bool(self.hdr_sdr_bad_fallback_check.isChecked()) if hasattr(self, "hdr_sdr_bad_fallback_check") else False
        return cfg

    def _apply_cfg(self, cfg: SessionConfig):
        self.cfg.seek_fast = bool(getattr(cfg, "seek_fast", True))
        try:
            self.cfg.seek_max_grabs = int(getattr(cfg, "seek_max_grabs", 12))
        except Exception:
            self.cfg.seek_max_grabs = 12
        try:
            self.cfg.preview_max_dim = int(getattr(cfg, "preview_max_dim", 1280))
        except Exception:
            self.cfg.preview_max_dim = 1280
        try:
            self.cfg.preview_fps_cap = int(getattr(cfg, "preview_fps_cap", 20))
        except Exception:
            self.cfg.preview_fps_cap = 20
        try:
            self.cfg.seek_preview_peek_every = int(getattr(cfg, "seek_preview_peek_every", 16))
        except Exception:
            self.cfg.seek_preview_peek_every = 16
        self.cfg.overlay_scores = bool(getattr(cfg, "overlay_scores", False))
        try:
            self.cfg.preview_every = int(getattr(cfg, "preview_every", 3))
        except Exception:
            self.cfg.preview_every = 3
        try:
            self.cfg.prescan_decode_max_w = int(getattr(cfg, "prescan_decode_max_w", 384))
        except Exception:
            self.cfg.prescan_decode_max_w = 384
        self.cfg.prescan_cache_dir = str(getattr(cfg, "prescan_cache_dir", "prescan_cache") or "prescan_cache")
        try:
            self.cfg.hdr_export_timeout_sec = max(5, int(getattr(cfg, "hdr_export_timeout_sec", 300) or 300))
        except Exception:
            self.cfg.hdr_export_timeout_sec = 300
        self.cfg.compose_crop_enable = bool(getattr(cfg, "compose_crop_enable", True))
        self.cfg.compose_detect_person_for_face = bool(getattr(cfg, "compose_detect_person_for_face", True))
        try:
            self.cfg.compose_close_face_h_frac = float(getattr(cfg, "compose_close_face_h_frac", 0.34))
        except Exception:
            self.cfg.compose_close_face_h_frac = 0.34
        try:
            self.cfg.compose_upper_face_h_frac = float(getattr(cfg, "compose_upper_face_h_frac", 0.22))
        except Exception:
            self.cfg.compose_upper_face_h_frac = 0.22
        try:
            self.cfg.compose_body_face_h_frac = float(getattr(cfg, "compose_body_face_h_frac", 0.085))
        except Exception:
            self.cfg.compose_body_face_h_frac = 0.085
        try:
            self.cfg.compose_landscape_face_penalty = float(getattr(cfg, "compose_landscape_face_penalty", 5.0))
        except Exception:
            self.cfg.compose_landscape_face_penalty = 5.0
        try:
            self.cfg.compose_body_every_n = int(getattr(cfg, "compose_body_every_n", 6))
        except Exception:
            self.cfg.compose_body_every_n = 6
        try:
            self.cfg.compose_person_detect_cadence = int(getattr(cfg, "compose_person_detect_cadence", 6))
        except Exception:
            self.cfg.compose_person_detect_cadence = 6
        self.video_edit.setText(cfg.video)
        paths = [part.strip() for part in (cfg.ref or "").split(';') if part.strip()]
        self._set_ref_paths(paths)
        self.out_edit.setText(cfg.out_dir)
        if hasattr(self, "chk_hdr_passthrough"):
            self.chk_hdr_passthrough.setChecked(bool(getattr(cfg, "hdr_passthrough", False)))
        if hasattr(self, "chk_hdr_screencap_fullres"):
            self.chk_hdr_screencap_fullres.setChecked(bool(getattr(cfg, "hdr_screencap_fullres", True)))
        if hasattr(self, "chk_hdr_archive_crops"):
            self.chk_hdr_archive_crops.setChecked(bool(getattr(cfg, "hdr_archive_crops", False)))
        if hasattr(self, "hdr_sdr_quality_combo"):
            val = str(getattr(cfg, "hdr_sdr_quality", "madvr_like") or "madvr_like")
            idx = self.hdr_sdr_quality_combo.findData(val)
            self.hdr_sdr_quality_combo.setCurrentIndex(idx if idx >= 0 else 0)
        if hasattr(self, "hdr_sdr_tonemap_combo"):
            val = str(getattr(cfg, "hdr_sdr_tonemap", "auto") or "auto")
            idx = self.hdr_sdr_tonemap_combo.findData(val)
            self.hdr_sdr_tonemap_combo.setCurrentIndex(idx if idx >= 0 else 0)
        if hasattr(self, "hdr_sdr_gamut_combo"):
            val = str(getattr(cfg, "hdr_sdr_gamut_mapping", "clip") or "clip")
            idx = self.hdr_sdr_gamut_combo.findData(val)
            self.hdr_sdr_gamut_combo.setCurrentIndex(idx if idx >= 0 else 0)
        if hasattr(self, "hdr_sdr_contrast_spin"):
            try:
                self.hdr_sdr_contrast_spin.setValue(float(getattr(cfg, "hdr_sdr_contrast_recovery", 0.30)))
            except Exception:
                self.hdr_sdr_contrast_spin.setValue(0.30)
        if hasattr(self, "hdr_sdr_peak_check"):
            self.hdr_sdr_peak_check.setChecked(bool(getattr(cfg, "hdr_sdr_peak_detect", True)))
        if hasattr(self, "hdr_sdr_bad_fallback_check"):
            self.hdr_sdr_bad_fallback_check.setChecked(bool(getattr(cfg, "hdr_sdr_allow_inaccurate_fallback", False)))
        self.ratio_edit.setText(cfg.ratio)
        try:
            self.sdr_nits_spin.setValue(float(getattr(cfg, "sdr_nits", SessionConfig.sdr_nits)))
        except Exception:
            self.sdr_nits_spin.setValue(float(SessionConfig.sdr_nits))
        try:
            self.tm_desat_spin.setValue(float(getattr(cfg, "tm_desat", 0.25)))
        except Exception:
            self.tm_desat_spin.setValue(0.25)
        try:
            self.tm_param_spin.setValue(float(getattr(cfg, "tm_param", 0.40)))
        except Exception:
            self.tm_param_spin.setValue(0.40)
        pref = str(getattr(cfg, "hdr_tonemap_pref", "auto") or "auto")
        idx = self.tonemap_pref_combo.findData(pref)
        if idx >= 0:
            self.tonemap_pref_combo.setCurrentIndex(idx)
        else:
            self.tonemap_pref_combo.setCurrentIndex(0)
        hw_mode = str(getattr(cfg, "ff_hwaccel", "off") or "off").strip().lower()
        hw_idx = self.hwaccel_combo.findData(hw_mode)
        if hw_idx >= 0:
            self.hwaccel_combo.setCurrentIndex(hw_idx)
        else:
            fallback_idx = self.hwaccel_combo.findData("off")
            self.hwaccel_combo.setCurrentIndex(fallback_idx if fallback_idx >= 0 else 0)
        self.cfg.ff_hwaccel = self.hwaccel_combo.currentData() or "off"
        fmt = str(getattr(cfg, "hdr_crop_format", "mkv") or "mkv").lower()
        fmt_idx = self.hdr_crop_format_combo.findText(fmt)
        if fmt_idx >= 0:
            self.hdr_crop_format_combo.setCurrentIndex(fmt_idx)
        else:
            self.hdr_crop_format_combo.setCurrentIndex(0)
        self.cfg.hdr_crop_format = self.hdr_crop_format_combo.currentText().lower()
        self.stride_spin.setValue(cfg.frame_stride)
        self.det_conf_spin.setValue(cfg.min_det_conf)
        self.face_thr_spin.setValue(cfg.face_thresh)
        self.face_det_conf_spin.setValue(cfg.face_det_conf)
        self.face_det_pad_spin.setValue(cfg.face_det_pad)
        self.face_quality_spin.setValue(cfg.face_quality_min)
        self.face_vis_quality_check.setChecked(cfg.face_visible_uses_quality)
        self.reid_thr_spin.setValue(cfg.reid_thresh)
        self.combine_combo.setCurrentText(cfg.combine)
        self.match_mode_combo.setCurrentText(cfg.match_mode)
        if hasattr(self, 'disable_reid_check'):
            self.disable_reid_check.setChecked(cfg.disable_reid)
        if hasattr(self, 'face_fullframe_check'):
            self.face_fullframe_check.setChecked(cfg.face_fullframe_when_missed)
        if hasattr(self, 'face_fullframe_imgsz_spin'):
            self.face_fullframe_imgsz_spin.setValue(int(cfg.face_fullframe_imgsz))
        if hasattr(self, 'rot_adaptive_check'):
            self.rot_adaptive_check.setChecked(bool(cfg.rot_adaptive))
        if hasattr(self, 'rot_every_spin'):
            self.rot_every_spin.setValue(int(cfg.rot_every_n))
        if hasattr(self, 'rot_after_hit_spin'):
            self.rot_after_hit_spin.setValue(int(cfg.rot_after_hit_frames))
        if hasattr(self, 'fast_no_face_spin'):
            self.fast_no_face_spin.setValue(int(cfg.fast_no_face_imgsz))
        # speed / I/O
        if hasattr(self, 'skip_yolo_faceonly_check'):
            self.skip_yolo_faceonly_check.setChecked(bool(getattr(cfg, 'skip_yolo_when_faceonly', True)))
        if hasattr(self, 'async_save_check'):
            self.async_save_check.setChecked(bool(getattr(cfg, 'async_save', True)))
        if hasattr(self, 'jpg_quality_spin'):
            self.jpg_quality_spin.setValue(int(getattr(cfg, 'jpg_quality', 85)))
        self.only_best_check.setChecked(cfg.only_best)
        if hasattr(self, 'chk_prescan'):
            self.chk_prescan.setChecked(cfg.prescan_enable)
        if hasattr(self, 'spin_prescan_stride'):
            self.spin_prescan_stride.setValue(int(cfg.prescan_stride))
        if hasattr(self, 'spin_prescan_fd_add'):
            self.spin_prescan_fd_add.setValue(float(cfg.prescan_fd_add))
        if hasattr(self, 'spin_prescan_max_width'):
            self.spin_prescan_max_width.setValue(int(cfg.prescan_max_width))
        if hasattr(self, 'spin_prescan_decode_max_w'):
            self.spin_prescan_decode_max_w.setValue(int(getattr(cfg, 'prescan_decode_max_w', 384)))
        if hasattr(self, "prescan_cache_combo"):
            mode = str(getattr(cfg, "prescan_cache_mode", "auto") or "auto")
            idx = self.prescan_cache_combo.findData(mode)
            self.prescan_cache_combo.setCurrentIndex(idx if idx >= 0 else 0)
        if hasattr(self, "edit_prescan_cache_dir"):
            self.edit_prescan_cache_dir.setText(str(getattr(cfg, "prescan_cache_dir", "prescan_cache") or "prescan_cache"))
        if hasattr(self, 'spin_prescan_face_conf'):
            self.spin_prescan_face_conf.setValue(float(cfg.prescan_face_conf))
        if hasattr(self, 'spin_prescan_fd_enter'):
            self.spin_prescan_fd_enter.setValue(float(cfg.prescan_fd_enter))
        if hasattr(self, 'spin_prescan_fd_exit'):
            self.spin_prescan_fd_exit.setValue(float(cfg.prescan_fd_exit))
        if hasattr(self, 'spin_prescan_add_cooldown'):
            self.spin_prescan_add_cooldown.setValue(int(cfg.prescan_add_cooldown_samples))
        if hasattr(self, 'spin_prescan_min_segment'):
            self.spin_prescan_min_segment.setValue(float(cfg.prescan_min_segment_sec))
        if hasattr(self, 'spin_prescan_pad'):
            self.spin_prescan_pad.setValue(float(cfg.prescan_pad_sec))
        if hasattr(self, 'spin_prescan_bridge'):
            self.spin_prescan_bridge.setValue(float(cfg.prescan_bridge_gap_sec))
        if hasattr(self, 'spin_prescan_exit_cooldown'):
            self.spin_prescan_exit_cooldown.setValue(float(cfg.prescan_exit_cooldown_sec))
        if hasattr(self, 'spin_prescan_refine_window'):
            self.spin_prescan_refine_window.setValue(float(cfg.prescan_boundary_refine_sec))
        if hasattr(self, 'spin_prescan_refine_stride'):
            self.spin_prescan_refine_stride.setValue(int(cfg.prescan_refine_stride_min))
        if hasattr(self, 'chk_prescan_trim_pad'):
            self.chk_prescan_trim_pad.setChecked(bool(cfg.prescan_trim_pad))
        if hasattr(self, 'spin_prescan_bank_max'):
            self.spin_prescan_bank_max.setValue(int(cfg.prescan_bank_max))
        if hasattr(self, 'spin_prescan_dedup'):
            self.spin_prescan_dedup.setValue(float(cfg.prescan_diversity_dedup_cos))
        if hasattr(self, 'spin_prescan_margin'):
            self.spin_prescan_margin.setValue(float(cfg.prescan_replace_margin))
        if hasattr(self, 'trt_edit'):
            self.trt_edit.setText(cfg.trt_lib_dir or r"D:\\tensorrt\\TensorRT-10.13.3.9")
        if hasattr(self, 'chk_trt_fp16'):
            self.chk_trt_fp16.setChecked(bool(getattr(cfg, 'trt_fp16_enable', True)))
        if hasattr(self, 'chk_trt_timing'):
            self.chk_trt_timing.setChecked(bool(getattr(cfg, 'trt_timing_cache_enable', True)))
        if hasattr(self, 'chk_trt_engine'):
            self.chk_trt_engine.setChecked(bool(getattr(cfg, 'trt_engine_cache_enable', True)))
        if hasattr(self, 'edit_trt_cache'):
            self.edit_trt_cache.setText(getattr(cfg, 'trt_cache_root', 'trt_cache') or 'trt_cache')
        if hasattr(self, 'spin_trt_level'):
            self.spin_trt_level.setValue(int(getattr(cfg, 'trt_builder_optimization_level', 5)))
        if hasattr(self, 'chk_trt_cuda_graph'):
            self.chk_trt_cuda_graph.setChecked(bool(getattr(cfg, 'trt_cuda_graph_enable', True)))
        if hasattr(self, 'chk_trt_ctx_share'):
            self.chk_trt_ctx_share.setChecked(bool(getattr(cfg, 'trt_context_memory_sharing_enable', True)))
        if hasattr(self, 'spin_trt_aux'):
            self.spin_trt_aux.setValue(int(getattr(cfg, 'trt_auxiliary_streams', -1)))
        if hasattr(self, 'chk_cuda_tf32'):
            self.chk_cuda_tf32.setChecked(bool(getattr(cfg, 'cuda_use_tf32', True)))
        self.min_sharp_spin.setValue(cfg.min_sharpness)
        self.min_gap_spin.setValue(cfg.min_gap_sec)
        self.min_box_pix_spin.setValue(cfg.min_box_pixels)
        self.auto_crop_check.setChecked(cfg.auto_crop_borders)
        self.border_thr_spin.setValue(cfg.border_threshold)
        self.require_face_check.setChecked(cfg.require_face_if_visible)
        self.lock_mom_spin.setValue(cfg.lock_momentum)
        self.suppress_neg_check.setChecked(cfg.suppress_negatives)
        self.neg_tol_spin.setValue(cfg.neg_tolerance)
        self.max_neg_spin.setValue(cfg.max_negatives)
        self.log_every_spin.setValue(cfg.log_interval_sec)
        self.lock_after_spin.setValue(cfg.lock_after_hits)
        self.lock_face_spin.setValue(cfg.lock_face_thresh)
        self.lock_reid_spin.setValue(cfg.lock_reid_thresh)
        self.margin_spin.setValue(cfg.score_margin)
        self.iou_gate_spin.setValue(cfg.iou_gate)
        self.use_arc_check.setChecked(cfg.use_arcface)
        self.device_combo.setCurrentText(cfg.device)
        self.yolo_edit.setText(cfg.yolo_model)
        self.face_yolo_edit.setText(cfg.face_model)
        self.annot_check.setChecked(cfg.save_annot)
        self.preview_every_spin.setValue(cfg.preview_every)
        if hasattr(self, 'preview_max_dim_spin'):
            self.preview_max_dim_spin.setValue(int(getattr(cfg, 'preview_max_dim', self.cfg.preview_max_dim)))
        if hasattr(self, 'preview_fps_cap_spin'):
            self.preview_fps_cap_spin.setValue(int(getattr(cfg, 'preview_fps_cap', self.cfg.preview_fps_cap)))
        if hasattr(self, 'seek_fast_check'):
            self.seek_fast_check.setChecked(bool(getattr(cfg, 'seek_fast', self.cfg.seek_fast)))
        if hasattr(self, 'seek_max_grabs_spin'):
            self.seek_max_grabs_spin.setValue(int(getattr(cfg, 'seek_max_grabs', self.cfg.seek_max_grabs)))
        if hasattr(self, 'seek_preview_peek_spin'):
            self.seek_preview_peek_spin.setValue(int(getattr(cfg, 'seek_preview_peek_every', self.cfg.seek_preview_peek_every)))
        if hasattr(self, 'overlay_scores_check'):
            self.overlay_scores_check.setChecked(bool(getattr(cfg, 'overlay_scores', self.cfg.overlay_scores)))
        if hasattr(self, 'pref_face_check'):
            self.pref_face_check.setChecked(cfg.prefer_face_when_available)
        if hasattr(self, 'face_quality_spin'):
            self.face_quality_spin.setValue(cfg.face_quality_min)
        if hasattr(self, 'face_vis_quality_check'):
            self.face_vis_quality_check.setChecked(cfg.face_visible_uses_quality)
        if hasattr(self, 'face_det_conf_spin'):
            self.face_det_conf_spin.setValue(cfg.face_det_conf)
        if hasattr(self, 'face_det_pad_spin'):
            self.face_det_pad_spin.setValue(cfg.face_det_pad)
        if hasattr(self, 'margin_spin'):
            self.margin_spin.setValue(cfg.face_margin_min)
        if hasattr(self, 'faceless_allow_check'):
            self.faceless_allow_check.setChecked(cfg.allow_faceless_when_locked)
        if hasattr(self, 'learn_bank_runtime_check'):
            self.learn_bank_runtime_check.setChecked(cfg.learn_bank_runtime)
        if hasattr(self, 'drop_reid_if_any_face_match_check'):
            self.drop_reid_if_any_face_match_check.setChecked(cfg.drop_reid_if_any_face_match)
        if hasattr(self, 'faceless_reid_spin'):
            self.faceless_reid_spin.setValue(cfg.faceless_reid_thresh)
        if hasattr(self, 'faceless_iou_spin'):
            self.faceless_iou_spin.setValue(cfg.faceless_iou_min)
        if hasattr(self, 'faceless_persist_spin'):
            self.faceless_persist_spin.setValue(cfg.faceless_persist_frames)
        if hasattr(self, 'faceless_min_area_spin'):
            self.faceless_min_area_spin.setValue(cfg.faceless_min_area_frac)
        if hasattr(self, 'faceless_max_area_spin'):
            self.faceless_max_area_spin.setValue(cfg.faceless_max_area_frac)
        if hasattr(self, 'faceless_center_spin'):
            self.faceless_center_spin.setValue(cfg.faceless_center_max_frac)
        if hasattr(self, 'faceless_motion_spin'):
            self.faceless_motion_spin.setValue(cfg.faceless_min_motion_frac)
        if hasattr(self, 'crop_face_side_margin_spin'):
            self.crop_face_side_margin_spin.setValue(cfg.crop_face_side_margin_frac)
        if hasattr(self, 'crop_top_headroom_spin'):
            self.crop_top_headroom_spin.setValue(cfg.crop_top_headroom_max_frac)
        if hasattr(self, 'crop_bottom_min_face_spin'):
            self.crop_bottom_min_face_spin.setValue(cfg.crop_bottom_min_face_heights)
        if hasattr(self, 'crop_penalty_weight_spin'):
            self.crop_penalty_weight_spin.setValue(cfg.crop_penalty_weight)
        try:
            self.cfg.crop_head_side_pad_frac = float(
                getattr(cfg, "crop_head_side_pad_frac", SessionConfig.crop_head_side_pad_frac)
            )
        except Exception:
            self.cfg.crop_head_side_pad_frac = float(SessionConfig.crop_head_side_pad_frac)
        try:
            self.cfg.crop_head_top_pad_frac = float(
                getattr(cfg, "crop_head_top_pad_frac", SessionConfig.crop_head_top_pad_frac)
            )
        except Exception:
            self.cfg.crop_head_top_pad_frac = float(SessionConfig.crop_head_top_pad_frac)
        try:
            self.cfg.crop_head_bottom_pad_frac = float(
                getattr(cfg, "crop_head_bottom_pad_frac", SessionConfig.crop_head_bottom_pad_frac)
            )
        except Exception:
            self.cfg.crop_head_bottom_pad_frac = float(SessionConfig.crop_head_bottom_pad_frac)
        try:
            self.cfg.wide_face_aspect_penalty_weight = float(
                getattr(cfg, "wide_face_aspect_penalty_weight", 10.0)
            )
        except Exception:
            self.cfg.wide_face_aspect_penalty_weight = 10.0
        try:
            self.cfg.wide_face_min_frame_frac = float(getattr(cfg, "wide_face_min_frame_frac", 0.12))
        except Exception:
            self.cfg.wide_face_min_frame_frac = 0.12
        try:
            self.cfg.wide_face_aspect_limit = float(getattr(cfg, "wide_face_aspect_limit", 1.05))
        except Exception:
            self.cfg.wide_face_aspect_limit = 1.05
        if hasattr(self, 'face_anchor_down_spin'):
            self.face_anchor_down_spin.setValue(cfg.face_anchor_down_frac)
        if hasattr(self, 'lambda_facefrac_spin'):
            self.lambda_facefrac_spin.setValue(cfg.lambda_facefrac)
        if hasattr(self, 'crop_center_weight_spin'):
            self.crop_center_weight_spin.setValue(cfg.crop_center_weight)
        if hasattr(self, 'area_gamma_spin'):
            self.area_gamma_spin.setValue(cfg.area_gamma)
        if hasattr(self, 'area_face_scale_w_spin'):
            self.area_face_scale_w_spin.setValue(cfg.area_face_scale_weight)
        if hasattr(self, 'square_pull_face_min_spin'):
            self.square_pull_face_min_spin.setValue(cfg.square_pull_face_min)
        if hasattr(self, 'square_pull_weight_spin'):
            self.square_pull_weight_spin.setValue(cfg.square_pull_weight)
        if hasattr(self, 'face_target_close_spin'):
            self.face_target_close_spin.setValue(cfg.face_target_close)
        if hasattr(self, 'face_target_upper_spin'):
            self.face_target_upper_spin.setValue(cfg.face_target_upper)
        if hasattr(self, 'face_target_cowboy_spin'):
            self.face_target_cowboy_spin.setValue(cfg.face_target_cowboy)
        if hasattr(self, 'face_target_body_spin'):
            self.face_target_body_spin.setValue(cfg.face_target_body)
        if hasattr(self, 'face_target_tol_spin'):
            self.face_target_tol_spin.setValue(cfg.face_target_tolerance)
        if hasattr(self, 'face_target_close_min_spin'):
            self.face_target_close_min_spin.setValue(cfg.face_target_close_min_frac)
        if hasattr(self, 'w_close_spin'):
            self.w_close_spin.setValue(cfg.w_close)
        if hasattr(self, 'w_upper_spin'):
            self.w_upper_spin.setValue(cfg.w_upper)
        if hasattr(self, 'w_cowboy_spin'):
            self.w_cowboy_spin.setValue(cfg.w_cowboy)
        if hasattr(self, 'w_body_spin'):
            self.w_body_spin.setValue(cfg.w_body)
        if hasattr(self, 'smart_crop_enable_check'):
            self.smart_crop_enable_check.setChecked(bool(cfg.smart_crop_enable))
        if hasattr(self, 'smart_crop_steps_spin'):
            self.smart_crop_steps_spin.setValue(int(cfg.smart_crop_steps))
        if hasattr(self, 'smart_crop_side_search_spin'):
            self.smart_crop_side_search_spin.setValue(float(cfg.smart_crop_side_search_frac))
        if hasattr(self, 'smart_crop_use_grad_check'):
            self.smart_crop_use_grad_check.setChecked(bool(cfg.smart_crop_use_grad))
        if hasattr(self, 'face_max_frac_spin'):
            self.face_max_frac_spin.setValue(cfg.face_max_frac_in_crop)
        if hasattr(self, 'face_min_frac_spin'):
            self.face_min_frac_spin.setValue(cfg.face_min_frac_in_crop)
        if hasattr(self, 'crop_min_height_frac_spin'):
            self.crop_min_height_frac_spin.setValue(cfg.crop_min_height_frac)
        if hasattr(self, 'compose_crop_enable_check'):
            self.compose_crop_enable_check.setChecked(bool(cfg.compose_crop_enable))
        if hasattr(self, 'compose_detect_person_for_face_check'):
            self.compose_detect_person_for_face_check.setChecked(bool(cfg.compose_detect_person_for_face))
        if hasattr(self, 'compose_close_face_h_frac_spin'):
            self.compose_close_face_h_frac_spin.setValue(float(cfg.compose_close_face_h_frac))
        if hasattr(self, 'compose_upper_face_h_frac_spin'):
            self.compose_upper_face_h_frac_spin.setValue(float(cfg.compose_upper_face_h_frac))
        if hasattr(self, 'compose_body_face_h_frac_spin'):
            self.compose_body_face_h_frac_spin.setValue(float(cfg.compose_body_face_h_frac))
        if hasattr(self, 'compose_landscape_face_penalty_spin'):
            self.compose_landscape_face_penalty_spin.setValue(float(cfg.compose_landscape_face_penalty))
        if hasattr(self, 'compose_body_every_n_spin'):
            self.compose_body_every_n_spin.setValue(int(cfg.compose_body_every_n))
        if hasattr(self, 'compose_person_detect_cadence_spin'):
            self.compose_person_detect_cadence_spin.setValue(int(getattr(cfg, "compose_person_detect_cadence", 6)))

    # Thread control
    def on_start(self):
        if self._thread is not None:
            QtWidgets.QMessageBox.warning(self, "Busy", "Processing already running")
            return

        cfg = self._collect_cfg()

        if cfg.hdr_passthrough:
            os.environ["PC_HDR_SWAPCHAIN_HDR"] = "1"
        else:
            os.environ.pop("PC_HDR_SWAPCHAIN_HDR", None)

        # Single source of truth: HDR passthrough is active iff the main pipeline
        # is in HDR passthrough mode AND the Vulkan widget is actually available.
        passthrough_active = bool(
            cfg.hdr_passthrough and getattr(self, "_hdr_passthrough_supported", False)
        )
        self._hdr_passthrough_enabled = passthrough_active
        self._last_hdr_preview_t = time.perf_counter()
        self._hdr_preview_seen = False

        # basic validation
        if not os.path.isfile(cfg.video):
            QtWidgets.QMessageBox.warning(self, "Missing", "Select a video file")
            return
        if not os.path.isfile(cfg.ref):
            QtWidgets.QMessageBox.warning(self, "Missing", "Select a reference image")
            return
        Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

        self.progress.setValue(0)
        self.status_lbl.setText("Starting...")
        self._log("Starting processing")
        self._update_buttons(state="running")

        self._thread = QtCore.QThread(self)
        self._worker = Processor(cfg)
        self._worker.moveToThread(self._thread)
        try:
            # Use the same flag on the worker so reader+UI stay in lockstep.
            self._worker.ui_hdr_passthrough_enabled = passthrough_active
        except Exception:
            self._worker.ui_hdr_passthrough_enabled = False

        # Wire signals
        self._thread.started.connect(self._worker.run)
        self._worker.setup.connect(self._on_setup)
        self._worker.progress.connect(self._on_progress)
        self._worker.status.connect(self._on_status)
        self._worker.preview.connect(self._on_preview)
        try:
            self._worker.preview_hdr_p010.connect(self._on_hdr_preview_p010)
        except Exception:
            pass
        self._worker.hit.connect(self._on_hit)
        self._worker.finished.connect(self._on_finished)
        self._worker.keyframes.connect(self._on_keyframes)

        self._on_speed_combo_changed(self.speed_combo.currentText())
        self._thread.start()

    def on_pause(self):
        self._pause(True)

    def on_stop(self):
        if not self._thread or not self._worker:
            return
        self._worker.request_abort()
        self._log("Stop requested")

    # Signal handlers
    def _on_setup(self, total_frames: int, fps: float):
        self._total_frames = int(total_frames)
        self._fps = float(fps)
        self._current_idx = 0
        self.progress.setRange(0, max(0, total_frames))
        self.meta_lbl.setText(f"fps={fps:.2f} frames={total_frames if total_frames>0 else 'unknown'}")
        if hasattr(self, "seek_slider"):
            self.seek_slider.setRange(0, max(0, total_frames - 1))
            self.seek_slider.setValue(0)
            self.seek_slider.setPageStep(max(1, self.seek_slider.maximum() // 100))
            if hasattr(self, "time_lbl"):
                self.time_lbl.setText(f"{self._fmt_time(0)} / {self._fmt_time(total_frames)}")
        self._update_play_toggle_ui(paused=False)
        # No ffprobe fallback here; worker will always emit a keyframe grid at minimum.
        # _on_keyframes will populate self._keyframes.

    def _on_keyframes(self, ks):
        try:
            self._keyframes = [int(x) for x in ks or []]
            if self._keyframes:
                self._log(f"Keyframes: {len(self._keyframes)} from worker")
            else:
                self._log("No keyframes available; arrows step ±1")
        except Exception:
            pass

    def _on_progress(self, idx: int):
        self._current_idx = int(idx)
        if self.progress.maximum() > 0 and idx <= self.progress.maximum():
            self.progress.setValue(idx)
        if hasattr(self, "seek_slider"):
            if not self.seek_slider.isSliderDown():
                self.seek_slider.setValue(int(idx))
            if self._total_frames is not None and self._fps:
                self.time_lbl.setText(f"{self._fmt_time(idx)} / {self._fmt_time(self._total_frames)}")

    def _fmt_time(self, frames: int) -> str:
        if not self._fps:
            return "00:00"
        secs = max(0, int(round(frames / float(self._fps))))
        m, s = divmod(secs, 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

    def _on_status(self, txt: str):
        self.status_lbl.setText(txt)
        self._log(txt)

    def _on_preview(self, img: Optional[QtGui.QImage]):
        """SDR preview path (tonemapped BGR). Always updates the label,
        but only takes over the stacked widget when HDR passthrough is OFF.
        """
        log = logging.getLogger(__name__)

        def _update_label(image: QtGui.QImage) -> None:
            if image is None:
                return
            try:
                pix = QtGui.QPixmap.fromImage(image)
            except Exception:
                return
            w = max(8, self.preview_label.width())
            h = max(8, self.preview_label.height())
            self.preview_label.setPixmap(
                pix.scaled(
                    w,
                    h,
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.FastTransformation,
                )
            )

        if img is None:
            self._last_preview_qimage = None
            self.preview_label.clear()
            return

        try:
            self._last_preview_qimage = img.copy()
        except Exception:
            self._last_preview_qimage = None

        _update_label(img)

        hdr_passthrough_enabled = bool(getattr(self, "_hdr_passthrough_enabled", False))
        if hdr_passthrough_enabled:
            ctx_ok = False
            try:
                widget = getattr(self, "hdr_widget", None)
                has_ctx = getattr(widget, "has_valid_ctx", None) if widget is not None else None
                ctx_ok = bool(has_ctx()) if callable(has_ctx) else False
            except Exception:
                ctx_ok = False
            hdr_seen = bool(getattr(self, "_hdr_preview_seen", False))
            try:
                last_hdr_t = float(getattr(self, "_last_hdr_preview_t", 0.0) or 0.0)
                hdr_stale = hdr_seen and ((time.perf_counter() - last_hdr_t) > 2.0)
            except Exception:
                hdr_stale = False
            if (hdr_seen and not ctx_ok) or hdr_stale:
                log.debug("SDR preview taking over preview_stack (HDR passthrough stale/dead)")
                self._hdr_passthrough_enabled = False
                hdr_passthrough_enabled = False
                # Propagate to worker: disable HDR passthrough and clean up reader
                self._queue_worker_disable_hdr_passthrough()

        # If HDR passthrough is not active, SDR preview owns the stack.
        if not hdr_passthrough_enabled:
            try:
                if hasattr(self, "preview_stack") and self.preview_stack.currentIndex() != 0:
                    log.debug("SDR preview taking over preview_stack (index 0)")
                    self.preview_stack.setCurrentIndex(0)
            except Exception:
                pass

    @QtCore.Slot()
    def on_save_preview_frame(self) -> None:
        """
        Save the last SDR/tone-mapped preview frame to disk.
        This produces a BGR/8-bit image in the same domain the face embedder sees.
        """
        img = self._last_preview_qimage
        if img is None:
            QtWidgets.QMessageBox.warning(
                self,
                "No preview",
                "No preview frame available to save.",
            )
            return
        try:
            # Be robust against older SessionConfig versions that do not expose outdir.
            outdir = ""
            try:
                outdir = getattr(self.cfg, "outdir", "") or ""
            except Exception:
                outdir = ""
            if not outdir:
                try:
                    outdir = self.outedit.text().strip()
                except Exception:
                    outdir = ""
            if not outdir:
                outdir = os.getcwd()

            fname, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save preview frame",
                os.path.join(outdir, "preview_ref.jpg"),
                "JPEG image (*.jpg);;PNG image (*.png);;All files (*)",
            )
            if not fname:
                return
            img.save(fname)
            self.statusbar.showMessage(f"Saved preview frame to {fname}", 5000)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Save failed",
                f"Could not save preview frame:\n{exc}",
            )

    @QtCore.Slot(object)
    def _on_hdr_preview_p010(self, frame: object) -> None:
        """HDR passthrough path: receives normalized P010 tuple
        (w, h, y_plane, uv_plane, stride_y, stride_uv) from the worker.
        Owns the stacked widget while HDR passthrough is enabled.
        """
        log = logging.getLogger(__name__)

        # Early check: if the worker has disabled HDR passthrough, stop processing frames.
        worker = getattr(self, "_worker", None)
        if worker is not None and not getattr(worker, "_hdr_passthrough_active", False):
            return

        widget = getattr(self, "hdr_widget", None)
        if widget is None:
            return

        try:
            width, height, y_plane, uv_plane, stride_y, stride_uv = frame  # type: ignore[misc]
        except Exception:
            return

        if not isinstance(y_plane, np.ndarray) or not isinstance(uv_plane, np.ndarray):
            return

        try:
            widget.init_hdr(int(width), int(height))
        except Exception:
            if self._hdr_passthrough_enabled:
                log.info("HDR passthrough: disabling (init failed)")
            self._hdr_passthrough_enabled = False
            # Propagate to worker: disable HDR passthrough and clean up reader
            self._queue_worker_disable_hdr_passthrough()
            if hasattr(self, "preview_stack") and self.preview_stack.currentIndex() != 0:
                log.debug("SDR preview taking over preview_stack (index 0)")
                self.preview_stack.setCurrentIndex(0)
            return

        try:
            frame_tuple = (
                int(width),
                int(height),
                y_plane,
                uv_plane,
                int(stride_y),
                int(stride_uv),
            )
            if hasattr(widget, "upload_p010_frame"):
                widget.upload_p010_frame(frame_tuple)
            else:
                widget.feed_p010(
                    int(y_plane.ctypes.data),
                    int(uv_plane.ctypes.data),
                    int(stride_y),
                    int(stride_uv),
                )

        except Exception:
            if self._hdr_passthrough_enabled:
                log.info("HDR passthrough: disabling (upload failed)")
            self._hdr_passthrough_enabled = False
            # Propagate to worker: disable HDR passthrough and clean up reader
            self._queue_worker_disable_hdr_passthrough()
            if hasattr(self, "preview_stack") and self.preview_stack.currentIndex() != 0:
                log.debug("SDR preview taking over preview_stack (index 0)")
                self.preview_stack.setCurrentIndex(0)
            return
        self._hdr_preview_seen = True
        self._last_hdr_preview_t = time.perf_counter()

        try:
            has_ctx = getattr(widget, "has_valid_ctx", None)
            ctx_ok = bool(has_ctx()) if callable(has_ctx) else bool(getattr(widget, "_ctx", None))
        except Exception:
            ctx_ok = False

        if not ctx_ok:
            if self._hdr_passthrough_enabled:
                log.info("HDR passthrough: disabling (no valid context)")
            self._hdr_passthrough_enabled = False
            # Propagate to worker: disable HDR passthrough and clean up reader
            self._queue_worker_disable_hdr_passthrough()
            if hasattr(self, "preview_stack") and self.preview_stack.currentIndex() != 0:
                log.debug("SDR preview taking over preview_stack (index 0)")
                self.preview_stack.setCurrentIndex(0)
            return

        # HDR context exists; mark passthrough active.
        if not self._hdr_passthrough_enabled:
            log.info("HDR passthrough: enabling (context OK)")
        self._hdr_passthrough_enabled = True

        if hasattr(self, "preview_stack") and self.preview_stack.currentIndex() != 1:
            log.debug("HDR preview taking over preview_stack (index 1)")
            self.preview_stack.setCurrentIndex(1)

    def _on_hit(self, crop_path: str):
        def _set(img: QtGui.QImage) -> bool:
            if img.isNull():
                return False
            label_size = self.hit_label.size()
            pix = QtGui.QPixmap.fromImage(img)
            # The saved-crop panel is a QC surface. Do not upscale the crop or
            # smooth-filter it; that made the UI itself look softer than the live
            # preview and hid whether the file was actually sharp.
            if img.width() > label_size.width() or img.height() > label_size.height():
                pix = pix.scaled(
                    label_size,
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.FastTransformation,
                )
            self.hit_label.setPixmap(pix)
            try:
                self.hit_label.setToolTip(f"{crop_path}\n{img.width()}x{img.height()}")
            except Exception:
                logging.getLogger(__name__).debug(
                    "Failed to set hit preview tooltip for %s (%sx%s)",
                    crop_path,
                    img.width(),
                    img.height(),
                    exc_info=True,
                )
            return True

        img = QtGui.QImage(crop_path)
        if _set(img):
            return
        try:
            reader = QtGui.QImageReader(crop_path)
            reader.setAutoTransform(True)
            try:
                reader.setDecideFormatFromContent(True)
            except Exception:
                pass
            img2 = reader.read()
            if _set(img2):
                return
        except Exception:
            pass
        self._log(f"Preview load failed: {crop_path}")

    def _on_finished(self, ok: bool, msg: str):
        self._log(msg)
        self.status_lbl.setText("Idle" if ok else "Error")
        QtWidgets.QMessageBox.information(self, "Finished" if ok else "Error", msg)
        if hasattr(self, "seek_slider"):
            self.seek_slider.setValue(0)
            self.seek_slider.setRange(0, 0)
        if hasattr(self, "time_lbl"):
            self.time_lbl.setText("00:00 / 00:00")
        self._fps = None
        self._total_frames = None
        self._keyframes = []
        self._current_idx = 0
        self._update_play_toggle_ui(paused=True)
        # Clean up thread
        try:
            if self._thread and self._thread.isRunning():
                self._thread.quit()
                self._thread.wait(3000)
        except Exception:
            pass
        self._thread = None
        self._worker = None
        self._update_buttons(state="idle")

    # Helpers
    def _log(self, s: str):
        ts = time.strftime("%H:%M:%S")
        self.log_edit.appendPlainText(f"[{ts}] {s}")

    def _read_keyframes(self, video_path: str, fps: Optional[float], total: Optional[int]) -> List[int]:
        if not video_path or not os.path.exists(video_path):
            return []
        try:
            cmd = [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-skip_frame", "nokey",
                "-show_entries", "frame=pkt_pts_time,best_effort_timestamp_time,pict_type",
                "-show_frames",
                "-of", "json",
                video_path,
            ]
            p = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(p.stdout or "{}")
            frames = data.get("frames", [])
            if not frames or not fps or fps <= 0:
                return []
            out: List[int] = []
            seen = set()
            tf = total if isinstance(total, int) and total > 0 else None
            for fr in frames:
                t = fr.get("best_effort_timestamp_time") or fr.get("pkt_pts_time")
                try:
                    sec = float(t)
                except Exception:
                    continue
                idx = int(round(sec * float(fps)))
                if tf is not None:
                    idx = max(0, min(tf - 1, idx))
                if idx not in seen:
                    seen.add(idx)
                    out.append(idx)
            out.sort()
            return out
        except Exception:
            return []

    def _jump_keyframe(self, forward: bool):
        ks = self._keyframes
        if not ks:
            if self._worker:
                self._worker.step(1 if forward else -1)
            return
        cur = int(self.seek_slider.value()) if hasattr(self, "seek_slider") else self._current_idx
        if forward:
            i = bisect.bisect_right(ks, cur)
            if i >= len(ks):
                # past last keyframe -> regular step forward
                if self._worker:
                    self._worker.step(1)
                return
        else:
            i = bisect.bisect_left(ks, cur) - 1
            if i < 0:
                # before first keyframe -> regular step back
                if self._worker:
                    self._worker.step(-1)
                return
        target = int(ks[i])
        if self._worker:
            self._pause(True)
            self._worker.seek_frame(target)

    def _shortcut_play_pause(self):
        if not self._worker or not hasattr(self, "play_toggle"):
            return
        paused = bool(getattr(self._worker, "_paused", False))
        # Reflect worker state in toggle; toggled signal drives play/pause
        self.play_toggle.setChecked(paused)

    def _shortcut_step(self, forward: bool):
        if not self._worker:
            return
        self._jump_keyframe(forward)

    def keyPressEvent(self, e: QtGui.QKeyEvent):
        # Space/←/→ handled by QShortcuts. Defer others to default.
        return super().keyPressEvent(e)

    def _update_buttons(self, state: str):
        running = (state == "running")
        self.start_btn.setEnabled(not running)
        self.pause_btn.setEnabled(running)
        self.resume_btn.setEnabled(running)
        self.stop_btn.setEnabled(running)
        if hasattr(self, "play_toggle"):
            self.play_toggle.setEnabled(running)
        if hasattr(self, "step_back_btn"):
            self.step_back_btn.setEnabled(running)
        if hasattr(self, "step_fwd_btn"):
            self.step_fwd_btn.setEnabled(running)
        if hasattr(self, "seek_slider"):
            self.seek_slider.setEnabled(running)
        if hasattr(self, "speed_combo"):
            self.speed_combo.setEnabled(running)
        if hasattr(self, "act_start"):
            self.act_start.setEnabled(not running)
        if hasattr(self, "act_pause"):
            self.act_pause.setEnabled(running)
        if hasattr(self, "act_stop"):
            self.act_stop.setEnabled(running)

    # Persist UI settings between runs
    def closeEvent(self, e: QtGui.QCloseEvent) -> None:
        try:
            try:
                s = QtCore.QSettings(APP_ORG, APP_NAME)
                s.setValue("dock_state", self.saveState())
                s.sync()
            except Exception:
                pass
            try:
                self._save_qsettings()
            except Exception:
                pass
            finally:
                # stop running job cleanly
                try:
                    if self._worker:
                        self._worker.request_abort()
                    if self._thread and self._thread.isRunning():
                        self._thread.quit()
                        if not self._thread.wait(4000):
                            self._thread.terminate()
                            self._thread.wait(1000)
                except Exception:
                    pass
                self._worker = None
                self._thread = None
        finally:
            super().closeEvent(e)

    def _load_qsettings(self):
        s = QtCore.QSettings(APP_ORG, APP_NAME)
        try:
            if hasattr(self, "_ffmpeg_edit"):
                stored = s.value(_SETTINGS_KEY_FFMPEG_DIR, getattr(self, "_initial_ffmpeg_dir", ""), type=str)
                self._ffmpeg_edit.setText(stored or "")
        except Exception:
            pass
        self.video_edit.setText(s.value("video", ""))
        refs = s.value("ref", "", type=str)
        paths = self._filter_images([part.strip() for part in refs.split(';') if part.strip()])
        self._set_ref_paths(paths)
        self.out_edit.setText(s.value("out_dir", "output"))
        _stored_ratio = str(s.value("ratio", getattr(self.cfg, "ratio", "1:1,2:3,3:2")) or "1:1,2:3,3:2")
        if _stored_ratio.strip() == "2:3":
            # Migrate the old single portrait default to the new composition-safe
            # candidate list.  A lone 2:3 default deprived the scorer of square
            # alternatives for close/prominent faces.
            _stored_ratio = "1:1,2:3,3:2"
        self.ratio_edit.setText(_stored_ratio)
        try:
            _sdr_nits = float(s.value("sdr_nits", self.cfg.sdr_nits))
            # One-time migration away from the old 125-nits default. Use a
            # sentinel so users who intentionally keep 125 are not reset on
            # every startup.
            if not bool(s.value(_SETTINGS_KEY_SDR_NITS_MIGRATED, False, type=bool)):
                if abs(_sdr_nits - 125.0) < 0.001:
                    _sdr_nits = float(self.cfg.sdr_nits)
                    s.setValue("sdr_nits", _sdr_nits)
                s.setValue(_SETTINGS_KEY_SDR_NITS_MIGRATED, True)
            self.sdr_nits_spin.setValue(_sdr_nits)
        except Exception:
            self.sdr_nits_spin.setValue(float(self.cfg.sdr_nits))
        try:
            # One-time migration for the face/head protection defaults so
            # persisted pre-change defaults pick up the final alignment fix.
            if not bool(s.value(_SETTINGS_KEY_CROP_HEAD_PAD_MIGRATED, False, type=bool)):
                def _migrate_crop_pad(key: str, old_default: float, new_default: float) -> None:
                    raw = s.value(key, None)
                    if raw is None:
                        s.setValue(key, float(new_default))
                        return
                    try:
                        current = float(raw)
                    except Exception:
                        return
                    if abs(current - old_default) < 1e-6:
                        s.setValue(key, float(new_default))

                _migrate_crop_pad("crop_head_side_pad_frac", 0.70, 0.88)
                _migrate_crop_pad("crop_head_top_pad_frac", 0.85, 0.95)
                s.setValue(_SETTINGS_KEY_CROP_HEAD_PAD_MIGRATED, True)
        except Exception:
            pass
        try:
            self.tm_desat_spin.setValue(float(s.value("tm_desat", self.cfg.tm_desat)))
        except Exception:
            self.tm_desat_spin.setValue(float(self.cfg.tm_desat))
        try:
            self.tm_param_spin.setValue(float(s.value("tm_param", self.cfg.tm_param)))
        except Exception:
            self.tm_param_spin.setValue(float(self.cfg.tm_param))
        pref = str(s.value("hdr_tonemap_pref", self.cfg.hdr_tonemap_pref))
        idx = self.tonemap_pref_combo.findData(pref)
        if idx >= 0:
            self.tonemap_pref_combo.setCurrentIndex(idx)
        else:
            self.tonemap_pref_combo.setCurrentIndex(0)
        hw_mode = str(s.value("ff_hwaccel", self.cfg.ff_hwaccel)).strip().lower()
        hw_idx = self.hwaccel_combo.findData(hw_mode)
        if hw_idx >= 0:
            self.hwaccel_combo.setCurrentIndex(hw_idx)
        else:
            fallback_idx = self.hwaccel_combo.findData("off")
            self.hwaccel_combo.setCurrentIndex(fallback_idx if fallback_idx >= 0 else 0)
        self.cfg.ff_hwaccel = self.hwaccel_combo.currentData() or "off"
        fmt = str(s.value("hdr_crop_format", self.cfg.hdr_crop_format)).lower()
        fmt_idx = self.hdr_crop_format_combo.findText(fmt)
        if fmt_idx >= 0:
            self.hdr_crop_format_combo.setCurrentIndex(fmt_idx)
        else:
            self.hdr_crop_format_combo.setCurrentIndex(0)
        self.cfg.hdr_crop_format = self.hdr_crop_format_combo.currentText().lower()
        if hasattr(self, "chk_hdr_screencap_fullres"):
            self.chk_hdr_screencap_fullres.setChecked(
                s.value("hdr_screencap_fullres", self.cfg.hdr_screencap_fullres, type=bool)
            )
            self.cfg.hdr_screencap_fullres = bool(self.chk_hdr_screencap_fullres.isChecked())
        if hasattr(self, "chk_hdr_archive_crops"):
            self.chk_hdr_archive_crops.setChecked(
                s.value("hdr_archive_crops", self.cfg.hdr_archive_crops, type=bool)
            )
            self.cfg.hdr_archive_crops = bool(self.chk_hdr_archive_crops.isChecked())
        if hasattr(self, "hdr_sdr_quality_combo"):
            val = str(s.value("hdr_sdr_quality", getattr(self.cfg, "hdr_sdr_quality", "madvr_like")) or "madvr_like")
            idx = self.hdr_sdr_quality_combo.findData(val)
            self.hdr_sdr_quality_combo.setCurrentIndex(idx if idx >= 0 else 0)
            self.cfg.hdr_sdr_quality = self.hdr_sdr_quality_combo.currentData() or "madvr_like"
        if hasattr(self, "hdr_sdr_tonemap_combo"):
            val = str(s.value("hdr_sdr_tonemap", getattr(self.cfg, "hdr_sdr_tonemap", "auto")) or "auto")
            idx = self.hdr_sdr_tonemap_combo.findData(val)
            self.hdr_sdr_tonemap_combo.setCurrentIndex(idx if idx >= 0 else 0)
            self.cfg.hdr_sdr_tonemap = self.hdr_sdr_tonemap_combo.currentData() or "auto"
        if hasattr(self, "hdr_sdr_gamut_combo"):
            val = str(s.value("hdr_sdr_gamut_mapping", getattr(self.cfg, "hdr_sdr_gamut_mapping", "clip")) or "clip")
            idx = self.hdr_sdr_gamut_combo.findData(val)
            self.hdr_sdr_gamut_combo.setCurrentIndex(idx if idx >= 0 else 0)
            self.cfg.hdr_sdr_gamut_mapping = self.hdr_sdr_gamut_combo.currentData() or "clip"
        if all(hasattr(self, name) for name in ("hdr_sdr_quality_combo", "hdr_sdr_tonemap_combo", "hdr_sdr_gamut_combo")):
            _old_hdr_defaults = (
                self.hdr_sdr_quality_combo.currentData(),
                self.hdr_sdr_tonemap_combo.currentData(),
                self.hdr_sdr_gamut_combo.currentData(),
            )
            if _old_hdr_defaults == ("resolve_like", "spline", "perceptual"):
                # Migrate the previous still-export defaults.  They were intentionally
                # different from the live libplacebo renderer and are the common cause
                # of saved crops looking brighter/flatter than the preview.
                for _combo, _data in (
                    (self.hdr_sdr_quality_combo, "madvr_like"),
                    (self.hdr_sdr_tonemap_combo, "auto"),
                    (self.hdr_sdr_gamut_combo, "clip"),
                ):
                    _idx = _combo.findData(_data)
                    if _idx >= 0:
                        _combo.setCurrentIndex(_idx)
                self.cfg.hdr_sdr_quality = self.hdr_sdr_quality_combo.currentData() or "madvr_like"
                self.cfg.hdr_sdr_tonemap = self.hdr_sdr_tonemap_combo.currentData() or "auto"
                self.cfg.hdr_sdr_gamut_mapping = self.hdr_sdr_gamut_combo.currentData() or "clip"
        if hasattr(self, "hdr_sdr_contrast_spin"):
            try:
                self.hdr_sdr_contrast_spin.setValue(float(s.value("hdr_sdr_contrast_recovery", getattr(self.cfg, "hdr_sdr_contrast_recovery", 0.30))))
            except Exception:
                self.hdr_sdr_contrast_spin.setValue(0.30)
            self.cfg.hdr_sdr_contrast_recovery = float(self.hdr_sdr_contrast_spin.value())
        if hasattr(self, "hdr_sdr_peak_check"):
            self.hdr_sdr_peak_check.setChecked(
                s.value("hdr_sdr_peak_detect", getattr(self.cfg, "hdr_sdr_peak_detect", True), type=bool)
            )
            self.cfg.hdr_sdr_peak_detect = bool(self.hdr_sdr_peak_check.isChecked())
        if hasattr(self, "hdr_sdr_bad_fallback_check"):
            self.hdr_sdr_bad_fallback_check.setChecked(
                s.value("hdr_sdr_allow_inaccurate_fallback", getattr(self.cfg, "hdr_sdr_allow_inaccurate_fallback", False), type=bool)
            )
            self.cfg.hdr_sdr_allow_inaccurate_fallback = bool(self.hdr_sdr_bad_fallback_check.isChecked())
        try:
            self.cfg.hdr_export_timeout_sec = max(
                5,
                int(s.value("hdr_export_timeout_sec", getattr(self.cfg, "hdr_export_timeout_sec", 300)) or 300),
            )
        except Exception:
            self.cfg.hdr_export_timeout_sec = 300
        self.cfg.seek_fast = s.value("seek_fast", True, type=bool)
        try:
            self.cfg.seek_max_grabs = int(s.value("seek_max_grabs", 12))
        except Exception:
            self.cfg.seek_max_grabs = 12
        try:
            self.cfg.preview_max_dim = int(s.value("preview_max_dim", 1280))
        except Exception:
            self.cfg.preview_max_dim = 1280
        try:
            self.cfg.preview_fps_cap = int(s.value("preview_fps_cap", 20))
        except Exception:
            self.cfg.preview_fps_cap = 20
        try:
            self.cfg.seek_preview_peek_every = int(s.value("seek_preview_peek_every", 16))
        except Exception:
            self.cfg.seek_preview_peek_every = 16
        self.cfg.overlay_scores = s.value("overlay_scores", False, type=bool)
        try:
            self.cfg.preview_every = int(s.value("preview_every", 3))
        except Exception:
            self.cfg.preview_every = 3
        self.stride_spin.setValue(int(s.value("frame_stride", 2)))
        self.det_conf_spin.setValue(float(s.value("min_det_conf", 0.35)))
        self.face_thr_spin.setValue(float(s.value("face_thresh", 0.45)))
        self.face_det_conf_spin.setValue(float(s.value("face_det_conf", 0.5)))
        self.face_det_pad_spin.setValue(float(s.value("face_det_pad", 0.08)))
        self.face_quality_spin.setValue(float(s.value("face_quality_min", 70.0)))
        self.face_vis_quality_check.setChecked(
            s.value("face_visible_uses_quality", True, type=bool)
        )
        self.reid_thr_spin.setValue(float(s.value("reid_thresh", 0.38)))
        self.combine_combo.setCurrentText(s.value("combine", "min"))
        self.match_mode_combo.setCurrentText(s.value("match_mode", "face_only"))
        if hasattr(self, 'disable_reid_check'):
            self.disable_reid_check.setChecked(s.value("disable_reid", True, type=bool))
        if hasattr(self, 'face_fullframe_check'):
            self.face_fullframe_check.setChecked(s.value("face_fullframe_when_missed", True, type=bool))
        if hasattr(self, 'face_fullframe_imgsz_spin'):
            self.face_fullframe_imgsz_spin.setValue(int(s.value("face_fullframe_imgsz", self.cfg.face_fullframe_imgsz)))
        if hasattr(self, 'rot_adaptive_check'):
            self.rot_adaptive_check.setChecked(
                s.value("rot_adaptive", self.cfg.rot_adaptive, type=bool)
            )
        if hasattr(self, 'rot_every_spin'):
            self.rot_every_spin.setValue(int(s.value("rot_every_n", self.cfg.rot_every_n)))
        if hasattr(self, 'rot_after_hit_spin'):
            self.rot_after_hit_spin.setValue(int(s.value("rot_after_hit_frames", self.cfg.rot_after_hit_frames)))
        if hasattr(self, 'fast_no_face_spin'):
            self.fast_no_face_spin.setValue(int(s.value("fast_no_face_imgsz", self.cfg.fast_no_face_imgsz)))
        self.only_best_check.setChecked(s.value("only_best", True, type=bool))
        if hasattr(self, 'chk_prescan'):
            self.chk_prescan.setChecked(s.value("prescan_enable", True, type=bool))
        if hasattr(self, 'spin_prescan_stride'):
            self.spin_prescan_stride.setValue(int(s.value("prescan_stride", self.cfg.prescan_stride)))
        if hasattr(self, 'spin_prescan_fd_add'):
            self.spin_prescan_fd_add.setValue(float(s.value("prescan_fd_add", self.cfg.prescan_fd_add)))
        if hasattr(self, 'spin_prescan_max_width'):
            self.spin_prescan_max_width.setValue(int(s.value("prescan_max_width", self.cfg.prescan_max_width)))
        if hasattr(self, 'spin_prescan_decode_max_w'):
            self.spin_prescan_decode_max_w.setValue(int(s.value("prescan_decode_max_w", getattr(self.cfg, "prescan_decode_max_w", 384))))
        if hasattr(self, "prescan_cache_combo"):
            mode = str(s.value("prescan_cache_mode", getattr(self.cfg, "prescan_cache_mode", "auto")) or "auto")
            idx = self.prescan_cache_combo.findData(mode)
            self.prescan_cache_combo.setCurrentIndex(idx if idx >= 0 else 0)
            self.cfg.prescan_cache_mode = self.prescan_cache_combo.currentData() or "auto"
        self.cfg.prescan_cache_dir = str(
            s.value("prescan_cache_dir", getattr(self.cfg, "prescan_cache_dir", "prescan_cache"))
            or "prescan_cache"
        )
        if hasattr(self, "edit_prescan_cache_dir"):
            self.edit_prescan_cache_dir.setText(self.cfg.prescan_cache_dir)
        if hasattr(self, 'spin_prescan_face_conf'):
            self.spin_prescan_face_conf.setValue(float(s.value("prescan_face_conf", self.cfg.prescan_face_conf)))
        if hasattr(self, 'spin_prescan_fd_enter'):
            self.spin_prescan_fd_enter.setValue(float(s.value("prescan_fd_enter", self.cfg.prescan_fd_enter)))
        if hasattr(self, 'spin_prescan_fd_exit'):
            self.spin_prescan_fd_exit.setValue(float(s.value("prescan_fd_exit", self.cfg.prescan_fd_exit)))
        if hasattr(self, 'spin_prescan_add_cooldown'):
            self.spin_prescan_add_cooldown.setValue(int(s.value("prescan_add_cooldown_samples", self.cfg.prescan_add_cooldown_samples)))
        if hasattr(self, 'spin_prescan_min_segment'):
            self.spin_prescan_min_segment.setValue(float(s.value("prescan_min_segment_sec", self.cfg.prescan_min_segment_sec)))
        if hasattr(self, 'spin_prescan_pad'):
            self.spin_prescan_pad.setValue(float(s.value("prescan_pad_sec", self.cfg.prescan_pad_sec)))
        if hasattr(self, 'spin_prescan_bridge'):
            self.spin_prescan_bridge.setValue(float(s.value("prescan_bridge_gap_sec", self.cfg.prescan_bridge_gap_sec)))
        if hasattr(self, 'spin_prescan_exit_cooldown'):
            self.spin_prescan_exit_cooldown.setValue(
                float(s.value("prescan_exit_cooldown_sec", self.cfg.prescan_exit_cooldown_sec))
            )
        if hasattr(self, 'spin_prescan_refine_window'):
            self.spin_prescan_refine_window.setValue(
                float(s.value("prescan_boundary_refine_sec", self.cfg.prescan_boundary_refine_sec))
            )
        if hasattr(self, 'spin_prescan_refine_stride'):
            self.spin_prescan_refine_stride.setValue(
                int(s.value("prescan_refine_stride_min", self.cfg.prescan_refine_stride_min))
            )
        if hasattr(self, 'chk_prescan_trim_pad'):
            self.chk_prescan_trim_pad.setChecked(
                s.value("prescan_trim_pad", self.cfg.prescan_trim_pad, type=bool)
            )
        if hasattr(self, 'spin_prescan_bank_max'):
            self.spin_prescan_bank_max.setValue(int(s.value("prescan_bank_max", self.cfg.prescan_bank_max)))
        if hasattr(self, 'spin_prescan_dedup'):
            self.spin_prescan_dedup.setValue(float(s.value("prescan_diversity_dedup_cos", self.cfg.prescan_diversity_dedup_cos)))
        if hasattr(self, 'spin_prescan_margin'):
            self.spin_prescan_margin.setValue(float(s.value("prescan_replace_margin", self.cfg.prescan_replace_margin)))
        self.min_sharp_spin.setValue(float(s.value("min_sharpness", 0.0)))
        self.min_gap_spin.setValue(float(s.value("min_gap_sec", 1.5)))
        self.min_box_pix_spin.setValue(int(s.value("min_box_pixels", 5000)))
        self.auto_crop_check.setChecked(s.value("auto_crop_borders", True, type=bool))
        self.border_thr_spin.setValue(int(s.value("border_threshold", 22)))
        self.require_face_check.setChecked(
            s.value("require_face_if_visible", True, type=bool)
        )
        self.lock_mom_spin.setValue(float(s.value("lock_momentum", 0.7)))
        self.suppress_neg_check.setChecked(
            s.value("suppress_negatives", False, type=bool)
        )
        self.neg_tol_spin.setValue(float(s.value("neg_tolerance", 0.35)))
        self.max_neg_spin.setValue(int(s.value("max_negatives", 5)))
        self.log_every_spin.setValue(float(s.value("log_interval_sec", 1.0)))
        self.lock_after_spin.setValue(int(s.value("lock_after_hits", 1)))
        self.lock_face_spin.setValue(float(s.value("lock_face_thresh", 0.28)))
        self.lock_reid_spin.setValue(float(s.value("lock_reid_thresh", 0.30)))
        self.margin_spin.setValue(float(s.value("score_margin", 0.03)))
        self.iou_gate_spin.setValue(float(s.value("iou_gate", 0.05)))
        self.use_arc_check.setChecked(s.value("use_arcface", True, type=bool))
        self.device_combo.setCurrentText(s.value("device", "cuda"))
        self.yolo_edit.setText(s.value("yolo_model", "yolov8n.pt"))
        self.face_yolo_edit.setText(s.value("face_model", "scrfd_10g_bnkps"))
        self.annot_check.setChecked(s.value("save_annot", False, type=bool))
        self.preview_every_spin.setValue(int(s.value("preview_every", self.cfg.preview_every)))
        if hasattr(self, 'preview_max_dim_spin'):
            self.preview_max_dim_spin.setValue(int(s.value("preview_max_dim", self.cfg.preview_max_dim)))
        if hasattr(self, 'preview_fps_cap_spin'):
            self.preview_fps_cap_spin.setValue(int(s.value("preview_fps_cap", self.cfg.preview_fps_cap)))
        if hasattr(self, 'seek_fast_check'):
            self.seek_fast_check.setChecked(s.value("seek_fast", self.cfg.seek_fast, type=bool))
        if hasattr(self, 'seek_max_grabs_spin'):
            self.seek_max_grabs_spin.setValue(int(s.value("seek_max_grabs", self.cfg.seek_max_grabs)))
        if hasattr(self, 'seek_preview_peek_spin'):
            self.seek_preview_peek_spin.setValue(int(s.value("seek_preview_peek_every", self.cfg.seek_preview_peek_every)))
        if hasattr(self, 'overlay_scores_check'):
            self.overlay_scores_check.setChecked(s.value("overlay_scores", self.cfg.overlay_scores, type=bool))
        if hasattr(self, 'faceless_allow_check'):
            self.faceless_allow_check.setChecked(
                s.value(
                    "allow_faceless_when_locked",
                    self.cfg.allow_faceless_when_locked,
                    type=bool,
                )
            )
        if hasattr(self, 'learn_bank_runtime_check'):
            self.learn_bank_runtime_check.setChecked(
                s.value(
                    "learn_bank_runtime",
                    self.cfg.learn_bank_runtime,
                    type=bool,
                )
            )
        if hasattr(self, 'drop_reid_if_any_face_match_check'):
            self.drop_reid_if_any_face_match_check.setChecked(
                s.value(
                    "drop_reid_if_any_face_match",
                    self.cfg.drop_reid_if_any_face_match,
                    type=bool,
                )
            )
        if hasattr(self, 'faceless_reid_spin'):
            self.faceless_reid_spin.setValue(float(s.value("faceless_reid_thresh", self.cfg.faceless_reid_thresh)))
        if hasattr(self, 'faceless_iou_spin'):
            self.faceless_iou_spin.setValue(float(s.value("faceless_iou_min", self.cfg.faceless_iou_min)))
        if hasattr(self, 'faceless_persist_spin'):
            self.faceless_persist_spin.setValue(int(s.value("faceless_persist_frames", self.cfg.faceless_persist_frames)))
        if hasattr(self, 'faceless_min_area_spin'):
            self.faceless_min_area_spin.setValue(float(s.value("faceless_min_area_frac", self.cfg.faceless_min_area_frac)))
        if hasattr(self, 'faceless_max_area_spin'):
            self.faceless_max_area_spin.setValue(float(s.value("faceless_max_area_frac", self.cfg.faceless_max_area_frac)))
        if hasattr(self, 'faceless_center_spin'):
            self.faceless_center_spin.setValue(float(s.value("faceless_center_max_frac", self.cfg.faceless_center_max_frac)))
        if hasattr(self, 'faceless_motion_spin'):
            self.faceless_motion_spin.setValue(float(s.value("faceless_min_motion_frac", self.cfg.faceless_min_motion_frac)))
        if hasattr(self, 'crop_face_side_margin_spin'):
            self.crop_face_side_margin_spin.setValue(
                float(s.value("crop_face_side_margin_frac", self.cfg.crop_face_side_margin_frac))
            )
        if hasattr(self, 'crop_top_headroom_spin'):
            self.crop_top_headroom_spin.setValue(
                float(s.value("crop_top_headroom_max_frac", self.cfg.crop_top_headroom_max_frac))
            )
        if hasattr(self, 'crop_bottom_min_face_spin'):
            self.crop_bottom_min_face_spin.setValue(
                float(s.value("crop_bottom_min_face_heights", self.cfg.crop_bottom_min_face_heights))
            )
        if hasattr(self, 'crop_penalty_weight_spin'):
            self.crop_penalty_weight_spin.setValue(
                float(s.value("crop_penalty_weight", self.cfg.crop_penalty_weight))
            )
        try:
            self.cfg.crop_head_side_pad_frac = float(
                s.value(
                    "crop_head_side_pad_frac",
                    getattr(self.cfg, "crop_head_side_pad_frac", SessionConfig.crop_head_side_pad_frac),
                )
            )
        except Exception:
            self.cfg.crop_head_side_pad_frac = float(SessionConfig.crop_head_side_pad_frac)
        try:
            self.cfg.crop_head_top_pad_frac = float(
                s.value(
                    "crop_head_top_pad_frac",
                    getattr(self.cfg, "crop_head_top_pad_frac", SessionConfig.crop_head_top_pad_frac),
                )
            )
        except Exception:
            self.cfg.crop_head_top_pad_frac = float(SessionConfig.crop_head_top_pad_frac)
        try:
            self.cfg.crop_head_bottom_pad_frac = float(
                s.value(
                    "crop_head_bottom_pad_frac",
                    getattr(self.cfg, "crop_head_bottom_pad_frac", SessionConfig.crop_head_bottom_pad_frac),
                )
            )
        except Exception:
            self.cfg.crop_head_bottom_pad_frac = float(SessionConfig.crop_head_bottom_pad_frac)
        try:
            self.cfg.wide_face_aspect_penalty_weight = float(
                s.value(
                    "wide_face_aspect_penalty_weight",
                    getattr(self.cfg, "wide_face_aspect_penalty_weight", 10.0),
                )
            )
        except Exception:
            self.cfg.wide_face_aspect_penalty_weight = 10.0
        try:
            self.cfg.wide_face_min_frame_frac = float(
                s.value("wide_face_min_frame_frac", getattr(self.cfg, "wide_face_min_frame_frac", 0.12))
            )
        except Exception:
            self.cfg.wide_face_min_frame_frac = 0.12
        try:
            self.cfg.wide_face_aspect_limit = float(
                s.value("wide_face_aspect_limit", getattr(self.cfg, "wide_face_aspect_limit", 1.05))
            )
        except Exception:
            self.cfg.wide_face_aspect_limit = 1.05
        self.cfg.compose_crop_enable = s.value(
            "compose_crop_enable",
            getattr(self.cfg, "compose_crop_enable", True),
            type=bool,
        )
        self.cfg.compose_detect_person_for_face = s.value(
            "compose_detect_person_for_face",
            getattr(self.cfg, "compose_detect_person_for_face", True),
            type=bool,
        )
        try:
            self.cfg.compose_close_face_h_frac = float(
                s.value("compose_close_face_h_frac", getattr(self.cfg, "compose_close_face_h_frac", 0.34))
            )
        except Exception:
            self.cfg.compose_close_face_h_frac = 0.34
        try:
            self.cfg.compose_upper_face_h_frac = float(
                s.value("compose_upper_face_h_frac", getattr(self.cfg, "compose_upper_face_h_frac", 0.22))
            )
        except Exception:
            self.cfg.compose_upper_face_h_frac = 0.22
        try:
            self.cfg.compose_body_face_h_frac = float(
                s.value("compose_body_face_h_frac", getattr(self.cfg, "compose_body_face_h_frac", 0.085))
            )
        except Exception:
            self.cfg.compose_body_face_h_frac = 0.085
        try:
            self.cfg.compose_landscape_face_penalty = float(
                s.value(
                    "compose_landscape_face_penalty",
                    getattr(self.cfg, "compose_landscape_face_penalty", 5.0),
                )
            )
        except Exception:
            self.cfg.compose_landscape_face_penalty = 5.0
        try:
            self.cfg.compose_body_every_n = int(
                s.value("compose_body_every_n", getattr(self.cfg, "compose_body_every_n", 6))
            )
        except Exception:
            self.cfg.compose_body_every_n = 6
        try:
            self.cfg.compose_person_detect_cadence = int(
                s.value("compose_person_detect_cadence", getattr(self.cfg, "compose_person_detect_cadence", 6))
            )
        except Exception:
            self.cfg.compose_person_detect_cadence = 6
        if hasattr(self, 'face_anchor_down_spin'):
            self.face_anchor_down_spin.setValue(
                float(s.value("face_anchor_down_frac", self.cfg.face_anchor_down_frac))
            )
        if hasattr(self, 'lambda_facefrac_spin'):
            self.lambda_facefrac_spin.setValue(
                float(s.value("lambda_facefrac", self.cfg.lambda_facefrac))
            )
        if hasattr(self, 'crop_center_weight_spin'):
            self.crop_center_weight_spin.setValue(
                float(s.value("crop_center_weight", self.cfg.crop_center_weight))
            )
        if hasattr(self, 'area_gamma_spin'):
            self.area_gamma_spin.setValue(
                float(s.value("area_gamma", self.cfg.area_gamma))
            )
        if hasattr(self, 'area_face_scale_w_spin'):
            self.area_face_scale_w_spin.setValue(
                float(s.value("area_face_scale_weight", self.cfg.area_face_scale_weight))
            )
        if hasattr(self, 'square_pull_face_min_spin'):
            self.square_pull_face_min_spin.setValue(
                float(s.value("square_pull_face_min", self.cfg.square_pull_face_min))
            )
        if hasattr(self, 'square_pull_weight_spin'):
            self.square_pull_weight_spin.setValue(
                float(s.value("square_pull_weight", self.cfg.square_pull_weight))
            )
        if hasattr(self, 'face_target_close_spin'):
            self.face_target_close_spin.setValue(
                float(s.value("face_target_close", self.cfg.face_target_close))
            )
        if hasattr(self, 'face_target_upper_spin'):
            self.face_target_upper_spin.setValue(
                float(s.value("face_target_upper", self.cfg.face_target_upper))
            )
        if hasattr(self, 'face_target_cowboy_spin'):
            self.face_target_cowboy_spin.setValue(
                float(s.value("face_target_cowboy", self.cfg.face_target_cowboy))
            )
        if hasattr(self, 'face_target_body_spin'):
            self.face_target_body_spin.setValue(
                float(s.value("face_target_body", self.cfg.face_target_body))
            )
        if hasattr(self, 'face_target_tol_spin'):
            self.face_target_tol_spin.setValue(
                float(s.value("face_target_tolerance", self.cfg.face_target_tolerance))
            )
        if hasattr(self, 'face_target_close_min_spin'):
            self.face_target_close_min_spin.setValue(
                float(s.value("face_target_close_min_frac", self.cfg.face_target_close_min_frac))
            )
        if hasattr(self, 'w_close_spin'):
            self.w_close_spin.setValue(float(s.value("w_close", self.cfg.w_close)))
        if hasattr(self, 'w_upper_spin'):
            self.w_upper_spin.setValue(float(s.value("w_upper", self.cfg.w_upper)))
        if hasattr(self, 'w_cowboy_spin'):
            self.w_cowboy_spin.setValue(float(s.value("w_cowboy", self.cfg.w_cowboy)))
        if hasattr(self, 'w_body_spin'):
            self.w_body_spin.setValue(float(s.value("w_body", self.cfg.w_body)))
        if hasattr(self, 'smart_crop_enable_check'):
            self.smart_crop_enable_check.setChecked(
                s.value("smart_crop_enable", self.cfg.smart_crop_enable, type=bool)
            )
        if hasattr(self, 'smart_crop_steps_spin'):
            self.smart_crop_steps_spin.setValue(
                int(s.value("smart_crop_steps", self.cfg.smart_crop_steps))
            )
        if hasattr(self, 'smart_crop_side_search_spin'):
            self.smart_crop_side_search_spin.setValue(
                float(s.value("smart_crop_side_search_frac", self.cfg.smart_crop_side_search_frac))
            )
        if hasattr(self, 'smart_crop_use_grad_check'):
            self.smart_crop_use_grad_check.setChecked(
                s.value("smart_crop_use_grad", self.cfg.smart_crop_use_grad, type=bool)
            )
        if hasattr(self, 'face_max_frac_spin'):
            self.face_max_frac_spin.setValue(
                float(s.value("face_max_frac_in_crop", self.cfg.face_max_frac_in_crop))
            )
        if hasattr(self, 'face_min_frac_spin'):
            self.face_min_frac_spin.setValue(
                float(s.value("face_min_frac_in_crop", self.cfg.face_min_frac_in_crop))
            )
        if hasattr(self, 'crop_min_height_frac_spin'):
            self.crop_min_height_frac_spin.setValue(
                float(s.value("crop_min_height_frac", self.cfg.crop_min_height_frac))
            )
        if hasattr(self, 'compose_crop_enable_check'):
            self.compose_crop_enable_check.setChecked(
                s.value("compose_crop_enable", self.cfg.compose_crop_enable, type=bool)
            )
        if hasattr(self, 'compose_detect_person_for_face_check'):
            self.compose_detect_person_for_face_check.setChecked(
                s.value(
                    "compose_detect_person_for_face",
                    self.cfg.compose_detect_person_for_face,
                    type=bool,
                )
            )
        if hasattr(self, 'compose_close_face_h_frac_spin'):
            self.compose_close_face_h_frac_spin.setValue(
                float(s.value("compose_close_face_h_frac", self.cfg.compose_close_face_h_frac))
            )
        if hasattr(self, 'compose_upper_face_h_frac_spin'):
            self.compose_upper_face_h_frac_spin.setValue(
                float(s.value("compose_upper_face_h_frac", self.cfg.compose_upper_face_h_frac))
            )
        if hasattr(self, 'compose_body_face_h_frac_spin'):
            self.compose_body_face_h_frac_spin.setValue(
                float(s.value("compose_body_face_h_frac", self.cfg.compose_body_face_h_frac))
            )
        if hasattr(self, 'compose_landscape_face_penalty_spin'):
            self.compose_landscape_face_penalty_spin.setValue(
                float(
                    s.value(
                        "compose_landscape_face_penalty",
                        self.cfg.compose_landscape_face_penalty,
                    )
                )
            )
        if hasattr(self, 'compose_body_every_n_spin'):
            self.compose_body_every_n_spin.setValue(
                int(s.value("compose_body_every_n", self.cfg.compose_body_every_n))
            )
        if hasattr(self, 'compose_person_detect_cadence_spin'):
            self.compose_person_detect_cadence_spin.setValue(
                int(s.value("compose_person_detect_cadence", getattr(self.cfg, "compose_person_detect_cadence", 6)))
            )
        # Face-first defaults if controls not present
        if hasattr(self, 'pref_face_check'):
            self.pref_face_check.setChecked(
                s.value("prefer_face_when_available", True, type=bool)
            )
        # Use existing controls to hold thresholds for convenience
        if hasattr(self, 'face_quality_spin'):
            self.face_quality_spin.setValue(float(s.value("face_quality_min", 70.0)))
        if hasattr(self, 'face_vis_quality_check'):
            self.face_vis_quality_check.setChecked(
                s.value("face_visible_uses_quality", True, type=bool)
            )
        if hasattr(self, 'face_det_conf_spin'):
            self.face_det_conf_spin.setValue(float(s.value("face_det_conf", 0.5)))
        if hasattr(self, 'face_det_pad_spin'):
            self.face_det_pad_spin.setValue(float(s.value("face_det_pad", 0.08)))
        self._on_ui_change()

    def _save_qsettings(self):
        s = QtCore.QSettings(APP_ORG, APP_NAME)
        cfg = self._collect_cfg()
        cfg_dict = asdict(cfg)
        cfg_dict["ref"] = "; ".join(self._get_ref_paths())
        for k, v in cfg_dict.items():
            s.setValue(k, v)
        try:
            s.setValue("sdr_nits", float(self.sdr_nits_spin.value()))
            s.setValue("tm_desat", float(self.tm_desat_spin.value()))
            s.setValue("tm_param", float(self.tm_param_spin.value()))
            s.setValue("hdr_tonemap_pref", self.tonemap_pref_combo.currentData())
        except Exception:
            pass
        s.setValue("ff_hwaccel", cfg.ff_hwaccel)
        if hasattr(self, "_ffmpeg_edit"):
            try:
                s.setValue(_SETTINGS_KEY_FFMPEG_DIR, self._ffmpeg_edit.text().strip())
            except Exception:
                pass
        if hasattr(self, 'spin_prescan_fd_add'):
            s.setValue("prescan_fd_add", float(self.spin_prescan_fd_add.value()))
        if hasattr(self, 'spin_prescan_exit_cooldown'):
            s.setValue(
                "prescan_exit_cooldown_sec",
                float(self.spin_prescan_exit_cooldown.value()),
            )
        if hasattr(self, 'preview_max_dim_spin'):
            s.setValue("preview_max_dim", int(self.preview_max_dim_spin.value()))
        if hasattr(self, 'preview_fps_cap_spin'):
            s.setValue("preview_fps_cap", int(self.preview_fps_cap_spin.value()))
        if hasattr(self, 'seek_fast_check'):
            s.setValue("seek_fast", bool(self.seek_fast_check.isChecked()))
        if hasattr(self, 'seek_max_grabs_spin'):
            s.setValue("seek_max_grabs", int(self.seek_max_grabs_spin.value()))
        if hasattr(self, 'seek_preview_peek_spin'):
            s.setValue("seek_preview_peek_every", int(self.seek_preview_peek_spin.value()))
        if hasattr(self, 'overlay_scores_check'):
            s.setValue("overlay_scores", bool(self.overlay_scores_check.isChecked()))
        if hasattr(self, 'preview_every_spin'):
            s.setValue("preview_every", int(self.preview_every_spin.value()))
        if hasattr(self, 'spin_prescan_refine_window'):
            s.setValue(
                "prescan_boundary_refine_sec",
                float(self.spin_prescan_refine_window.value()),
            )
        if hasattr(self, 'spin_prescan_refine_stride'):
            s.setValue(
                "prescan_refine_stride_min",
                int(self.spin_prescan_refine_stride.value()),
            )
        if hasattr(self, 'chk_prescan_trim_pad'):
            s.setValue(
                "prescan_trim_pad",
                bool(self.chk_prescan_trim_pad.isChecked()),
            )
        if hasattr(self, 'disable_reid_check'):
            s.setValue("disable_reid", self.disable_reid_check.isChecked())
        if hasattr(self, 'face_fullframe_check'):
            s.setValue("face_fullframe_when_missed", self.face_fullframe_check.isChecked())
        if hasattr(self, 'faceless_allow_check'):
            s.setValue(
                "allow_faceless_when_locked",
                bool(self.faceless_allow_check.isChecked()),
            )
        if hasattr(self, 'learn_bank_runtime_check'):
            s.setValue(
                "learn_bank_runtime",
                bool(self.learn_bank_runtime_check.isChecked()),
            )
        if hasattr(self, 'drop_reid_if_any_face_match_check'):
            s.setValue(
                "drop_reid_if_any_face_match",
                bool(self.drop_reid_if_any_face_match_check.isChecked()),
            )
        if hasattr(self, 'rot_adaptive_check'):
            s.setValue(
                "rot_adaptive",
                bool(self.rot_adaptive_check.isChecked()),
            )
        if hasattr(self, 'rot_every_spin'):
            s.setValue(
                "rot_every_n",
                int(self.rot_every_spin.value()),
            )
        if hasattr(self, 'rot_after_hit_spin'):
            s.setValue(
                "rot_after_hit_frames",
                int(self.rot_after_hit_spin.value()),
            )
        if hasattr(self, 'fast_no_face_spin'):
            s.setValue(
                "fast_no_face_imgsz",
                int(self.fast_no_face_spin.value()),
            )
        if hasattr(self, 'smart_crop_enable_check'):
            s.setValue(
                "smart_crop_enable",
                bool(self.smart_crop_enable_check.isChecked()),
            )
        if hasattr(self, 'smart_crop_steps_spin'):
            s.setValue(
                "smart_crop_steps",
                int(self.smart_crop_steps_spin.value()),
            )
        if hasattr(self, 'smart_crop_side_search_spin'):
            s.setValue(
                "smart_crop_side_search_frac",
                float(self.smart_crop_side_search_spin.value()),
            )
        if hasattr(self, 'smart_crop_use_grad_check'):
            s.setValue(
                "smart_crop_use_grad",
                bool(self.smart_crop_use_grad_check.isChecked()),
            )
        if hasattr(self, 'lambda_facefrac_spin'):
            s.setValue("lambda_facefrac", self.lambda_facefrac_spin.value())
        if hasattr(self, 'crop_center_weight_spin'):
            s.setValue("crop_center_weight", self.crop_center_weight_spin.value())
        if hasattr(self, 'area_gamma_spin'):
            s.setValue("area_gamma", self.area_gamma_spin.value())
        if hasattr(self, 'area_face_scale_w_spin'):
            s.setValue("area_face_scale_weight", self.area_face_scale_w_spin.value())
        if hasattr(self, 'square_pull_face_min_spin'):
            s.setValue("square_pull_face_min", self.square_pull_face_min_spin.value())
        if hasattr(self, 'square_pull_weight_spin'):
            s.setValue("square_pull_weight", self.square_pull_weight_spin.value())
        if hasattr(self, 'face_target_close_spin'):
            s.setValue("face_target_close", self.face_target_close_spin.value())
        if hasattr(self, 'face_target_upper_spin'):
            s.setValue("face_target_upper", self.face_target_upper_spin.value())
        if hasattr(self, 'face_target_cowboy_spin'):
            s.setValue("face_target_cowboy", self.face_target_cowboy_spin.value())
        if hasattr(self, 'face_target_body_spin'):
            s.setValue("face_target_body", self.face_target_body_spin.value())
        if hasattr(self, 'face_target_tol_spin'):
            s.setValue("face_target_tolerance", self.face_target_tol_spin.value())
        if hasattr(self, 'face_target_close_min_spin'):
            s.setValue("face_target_close_min_frac", self.face_target_close_min_spin.value())
        if hasattr(self, 'w_close_spin'):
            s.setValue("w_close", self.w_close_spin.value())
        if hasattr(self, 'w_upper_spin'):
            s.setValue("w_upper", self.w_upper_spin.value())
        if hasattr(self, 'w_cowboy_spin'):
            s.setValue("w_cowboy", self.w_cowboy_spin.value())
        if hasattr(self, 'w_body_spin'):
            s.setValue("w_body", self.w_body_spin.value())
        s.sync()


def main():
    # Suppress noisy HF hub warnings in GUI
    import os
    import logging
    os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
    try:
        logging.getLogger('huggingface_hub').setLevel(logging.ERROR)
        logging.getLogger('huggingface_hub.file_download').setLevel(logging.ERROR)
    except Exception:
        pass
    QtCore.QCoreApplication.setOrganizationName(APP_ORG)
    QtCore.QCoreApplication.setApplicationName(APP_NAME)
    app = QtWidgets.QApplication(sys.argv)
    # Apply staged updates as early as possible (no windows/dialogs shown yet)
    try:
        # Import late to avoid circulars if any
        try:
            from .updater import UpdateManager  # type: ignore
        except Exception:
            from updater import UpdateManager  # type: ignore
        if UpdateManager is not None:
            UpdateManager(APP_NAME).maybe_apply_pending_at_start()
    except Exception:
        pass
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    try:
        import cv2

        cv2.setNumThreads(1)
    except Exception:
        pass
    main()
