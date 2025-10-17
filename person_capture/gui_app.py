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

import os, sys, subprocess, shutil, threading, struct
import json, csv, traceback
import time
import logging
import cv2
import numpy as np
import bisect
import queue
import math
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List
from pathlib import Path

# Robust imports: support both package ("from .module") and flat files ("import module").
def _imp():
    try:
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
) = _imp()
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


if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


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
    seek_max_grabs: int = 45
    out_dir: str = "output"
    ratio: str = "2:3,1:1,3:2"
    frame_stride: int = 2
    min_det_conf: float = 0.35
    face_thresh: float = 0.45
    reid_thresh: float = 0.42
    combine: str = "min"            # min | avg | face_priority
    match_mode: str = "face_only"        # either | both | face_only | reid_only
    only_best: bool = True
    min_sharpness: float = 0.0
    min_gap_sec: float = 1.5
    min_box_pixels: int = 8000
    auto_crop_borders: bool = True
    log_interval_sec: float = 1.0
    lock_after_hits: int = 1
    lock_face_thresh: float = 0.28
    lock_reid_thresh: float = 0.30
    score_margin: float = 0.03
    iou_gate: float = 0.05
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
    square_pull_face_min: float = 0.45       # activate square pull when fh/frame_h > this; 0..1
    square_pull_weight: float = 0.25         # strength of square pull
    tight_face_relax_thresh: float = 0.48    # if face_h / crop_h ≥ thresh, relax bottom
    tight_face_relax_scale: float = 0.5      # scale want_bottom by this when tight
    device: str = "cuda"            # cuda | cpu
    yolo_model: str = "yolov8n.pt"
    face_model: str = "scrfd_10g_bnkps"
    save_annot: bool = False
    preview_every: int = 120
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
    # Final safety guards (post-trim)
    side_guard_drop_enable: bool = True          # drop frames that still violate side margin after all steps
    side_guard_drop_factor: float = 0.66         # require at least this * desired margin on both sides before saving
    face_anchor_down_frac: float = 1.1           # shift center downward by this * face_h (torso bias)
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
    overlay_scores: bool = True
    overlay_face_fd: bool = True
    lock_momentum: float = 0.7
    suppress_negatives: bool = False
    neg_tolerance: float = 0.35
    max_negatives: int = 5         # emit preview every N processed frames
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
        return len(spans)

    def _prescan(self, cap, fps, total_frames, face: "FaceEmbedder", ref_feat, cfg):
        """
        Fast pass to find keep-spans. Now:
          - obeys play/pause/seek/step/speed from _cmd_q
          - grows ref face bank on confident matches (dedup + cap)
        Returns: (spans, updated_ref_feat)
        """
        import numpy as _np
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
        pos0 = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
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
            preview_step = max(stride, int(getattr(cfg, "preview_every", 120)))
            last_preview_idx = -preview_step
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
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
                            try:
                                last_seek = int(arg)
                            except Exception:
                                pass
                        elif cmd == "step":
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
                        max_grabs=int(getattr(cfg, "seek_max_grabs", 45)),
                    )
                    i = (new_pos // stride) * stride
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
                    i = self._seek_to(
                        cap,
                        i,
                        next_i,
                        fast=bool(getattr(cfg, "seek_fast", True)),
                        max_grabs=int(getattr(cfg, "seek_max_grabs", 45)),
                    )
                    continue
                # Preempt before IO to honor newly queued seeks/steps
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
                        max_grabs=int(getattr(cfg, "seek_max_grabs", 45)),
                    )
                    i = (new_pos // stride) * stride
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
                if not cap.grab():
                    break
                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    i += 1
                    continue
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
                        self.preview.emit(self._cv_bgr_to_qimage(vis))
                    except Exception:
                        pass
                i += 1
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
                            max_grabs=int(getattr(cfg, "seek_max_grabs", 45)),
                        )
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            j += stride_ref
                            continue
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
                                max_grabs=int(getattr(cfg, "seek_max_grabs", 45)),
                            )
                            ret, frame = cap.read()
                            if not ret or frame is None:
                                j += stride_ref
                                continue
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
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos0)
        try:
            # Restore progress to the position we were at before pre-scan so the UI
            # doesn't jump to EOF and queue a seek there.
            self.progress.emit(max(0, pos0))
        except Exception:
            pass
        self._status(
            f"Pre-scan 100% ({total_frames}/{total_frames}) • segments={len(spans)}",
            key="prescan_progress",
            interval=0.1,
        )
        self._status(
            f"Pre-scan ref bank added {added_vecs} vector(s); size={len(ref_bank_list)} (start={initial_bank_len})",
            key="prescan_bank_summary",
            interval=0.5,
        )
        return spans, (ref_feat_local if ref_feat_local is not None else ref_feat)

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

            pen = _penalty((ex1, ey1, ex2, ey2), face_box)
            total = area_term + float(cfg.crop_penalty_weight) * pen
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
    ) -> int:
        """Keyframe-aware seek. If fast=True, cap forward grabs to max_grabs to avoid long stalls."""
        if self._total_frames is not None:
            tgt_idx = max(0, min(self._total_frames - 1, int(tgt_idx)))
        base = tgt_idx
        ks = self._keyframes
        if ks:
            i = bisect.bisect_right(ks, tgt_idx) - 1
            if i >= 0:
                base = int(ks[i])
        # If we have no KF info or landed exactly on tgt, choose a near target base in fast mode
        if fast and (not ks or base == tgt_idx):
            mg = max_grabs
            if mg <= 0:
                f = float(self._fps or 30.0)
                mg = max(15, min(240, int(round(f))))
            base = max(0, tgt_idx - int(mg))
        if base != cur_idx:
            # time-based reposition is often faster on some backends; fall back if it fails
            if fast and self._fps:
                ok = cap.set(cv2.CAP_PROP_POS_MSEC, (base / float(self._fps)) * 1000.0)
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, base)
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, base)
        idx = base
        limit = max(0, tgt_idx - base)
        # Cap forward grabs in fast mode to keep UI responsive
        if fast:
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
        # Time-budgeted forward grabs to avoid long decode stalls
        t0 = time.perf_counter() if fast else None
        budget = 0.15  # ~150 ms max spent per seek
        for _ in range(limit):
            if not cap.grab():
                break
            idx += 1
            if fast and (time.perf_counter() - t0) > budget:
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
        if not ref_path:
            self._status("Curator: reference image missing.", key="curate", interval=5.0)
            return
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
                ref_image=ref_path,
                device=str(getattr(cfg, "device", "cuda")),
                trt_lib_dir=(getattr(cfg, "trt_lib_dir", "") or None),
                face_model=str(getattr(cfg, "face_model", "scrfd_10g_bnkps")),
                face_det_conf=float(getattr(cfg, "face_det_conf", 0.50)),
                progress=_progress,
            )
            if getattr(curator, "ref_feat", None) is None:
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
        cap = save_q = saver_thread = csv_f = dbg_f = hit_q = None
        try:
            # Apply TRT/ORT env from cfg early
            def _env_set(k, v):
                if v is None:
                    return
                os.environ[k] = str(v)

            _env_set("PERSON_CAPTURE_TRT_CACHE_ROOT", getattr(self.cfg, "trt_cache_root", "trt_cache"))
            _env_set("PERSON_CAPTURE_TRT_FP16", "1" if getattr(self.cfg, "trt_fp16_enable", True) else "0")
            _env_set("PERSON_CAPTURE_TRT_TIMING_CACHE_ENABLE", "1" if getattr(self.cfg, "trt_timing_cache_enable", True) else "0")
            _env_set("PERSON_CAPTURE_TRT_ENGINE_CACHE_ENABLE", "1" if getattr(self.cfg, "trt_engine_cache_enable", True) else "0")
            _env_set("PERSON_CAPTURE_TRT_BUILDER_OPT_LEVEL", int(getattr(self.cfg, "trt_builder_optimization_level", 5)))
            _env_set("PERSON_CAPTURE_TRT_CUDA_GRAPH_ENABLE", "1" if getattr(self.cfg, "trt_cuda_graph_enable", True) else "0")
            _env_set(
                "PERSON_CAPTURE_TRT_CONTEXT_MEMORY_SHARING_ENABLE",
                "1" if getattr(self.cfg, "trt_context_memory_sharing_enable", True) else "0",
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
            try:
                os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "hwaccel;cuda")
                cap = cv2.VideoCapture(cfg.video, cv2.CAP_FFMPEG)
                try:
                    cap.set(cv2.CAP_PROP_HW_ACCELERATION, 1)
                except Exception:
                    pass
                if not cap.isOpened():
                    cap.release()
                    cap = cv2.VideoCapture(cfg.video)
            except Exception:
                cap = cv2.VideoCapture(cfg.video)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {cfg.video}")
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            except Exception:
                pass
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
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
            if bool(getattr(cfg, "prescan_enable", True)) and total_frames > 0:
                self._status("Pre-scan.", key="phase", interval=2.0)
                keep_spans, ref_face_feat = self._prescan(
                    cap, int(round(fps)), total_frames, face, ref_face_feat, cfg
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
                        max_grabs=int(getattr(cfg, "seek_max_grabs", 45)),
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
                                self.preview.emit(self._cv_bgr_to_qimage(frame))
                        # Do not advance past s0 for processing—restore read head.
                        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                    except Exception:
                        pass
                    self._status(f"Pre-scan segments: {len(keep_spans)}", key="prescan", interval=30.0)
                else:
                    self._status("Pre-scan found no matches; full scan", key="prescan", interval=30.0)
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

                        img_path, img, row = item
                        ok, why = _atomic_jpeg_write(img, img_path, jpg_q)

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

                saver_thread = threading.Thread(target=_saver, name="pc.saver", daemon=True)
                saver_thread.start()

            hit_count = 0
            lock_hits = 0
            locked_face = None
            locked_reid = None
            prev_box = None
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
                            "crop_face_side_margin_frac",
                            "crop_top_headroom_max_frac",
                            "crop_bottom_min_face_heights",
                            "crop_penalty_weight",
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
                        max_grabs=int(getattr(self.cfg, "seek_max_grabs", 45)),
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
                            max_grabs=int(getattr(self.cfg, "seek_max_grabs", 45)),
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
                            max_grabs=int(getattr(self.cfg, "seek_max_grabs", 45)),
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
                H, W = frame.shape[:2]
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                
                # Optional black border crop
                frame_for_det = frame
                off_x, off_y = 0, 0
                if cfg.auto_crop_borders:
                    frame_for_det, (off_x, off_y) = self._autocrop_borders(frame, cfg.border_threshold)
                    H2, W2 = frame_for_det.shape[:2]
                else:
                    H2, W2 = H, W

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
                                (ex1, ey1, ex2, ey2), chosen_ratio, chosen_tloss = self._choose_best_ratio(
                                    face_box_abs, ratios, W2, H2, anchor=(acx, acy), face_box=face_box_abs
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
                                    fx1i = int(round(float(np.clip(fx1 + off_x, 0, W - 1))))
                                    fy1i = int(round(float(np.clip(fy1 + off_y, 0, H - 1))))
                                    fx2i = int(round(float(np.clip(fx2 + off_x, 0, W - 1))))
                                    fy2i = int(round(float(np.clip(fy2 + off_y, 0, H - 1))))
                                    face_box_global = (fx1i, fy1i, fx2i, fy2i)
                                    short_circuit_candidate = dict(
                                        score=fd_val,
                                        fd=fd_val,
                                        rd=None,
                                        sharp=sharp,
                                        box=(ox1, oy1, ox2, oy2),
                                        area=(ox2 - ox1) * (oy2 - oy1),
                                        show_box=face_box_global,
                                        face_box=face_box_global,
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
                                    diag_persons = 0
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
                            face_box=face_box_global,
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
                def save_hit(c, idx):
                    nonlocal hit_count, lock_hits, locked_face, locked_reid, prev_box, ref_face_feat, ref_bank_list
                    crop_img_path = os.path.join(crops_dir, f"f{idx:08d}.jpg")
                    # Start from candidate box in GLOBAL coords
                    gx1, gy1, gx2, gy2 = c["box"]
                    ratio_str = str(
                        c.get("ratio")
                        or (
                            ratios[0]
                            if 'ratios' in locals() and ratios
                            else (self.cfg.ratio.split(',')[0] if self.cfg.ratio else '2:3')
                        )
                    )

                    # If auto-crop borders were applied for detection, run smart-crop inside the ROI
                    use_roi = bool(getattr(cfg, "auto_crop_borders", True)) and (
                        (W2 != W) or (H2 != H) or (off_x != 0) or (off_y != 0)
                    )
                    if use_roi:
                        rx1 = max(0.0, min(float(W2), gx1 - off_x))
                        ry1 = max(0.0, min(float(H2), gy1 - off_y))
                        rx2 = min(float(W2), max(rx1 + 1.0, gx2 - off_x))
                        ry2 = min(float(H2), max(ry1 + 1.0, gy2 - off_y))
                        # Keep the candidate's chosen ratio to avoid re-selecting here.
                        ratio_str = str(c.get("ratio") or ratio_str)
                        face_box_roi = None
                        if c.get("face_box") is not None:
                            fx1, fy1, fx2, fy2 = c["face_box"]
                            face_box_roi = (
                                max(0.0, min(float(W2), fx1 - off_x)),
                                max(0.0, min(float(H2), fy1 - off_y)),
                                max(0.0, min(float(W2), fx2 - off_x)),
                                max(0.0, min(float(H2), fy2 - off_y)),
                            )
                        if bool(getattr(cfg, "smart_crop_enable", True)):
                            rx1, ry1, rx2, ry2 = self._smart_crop_box(
                                frame_for_det, (rx1, ry1, rx2, ry2), face_box_roi, ratio_str, cfg
                            )
                            # nudge center downwards toward torso when a face is present
                            anchor_roi = None
                            if face_box_roi is not None:
                                fcx = 0.5 * (face_box_roi[0] + face_box_roi[2])
                                fcy = 0.5 * (face_box_roi[1] + face_box_roi[3])
                                fh = max(1.0, face_box_roi[3] - face_box_roi[1])
                                anchor_roi = (
                                    fcx,
                                    fcy + 0.5 * float(getattr(cfg, "face_anchor_down_frac", 1.1)) * fh,
                                )
                            rx1, ry1, rx2, ry2 = self._enforce_scale_and_margins(
                                (rx1, ry1, rx2, ry2),
                                ratio_str,
                                W2,
                                H2,
                                face_box=face_box_roi,
                                anchor=anchor_roi,
                            )
                        else:
                            try:
                                tw, th = parse_ratio(ratio_str)
                            except Exception:
                                tw, th = 2.0, 3.0
                            w = rx2 - rx1
                            h = ry2 - ry1
                            target = float(tw) / float(th)
                            cur = w / float(h) if h > 0 else target
                            if abs(cur - target) > 1e-3 and w > 2 and h > 2:
                                if cur < target:
                                    new_h = int(round(w / target))
                                    dy = (h - new_h) // 2
                                    ry1 = max(0, min(H2 - new_h, ry1 + dy))
                                    ry2 = ry1 + new_h
                                else:
                                    new_w = int(round(h * target))
                                    dx = (w - new_w) // 2
                                    rx1 = max(0, min(W2 - new_w, rx1 + dx))
                                    rx2 = rx1 + new_w
                        try:
                            rw, rh = parse_ratio(ratio_str)
                            rx1, ry1, rx2, ry2 = expand_box_to_ratio(
                                rx1, ry1, rx2, ry2, rw, rh, W2, H2, anchor=None, head_bias=0.0
                            )
                        except Exception:
                            pass
                        cx1, cy1, cx2, cy2 = rx1 + off_x, ry1 + off_y, rx2 + off_x, ry2 + off_y
                    else:
                        cx1, cy1, cx2, cy2 = gx1, gy1, gx2, gy2
                        if bool(getattr(cfg, "smart_crop_enable", True)):
                            cx1, cy1, cx2, cy2 = self._smart_crop_box(
                                frame, (cx1, cy1, cx2, cy2), c.get("face_box"), ratio_str, cfg
                            )
                            H_, W_ = frame.shape[:2]
                            anchor_final = None
                            fb = c.get("face_box")
                            if fb is not None:
                                fcx = 0.5 * (fb[0] + fb[2])
                                fcy = 0.5 * (fb[1] + fb[3])
                                fh = max(1.0, fb[3] - fb[1])
                                anchor_final = (
                                    fcx,
                                    fcy + 0.5 * float(getattr(cfg, "face_anchor_down_frac", 1.1)) * fh,
                                )
                            cx1, cy1, cx2, cy2 = self._enforce_scale_and_margins(
                                (cx1, cy1, cx2, cy2),
                                ratio_str,
                                W_,
                                H_,
                                face_box=c.get("face_box"),
                                anchor=anchor_final,
                            )
                        else:
                            try:
                                tw, th = parse_ratio(ratio_str)
                            except Exception:
                                tw, th = 2.0, 3.0
                            w = cx2 - cx1
                            h = cy2 - cy1
                            target = float(tw) / float(th)
                            cur = w / float(h) if h > 0 else target
                            if abs(cur - target) > 1e-3 and w > 2 and h > 2:
                                if cur < target:
                                    new_h = int(round(w / target))
                                    dy = (h - new_h) // 2
                                    cy1 += dy
                                    cy2 = cy1 + new_h
                                else:
                                    new_w = int(round(h * target))
                                    dx = (w - new_w) // 2
                                    cx1 += dx
                                    cx2 = cx1 + new_w
                        try:
                            rw, rh = parse_ratio(ratio_str)
                            cx1, cy1, cx2, cy2 = expand_box_to_ratio(
                                cx1, cy1, cx2, cy2, rw, rh, W, H, anchor=None, head_bias=0.0
                            )
                        except Exception:
                            pass

                    # Final black-border trim on the saved crop, then re-expand to exact ratio
                    try:
                        try:
                            from .utils import detect_black_borders  # type: ignore
                        except Exception:
                            from utils import detect_black_borders  # type: ignore
                        sub = frame[int(round(cy1)):int(round(cy2)), int(round(cx1)):int(round(cx2))]
                        if sub.size > 0:
                            scan_frac = max(0.0, float(getattr(cfg, "border_scan_frac", 0.25)))
                            # full depth on the final crop to guarantee complete removal
                            max_scan = min(sub.shape[0], sub.shape[1])
                            if scan_frac == 0.0:
                                l, t, r, b = 0, 0, sub.shape[1], sub.shape[0]
                            else:
                                l, t, r, b = detect_black_borders(
                                    sub,
                                    thr=int(getattr(cfg, "border_threshold", 22)),
                                    max_scan=max_scan,
                                )
                            if (l > 0) or (t > 0) or (r < sub.shape[1]) or (b < sub.shape[0]):
                                nx1, ny1 = int(round(cx1)) + int(l), int(round(cy1)) + int(t)
                                nx2, ny2 = int(round(cx1)) + int(r), int(round(cy1)) + int(b)
                                try:
                                    tw, th = parse_ratio(ratio_str)
                                except Exception:
                                    tw, th = 2, 3
                                # anchor-aware shrink INSIDE trimmed ROI to exact ratio
                                anchor_glob = None
                                fb = c.get("face_box")
                                if fb is not None:
                                    fcx = 0.5 * (fb[0] + fb[2])
                                    fcy = 0.5 * (fb[1] + fb[3])
                                    fh  = max(1.0, fb[3] - fb[1])
                                    anchor_glob = (
                                        fcx,
                                        fcy + 0.5 * float(getattr(cfg, "face_anchor_down_frac", 1.1)) * fh,
                                    )
                                cx1, cy1, cx2, cy2 = self._shrink_to_ratio_inside(
                                    (nx1, ny1, nx2, ny2), int(tw), int(th),
                                    (nx1, ny1, nx2, ny2), anchor=anchor_glob
                                )
                                # Final guard: ensure the shrunken crop still honors face side margins.
                                try:
                                    cx1, cy1, cx2, cy2 = self._enforce_scale_and_margins(
                                        (cx1, cy1, cx2, cy2),
                                        ratio_str,
                                        W,
                                        H,
                                        face_box=fb,
                                        anchor=anchor_glob,
                                    )
                                except Exception:
                                    pass
                    except Exception:
                        pass

                    cx1 = max(0, min(W - 1, int(round(cx1))))
                    cy1 = max(0, min(H - 1, int(round(cy1))))
                    cx2 = max(cx1 + 1, min(W, int(round(cx2))))
                    cy2 = max(cy1 + 1, min(H, int(round(cy2))))

                    if bool(getattr(cfg, "auto_crop_borders", False)):
                        bx1 = int(locals().get("off_x", 0))
                        by1 = int(locals().get("off_y", 0))
                        bx2 = int(bx1 + locals().get("W2", W))
                        by2 = int(by1 + locals().get("H2", H))
                    else:
                        bx1, by1, bx2, by2 = 0, 0, W, H

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
                            cx1 = max(bx1, min(bx2 - target_w, cx1 + (w - target_w) // 2))
                            cx2 = cx1 + target_w
                        # Height correction stays inside the same content window
                        target_h = max(1, int(round((cx2 - cx1) * float(rh) / float(rw))))
                        if abs((cy2 - cy1) - target_h) > 1:
                            cy1 = max(by1, min(by2 - target_h, cy1 + ((cy2 - cy1) - target_h) // 2))
                            cy2 = cy1 + target_h
                    except Exception:
                        pass

                    # Edge-aware side-margin safety: avoid saving half-face crops pinned to an edge.
                    fb = c.get("face_box")
                    if fb is not None and bool(getattr(cfg, "side_guard_drop_enable", True)):
                        fw = max(1.0, float(fb[2]) - float(fb[0]))
                        # Use face *edges* vs crop edges
                        left_margin  = max(0.0, float(fb[0]) - float(cx1))
                        right_margin = max(0.0, float(cx2) - float(fb[2]))
                        desired = float(cfg.crop_face_side_margin_frac) * fw
                        required = float(getattr(cfg, "side_guard_drop_factor", 0.66)) * desired

                        fd_val = float(c.get("fd")) if c.get("fd") is not None else 9.0
                        reasons = set(c.get("reasons", []))
                        is_rescue = ("face_short_circuit" in reasons) or ("global_face" in reasons)
                        relax_fd = float(getattr(cfg, "side_guard_relax_fd", 0.22))
                        relax_factor = float(getattr(cfg, "side_guard_relax_factor", 0.50))
                        if (fd_val <= relax_fd) or is_rescue:
                            required *= relax_factor

                        inner_px = float(getattr(cfg, "face_edge_inner_px", 1.0))
                        if (fd_val <= relax_fd) or is_rescue:
                            inner_px *= float(getattr(cfg, "face_edge_inner_relax", 0.25))

                        self._status(
                            f"side_guard L={left_margin:.1f} R={right_margin:.1f} "
                            f"req={required:.1f} fw={fw:.1f} fd={fd_val:.3f} "
                            f"inner={inner_px:.2f}",
                            key="side_guard_dbg",
                            interval=0.8,
                        )
                        # Try a minimal salvage shift before drop
                        if min(left_margin, right_margin) < required:
                            reqL = max(required, inner_px)
                            reqR = max(required, inner_px)
                            needL = max(0.0, reqL - left_margin)
                            needR = max(0.0, reqR - right_margin)
                            if (needL > 0.0) or (needR > 0.0):
                                width = cx2 - cx1
                                # positive shift → move right; negative → left
                                shift = needR - needL
                                nx1 = max(bx1, min(bx2 - width, int(round(cx1 + shift))))
                                nx2 = nx1 + width
                                # recompute
                                left_margin  = max(0.0, float(fb[0]) - float(nx1))
                                right_margin = max(0.0, float(nx2) - float(fb[2]))
                                if min(left_margin, right_margin) >= required:
                                    cx1, cx2 = nx1, nx2
                                    self._status(
                                        f"side_guard after-shift L={left_margin:.1f} R={right_margin:.1f}",
                                        key="side_guard_dbg2",
                                        interval=0.8,
                                    )
                        # If we still can't meet margins, try a tiny ratio-preserving shrink around the face.
                        if (
                            (min(left_margin, right_margin) < required)
                            and ((fd_val <= relax_fd) or is_rescue)
                            and bool(getattr(cfg, "side_guard_allow_shrink", True))
                        ):
                            deficit = max(0.0, required - min(left_margin, right_margin))
                            max_shrink_px = float(getattr(cfg, "side_guard_max_shrink_px", 32))
                            min_shrink_frac = float(getattr(cfg, "side_guard_min_shrink_frac", 0.02))
                            shrink_px = min(
                                max_shrink_px,
                                max(2.0, max(deficit * 2.0, (cx2 - cx1) * min_shrink_frac)),
                            )

                            old_w = (cx2 - cx1)
                            new_w = max(2, int(round(old_w - shrink_px)))
                            if new_w >= old_w:
                                new_w = max(2, old_w - 2)
                            new_h = max(2, int(round(new_w * float(rh) / float(rw))))

                            face_cx = 0.5 * (float(fb[0]) + float(fb[2]))
                            cx1_try = int(round(face_cx - new_w * 0.5))
                            cx1_try = max(bx1, min(bx2 - new_w, cx1_try))
                            cx2_try = cx1_try + new_w

                            cy_mid = int(round((cy1 + cy2) * 0.5))
                            cy1_try = max(by1, min(by2 - new_h, cy_mid - new_h // 2))
                            cy2_try = cy1_try + new_h

                            left_margin_try = max(0.0, float(fb[0]) - float(cx1_try))
                            right_margin_try = max(0.0, float(cx2_try) - float(fb[2]))

                            if min(left_margin_try, right_margin_try) >= required:
                                cx1, cx2, cy1, cy2 = cx1_try, cx2_try, cy1_try, cy2_try
                                left_margin, right_margin = left_margin_try, right_margin_try
                                self._status(
                                    f"side_guard after-shrink L={left_margin:.1f} R={right_margin:.1f}",
                                    key="side_guard_dbg2",
                                    interval=0.8,
                                )
                        # final guard (also catches actual face cuts)
                        edge_cut = (float(fb[0]) < float(cx1) + inner_px) or (float(fb[2]) > float(cx2) - inner_px)
                        if min(left_margin, right_margin) < required or edge_cut:
                            drop_reasons = ["side_guard_drop", *list(c.get("reasons", []))]
                            self._status(
                                f"reject_reasons={drop_reasons[:6]}",
                                key="rej_reasons",
                                interval=1.0,
                            )
                            self._status(
                                "skip: side_guard_drop (insufficient face margin)",
                                key="cap",
                                interval=1.5,
                            )
                            return False

                    # Final clamp inside de-barred content window (prevents 1px bar re-entry)
                    cx1 = max(bx1, min(bx2 - 1, cx1))
                    cy1 = max(by1, min(by2 - 1, cy1))
                    cx2 = max(cx1 + 1, min(bx2, cx2))
                    cy2 = max(cy1 + 1, min(by2, cy2))
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

                    crop_img2 = frame[cy1:cy2, cx1:cx2]
                    row = [
                        idx,
                        idx / float(fps),
                        c.get("score"),
                        c.get("fd"),
                        c.get("rd"),
                        cx1,
                        cy1,
                        cx2,
                        cy2,
                        crop_img_path,
                        c.get("sharp"),
                        str(ratio_str),
                    ]
                    if save_q is not None:
                        try:
                            # enqueue a contiguous copy; slices are views into `frame`
                            buf = np.ascontiguousarray(crop_img2)
                            if not buf.flags.owndata:
                                buf = buf.copy()
                            save_q.put_nowait((crop_img_path, buf, row))
                        except queue.Full:
                            pass  # drop if saver is behind
                    else:
                        ok, why = _atomic_jpeg_write(crop_img2, crop_img_path, jpg_q)
                        if ok:
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
                            # skip this hit, keep processing
                            pass
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
                                (ex1, ey1, ex2, ey2), chosen_ratio, chosen_tloss = self._choose_best_ratio(
                                    face_box_abs, ratios, W2, H2, anchor=(acx, acy), face_box=face_box_abs
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
                                        show_box=(sfx1, sfy1, sfx2, sfy2),
                                        face_box=(sfx1, sfy1, sfx2, sfy2),
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
                        if save_hit(chosen, current_idx):
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
                        bvals = np.asarray(candidates[0]["box"], dtype=float)
                        if np.isfinite(bvals).all():
                            bx1, by1, bx2, by2 = [int(round(v)) for v in bvals]
                            bx1 = max(0, min(W - 1, bx1))
                            by1 = max(0, min(H - 1, by1))
                            bx2 = max(0, min(W - 1, bx2))
                            by2 = max(0, min(H - 1, by2))
                            if bx2 > bx1 and by2 > by1:
                                cv2.rectangle(show, (bx1, by1), (bx2, by2), (255,0,0), 2)
                                rs = candidates[0].get("reasons")
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
                    qimg = self._cv_bgr_to_qimage(show)
                    self.preview.emit(qimg)

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
                    qimg = self._cv_bgr_to_qimage(base)
                    self.preview.emit(qimg)

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
                self.finished.emit(False, "Aborted")
            else:
                self.finished.emit(True, f"Done. Hits: {hit_count}. Index: {csv_path}")
        except Exception as e:
            err = f"Error: {e}\n{traceback.format_exc()}"
            self.finished.emit(False, err)
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
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
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
        self._curator_fallback: Optional[Processor] = None
        self._fps: Optional[float] = None
        self._total_frames: Optional[int] = None
        self._keyframes: List[int] = []
        self._current_idx: int = 0

        self.cfg = SessionConfig()
        self._updating_refs = False
        self._build_ui()
        self._load_qsettings()
        self.statusbar = self.statusBar()
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

        # Params
        param_group = QtWidgets.QGroupBox("Parameters")
        grid = QtWidgets.QGridLayout(param_group)

        self.ratio_edit = QtWidgets.QLineEdit("2:3")
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
        self.preview_every_spin = QtWidgets.QSpinBox(); self.preview_every_spin.setRange(0, 5000); self.preview_every_spin.setValue(120)

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

        def _mk_fspin(minv: float, maxv: float, step: float, decimals: int, tooltip: str) -> QtWidgets.QDoubleSpinBox:
            sb = QtWidgets.QDoubleSpinBox()
            sb.setRange(minv, maxv)
            sb.setSingleStep(step)
            sb.setDecimals(decimals)
            sb.setToolTip(tooltip)
            sb.setKeyboardTracking(False)
            return sb

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

        labels = [
            ("Aspect ratio W:H", self.ratio_edit),
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
        prev_group = QtWidgets.QGroupBox("Live preview")
        prev_layout = QtWidgets.QVBoxLayout(prev_group)
        self.preview_label = QtWidgets.QLabel()
        self.preview_label.setAlignment(QtCore.Qt.AlignCenter)
        self.preview_label.setMinimumHeight(1)
        prev_layout.addWidget(self.preview_label)
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
            "crop_face_side_margin_frac",
            "crop_top_headroom_max_frac",
            "crop_bottom_min_face_heights",
            "crop_penalty_weight",
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
            "rot_adaptive",
            "rot_every_n",
            "rot_after_hit_frames",
            "fast_no_face_imgsz",
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
            ratio=self.ratio_edit.text().strip() or "2:3",
            seek_fast=bool(getattr(self.cfg, "seek_fast", True)),
            seek_max_grabs=max(0, int(getattr(self.cfg, "seek_max_grabs", 45))),
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
            preview_every=int(self.preview_every_spin.value()),
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
        )
        return cfg

    def _apply_cfg(self, cfg: SessionConfig):
        self.cfg.seek_fast = bool(getattr(cfg, "seek_fast", True))
        try:
            self.cfg.seek_max_grabs = int(getattr(cfg, "seek_max_grabs", 45))
        except Exception:
            self.cfg.seek_max_grabs = 45
        self.video_edit.setText(cfg.video)
        paths = [part.strip() for part in (cfg.ref or "").split(';') if part.strip()]
        self._set_ref_paths(paths)
        self.out_edit.setText(cfg.out_dir)
        self.ratio_edit.setText(cfg.ratio)
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

    # Thread control
    def on_start(self):
        if self._thread is not None:
            QtWidgets.QMessageBox.warning(self, "Busy", "Processing already running")
            return

        cfg = self._collect_cfg()
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

        # Wire signals
        self._thread.started.connect(self._worker.run)
        self._worker.setup.connect(self._on_setup)
        self._worker.progress.connect(self._on_progress)
        self._worker.status.connect(self._on_status)
        self._worker.preview.connect(self._on_preview)
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

    def _on_preview(self, img: QtGui.QImage):
        pix = QtGui.QPixmap.fromImage(img)
        self.preview_label.setPixmap(pix.scaled(self.preview_label.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation))

    def _on_hit(self, crop_path: str):
        def _set(img: QtGui.QImage) -> bool:
            if img.isNull():
                return False
            self.hit_label.setPixmap(
                QtGui.QPixmap.fromImage(img).scaled(
                    self.hit_label.size(),
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation,
                )
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
        self.video_edit.setText(s.value("video", ""))
        refs = s.value("ref", "", type=str)
        paths = self._filter_images([part.strip() for part in refs.split(';') if part.strip()])
        self._set_ref_paths(paths)
        self.out_edit.setText(s.value("out_dir", "output"))
        self.ratio_edit.setText(s.value("ratio", "2:3"))
        self.cfg.seek_fast = s.value("seek_fast", True, type=bool)
        try:
            self.cfg.seek_max_grabs = int(s.value("seek_max_grabs", 45))
        except Exception:
            self.cfg.seek_max_grabs = 45
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
        self.preview_every_spin.setValue(int(s.value("preview_every", 120)))
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
        if hasattr(self, 'spin_prescan_fd_add'):
            s.setValue("prescan_fd_add", float(self.spin_prescan_fd_add.value()))
        if hasattr(self, 'spin_prescan_exit_cooldown'):
            s.setValue(
                "prescan_exit_cooldown_sec",
                float(self.spin_prescan_exit_cooldown.value()),
            )
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
    import os, logging
    os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
    try:
        import huggingface_hub
        logging.getLogger('huggingface_hub').setLevel(logging.ERROR)
        logging.getLogger('huggingface_hub.file_download').setLevel(logging.ERROR)
    except Exception:
        pass
    QtCore.QCoreApplication.setOrganizationName(APP_ORG)
    QtCore.QCoreApplication.setApplicationName(APP_NAME)
    app = QtWidgets.QApplication(sys.argv)
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
