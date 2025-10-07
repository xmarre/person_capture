#!/usr/bin/env python3
"""
GUI for PersonCapture: target-person finder and 2:3 crops from video.

Requirements:
  pip install PySide6
  # plus the project's requirements (torch, ultralytics, open-clip-torch, opencv-python, etc.)

Run:
  python gui_app.py
"""

import sys, os, json, time, csv, traceback
import queue
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List
from pathlib import Path

# Robust imports: support both package ("from .module") and flat files ("import module").
def _imp():
    try:
        from .detectors import PersonDetector  # type: ignore
        from .face_embedder import FaceEmbedder  # type: ignore
        from .reid_embedder import ReIDEmbedder  # type: ignore
        from .utils import ensure_dir, parse_ratio, expand_box_to_ratio
        return PersonDetector, FaceEmbedder, ReIDEmbedder, ensure_dir, parse_ratio, expand_box_to_ratio
    except Exception:
        # Add script dir to sys.path and try again as flat modules
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.insert(0, here)
        from detectors import PersonDetector  # type: ignore
        from face_embedder import FaceEmbedder  # type: ignore
        from reid_embedder import ReIDEmbedder  # type: ignore
        from utils import ensure_dir, parse_ratio, expand_box_to_ratio
        return PersonDetector, FaceEmbedder, ReIDEmbedder, ensure_dir, parse_ratio, expand_box_to_ratio

PersonDetector, FaceEmbedder, ReIDEmbedder, ensure_dir, parse_ratio, expand_box_to_ratio = _imp()

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtWidgets import QDockWidget

APP_ORG = "PersonCapture"
APP_NAME = "PersonCapture GUI"

# ---------------------- Data & Settings ----------------------

@dataclass
class SessionConfig:
    video: str = ""
    ref: str = ""
    out_dir: str = "output"
    ratio: str = "2:3"
    frame_stride: int = 2
    min_det_conf: float = 0.35
    face_thresh: float = 0.36
    reid_thresh: float = 0.42
    combine: str = "min"            # min | avg | face_priority
    match_mode: str = "both"        # either | both | face_only | reid_only
    only_best: bool = True
    min_sharpness: float = 160.0
    min_gap_sec: float = 1.5
    min_box_pixels: int = 8000
    auto_crop_borders: bool = False
    border_threshold: int = 10
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
    use_arcface: bool = False
    device: str = "cuda"            # cuda | cpu
    yolo_model: str = "yolov8n.pt"
    face_model: str = "yolov8n-face.pt"
    save_annot: bool = False
    preview_every: int = 30
    prefer_face_when_available: bool = True
    face_quality_min: float = 120.0
    face_visible_uses_quality: bool = True      # if False, any detected face counts as "visible"
    face_det_conf: float = 0.22                 # YOLOv8-face confidence
    face_det_pad: float = 0.08                  # expand person box before face detect (fraction of w/h)
    face_margin_min: float = 0.05
    require_face_if_visible: bool = True
    drop_reid_if_any_face_match: bool = True
    # Debug/diagnostics
    debug_dump: bool = True
    debug_dir: str = "debug"
    overlay_scores: bool = True
    overlay_face_fd: bool = True
    lock_momentum: float = 0.7
    suppress_negatives: bool = False
    neg_tolerance: float = 0.35
    max_negatives: int = 5         # emit preview every N processed frames

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(s: str) -> "SessionConfig":
        d = json.loads(s)
        c = SessionConfig()
        for k, v in d.items():
            if hasattr(c, k):
                setattr(c, k, v)
        return c

# ---------------------- Worker Thread ----------------------

class Processor(QtCore.QObject):
    def _ema(self, prev, new, m):
        if prev is None:
            return new
        return (m * prev + (1.0 - m) * new) if isinstance(prev, np.ndarray) else new

    def _choose_best_ratio(self, det_box, ratios, frame_w, frame_h, anchor=None):
        # Return expanded box and chosen ratio string that minimally enlarges the det box
        (x1,y1,x2,y2) = det_box
        det_area = max(1, (x2-x1)*(y2-y1))
        best = None
        best_ratio = None
        best_factor = 1e9
        for rs in ratios:
            try:
                rw, rh = parse_ratio(rs)
            except Exception:
                continue
            ex1,ey1,ex2,ey2 = expand_box_to_ratio(x1,y1,x2,y2, rw, rh, frame_w, frame_h, anchor=anchor, head_bias=0.12)
            area = max(1, (ex2-ex1)*(ey2-ey1))
            factor = float(area)/float(det_area)
            if factor < best_factor:
                best_factor = factor
                best = (ex1,ey1,ex2,ey2)
                best_ratio = rs
        return best, (best_ratio or (f"{int(rw)}:{int(rh)}" if 'rw' in locals() else "unk"))

    def _calc_sharpness(self, bgr):
        if bgr is None or bgr.size == 0:
            return 0.0
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(g, cv2.CV_64F).var())

    def _autocrop_borders(self, frame, thr):
        from utils import detect_black_borders
        x1,y1,x2,y2 = detect_black_borders(frame, thr=int(thr))
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
        from utils import expand_box_to_ratio
        x1,y1,x2,y2 = box
        ex1,ey1,ex2,ey2 = expand_box_to_ratio(x1,y1,x2,y2, ratio_w, ratio_h, frame_w, frame_h, anchor=anchor, head_bias=0.12)
        return ex1,ey1,ex2,ey2

    # Signals for UI
    setup = QtCore.Signal(int, float)                # total_frames, fps
    progress = QtCore.Signal(int)                    # current frame idx
    status = QtCore.Signal(str)                      # status text
    preview = QtCore.Signal(QtGui.QImage)            # annotated frame
    hit = QtCore.Signal(str)                         # crop path
    finished = QtCore.Signal(bool, str)              # success, message

    # --- Player control slots (queued, thread-safe) ---
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

    @QtCore.Slot()
    def request_abort(self):
        self._abort = True

    @QtCore.Slot(bool)
    def request_pause(self, p: bool):
        self._pause = bool(p)
        self._paused = bool(p)

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
        try:
            self._paused = False
            self._init_status()
            cfg = self.cfg
            if not os.path.isfile(cfg.video):
                raise FileNotFoundError(f"Video not found: {cfg.video}")
            if not os.path.isfile(cfg.ref):
                raise FileNotFoundError(f"Reference image not found: {cfg.ref}")

            ensure_dir(cfg.out_dir)
            crops_dir = os.path.join(cfg.out_dir, "crops")
            ann_dir = os.path.join(cfg.out_dir, "annot") if cfg.save_annot else None
            ensure_dir(crops_dir)
            if ann_dir:
                ensure_dir(ann_dir)
            # Debug I/O
            dbg_f = None
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
            )
            reid = ReIDEmbedder(device=cfg.device, model_name=cfg.reid_backbone, pretrained=cfg.reid_pretrained, progress=self.status.emit)

            # Reference
            self._status("Preparing reference features...", key="phase", interval=5.0)
            ref_img = cv2.imread(cfg.ref, cv2.IMREAD_COLOR)
            if ref_img is None:
                raise RuntimeError("Cannot read reference image")

            ref_faces = face.extract(ref_img)
            ref_face = FaceEmbedder.best_face(ref_faces)
            ref_face_feat = ref_face['feat'] if ref_face else None
            if ref_face_feat is not None:
                ref_norm = float(np.linalg.norm(ref_face_feat))
                self._status(
                    f"Ref face: ok | ||ref||={ref_norm:.3f} backend={getattr(face, 'backend', None)}",
                    key="ref",
                    interval=60.0,
                )
            else:
                self._status(
                    "Ref face: missing",
                    key="ref",
                    interval=60.0,
                )

            ref_persons = det.detect(ref_img, conf=0.1)
            if ref_persons:
                ref_persons.sort(key=lambda d: (d['xyxy'][2]-d['xyxy'][0])*(d['xyxy'][3]-d['xyxy'][1]), reverse=True)
                rx1, ry1, rx2, ry2 = [int(v) for v in ref_persons[0]['xyxy']]
                ref_crop = ref_img[ry1:ry2, rx1:rx2]
                ref_reid_feat = reid.extract([ref_crop])[0]
            else:
                ref_reid_feat = reid.extract([ref_img])[0]
            self._status(
                f"Ref ReID: {'ok' if ref_reid_feat is not None else 'missing'}",
                key="ref_reid",
                interval=60.0,
            )

            # Video
            cap = cv2.VideoCapture(cfg.video)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {cfg.video}")
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            self._fps = float(fps)
            self._total_frames = int(total_frames)
            self.setup.emit(total_frames, float(fps))

            ratios = [r.strip() for r in str(cfg.ratio).split(',') if r.strip()]
            if not ratios:
                ratios = ["2:3"]

            # CSV
            csv_path = os.path.join(cfg.out_dir, "index.csv")
            csv_f = open(csv_path, "w", newline="")
            writer = csv.writer(csv_f)
            writer.writerow(["frame","time_secs","score","face_dist","reid_dist","x1","y1","x2","y2","crop_path","sharpness","ratio"])

            hit_count = 0
            lock_hits = 0
            locked_face = None
            locked_reid = None
            prev_box = None
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
            last_preview_emit = -999999

            self._status("Processing...", key="phase", interval=5.0)
            while True:
                if self._abort:
                    self._status("Aborting...", key="phase")
                    break
                force_process = False
                try:
                    while True:
                        cmd, arg = self._cmd_q.get_nowait()
                        if cmd == "seek":
                            tgt = int(arg)
                            if self._total_frames is not None:
                                tgt = max(0, min(self._total_frames - 1, tgt))
                            cap.set(cv2.CAP_PROP_POS_FRAMES, tgt)
                            frame_idx = tgt
                            self.progress.emit(frame_idx)
                            force_process = True
                        elif cmd == "pause":
                            self._paused = True
                        elif cmd == "play":
                            self._paused = False
                        elif cmd == "step":
                            stepn = int(arg) if arg is not None else 1
                            tgt = frame_idx + stepn
                            if self._total_frames is not None:
                                tgt = max(0, min(self._total_frames - 1, tgt))
                            cap.set(cv2.CAP_PROP_POS_FRAMES, tgt)
                            frame_idx = tgt
                            self.progress.emit(frame_idx)
                            self._paused = True
                            force_process = True
                        elif cmd == "speed":
                            try:
                                self._speed = max(0.1, min(4.0, float(arg)))
                            except Exception:
                                pass
                except queue.Empty:
                    pass

                if self._pause:
                    self._paused = True
                if self._paused and not force_process:
                    time.sleep(0.01)
                    continue

                ret = cap.grab()
                if not ret:
                    break
                current_idx = frame_idx
                if not force_process and current_idx % max(1, int(cfg.frame_stride)) != 0:
                    frame_idx = current_idx + 1
                    self.progress.emit(current_idx)
                    continue
                ret, frame = cap.retrieve()
                if not ret:
                    break
                H, W = frame.shape[:2]

                
                # Optional black border crop
                frame_for_det = frame
                off_x, off_y = 0, 0
                if cfg.auto_crop_borders:
                    frame_for_det, (off_x, off_y) = self._autocrop_borders(frame, cfg.border_threshold)
                    H2, W2 = frame_for_det.shape[:2]
                else:
                    H2, W2 = H, W

                persons = det.detect(frame_for_det, conf=float(cfg.min_det_conf))
                if not persons and cfg.auto_crop_borders:
                    # Fallback to full frame if border-cropped frame yields nothing
                    persons = det.detect(frame, conf=float(cfg.min_det_conf))
                    frame_for_det = frame
                    off_x, off_y = 0, 0
                    H2, W2 = H, W
                    self._status("Border-crop yielded no detections. Fallback to full frame.", key="fallback", interval=2.0)
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

                reid_feats = reid.extract(crops) if crops else []

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
                        if ref_face_feat is not None and f.get("feat") is not None:
                            fd_all = 1.0 - float(np.dot(f["feat"], ref_face_feat))
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
                                key=lambda f: 1.0 - float(np.dot(f["feat"], ref_face_feat)),
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
                    fd_tmp = 1.0 - float(np.dot(bf["feat"], ref_face_feat))
                    face_dists_all.append(fd_tmp)
                    if bf.get("quality", 0.0) >= float(cfg.face_quality_min):
                        face_dists_quality.append(fd_tmp)
                    if fd_tmp <= float(cfg.face_thresh):
                        any_face_match = True
                any_face_match_qual = any((d <= float(cfg.face_thresh)) for d in face_dists_quality)
                best_face_dist = min(face_dists_quality) if face_dists_quality else None
                if min_fd_all is None and face_dists_all:
                    min_fd_all = min(face_dists_all)

                # Evaluate candidates
                for i, feat in enumerate(reid_feats):
                    cand_reason = []
                    rd = None
                    if ref_reid_feat is not None:
                        f = feat / max(np.linalg.norm(feat), 1e-9)
                        r = ref_reid_feat / max(np.linalg.norm(ref_reid_feat), 1e-9)
                        rd = 1.0 - float(np.dot(f, r))
                    if rd is None:
                        cand_reason.append("no_reid_ref_or_feat")
                    else:
                        cand_reason.append(f"rd={rd:.3f} thr={float(cfg.reid_thresh):.3f}")

                    bf = faces_local.get(i, None)
                    fd = None
                    if bf is not None and ref_face_feat is not None:
                        fd = 1.0 - float(np.dot(bf["feat"], ref_face_feat))
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

                    eff_mode = cfg.match_mode
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

                    if (
                        getattr(cfg, "drop_reid_if_any_face_match", True)
                        and (any_face_match_qual if 'any_face_match_qual' in locals() else any_face_match)
                        and not face_ok
                        and accept
                    ):
                        accept = False
                        cand_reason.append("drop_reid_due_to_face_match_present")

                    accept_before_face_policy = accept

                    # Face-first policy: candidate-level strict gate
                    if (
                        cfg.require_face_if_visible
                        and any_face_detected
                        and (ref_face_feat is not None)
                    ):
                        if bf is None:
                            # Allow ReID only when no quality‑passing face elsewhere matches.
                            if not (reid_ok and not (any_face_match_qual if 'any_face_match_qual' in locals() else any_face_match)):
                                accept = False
                                cand_reason.append("gate_no_cand_face")
                        else:
                            if not face_ok:
                                fd_tol = float(cfg.face_thresh) + float(cfg.score_margin)
                                reid_escape = (reid_ok and rd is not None and rd <= max(0.0, float(cfg.reid_thresh) - float(cfg.score_margin)) and (fd is None or fd <= fd_tol))
                                if not reid_escape:
                                    accept = False
                                    cand_reason.append("gate_face_present_no_escape")
                    elif cfg.prefer_face_when_available and any_face_visible and (bf is None):
                        cand_reason.append("soft_pref_face_missing")

                    if not accept:
                        cand_reason.append("reject")
                        continue

                    score = self._combine_scores(fd, rd, mode=cfg.combine)
                    x1,y1,x2,y2 = boxes[i]
                    # map anchor to frame_for_det coords
                    anchor = None
                    if bf is not None:
                        fb = bf["bbox"]
                        acx = x1 + (fb[0]+fb[2])/2.0
                        acy = y1 + (fb[1]+fb[3])/2.0
                        anchor = (acx, acy)

                    # expand to ratio within the DET frame
                    (ex1,ey1,ex2,ey2), chosen_ratio = self._choose_best_ratio((x1,y1,x2,y2), ratios, W2, H2, anchor=anchor)

                    # Compute sharpness on final crop
                    crop_img = frame_for_det[ey1:ey2, ex1:ex2]
                    sharp = self._calc_sharpness(crop_img)
                    if sharp < float(cfg.min_sharpness):
                        continue

                    # Map to original frame coords for annotation
                    ox1, oy1, ox2, oy2 = ex1+off_x, ey1+off_y, ex2+off_x, ey2+off_y
                    area = (ox2-ox1)*(oy2-oy1)
                    candidates.append(
                        dict(
                            score=score,
                            fd=fd,
                            rd=rd,
                            sharp=sharp,
                            box=(ox1, oy1, ox2, oy2),
                            area=area,
                            show_box=(x1 + off_x, y1 + off_y, x2 + off_x, y2 + off_y),
                            face_feat=(bf["feat"] if bf is not None else None),
                            reid_feat=(reid_feats[i] if i < len(reid_feats) else None),
                            ratio=chosen_ratio,
                            reasons=cand_reason,
                            face_quality=(bf.get('quality', 0.0) if bf is not None else None),
                            accept_pre=accept_before_face_policy,
                        )
                    )

                # Choose best and save with cadence + lock + margin + IoU gate
                def save_hit(c, idx):
                    nonlocal hit_count, lock_hits, locked_face, locked_reid, prev_box
                    crop_img_path = os.path.join(crops_dir, f"f{idx:08d}.jpg")
                    cx1,cy1,cx2,cy2 = c["box"]
                     # final ratio enforcement
                    ratio_str = c.get("ratio") or (ratios[0] if 'ratios' in locals() and ratios else (self.cfg.ratio.split(',')[0] if self.cfg.ratio else '2:3'))
                    try:
                        tw, th = parse_ratio(ratio_str)
                    except Exception:
                        tw, th = 2.0, 3.0
                    w = cx2 - cx1; h = cy2 - cy1
                    target = float(tw)/float(th)
                    cur = w/float(h) if h>0 else target
                    if abs(cur - target) > 1e-3 and w>2 and h>2:
                        if cur < target:
                            new_h = int(round(w/target))
                            dy = (h - new_h)//2
                            cy1 += dy; cy2 = cy1 + new_h
                        else:
                            new_w = int(round(h*target))
                            dx = (w - new_w)//2
                            cx1 += dx; cx2 = cx1 + new_w
                    crop_img2 = frame[cy1:cy2, cx1:cx2]
                    cv2.imwrite(crop_img_path, crop_img2)
                    writer.writerow([idx, idx/float(fps), c.get("score"), c.get("fd"), c.get("rd"), cx1, cy1, cx2, cy2, crop_img_path, c.get("sharp"), c.get("ratio", "")])
                    reasons_list = c.get("reasons") or []
                    reasons = "|".join(reasons_list)
                    area = (cx2 - cx1) * (cy2 - cy1)
                    self._status(
                        (
                            f"CAPTURE idx={idx} t={idx/float(fps):.2f} "
                            f"fd={c.get('fd')} rd={c.get('rd')} score={c.get('score')} "
                            f"area={area} reasons={reasons}"
                        ),
                        key="cap",
                    )
                    self.hit.emit(crop_img_path)
                    # update lock source
                    if c.get("face_feat") is not None:
                        locked_face = c["face_feat"]
                    if c.get("reid_feat") is not None:
                        lr_prev = locked_reid
                        locked_reid = self._ema(lr_prev, c["reid_feat"], float(cfg.lock_momentum))
                    lock_hits += 1
                    prev_box = (cx1,cy1,cx2,cy2)
                    return True

                # initialize cadence tracking
                if not hasattr(self, "_last_hit_t"):
                    self._last_hit_t = -1e9

                if not candidates:
                    self._status(
                        f"No match. persons={diag_persons} faces={faces_detected} pass_q={faces_passing_quality} "
                        f"visible={any_face_visible} best_fd_qonly={best_face_dist if best_face_dist is not None else 'n/a'} "
                        f"min_fd_all={min_fd_all if min_fd_all is not None else 'n/a'}",
                        key="no_match",
                        interval=1.0,
                    )
                else:
                    # Lock-aware scoring
                    # face margin check: chosen must be best face by a margin if faces are present
                    if cfg.prefer_face_when_available and any_face_visible:
                        # filter to face-bearing candidates
                        face_cands = [c for c in candidates if c.get('fd') is not None]
                        if len(face_cands) >= 2:
                            face_cands.sort(key=lambda d: d['fd'])
                            if (face_cands[1]['fd'] - face_cands[0]['fd']) < float(cfg.face_margin_min):
                                # ambiguous faces -> drop frame
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
                    use_lock = lock_hits >= int(cfg.lock_after_hits) and (locked_face is not None or locked_reid is not None)
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

                    if chosen is not None and now_t - self._last_hit_t >= float(cfg.min_gap_sec):
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
                    # draw person boxes
                    for c in candidates:
                        x1,y1,x2,y2 = c["show_box"]
                        cv2.rectangle(show, (x1,y1),(x2,y2),(0,255,0),1)
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
                                    (x1, max(0, y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.45,
                                    (0, 255, 0),
                                    1,
                                    cv2.LINE_AA,
                                )
                    if candidates:
                        bx1,by1,bx2,by2 = candidates[0]["box"]
                        cv2.rectangle(show, (bx1,by1),(bx2,by2),(255,0,0),2)
                    qimg = self._cv_bgr_to_qimage(show)
                    self.preview.emit(qimg)
                # Always-on preview cadence
                if preview_due:
                    base = frame.copy()
                    qimg = self._cv_bgr_to_qimage(base)
                    self.preview.emit(qimg)

                # single progress update per loop
                self.progress.emit(current_idx)
                frame_idx = current_idx + 1
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

            cap.release()
            # finalize CSV
            csv_f.close()
            if dbg_f:
                dbg_f.close()

            if self._abort:
                self.finished.emit(False, "Aborted")
            else:
                self.finished.emit(True, f"Done. Hits: {hit_count}. Index: {csv_path}")
        except Exception as e:
            if 'dbg_f' in locals() and dbg_f:
                try:
                    dbg_f.close()
                except Exception:
                    pass
            err = f"Error: {e}\n{traceback.format_exc()}"
            self.finished.emit(False, err)

    def _init_status(self):
        self._last_status_time = 0.0
        self._last_status_text = None

    def _init_status(self):
        # Per-key throttle timestamps and last texts
        self._status_last_time = {}
        self._status_last_text = {}

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
        self.toolbar.setObjectName("Main")
        self.toolbar.setMovable(True)
        self.act_start = QtGui.QAction("Start", self); self.act_start.triggered.connect(self.on_start)
        self.act_pause = QtGui.QAction("Pause", self); self.act_pause.triggered.connect(self.on_pause)
        self.act_stop = QtGui.QAction("Stop", self); self.act_stop.triggered.connect(self.on_stop)
        self.act_compact = QtGui.QAction("Compact", self); self.act_compact.setCheckable(True); self.act_compact.toggled.connect(self.toggle_compact_mode)
        self.act_reset_layout = QtGui.QAction("Reset layout", self); self.act_reset_layout.triggered.connect(self.reset_layout)
        self.toolbar.addActions([self.act_start, self.act_pause, self.act_stop, self.act_compact, self.act_reset_layout])
        self._thread: Optional[QtCore.QThread] = None
        self._worker: Optional[Processor] = None
        self._fps: Optional[float] = None
        self._total_frames: Optional[int] = None

        self.cfg = SessionConfig()
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
        self.out_edit = QtWidgets.QLineEdit("output")

        vid_btn = QtWidgets.QPushButton("Browse...")
        ref_btn = QtWidgets.QPushButton("Browse...")
        out_btn = QtWidgets.QPushButton("Browse...")

        file_layout.addWidget(QtWidgets.QLabel("Video"), 0, 0)
        file_layout.addWidget(self.video_edit, 0, 1)
        file_layout.addWidget(vid_btn, 0, 2)

        file_layout.addWidget(QtWidgets.QLabel("Reference image"), 1, 0)
        file_layout.addWidget(self.ref_edit, 1, 1)
        file_layout.addWidget(ref_btn, 1, 2)

        file_layout.addWidget(QtWidgets.QLabel("Output directory"), 2, 0)
        file_layout.addWidget(self.out_edit, 2, 1)
        file_layout.addWidget(out_btn, 2, 2)

        # Params
        param_group = QtWidgets.QGroupBox("Parameters")
        grid = QtWidgets.QGridLayout(param_group)

        self.ratio_edit = QtWidgets.QLineEdit("2:3")
        self.stride_spin = QtWidgets.QSpinBox(); self.stride_spin.setRange(1, 1000); self.stride_spin.setValue(2)
        self.det_conf_spin = QtWidgets.QDoubleSpinBox(); self.det_conf_spin.setDecimals(3); self.det_conf_spin.setRange(0.0, 1.0); self.det_conf_spin.setSingleStep(0.01); self.det_conf_spin.setValue(0.35)
        self.face_thr_spin = QtWidgets.QDoubleSpinBox(); self.face_thr_spin.setDecimals(3); self.face_thr_spin.setRange(0.0, 2.0); self.face_thr_spin.setSingleStep(0.01); self.face_thr_spin.setValue(0.32)
        self.face_det_conf_spin = QtWidgets.QDoubleSpinBox(); self.face_det_conf_spin.setDecimals(3); self.face_det_conf_spin.setRange(0.0, 1.0); self.face_det_conf_spin.setSingleStep(0.01); self.face_det_conf_spin.setValue(0.22)
        self.face_det_pad_spin = QtWidgets.QDoubleSpinBox(); self.face_det_pad_spin.setDecimals(3); self.face_det_pad_spin.setRange(0.0, 1.0); self.face_det_pad_spin.setSingleStep(0.01); self.face_det_pad_spin.setValue(0.08)
        self.face_quality_spin = QtWidgets.QDoubleSpinBox(); self.face_quality_spin.setRange(0.0, 1000.0); self.face_quality_spin.setSingleStep(1.0); self.face_quality_spin.setValue(120.0)
        self.face_vis_quality_check = QtWidgets.QCheckBox(); self.face_vis_quality_check.setChecked(True)
        self.reid_thr_spin = QtWidgets.QDoubleSpinBox(); self.reid_thr_spin.setDecimals(3); self.reid_thr_spin.setRange(0.0, 2.0); self.reid_thr_spin.setSingleStep(0.01); self.reid_thr_spin.setValue(0.38)
        self.combine_combo = QtWidgets.QComboBox(); self.combine_combo.addItems(["min","avg","face_priority"])
        self.match_mode_combo = QtWidgets.QComboBox(); self.match_mode_combo.addItems(["either","both","face_only","reid_only"])
        self.only_best_check = QtWidgets.QCheckBox("Only best per frame")
        self.only_best_check.setChecked(True)
        self.min_sharp_spin = QtWidgets.QDoubleSpinBox(); self.min_sharp_spin.setRange(0.0, 5000.0); self.min_sharp_spin.setValue(120.0)
        self.min_gap_spin = QtWidgets.QDoubleSpinBox(); self.min_gap_spin.setDecimals(2); self.min_gap_spin.setRange(0.0, 30.0); self.min_gap_spin.setValue(1.5)
        self.min_box_pix_spin = QtWidgets.QSpinBox(); self.min_box_pix_spin.setRange(0, 5000000); self.min_box_pix_spin.setValue(5000)
        self.auto_crop_check = QtWidgets.QCheckBox("Auto‑crop black borders")
        self.auto_crop_check.setChecked(True)
        self.border_thr_spin = QtWidgets.QSpinBox(); self.border_thr_spin.setRange(0, 50); self.border_thr_spin.setValue(10)
        self.require_face_check = QtWidgets.QCheckBox("Require face if visible"); self.require_face_check.setChecked(True)
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
        self.use_arc_check.setChecked(False)
        self.device_combo = QtWidgets.QComboBox(); self.device_combo.addItems(["cuda","cpu"])
        self.yolo_edit = QtWidgets.QLineEdit("yolov8n.pt")
        self.face_yolo_edit = QtWidgets.QLineEdit("yolov8n-face.pt")
        self.annot_check = QtWidgets.QCheckBox("Save annotated frames")
        self.preview_every_spin = QtWidgets.QSpinBox(); self.preview_every_spin.setRange(0, 5000); self.preview_every_spin.setValue(30)

        labels = [
            ("Aspect ratio W:H", self.ratio_edit),
            ("Frame stride", self.stride_spin),
            ("YOLO min conf", self.det_conf_spin),
            ("Face max dist (face_thresh)", self.face_thr_spin),
            ("Face detector conf", self.face_det_conf_spin),
            ("Face detector pad", self.face_det_pad_spin),
            ("Face quality min", self.face_quality_spin),
            ("Face visible uses quality", self.face_vis_quality_check),
            ("ReID max dist (reid_thresh)", self.reid_thr_spin),
            ("Combine", self.combine_combo),
            ("Match mode", self.match_mode_combo),
            ("Only best", self.only_best_check),
            ("Min sharpness", self.min_sharp_spin),
            ("Min seconds between hits", self.min_gap_spin),
            ("Min box area (px)", self.min_box_pix_spin),
            ("Auto‑crop black borders", self.auto_crop_check),
            ("Border threshold", self.border_thr_spin),
            ("Require face if visible", self.require_face_check),
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
            ("", self.annot_check),
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
        for b in (self.start_btn, self.pause_btn, self.resume_btn, self.stop_btn, self.open_out_btn):
            ctrl_layout.addWidget(b)

        # Player
        player_group = QtWidgets.QGroupBox("Player")
        player_layout = QtWidgets.QHBoxLayout(player_group)
        self.play_btn = QtWidgets.QToolButton(); self.play_btn.setText("Play")
        self.pause_btn2 = QtWidgets.QToolButton(); self.pause_btn2.setText("Pause")
        self.step_back_btn = QtWidgets.QToolButton(); self.step_back_btn.setText("⟨⟨")
        self.step_fwd_btn = QtWidgets.QToolButton(); self.step_fwd_btn.setText("⟩⟩")
        self.speed_combo = QtWidgets.QComboBox(); self.speed_combo.addItems(["0.25x", "0.5x", "1.0x", "1.5x", "2.0x"])
        self.speed_combo.setCurrentText("1.0x")
        self.seek_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.seek_slider.setRange(0, 0)
        self.seek_slider.setSingleStep(1)
        self.time_lbl = QtWidgets.QLabel("00:00 / 00:00")
        for w in (self.play_btn, self.pause_btn2, self.step_back_btn, self.step_fwd_btn, self.speed_combo):
            player_layout.addWidget(w)
        player_layout.addWidget(self.seek_slider, 1)
        player_layout.addWidget(self.time_lbl)
        self.play_btn.clicked.connect(self._handle_play_clicked)
        self.pause_btn2.clicked.connect(self._handle_pause_clicked)
        self.step_back_btn.clicked.connect(self._handle_step_back)
        self.step_fwd_btn.clicked.connect(self._handle_step_forward)
        self.seek_slider.sliderReleased.connect(lambda: self._handle_seek_slider(self.seek_slider.value()))
        self.seek_slider.sliderMoved.connect(lambda _v: None)  # keep if you prefer release-only
        self.speed_combo.currentTextChanged.connect(self._on_speed_combo_changed)

        # Progress + status
        prog_layout = QtWidgets.QHBoxLayout()
        self.progress = QtWidgets.QProgressBar()
        self.status_lbl = QtWidgets.QLabel("Idle")
        self.meta_lbl = QtWidgets.QLabel("")  # fps + total frames
        prog_layout.addWidget(self.progress, 3)
        prog_layout.addWidget(self.meta_lbl, 1)
        prog_layout.addWidget(self.status_lbl, 2)

        # Center: preview + last hit
        mid_split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        # Preview
        prev_group = QtWidgets.QGroupBox("Live preview")
        prev_layout = QtWidgets.QVBoxLayout(prev_group)
        self.preview_label = QtWidgets.QLabel()
        self.preview_label.setAlignment(QtCore.Qt.AlignCenter)
        self.preview_label.setMinimumHeight(1)
        prev_layout.addWidget(self.preview_label)
        # Last hit
        hit_group = QtWidgets.QGroupBox("Last saved crop")
        hit_layout = QtWidgets.QVBoxLayout(hit_group)
        self.hit_label = QtWidgets.QLabel()
        self.hit_label.setAlignment(QtCore.Qt.AlignCenter)
        self.hit_label.setMinimumHeight(1)
        hit_layout.addWidget(self.hit_label)
        mid_split.addWidget(prev_group)
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
        controls_layout.addWidget(self.tabs, 1)
        controls_layout.addLayout(ctrl_layout)
        controls_layout.addWidget(player_group)
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
        ref_btn.clicked.connect(lambda: self._pick_file(self.ref_edit, "Images (*.jpg *.jpeg *.png *.bmp *.webp);;All files (*)"))
        out_btn.clicked.connect(lambda: self._pick_dir(self.out_edit))
        self.start_btn.clicked.connect(self.on_start)
        self.stop_btn.clicked.connect(self.on_stop)
        self.pause_btn.clicked.connect(lambda: self._pause(True))
        self.resume_btn.clicked.connect(lambda: self._pause(False))
        self.open_out_btn.clicked.connect(self._open_out_dir)

        self._update_buttons(state="idle")

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

    def _on_speed_combo_changed(self, txt: str):
        if not self._worker:
            return
        try:
            speed = float(txt.replace("x", ""))
        except Exception:
            speed = 1.0
        self._worker.set_speed(speed)

    def _handle_play_clicked(self, _checked: bool = False):
        if self._worker:
            self._worker.play()

    def _handle_pause_clicked(self, _checked: bool = False):
        if self._worker:
            self._worker.pause()

    def _handle_step_back(self, _checked: bool = False):
        if self._worker:
            self._worker.step(-1)

    def _handle_step_forward(self, _checked: bool = False):
        if self._worker:
            self._worker.step(1)

    def _handle_seek_slider(self, value: int):
        if self._worker:
            self._worker.seek_frame(int(value))

    def _pause(self, flag: bool):
        if self._worker:
            self._worker.request_pause(flag)
            self._log(f"{'Paused' if flag else 'Resumed'}")

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
        with open(path, "r", encoding="utf-8") as f:
            cfg = SessionConfig.from_json(f.read())
        self._apply_cfg(cfg)
        self._log(f"Preset loaded: {path}")

    def _about(self):
        QtWidgets.QMessageBox.information(self, "About", "PersonCapture GUI\nControls the video→crops pipeline with live preview, presets, and CSV index.")

    def _collect_cfg(self) -> SessionConfig:
        cfg = SessionConfig(
            video=self.video_edit.text().strip(),
            ref=self.ref_edit.text().strip(),
            out_dir=self.out_edit.text().strip() or "output",
            ratio=self.ratio_edit.text().strip() or "2:3",
            frame_stride=int(self.stride_spin.value()),
            min_det_conf=float(self.det_conf_spin.value()),
            face_thresh=float(self.face_thr_spin.value()),
            reid_thresh=float(self.reid_thr_spin.value()),
            combine=self.combine_combo.currentText(),
            match_mode=self.match_mode_combo.currentText(),
            only_best=bool(self.only_best_check.isChecked()),
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
            face_model=self.face_yolo_edit.text().strip() or "yolov8n-face.pt",
            save_annot=bool(self.annot_check.isChecked()),
            preview_every=int(self.preview_every_spin.value()),
            prefer_face_when_available=bool(self.require_face_check.isChecked()) if hasattr(self, "require_face_check") else True,
            face_quality_min=float(self.face_quality_spin.value()) if hasattr(self, "face_quality_spin") else 120.0,
            face_visible_uses_quality=bool(self.face_vis_quality_check.isChecked()) if hasattr(self, "face_vis_quality_check") else True,
            face_det_conf=float(self.face_det_conf_spin.value()) if hasattr(self, "face_det_conf_spin") else 0.22,
            face_det_pad=float(self.face_det_pad_spin.value()) if hasattr(self, "face_det_pad_spin") else 0.08,
            face_margin_min=float(getattr(self, 'margin_spin', QtWidgets.QDoubleSpinBox()).value()) if hasattr(self, 'margin_spin') else 0.05,
        )
        return cfg

    def _apply_cfg(self, cfg: SessionConfig):
        self.video_edit.setText(cfg.video)
        self.ref_edit.setText(cfg.ref)
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
        self.only_best_check.setChecked(cfg.only_best)
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
        if hasattr(self, 'require_face_check'):
            self.require_face_check.setChecked(cfg.prefer_face_when_available)
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
        self.progress.setRange(0, max(0, total_frames))
        self.meta_lbl.setText(f"fps={fps:.2f} frames={total_frames if total_frames>0 else 'unknown'}")
        if hasattr(self, "seek_slider"):
            self.seek_slider.setRange(0, max(0, total_frames - 1))
            self.seek_slider.setValue(0)
            if hasattr(self, "time_lbl"):
                self.time_lbl.setText(f"{self._fmt_time(0)} / {self._fmt_time(total_frames)}")

    def _on_progress(self, idx: int):
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
        img = QtGui.QImage(crop_path)
        if not img.isNull():
            self.hit_label.setPixmap(QtGui.QPixmap.fromImage(img).scaled(self.hit_label.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation))

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

    def keyPressEvent(self, e: QtGui.QKeyEvent):
        try:
            if e.key() == QtCore.Qt.Key_Space and self._worker:
                if getattr(self._worker, "_paused", False):
                    self._worker.play()
                else:
                    self._worker.pause()
                e.accept()
                return
            if e.key() == QtCore.Qt.Key_Right and self._worker:
                self._worker.step(1)
                e.accept()
                return
            if e.key() == QtCore.Qt.Key_Left and self._worker:
                self._worker.step(-1)
                e.accept()
                return
        except Exception:
            pass
        return super().keyPressEvent(e)

    def _update_buttons(self, state: str):
        running = (state == "running")
        self.start_btn.setEnabled(not running)
        self.pause_btn.setEnabled(running)
        self.resume_btn.setEnabled(running)
        self.stop_btn.setEnabled(running)
        if hasattr(self, "play_btn"):
            self.play_btn.setEnabled(running)
        if hasattr(self, "pause_btn2"):
            self.pause_btn2.setEnabled(running)
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
            s = QtCore.QSettings(APP_ORG, APP_NAME)
            s.setValue("dock_state", self.saveState())
        except Exception:
            pass
        self._save_qsettings()
        try:
            if getattr(self, "_worker", None):
                try:
                    self._worker.request_abort()
                except Exception:
                    pass
            if getattr(self, "_thread", None) and self._thread.isRunning():
                self._thread.quit()
                self._thread.wait(5000)
        except Exception:
            pass
        return super().closeEvent(e)

    def _load_qsettings(self):
        s = QtCore.QSettings(APP_ORG, APP_NAME)
        self.video_edit.setText(s.value("video", ""))
        self.ref_edit.setText(s.value("ref", ""))
        self.out_edit.setText(s.value("out_dir", "output"))
        self.ratio_edit.setText(s.value("ratio", "2:3"))
        self.stride_spin.setValue(int(s.value("frame_stride", 2)))
        self.det_conf_spin.setValue(float(s.value("min_det_conf", 0.35)))
        self.face_thr_spin.setValue(float(s.value("face_thresh", 0.32)))
        self.face_det_conf_spin.setValue(float(s.value("face_det_conf", 0.22)))
        self.face_det_pad_spin.setValue(float(s.value("face_det_pad", 0.08)))
        self.face_quality_spin.setValue(float(s.value("face_quality_min", 120.0)))
        self.face_vis_quality_check.setChecked(bool(s.value("face_visible_uses_quality", True)))
        self.reid_thr_spin.setValue(float(s.value("reid_thresh", 0.38)))
        self.combine_combo.setCurrentText(s.value("combine", "min"))
        self.match_mode_combo.setCurrentText(s.value("match_mode", "either"))
        self.only_best_check.setChecked(bool(s.value("only_best", True)))
        self.min_sharp_spin.setValue(float(s.value("min_sharpness", 120.0)))
        self.min_gap_spin.setValue(float(s.value("min_gap_sec", 1.5)))
        self.min_box_pix_spin.setValue(int(s.value("min_box_pixels", 5000)))
        self.auto_crop_check.setChecked(bool(s.value("auto_crop_borders", False)))
        self.border_thr_spin.setValue(int(s.value("border_threshold", 10)))
        self.require_face_check.setChecked(bool(s.value("require_face_if_visible", True)))
        self.lock_mom_spin.setValue(float(s.value("lock_momentum", 0.7)))
        self.suppress_neg_check.setChecked(bool(s.value("suppress_negatives", False)))
        self.neg_tol_spin.setValue(float(s.value("neg_tolerance", 0.35)))
        self.max_neg_spin.setValue(int(s.value("max_negatives", 5)))
        self.log_every_spin.setValue(float(s.value("log_interval_sec", 1.0)))
        self.lock_after_spin.setValue(int(s.value("lock_after_hits", 1)))
        self.lock_face_spin.setValue(float(s.value("lock_face_thresh", 0.28)))
        self.lock_reid_spin.setValue(float(s.value("lock_reid_thresh", 0.30)))
        self.margin_spin.setValue(float(s.value("score_margin", 0.03)))
        self.iou_gate_spin.setValue(float(s.value("iou_gate", 0.05)))
        self.use_arc_check.setChecked(bool(s.value("use_arcface", True)))
        self.device_combo.setCurrentText(s.value("device", "cuda"))
        self.yolo_edit.setText(s.value("yolo_model", "yolov8n.pt"))
        self.face_yolo_edit.setText(s.value("face_model", "yolov8n-face.pt"))
        self.annot_check.setChecked(bool(s.value("save_annot", False)))
        self.preview_every_spin.setValue(int(s.value("preview_every", 30)))
        # Face-first defaults if controls not present
        if hasattr(self, 'require_face_check'):
            self.require_face_check.setChecked(bool(s.value("prefer_face_when_available", True)))
        # Use existing controls to hold thresholds for convenience
        if hasattr(self, 'face_quality_spin'):
            self.face_quality_spin.setValue(float(s.value("face_quality_min", 120.0)))
        if hasattr(self, 'face_vis_quality_check'):
            self.face_vis_quality_check.setChecked(bool(s.value("face_visible_uses_quality", True)))
        if hasattr(self, 'face_det_conf_spin'):
            self.face_det_conf_spin.setValue(float(s.value("face_det_conf", 0.22)))
        if hasattr(self, 'face_det_pad_spin'):
            self.face_det_pad_spin.setValue(float(s.value("face_det_pad", 0.08)))
        if hasattr(self, 'margin_spin'):
            self.margin_spin.setValue(float(s.value("face_margin_min", 0.05)))

    def _save_qsettings(self):
        s = QtCore.QSettings(APP_ORG, APP_NAME)
        cfg = self._collect_cfg()
        for k, v in asdict(cfg).items():
            s.setValue(k, v)


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
    main()
