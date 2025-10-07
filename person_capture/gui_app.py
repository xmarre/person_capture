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
    match_mode: str = "either"      # either | both | face_only | reid_only
    only_best: bool = True
    min_sharpness: float = 120.0
    min_gap_sec: float = 1.5
    min_box_pixels: int = 4000
    auto_crop_borders: bool = False
    border_threshold: int = 10
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
    device: str = "cuda"            # cuda | cpu
    yolo_model: str = "yolov8n.pt"
    face_model: str = "yolov8n-face.pt"
    save_annot: bool = False
    preview_every: int = 30         # emit preview every N processed frames

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

    def __init__(self, cfg: SessionConfig, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self._abort = False
        self._pause = False

    @QtCore.Slot()
    def request_abort(self):
        self._abort = True

    @QtCore.Slot(bool)
    def request_pause(self, p: bool):
        self._pause = bool(p)

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

            self._status("Loading models...", key="phase", interval=5.0)
            det = PersonDetector(model_name=cfg.yolo_model, device=cfg.device)
            face = FaceEmbedder(ctx=cfg.device, yolo_model=cfg.face_model, use_arcface=cfg.use_arcface, clip_model_name=cfg.clip_face_backbone, clip_pretrained=cfg.clip_face_pretrained, progress=self.status.emit)
            reid = ReIDEmbedder(device=cfg.device, model_name=cfg.reid_backbone, pretrained=cfg.reid_pretrained, progress=self.status.emit)

            # Reference
            self._status("Preparing reference features...", key="phase", interval=5.0)
            ref_img = cv2.imread(cfg.ref, cv2.IMREAD_COLOR)
            if ref_img is None:
                raise RuntimeError("Cannot read reference image")

            ref_faces = face.extract(ref_img)
            ref_face = FaceEmbedder.best_face(ref_faces)
            ref_face_feat = ref_face['feat'] if ref_face else None

            ref_persons = det.detect(ref_img, conf=0.1)
            if ref_persons:
                ref_persons.sort(key=lambda d: (d['xyxy'][2]-d['xyxy'][0])*(d['xyxy'][3]-d['xyxy'][1]), reverse=True)
                rx1, ry1, rx2, ry2 = [int(v) for v in ref_persons[0]['xyxy']]
                ref_crop = ref_img[ry1:ry2, rx1:rx2]
                ref_reid_feat = reid.extract([ref_crop])[0]
            else:
                ref_reid_feat = reid.extract([ref_img])[0]

            # Video
            cap = cv2.VideoCapture(cfg.video)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {cfg.video}")
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            self.setup.emit(total_frames, float(fps))

            ratio_w, ratio_h = parse_ratio(cfg.ratio)

            # CSV
            csv_path = os.path.join(cfg.out_dir, "index.csv")
            csv_f = open(csv_path, "w", newline="")
            writer = csv.writer(csv_f)
            writer.writerow(["frame","time_secs","score","face_dist","reid_dist","x1","y1","x2","y2","crop_path","sharpness"])

            hit_count = 0
            lock_hits = 0
            locked_face = None
            locked_reid = None
            prev_box = None
            frame_idx = 0
            last_preview_emit = -999999

            self._status("Processing...", key="phase", interval=5.0)
            while True:
                if self._abort:
                    self._status("Aborting...")
                    break
                if self._pause:
                    time.sleep(0.05)
                    continue

                ret = cap.grab()
                if not ret:
                    break
                if frame_idx % max(1, int(cfg.frame_stride)) != 0:
                    frame_idx += 1
                    self.progress.emit(frame_idx)
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
                    if (x2-x1)*(y2-y1) < int(cfg.min_box_pixels):
                        continue
                    crop = frame_for_det[y1:y2, x1:x2]
                    crops.append(crop); boxes.append((x1,y1,x2,y2))

                reid_feats = reid.extract(crops) if crops else []

                # Face features per person
                for i, crop in enumerate(crops):
                    ffaces = face.extract(crop)
                    bestf = FaceEmbedder.best_face(ffaces)
                    faces_local[i] = bestf

                # Evaluate candidates
                for i, feat in enumerate(reid_feats):
                    rd = None
                    if ref_reid_feat is not None:
                        f = feat / max(np.linalg.norm(feat), 1e-9)
                        r = ref_reid_feat / max(np.linalg.norm(ref_reid_feat), 1e-9)
                        rd = 1.0 - float(np.dot(f, r))

                    bf = faces_local.get(i, None)
                    fd = None
                    if bf is not None and ref_face_feat is not None:
                        fd = 1.0 - float(np.dot(bf["feat"], ref_face_feat))

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

                    if not accept:
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
                    ex1,ey1,ex2,ey2 = self._expand_to_ratio((x1,y1,x2,y2), ratio_w, ratio_h, W2, H2, anchor=anchor)

                    # Compute sharpness on final crop
                    crop_img = frame_for_det[ey1:ey2, ex1:ex2]
                    sharp = self._calc_sharpness(crop_img)
                    if sharp < float(cfg.min_sharpness):
                        continue

                    # Map to original frame coords for annotation
                    ox1, oy1, ox2, oy2 = ex1+off_x, ey1+off_y, ex2+off_x, ey2+off_y
                    area = (ox2-ox1)*(oy2-oy1)
                    candidates.append(dict(score=score, fd=fd, rd=rd, sharp=sharp, box=(ox1,oy1,ox2,oy2), area=area, show_box=(x1+off_x,y1+off_y,x2+off_x,y2+off_y), face_feat=(bf["feat"] if bf is not None else None), reid_feat=(reid_feats[i] if i < len(reid_feats) else None)))

                # Choose best and save with cadence + lock + margin + IoU gate
                def save_hit(c):
                    nonlocal hit_count, lock_hits, locked_face, locked_reid, prev_box
                    crop_img_path = os.path.join(crops_dir, f"f{frame_idx:08d}.jpg")
                    cx1,cy1,cx2,cy2 = c["box"]
                    crop_img2 = frame[cy1:cy2, cx1:cx2]
                    cv2.imwrite(crop_img_path, crop_img2)
                    writer.writerow([frame_idx, frame_idx/float(fps), c["score"], c["fd"], c["rd"], cx1, cy1, cx2, cy2, crop_img_path, c["sharp"]])
                    self.hit.emit(crop_img_path)
                    # update lock source
                    if c.get("face_feat") is not None:
                        locked_face = c["face_feat"]
                    if c.get("reid_feat") is not None:
                        locked_reid = c["reid_feat"]
                    lock_hits += 1
                    prev_box = (cx1,cy1,cx2,cy2)
                    return True

                # initialize cadence tracking
                if not hasattr(self, "_last_hit_t"):
                    self._last_hit_t = -1e9

                if not candidates:
                    self._status(f"No match. persons={diag_persons}", key="no_match", interval=1.0)
                else:
                    # Lock-aware scoring
                    def eff_score(c):
                        s = c["score"] if c["score"] is not None else 1e9
                        # prefer higher sharpness slightly
                        return (s, -c["area"], -c["sharp"]) 

                    candidates.sort(key=lambda d: eff_score(d))

                    # Disambiguation margin vs #2
                    if len(candidates) >= 2 and candidates[0]["score"] is not None and candidates[1]["score"] is not None:
                        if abs(candidates[0]["score"] - candidates[1]["score"]) < float(cfg.score_margin):
                            candidates = [c for c in candidates if c is candidates[0]]  # keep best only

                    now_t = frame_idx/float(fps)

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
                        if save_hit(chosen):
                            hit_count += 1
                            self._last_hit_t = now_t

                # Annotated preview
                if cfg.save_annot or (cfg.preview_every > 0 and (frame_idx - last_preview_emit) >= int(cfg.preview_every)):
                    show = frame.copy()
                    # draw person boxes
                    for c in candidates:
                        x1,y1,x2,y2 = c["show_box"]
                        cv2.rectangle(show, (x1,y1),(x2,y2),(0,255,0),1)
                    if candidates:
                        bx1,by1,bx2,by2 = candidates[0]["box"]
                        cv2.rectangle(show, (bx1,by1),(bx2,by2),(255,0,0),2)
                    qimg = self._cv_bgr_to_qimage(show)
                    self.preview.emit(qimg)
                # Always-on preview cadence
                if cfg.preview_every > 0 and (frame_idx - last_preview_emit) >= int(cfg.preview_every):
                    last_preview_emit = frame_idx
                    base = frame.copy()
                    qimg = self._cv_bgr_to_qimage(base)
                    self.preview.emit(qimg)

                frame_idx += 1
                self.progress.emit(frame_idx)

            cap.release()
            # finalize CSV
            csv_f.close()

            if self._abort:
                self.finished.emit(False, "Aborted")
            else:
                self.finished.emit(True, f"Done. Hits: {hit_count}. Index: {csv_path}")
        except Exception as e:
            err = f"Error: {e}\n{traceback.format_exc()}"
            self.finished.emit(False, err)

    def _status(self, msg: str):
        now = time.time()
        if not hasattr(self, '_last_status_time'):
            self._last_status_time = 0.0
            self._last_status_text = None
        if (now - self._last_status_time) >= float(self.cfg.log_interval_sec) or msg != self._last_status_text:
            self.status.emit(msg)
            self._last_status_time = now
            self._last_status_text = msg

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
        self.resize(1200, 800)
        self._thread: Optional[QtCore.QThread] = None
        self._worker: Optional[Processor] = None

        self.cfg = SessionConfig()
        self._build_ui()
        self._load_qsettings()

    # UI Construction
    def _build_ui(self):
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

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
            ("Face max dist", self.face_thr_spin),
            ("ReID max dist", self.reid_thr_spin),
            ("Combine", self.combine_combo),
            ("Match mode", self.match_mode_combo),
            ("Only best", self.only_best_check),
            ("Min sharpness", self.min_sharp_spin),
            ("Min seconds between hits", self.min_gap_spin),
            ("Min box area (px)", self.min_box_pix_spin),
            ("Auto‑crop black borders", self.auto_crop_check),
            ("Border threshold", self.border_thr_spin),
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
        self.preview_label.setMinimumHeight(300)
        prev_layout.addWidget(self.preview_label)
        # Last hit
        hit_group = QtWidgets.QGroupBox("Last saved crop")
        hit_layout = QtWidgets.QVBoxLayout(hit_group)
        self.hit_label = QtWidgets.QLabel()
        self.hit_label.setAlignment(QtCore.Qt.AlignCenter)
        self.hit_label.setMinimumHeight(300)
        hit_layout.addWidget(self.hit_label)
        mid_split.addWidget(prev_group)
        mid_split.addWidget(hit_group)
        mid_split.setSizes([600, 600])

        # Bottom: log
        log_group = QtWidgets.QGroupBox("Log")
        log_layout = QtWidgets.QVBoxLayout(log_group)
        self.log_edit = QtWidgets.QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        log_layout.addWidget(self.log_edit)

        # Assemble
        root.addWidget(file_group)
        root.addWidget(param_group)
        root.addLayout(ctrl_layout)
        root.addLayout(prog_layout)
        root.addWidget(mid_split, 1)
        root.addWidget(log_group, 1)

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
        self.reid_thr_spin.setValue(cfg.reid_thresh)
        self.combine_combo.setCurrentText(cfg.combine)
        self.match_mode_combo.setCurrentText(cfg.match_mode)
        self.only_best_check.setChecked(cfg.only_best)
        self.min_sharp_spin.setValue(cfg.min_sharpness)
        self.min_gap_spin.setValue(cfg.min_gap_sec)
        self.min_box_pix_spin.setValue(cfg.min_box_pixels)
        self.auto_crop_check.setChecked(cfg.auto_crop_borders)
        self.border_thr_spin.setValue(cfg.border_threshold)
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

        self._thread.start()

    def on_stop(self):
        if not self._thread or not self._worker:
            return
        self._worker.request_abort()
        self._log("Stop requested")

    # Signal handlers
    def _on_setup(self, total_frames: int, fps: float):
        self.progress.setRange(0, max(0, total_frames))
        self.meta_lbl.setText(f"fps={fps:.2f} frames={total_frames if total_frames>0 else 'unknown'}")

    def _on_progress(self, idx: int):
        if self.progress.maximum() > 0 and idx <= self.progress.maximum():
            self.progress.setValue(idx)

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

    def _update_buttons(self, state: str):
        running = (state == "running")
        self.start_btn.setEnabled(not running)
        self.pause_btn.setEnabled(running)
        self.resume_btn.setEnabled(running)
        self.stop_btn.setEnabled(running)

    # Persist UI settings between runs
    def closeEvent(self, e: QtGui.QCloseEvent) -> None:
        self._save_qsettings()
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
        self.reid_thr_spin.setValue(float(s.value("reid_thresh", 0.38)))
        self.combine_combo.setCurrentText(s.value("combine", "min"))
        self.match_mode_combo.setCurrentText(s.value("match_mode", "either"))
        self.only_best_check.setChecked(bool(s.value("only_best", True)))
        self.min_sharp_spin.setValue(float(s.value("min_sharpness", 120.0)))
        self.min_gap_spin.setValue(float(s.value("min_gap_sec", 1.5)))
        self.min_box_pix_spin.setValue(int(s.value("min_box_pixels", 5000)))
        self.auto_crop_check.setChecked(bool(s.value("auto_crop_borders", False)))
        self.border_thr_spin.setValue(int(s.value("border_threshold", 10)))
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
