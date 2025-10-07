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
    face_thresh: float = 0.32
    reid_thresh: float = 0.38
    combine: str = "min"            # min | avg | face_priority
    device: str = "cuda"            # cuda | cpu
    yolo_model: str = "yolov8n.pt"
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

            self.status.emit("Loading models...")
            det = PersonDetector(model_name=cfg.yolo_model, device=cfg.device)
            face = FaceEmbedder(ctx=cfg.device)
            reid = ReIDEmbedder(device=cfg.device)

            # Reference
            self.status.emit("Preparing reference features...")
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
            writer.writerow(["frame","time_secs","score","face_dist","reid_dist","x1","y1","x2","y2","crop_path"])

            hit_count = 0
            frame_idx = 0
            last_preview_emit = -999999

            self.status.emit("Processing...")
            while True:
                if self._abort:
                    self.status.emit("Aborting...")
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

                persons = det.detect(frame, conf=float(cfg.min_det_conf))
                crops = []
                boxes = []
                for p in persons:
                    x1,y1,x2,y2 = [int(v) for v in p["xyxy"]]
                    x1 = max(0, x1); y1 = max(0, y1); x2 = min(W-1, x2); y2 = min(H-1, y2)
                    if x2 <= x1+2 or y2 <= y1+2:
                        continue
                    crop = frame[y1:y2, x1:x2]
                    crops.append(crop); boxes.append((x1,y1,x2,y2))

                reid_feats = reid.extract(crops) if crops else []

                # Face features per person
                face_map = {}
                for i, crop in enumerate(crops):
                    ffaces = face.extract(crop)
                    bestf = FaceEmbedder.best_face(ffaces)
                    if bestf is not None and ref_face_feat is not None:
                        fd = 1.0 - float(np.dot(bestf["feat"], ref_face_feat))
                        face_map[i] = (bestf, fd)

                for i, feat in enumerate(reid_feats):
                    rd = None
                    if ref_reid_feat is not None:
                        # cosine distance using L2-norm
                        f = feat / max(np.linalg.norm(feat), 1e-9)
                        r = ref_reid_feat / max(np.linalg.norm(ref_reid_feat), 1e-9)
                        sim = float(np.dot(f, r))
                        rd = 1.0 - sim
                    fd = face_map.get(i, (None, None))[1]
                    score = self._combine_scores(fd, rd, mode=cfg.combine)

                    accept = False
                    if score is not None:
                        face_ok = (fd is not None and fd <= float(cfg.face_thresh))
                        reid_ok = (rd is not None and rd <= float(cfg.reid_thresh))
                        accept = face_ok or reid_ok

                    if accept:
                        x1,y1,x2,y2 = boxes[i]
                        anchor = None
                        bf = face_map.get(i, (None,None))[0]
                        if bf is not None:
                            fb = bf["bbox"]
                            acx = x1 + (fb[0]+fb[2])/2.0
                            acy = y1 + (fb[1]+fb[3])/2.0
                            anchor = (acx, acy)
                        ex1,ey1,ex2,ey2 = expand_box_to_ratio(x1,y1,x2,y2, ratio_w, ratio_h, W, H, anchor=anchor, head_bias=0.12)
                        crop_img_path = os.path.join(crops_dir, f"f{frame_idx:08d}.jpg")
                        crop_img = frame[ey1:ey2, ex1:ex2]
                        cv2.imwrite(crop_img_path, crop_img)
                        hit_count += 1
                        self.hit.emit(crop_img_path)

                        if cfg.save_annot:
                            vis = frame.copy()
                            cv2.rectangle(vis, (x1,y1),(x2,y2),(0,255,0),2)
                            cv2.rectangle(vis, (ex1,ey1),(ex2,ey2),(255,0,0),2)
                            if bf is not None:
                                fb = bf["bbox"]
                                cv2.rectangle(vis, (x1+fb[0], y1+fb[1]), (x1+fb[2], y1+fb[3]), (0,0,255), 2)
                            txt = f"score={score:.3f} fd={fd if fd is not None else -1:.3f} rd={rd if rd is not None else -1:.3f}"
                            cv2.putText(vis, txt, (15,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                            ann_path = os.path.join(ann_dir, f"f{frame_idx:08d}.jpg") if ann_dir else None
                            if ann_path:
                                cv2.imwrite(ann_path, vis)

                    # Preview
                    if cfg.preview_every > 0 and (frame_idx - last_preview_emit) >= int(cfg.preview_every):
                        last_preview_emit = frame_idx
                        show = frame.copy()
                        # draw boxes for visibility
                        for (x1,y1,x2,y2) in boxes:
                            cv2.rectangle(show, (x1,y1),(x2,y2),(0,255,0),1)
                        qimg = self._cv_bgr_to_qimage(show)
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
        self.device_combo = QtWidgets.QComboBox(); self.device_combo.addItems(["cuda","cpu"])
        self.yolo_edit = QtWidgets.QLineEdit("yolov8n.pt")
        self.annot_check = QtWidgets.QCheckBox("Save annotated frames")
        self.preview_every_spin = QtWidgets.QSpinBox(); self.preview_every_spin.setRange(0, 5000); self.preview_every_spin.setValue(30)

        labels = [
            ("Aspect ratio W:H", self.ratio_edit),
            ("Frame stride", self.stride_spin),
            ("YOLO min conf", self.det_conf_spin),
            ("Face max dist", self.face_thr_spin),
            ("ReID max dist", self.reid_thr_spin),
            ("Combine", self.combine_combo),
            ("Device", self.device_combo),
            ("YOLO model", self.yolo_edit),
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
            device=self.device_combo.currentText(),
            yolo_model=self.yolo_edit.text().strip() or "yolov8n.pt",
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
        self.device_combo.setCurrentText(cfg.device)
        self.yolo_edit.setText(cfg.yolo_model)
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
        self.device_combo.setCurrentText(s.value("device", "cuda"))
        self.yolo_edit.setText(s.value("yolo_model", "yolov8n.pt"))
        self.annot_check.setChecked(bool(s.value("save_annot", False)))
        self.preview_every_spin.setValue(int(s.value("preview_every", 30)))

    def _save_qsettings(self):
        s = QtCore.QSettings(APP_ORG, APP_NAME)
        cfg = self._collect_cfg()
        for k, v in asdict(cfg).items():
            s.setValue(k, v)


def main():
    QtCore.QCoreApplication.setOrganizationName(APP_ORG)
    QtCore.QCoreApplication.setApplicationName(APP_NAME)
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
