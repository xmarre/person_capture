#!/usr/bin/env python3
"""PersonCapture GUI."""
import os, sys, json, time, csv, traceback
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List
from pathlib import Path

# Flexible imports (package or flat files)
def _imp():
    try:
        from .detectors import PersonDetector  # type: ignore
        from .face_embedder import FaceEmbedder  # type: ignore
        from .reid_embedder import ReIDEmbedder  # type: ignore
        from .utils import ensure_dir, parse_ratio, expand_box_to_ratio, detect_black_borders
        return PersonDetector, FaceEmbedder, ReIDEmbedder, ensure_dir, parse_ratio, expand_box_to_ratio, detect_black_borders
    except Exception:
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.insert(0, here)
        from detectors import PersonDetector  # type: ignore
        from face_embedder import FaceEmbedder  # type: ignore
        from reid_embedder import ReIDEmbedder  # type: ignore
        from utils import ensure_dir, parse_ratio, expand_box_to_ratio, detect_black_borders
        return PersonDetector, FaceEmbedder, ReIDEmbedder, ensure_dir, parse_ratio, expand_box_to_ratio, detect_black_borders

PersonDetector, FaceEmbedder, ReIDEmbedder, ensure_dir, parse_ratio, expand_box_to_ratio, detect_black_borders = _imp()

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

APP_ORG = "PersonCapture"
APP_NAME = "PersonCapture GUI"

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
    device: str = "cuda"
    yolo_model: str = "yolov8n.pt"
    save_annot: bool = False
    preview_every: int = 30
    auto_crop_borders: bool = False
    border_threshold: int = 10
    min_sharpness: float = 160.0
    min_box_pixels: int = 8000
    min_gap_sec: float = 1.5
    # Face specifics
    clip_face_backbone: str = "ViT-L-14"
    clip_face_pretrained: str = "laion2b_s32b_b82k"
    use_arcface: bool = False
    face_det_conf: float = 0.22
    face_quality_min: float = 120.0

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

class Processor(QtCore.QObject):
    setup = QtCore.Signal(int, float)   # total_frames, fps
    progress = QtCore.Signal(int)
    status = QtCore.Signal(str)
    preview = QtCore.Signal(QtGui.QImage)
    hit = QtCore.Signal(str)
    finished = QtCore.Signal(bool, str)

    def __init__(self, cfg: SessionConfig, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self._abort = False

    def _cv_to_qimg(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QtGui.QImage(rgb.data, w, h, 3*w, QtGui.QImage.Format_RGB888)
        return qimg.copy()

    def _status(self, msg: str, key: str = "global", interval: float = 1.0):
        now = time.time()
        last_t = getattr(self, "_status_last_time", {}).get(key, 0.0)
        last_msg = getattr(self, "_status_last_text", {}).get(key, None)
        if (now - last_t) >= float(interval) or msg != last_msg:
            self.status.emit(msg)
            if not hasattr(self, "_status_last_time"):
                self._status_last_time = {}
                self._status_last_text = {}
            self._status_last_time[key] = now
            self._status_last_text[key] = msg

    @QtCore.Slot()
    def request_abort(self):
        self._abort = True

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
            if ann_dir: ensure_dir(ann_dir)

            self._status("Loading models...", key="phase", interval=2.0)
            det = PersonDetector(model_name=cfg.yolo_model, device=cfg.device, progress=self.status.emit)
            face = FaceEmbedder(ctx=cfg.device, conf=float(cfg.face_det_conf),
                                clip_model_name=cfg.clip_face_backbone, clip_pretrained=cfg.clip_face_pretrained,
                                use_arcface=bool(cfg.use_arcface), progress=self.status.emit)
            reid = ReIDEmbedder(device=cfg.device, progress=self.status.emit)

            # Reference
            ref_img = cv2.imread(cfg.ref, cv2.IMREAD_COLOR)
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

            cap = cv2.VideoCapture(cfg.video)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {cfg.video}")
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            self.setup.emit(total, float(fps))

            ratios = [r.strip() for r in str(cfg.ratio).split(',') if r.strip()] or ["2:3"]

            csv_path = os.path.join(cfg.out_dir, "index.csv")
            fout = open(csv_path, "w", newline="")
            writer = csv.writer(fout)
            writer.writerow(["frame","time_secs","score","face_dist","reid_dist","x1","y1","x2","y2","crop_path","ratio"])

            last_hit_t = -1e9
            frame_idx = 0
            while True:
                if self._abort:
                    self._status("Aborting...", key="phase")
                    break
                ok = cap.grab()
                if not ok:
                    break
                if frame_idx % max(1, int(cfg.frame_stride)) != 0:
                    frame_idx += 1
                    self.progress.emit(frame_idx)
                    continue
                ok, frame = cap.retrieve()
                if not ok:
                    break
                H, W = frame.shape[:2]

                frame_for_det = frame
                off_x, off_y = 0, 0
                if cfg.auto_crop_borders:
                    x1b, y1b, x2b, y2b = detect_black_borders(frame, thr=int(cfg.border_threshold))
                    if (x2b - x1b) > 32 and (y2b - y1b) > 32:
                        frame_for_det = frame[y1b:y2b, x1b:x2b]
                        off_x, off_y = x1b, y1b
                        H, W = frame_for_det.shape[:2]

                persons = det.detect(frame_for_det, conf=float(cfg.min_det_conf))
                crops, boxes = [], []
                for p in persons:
                    x1,y1,x2,y2 = [int(v) for v in p["xyxy"]]
                    x1 = max(0, x1); y1 = max(0, y1); x2 = min(W-1, x2); y2 = min(H-1, y2)
                    if x2 <= x1+2 or y2 <= y1+2:
                        continue
                    if (x2-x1)*(y2-y1) < int(cfg.min_box_pixels):
                        continue
                    crops.append(frame_for_det[y1:y2, x1:x2])
                    boxes.append((x1,y1,x2,y2))

                if not boxes:
                    self.progress.emit(frame_idx); frame_idx += 1
                    continue

                reid_feats = reid.extract(crops)
                # face best + distance if possible
                faces_local = {}
                face_dists = {}
                for i, crop in enumerate(crops):
                    ffaces = face.extract(crop)
                    bf = FaceEmbedder.best_face(ffaces)
                    faces_local[i] = bf
                    if bf is not None and bf.get("feat") is not None and ref_face_feat is not None:
                        # cosine distance on normalized vectors
                        f = bf["feat"]; r = ref_face_feat
                        nf = max(np.linalg.norm(f), 1e-9); nr = max(np.linalg.norm(r), 1e-9)
                        fd = 1.0 - float(np.dot(f/nf, r/nr))
                        face_dists[i] = fd

                candidates = []
                for i, feat in enumerate(reid_feats):
                    rd = None
                    if feat is not None and ref_reid_feat is not None:
                        f = feat / max(np.linalg.norm(feat), 1e-9)
                        r = ref_reid_feat / max(np.linalg.norm(ref_reid_feat), 1e-9)
                        rd = 1.0 - float(np.dot(f, r))
                    fd = face_dists.get(i, None)
                    vals = [v for v in (fd, rd) if v is not None]
                    if not vals:
                        continue
                    # thresholds
                    if fd is not None and fd > float(cfg.face_thresh) and rd is not None and rd > float(cfg.reid_thresh):
                        continue
                    score = min(vals) if cfg.combine == "min" else (sum(vals)/len(vals) if cfg.combine == "avg" else (0.7*fd + 0.3*(rd if rd is not None else 0.5)) if fd is not None else rd)
                    # choose ratio with minimal expansion
                    bx1,by1,bx2,by2 = boxes[i]
                    det_area = max(1, (bx2-bx1)*(by2-by1))
                    best = None; best_ratio = None; best_factor = 1e9
                    for rs in ratios:
                        rw, rh = parse_ratio(rs)
                        ex1,ey1,ex2,ey2 = expand_box_to_ratio(bx1,by1,bx2,by2,rw,rh,W,H,anchor=None,head_bias=0.12)
                        area = max(1, (ex2-ex1)*(ey2-ey1))
                        fct = area/det_area
                        if fct < best_factor:
                            best_factor = fct; best = (ex1,ey1,ex2,ey2); best_ratio = rs
                    if best is None:
                        continue
                    # map to original frame
                    ox1,oy1,ox2,oy2 = best[0]+off_x, best[1]+off_y, best[2]+off_x, best[3]+off_y
                    candidates.append(dict(score=score, fd=fd, rd=rd, box=(ox1,oy1,ox2,oy2), ratio=best_ratio))

                if not candidates:
                    self.progress.emit(frame_idx); frame_idx += 1
                    continue

                candidates.sort(key=lambda c: (c["score"], -((c["box"][2]-c["box"][0])*(c["box"][3]-c["box"][1]))))
                chosen = candidates[0]
                now_t = frame_idx/float(fps)
                if now_t - last_hit_t >= float(cfg.min_gap_sec):
                    cx1,cy1,cx2,cy2 = chosen["box"]
                    crop = frame[cy1:cy2, cx1:cx2]
                    out_path = os.path.join(crops_dir, f"f{frame_idx:08d}.jpg")
                    cv2.imwrite(out_path, crop)
                    writer.writerow([frame_idx, now_t, chosen["score"], chosen["fd"], chosen["rd"], cx1, cy1, cx2, cy2, out_path, chosen["ratio"]])
                    self.hit.emit(out_path)
                    last_hit_t = now_t

                # preview
                show = frame.copy()
                bx1,by1,bx2,by2 = chosen["box"]
                cv2.rectangle(show, (bx1,by1),(bx2,by2),(0,255,0),2)
                self.preview.emit(self._cv_to_qimg(show))
                self.progress.emit(frame_idx)
                frame_idx += 1

            cap.release()
            fout.close()
            if self._abort:
                self.finished.emit(False, "Aborted")
            else:
                self.finished.emit(True, "Done")
        except Exception as e:
            self.finished.emit(False, f"Error: {e}\n{traceback.format_exc()}")

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1100, 700)

        # Controls
        w = QtWidgets.QWidget(self)
        lay = QtWidgets.QGridLayout(w)
        self.setCentralWidget(w)

        self.video_edit = QtWidgets.QLineEdit()
        self.ref_edit = QtWidgets.QLineEdit()
        self.out_edit = QtWidgets.QLineEdit("output")
        self.ratio_edit = QtWidgets.QLineEdit("2:3")
        self.det_spin = QtWidgets.QDoubleSpinBox(); self.det_spin.setRange(0.01, 0.99); self.det_spin.setValue(0.35); self.det_spin.setSingleStep(0.01)
        self.face_thr_spin = QtWidgets.QDoubleSpinBox(); self.face_thr_spin.setRange(0.01, 1.5); self.face_thr_spin.setValue(0.36); self.face_thr_spin.setSingleStep(0.01)
        self.reid_thr_spin = QtWidgets.QDoubleSpinBox(); self.reid_thr_spin.setRange(0.01, 1.5); self.reid_thr_spin.setValue(0.42); self.reid_thr_spin.setSingleStep(0.01)
        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.preview_lbl = QtWidgets.QLabel(); self.preview_lbl.setMinimumHeight(420); self.preview_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.status_edit = QtWidgets.QPlainTextEdit(); self.status_edit.setReadOnly(True); self.status_edit.setMaximumBlockCount(400)

        row = 0
        lay.addWidget(QtWidgets.QLabel("Video:"), row,0); lay.addWidget(self.video_edit, row,1,1,3)
        row += 1
        lay.addWidget(QtWidgets.QLabel("Reference:"), row,0); lay.addWidget(self.ref_edit, row,1,1,3)
        row += 1
        lay.addWidget(QtWidgets.QLabel("Output dir:"), row,0); lay.addWidget(self.out_edit, row,1,1,1)
        lay.addWidget(QtWidgets.QLabel("Ratio(s):"), row,2); lay.addWidget(self.ratio_edit, row,3,1,1)
        row += 1
        lay.addWidget(QtWidgets.QLabel("Det conf:"), row,0); lay.addWidget(self.det_spin, row,1)
        lay.addWidget(QtWidgets.QLabel("Face thr:"), row,2); lay.addWidget(self.face_thr_spin, row,3)
        row += 1
        lay.addWidget(QtWidgets.QLabel("ReID thr:"), row,0); lay.addWidget(self.reid_thr_spin, row,1)
        lay.addWidget(self.start_btn, row,2); lay.addWidget(self.stop_btn, row,3)
        row += 1
        lay.addWidget(self.preview_lbl, row,0,1,4)
        row += 1
        lay.addWidget(self.status_edit, row,0,1,4)

        self.thread = None
        self.worker = None

        self.start_btn.clicked.connect(self.start_proc)
        self.stop_btn.clicked.connect(self.stop_proc)

        # load preset if exists
        preset_path = Path("preset.json")
        if preset_path.exists():
            try:
                cfg = SessionConfig.from_json(preset_path.read_text(encoding="utf-8"))
                self.video_edit.setText(cfg.video)
                self.ref_edit.setText(cfg.ref)
                self.out_edit.setText(cfg.out_dir)
                self.ratio_edit.setText(cfg.ratio)
                self.det_spin.setValue(float(cfg.min_det_conf))
                self.face_thr_spin.setValue(float(cfg.face_thresh))
                self.reid_thr_spin.setValue(float(cfg.reid_thresh))
            except Exception:
                pass

    def log(self, msg: str):
        self.status_edit.appendPlainText(msg)

    def start_proc(self):
        cfg = SessionConfig(
            video=self.video_edit.text().strip(),
            ref=self.ref_edit.text().strip(),
            out_dir=self.out_edit.text().strip(),
            ratio=self.ratio_edit.text().strip(),
            min_det_conf=float(self.det_spin.value()),
            face_thresh=float(self.face_thr_spin.value()),
            reid_thresh=float(self.reid_thr_spin.value()),
        )
        self.thread = QtCore.QThread(self)
        self.worker = Processor(cfg)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.preview.connect(self.on_preview)
        self.worker.status.connect(self.log)
        self.worker.progress.connect(lambda i: self.statusBar().showMessage(f"Frame {i}"))
        self.worker.hit.connect(lambda p: self.log(f"HIT: {p}"))
        self.worker.finished.connect(self.on_finished)
        self.thread.start()

    @QtCore.Slot()
    def stop_proc(self):
        if self.worker:
            self.worker.request_abort()

    def on_preview(self, qimg: QtGui.QImage):
        self.preview_lbl.setPixmap(QtGui.QPixmap.fromImage(qimg).scaled(self.preview_lbl.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def on_finished(self, ok: bool, msg: str):
        self.log(msg)
        if self.thread:
            self.thread.quit()
            self.thread.wait()
        self.thread = None
        self.worker = None

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setOrganizationName(APP_ORG)
    app.setApplicationName(APP_NAME)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
