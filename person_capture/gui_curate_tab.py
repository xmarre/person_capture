
from __future__ import annotations

import os, traceback
from pathlib import Path
from typing import Optional, List

from PySide6 import QtCore, QtWidgets

# robust imports
def _imp():
    try:
        from .dataset_curator import Curator  # type: ignore
    except Exception:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from dataset_curator import Curator  # type: ignore
    return Curator

Curator = _imp()


class CurateWorker(QtCore.QObject):
    progress = QtCore.Signal(str, int, int)   # phase, done, total
    finished = QtCore.Signal(str, list)  # out_dir, manifest rows
    failed = QtCore.Signal(str)

    def __init__(
        self,
        pool_dir: str,
        ref_path: str,
        out_dir: str,
        max_images: int,
        *,
        device: str = "cuda",
        scene_aware: bool = True,
        scene_sim: float = 0.92,
        scene_time_gap: float = 4.0,
        scene_nn_window: int = 64,
        dedup_hamm: int = 7,
        scene_dedup: int = 4,
        scene_soft_cap: int = 0,
        scene_soft_penalty: float = 0.08,
        mmr_alpha: float = 0.75,
        profile_yaw: float = 50.0,
    ):
        super().__init__()
        self.pool_dir = pool_dir
        self.ref_path = ref_path
        self.out_dir = out_dir
        self.max_images = max_images
        self.device = device
        self.scene_aware = bool(scene_aware)
        self.scene_sim = float(scene_sim)
        self.scene_time_gap = float(scene_time_gap)
        self.scene_nn_window = int(scene_nn_window)
        self.dedup_hamm = int(dedup_hamm)
        self.scene_dedup = int(scene_dedup)
        self.scene_soft_cap = int(scene_soft_cap)
        self.scene_soft_penalty = float(scene_soft_penalty)
        self.mmr_alpha = float(mmr_alpha)
        self.profile_yaw = float(profile_yaw)

    @QtCore.Slot()
    def run(self):
        try:
            # let the UI show something immediately
            try:
                self.progress.emit("init: starting", 0, 0)
            except Exception:
                pass
            trt_dir = os.getenv("TRT_LIB_DIR") or os.getenv("TENSORRT_DIR") or ""
            try:
                s = QtCore.QSettings("PersonCapture", "PersonCapture GUI")
                face_model = s.value("face_model", "scrfd_10g_bnkps")
                face_det_conf = float(s.value("face_det_conf", 0.50))
            except Exception:
                face_model = "scrfd_10g_bnkps"
                face_det_conf = 0.50
            if not self.ref_path:
                try:
                    self.progress.emit("init: identity gating assumed (no ref)", 0, 0)
                except Exception:
                    pass
            cur = Curator(
                ref_image=(self.ref_path or None),
                device=self.device,
                trt_lib_dir=(trt_dir or None),
                face_model=str(face_model),
                face_det_conf=float(face_det_conf),
                assume_identity=bool(not self.ref_path),
                progress=lambda phase, done, total: self.progress.emit(str(phase), int(done), int(total)),
            )
            # If user provided a ref but it yielded no face, abort (matches main GUI behavior).
            if (
                self.ref_path
                and not getattr(cur, "id_already_passed", False)
                and getattr(cur, "ref_feat", None) is None
            ):
                raise RuntimeError("No face in the reference image.")
            out = cur.run(
                self.pool_dir,
                self.out_dir,
                max_images=self.max_images,
                scene_aware_override=self.scene_aware,
                scene_sim_override=self.scene_sim,
                scene_time_gap_override=self.scene_time_gap,
                scene_nn_window_override=self.scene_nn_window,
                dedup_hamm_override=self.dedup_hamm,
                scene_dedup_override=self.scene_dedup,
                scene_soft_cap_override=self.scene_soft_cap,
                scene_soft_penalty_override=self.scene_soft_penalty,
                alpha_override=self.mmr_alpha,
                profile_yaw_override=self.profile_yaw,
            )
            # read manifest for UI
            rows = []
            mp = Path(out) / "dataset_manifest.csv"
            if mp.exists():
                import csv

                with open(mp, "r", encoding="utf-8", newline="") as f:
                    reader = csv.reader(f)
                    next(reader, None)
                    for parts in reader:
                        rows.append([p.strip() for p in parts])
            self.finished.emit(out, rows)
        except Exception:
            self.failed.emit(traceback.format_exc())


class CurateTab(QtWidgets.QWidget):
    def __init__(self, parent=None, default_pool:str="", default_ref:str=""):
        super().__init__(parent)
        self.setObjectName("CurateTab")
        self._build_ui(default_pool, default_ref)
        self._thread: Optional[QtCore.QThread] = None
        self._worker: Optional[CurateWorker] = None

    def _build_ui(self, default_pool: str, default_ref: str):
        L = QtWidgets.QFormLayout(self)

        self.editPool = QtWidgets.QLineEdit(default_pool)
        self.btnPool = QtWidgets.QPushButton("Browse…")
        self.btnPool.clicked.connect(self._pick_pool)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.editPool, 1); row.addWidget(self.btnPool, 0)
        L.addRow("Pool folder:", row)

        self.editRef = QtWidgets.QLineEdit(default_ref)
        self.btnRef = QtWidgets.QPushButton("Browse…")
        self.btnRef.clicked.connect(self._pick_ref)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.editRef, 1); row.addWidget(self.btnRef, 0)
        L.addRow("Reference image:", row)

        self.editOut = QtWidgets.QLineEdit("dataset_out")
        self.btnOut = QtWidgets.QPushButton("Browse…")
        self.btnOut.clicked.connect(self._pick_out)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.editOut, 1); row.addWidget(self.btnOut, 0)
        L.addRow("Output folder:", row)

        self.spinMax = QtWidgets.QSpinBox()
        self.spinMax.setRange(20, 500)
        self.spinMax.setValue(200)
        L.addRow("Max images:", self.spinMax)

        self.comboDevice = QtWidgets.QComboBox()
        self.comboDevice.addItems(["cuda","cpu"])
        self.comboDevice.setCurrentText("cuda")
        L.addRow("Device:", self.comboDevice)

        self.chkScene = QtWidgets.QCheckBox("Scene-aware variety (group shots, pick best per scene)")
        self.chkScene.setChecked(True)
        L.addRow(self.chkScene)

        self.dspinSceneSim = QtWidgets.QDoubleSpinBox()
        self.dspinSceneSim.setRange(0.0, 1.0)
        self.dspinSceneSim.setDecimals(3)
        self.dspinSceneSim.setSingleStep(0.01)
        self.dspinSceneSim.setValue(0.92)
        self.dspinSceneSim.setToolTip("Cosine similarity threshold for grouping scenes (higher = stricter)")
        L.addRow("Scene similarity:", self.dspinSceneSim)

        self.dspinTimeGap = QtWidgets.QDoubleSpinBox()
        self.dspinTimeGap.setRange(0.0, 3600.0)
        self.dspinTimeGap.setDecimals(2)
        self.dspinTimeGap.setSingleStep(0.25)
        self.dspinTimeGap.setValue(4.0)
        self.dspinTimeGap.setToolTip("Max seconds gap when stitching adjacent frames into a scene")
        L.addRow("Scene time gap (s):", self.dspinTimeGap)

        self.spinNNWindow = QtWidgets.QSpinBox()
        self.spinNNWindow.setRange(0, 4096)
        self.spinNNWindow.setValue(64)
        self.spinNNWindow.setToolTip("Neighbor window when merging nearby scenes (0 disables)")
        L.addRow("Scene stitch window:", self.spinNNWindow)

        self.spinDedupGlobal = QtWidgets.QSpinBox()
        self.spinDedupGlobal.setRange(0, 64)
        self.spinDedupGlobal.setValue(7)
        self.spinDedupGlobal.setToolTip("Global pHash dedup threshold (Hamming distance)")
        L.addRow("Dedup Hamming (global):", self.spinDedupGlobal)

        self.spinDedupScene = QtWidgets.QSpinBox()
        self.spinDedupScene.setRange(0, 64)
        self.spinDedupScene.setValue(4)
        self.spinDedupScene.setToolTip("In-scene pHash dedup threshold (Hamming distance)")
        L.addRow("Dedup Hamming (in-scene):", self.spinDedupScene)

        self.spinSceneSoftCap = QtWidgets.QSpinBox()
        self.spinSceneSoftCap.setRange(0, 9999)
        self.spinSceneSoftCap.setValue(0)
        self.spinSceneSoftCap.setToolTip("Soft cap per scene; 0 disables penalty")
        L.addRow("Scene soft cap:", self.spinSceneSoftCap)

        self.dspinSceneSoftPenalty = QtWidgets.QDoubleSpinBox()
        self.dspinSceneSoftPenalty.setRange(0.0, 1.0)
        self.dspinSceneSoftPenalty.setDecimals(3)
        self.dspinSceneSoftPenalty.setSingleStep(0.01)
        self.dspinSceneSoftPenalty.setValue(0.08)
        self.dspinSceneSoftPenalty.setToolTip("Penalty subtracted when a scene exceeds the soft cap")
        L.addRow("Scene soft penalty:", self.dspinSceneSoftPenalty)

        self.dspinAlpha = QtWidgets.QDoubleSpinBox()
        self.dspinAlpha.setRange(0.0, 1.0)
        self.dspinAlpha.setDecimals(2)
        self.dspinAlpha.setSingleStep(0.05)
        self.dspinAlpha.setValue(0.75)
        self.dspinAlpha.setToolTip("MMR trade-off between quality (1) and diversity (0)")
        L.addRow("MMR alpha:", self.dspinAlpha)

        self.dspinProfileYaw = QtWidgets.QDoubleSpinBox()
        self.dspinProfileYaw.setRange(0.0, 180.0)
        self.dspinProfileYaw.setDecimals(1)
        self.dspinProfileYaw.setSingleStep(1.0)
        self.dspinProfileYaw.setValue(50.0)
        self.dspinProfileYaw.setToolTip("Yaw threshold (degrees) for treating a face as profile")
        L.addRow("Profile yaw (deg):", self.dspinProfileYaw)

        # Load persisted values
        self._load_settings()

        def _sync_scene_enabled(on: bool) -> None:
            for w in (
                self.dspinSceneSim,
                self.dspinTimeGap,
                self.spinNNWindow,
                self.spinDedupScene,
                self.spinSceneSoftCap,
                self.dspinSceneSoftPenalty,
            ):
                w.setEnabled(on)

        self.chkScene.toggled.connect(_sync_scene_enabled)
        _sync_scene_enabled(self.chkScene.isChecked())

        self.btnRun = QtWidgets.QPushButton("Score + Build Dataset")
        self.btnRun.clicked.connect(self._start)
        L.addRow(self.btnRun)
        self.prog = QtWidgets.QProgressBar()
        self.prog.setRange(0, 100)
        self.prog.setValue(0)
        L.addRow(self.prog)

        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["#", "file", "fd", "sharp", "face_frac", "ratio"])
        self.table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        L.addRow(self.table)

        self.status = QtWidgets.QLabel("Idle")
        L.addRow(self.status)

    def _pick_pool(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Pick pool folder")
        if d:
            self.editPool.setText(d)

    def _pick_ref(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Pick reference image", "", "Images (*.jpg *.jpeg *.png *.webp)")
        if f:
            self.editRef.setText(f)

    def _pick_out(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Pick output folder")
        if d:
            self.editOut.setText(d)

    def _start(self):
        # Save settings before kicking off the worker
        self._save_settings()
        pool = self.editPool.text().strip()
        ref = self.editRef.text().strip()
        out = self.editOut.text().strip()
        if not pool or not os.path.isdir(pool):
            self._set_status("Invalid pool folder")
            return
        # Reference is optional: if blank, we assume identity is already passed (PersonCapture crops).
        if ref and not os.path.isfile(ref):
            self._set_status("Invalid reference image")
            return
        if not out:
            self._set_status("Invalid output folder")
            return
        self.btnRun.setEnabled(False)
        # show activity immediately; _on_progress will keep it busy during init
        self.prog.setRange(0, 0); self.prog.setValue(0)
        self._set_status("Running…")

        self._thread = QtCore.QThread(self)
        self._worker = CurateWorker(
            pool,
            ref,
            out,
            int(self.spinMax.value()),
            device=self.comboDevice.currentText(),
            scene_aware=self.chkScene.isChecked(),
            scene_sim=float(self.dspinSceneSim.value()),
            scene_time_gap=float(self.dspinTimeGap.value()),
            scene_nn_window=int(self.spinNNWindow.value()),
            dedup_hamm=int(self.spinDedupGlobal.value()),
            scene_dedup=int(self.spinDedupScene.value()),
            scene_soft_cap=int(self.spinSceneSoftCap.value()),
            scene_soft_penalty=float(self.dspinSceneSoftPenalty.value()),
            mmr_alpha=float(self.dspinAlpha.value()),
            profile_yaw=float(self.dspinProfileYaw.value()),
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._done)
        self._worker.failed.connect(self._fail)
        self._worker.progress.connect(self._on_progress)
        # cleanup to prevent leaks on repeated runs
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.failed.connect(self._worker.deleteLater)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _load_settings(self) -> None:
        s = QtCore.QSettings("PersonCapture", "CurateTab")
        self.chkScene.setChecked(s.value("scene_aware", True, type=bool))

        def _float(key: str, default: float) -> float:
            try:
                v = s.value(key, default)
                if v is None or v == "":
                    return float(default)
                return float(v)
            except Exception:
                return float(default)

        def _int(key: str, default: int) -> int:
            try:
                v = s.value(key, default)
                if v is None or v == "":
                    return int(default)
                return int(float(v))
            except Exception:
                return int(default)

        self.dspinSceneSim.setValue(_float("scene_sim", 0.92))
        self.dspinTimeGap.setValue(_float("scene_time_gap", 4.0))
        self.spinNNWindow.setValue(_int("scene_nn_window", 64))
        self.spinDedupGlobal.setValue(_int("dedup_hamm", 7))
        self.spinDedupScene.setValue(_int("scene_dedup", 4))
        self.spinSceneSoftCap.setValue(_int("scene_soft_cap", 0))
        try:
            raw_pen = s.value("scene_soft_penalty", "")
            if raw_pen in ("", None):
                raw_pen = s.value("scene_soft_bonus", 0.08)
            self.dspinSceneSoftPenalty.setValue(float(raw_pen))
        except Exception:
            self.dspinSceneSoftPenalty.setValue(0.08)
        self.dspinAlpha.setValue(_float("mmr_alpha", 0.75))
        self.dspinProfileYaw.setValue(_float("profile_yaw", 50.0))

    def _save_settings(self) -> None:
        s = QtCore.QSettings("PersonCapture", "CurateTab")
        s.setValue("scene_aware", self.chkScene.isChecked())
        s.setValue("scene_sim", self.dspinSceneSim.value())
        s.setValue("scene_time_gap", self.dspinTimeGap.value())
        s.setValue("scene_nn_window", self.spinNNWindow.value())
        s.setValue("dedup_hamm", self.spinDedupGlobal.value())
        s.setValue("scene_dedup", self.spinDedupScene.value())
        s.setValue("scene_soft_cap", self.spinSceneSoftCap.value())
        s.setValue("scene_soft_penalty", self.dspinSceneSoftPenalty.value())
        s.setValue("scene_soft_bonus", self.dspinSceneSoftPenalty.value())
        s.setValue("mmr_alpha", self.dspinAlpha.value())
        s.setValue("profile_yaw", self.dspinProfileYaw.value())

    def _done(self, out_dir: str, rows: List[List[str]]):
        self._thread = None
        self._worker = None
        self._set_status(f"Done → {out_dir}")
        self.prog.setRange(0, 100)
        self.prog.setValue(100)
        self.btnRun.setEnabled(True)
        self.table.setRowCount(0)
        for i, parts in enumerate(rows, 0):
            r = self.table.rowCount()
            self.table.insertRow(r)
            # parts: [rank,file,face_fd,sharpness,exposure,face_frac,yaw,roll,ratio,quality,category,is_profile]
            self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(parts[0]))
            self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(parts[1]))
            self.table.setItem(r, 2, QtWidgets.QTableWidgetItem(parts[2]))
            self.table.setItem(r, 3, QtWidgets.QTableWidgetItem(parts[3]))
            self.table.setItem(r, 4, QtWidgets.QTableWidgetItem(parts[5]))
            self.table.setItem(r, 5, QtWidgets.QTableWidgetItem(parts[8]))

    def _fail(self, msg: str):
        self._thread = None
        self._worker = None
        self._set_status(f"Failed: {msg}")
        self.prog.setRange(0, 100)
        self.prog.setValue(0)
        self.btnRun.setEnabled(True)

    def _set_status(self, s: str):
        self.status.setText(s)

    @QtCore.Slot(str, int, int)
    def _on_progress(self, phase: str, done: int, total: int):
        # During init (total==0), show an indeterminate (marquee) bar and just print the phase text.
        if total <= 0:
            if self.prog.minimum() != 0 or self.prog.maximum() != 0:
                self.prog.setRange(0, 0)  # busy
            self._set_status(str(phase))
            return
        # Determinate progress once we know totals
        if self.prog.minimum() == 0 and self.prog.maximum() == 0:
            self.prog.setRange(0, 100)
        pct = int(round(100.0 * done / max(1, total)))
        self.prog.setValue(max(0, min(100, pct)))
        self._set_status(f"{phase}: {done}/{total}")


def add_tab_to(main_window, default_pool:str="", default_ref:str=""):
    """
    Call from your MainWindow after constructing QTabWidget 'tabs' or similar.
    Example:
        from gui_curate_tab import add_tab_to
        add_tab_to(self, default_pool=last_out_dir, default_ref=self.cfg.ref_path)
    """
    # Find a QTabWidget child
    tabs = main_window.findChild(QtWidgets.QTabWidget)
    if tabs is None:
        # create one if not present
        tabs = QtWidgets.QTabWidget(main_window)
        main_window.setCentralWidget(tabs)
    tab = CurateTab(tabs, default_pool, default_ref)
    tabs.addTab(tab, "Curate")
    return tab
