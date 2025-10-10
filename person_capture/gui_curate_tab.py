
from __future__ import annotations

import os, json, threading
from pathlib import Path
from typing import Optional, List

from PySide6 import QtCore, QtGui, QtWidgets

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

    def __init__(self, pool_dir: str, ref_path: str, out_dir: str, max_images: int, device: str = "cuda"):
        super().__init__()
        self.pool_dir = pool_dir
        self.ref_path = ref_path
        self.out_dir = out_dir
        self.max_images = max_images
        self.device = device

    @QtCore.Slot()
    def run(self):
        try:
            cur = Curator(ref_image=self.ref_path, device=self.device,
                          progress=lambda phase, done, total: self.progress.emit(phase, done, total))
            out = cur.run(self.pool_dir, self.out_dir, max_images=self.max_images)
            # read manifest for UI
            rows = []
            mp = Path(out) / "dataset_manifest.csv"
            if mp.exists():
                for i, line in enumerate(mp.read_text(encoding="utf-8").splitlines()[1:], 1):
                    parts = [p.strip() for p in line.split(",")]
                    rows.append(parts)
            self.finished.emit(out, rows)
        except Exception as e:
            self.failed.emit(str(e))


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
        pool = self.editPool.text().strip()
        ref = self.editRef.text().strip()
        out = self.editOut.text().strip()
        if not pool or not os.path.isdir(pool):
            self._set_status("Invalid pool folder")
            return
        if not ref or not os.path.isfile(ref):
            self._set_status("Invalid reference image")
            return
        if not out:
            self._set_status("Invalid output folder")
            return
        self.btnRun.setEnabled(False)
        self.prog.setValue(0)
        self._set_status("Running…")

        self._thread = QtCore.QThread(self)
        self._worker = CurateWorker(pool, ref, out, int(self.spinMax.value()), device=self.comboDevice.currentText())
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._done)
        self._worker.failed.connect(self._fail)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _done(self, out_dir: str, rows: list):
        self._set_status(f"Done → {out_dir}")
        self.prog.setValue(100)
        self.btnRun.setEnabled(True)
        self.table.setRowCount(0)
        for i, parts in enumerate(rows, 0):
            r = self.table.rowCount()
            self.table.insertRow(r)
            # parts: [rank,file,face_fd,sharpness,exposure,face_frac,yaw,roll,ratio,quality,category]
            self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(parts[0]))
            self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(parts[1]))
            self.table.setItem(r, 2, QtWidgets.QTableWidgetItem(parts[2]))
            self.table.setItem(r, 3, QtWidgets.QTableWidgetItem(parts[3]))
            self.table.setItem(r, 4, QtWidgets.QTableWidgetItem(parts[5]))
            self.table.setItem(r, 5, QtWidgets.QTableWidgetItem(parts[8]))

    def _fail(self, msg: str):
        self._set_status(f"Failed: {msg}")
        self.prog.setValue(0)
        self.btnRun.setEnabled(True)

    def _set_status(self, s: str):
        self.status.setText(s)

    @QtCore.Slot(str, int, int)
    def _on_progress(self, phase: str, done: int, total: int):
        pct = 0 if total <= 0 else int(round(100.0 * done / max(1, total)))
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
