"""Qt wrapper for the Vulkan HDR preview DLL."""

from __future__ import annotations

import ctypes
from ctypes import POINTER, c_int, c_uint16, c_void_p
import logging
import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from PySide6 import QtCore, QtGui, QtWidgets

_log = logging.getLogger(__name__)


def _load_hdr_dll() -> Optional[ctypes.CDLL]:
    """Try PERSON_CAPTURE_ROOT, repo root, then CWD when loading pc_hdr_vulkan.dll."""

    root_env = os.getenv("PERSON_CAPTURE_ROOT")
    candidates = []
    if root_env:
        candidates.append(Path(root_env) / "pc_hdr_vulkan.dll")

    candidates.append(Path(__file__).resolve().parent.parent / "pc_hdr_vulkan.dll")
    candidates.append(Path("pc_hdr_vulkan.dll"))

    last_err: Optional[BaseException] = None
    for candidate in candidates:
        try:
            dll = ctypes.CDLL(str(candidate))
            _log.info("HDR: loaded pc_hdr_vulkan.dll from %s", candidate)
            return dll
        except OSError as exc:
            last_err = exc

    _log.warning(
        "HDR: failed to load pc_hdr_vulkan.dll; tried %s (last error: %s)",
        ", ".join(str(c) for c in candidates),
        last_err,
    )
    return None


_dll = _load_hdr_dll()


class _HDRContext(ctypes.Structure):
    pass


_CTX_PTR = ctypes.POINTER(_HDRContext)


def _setup_prototypes():
    if _dll is None:
        return None, None, None, None, None

    pc_hdr_init = _dll.pc_hdr_init
    pc_hdr_init.argtypes = [c_void_p, c_int, c_int]
    pc_hdr_init.restype = _CTX_PTR

    pc_hdr_resize = _dll.pc_hdr_resize
    pc_hdr_resize.argtypes = [_CTX_PTR, c_int, c_int]
    pc_hdr_resize.restype = None

    pc_hdr_upload_p010 = _dll.pc_hdr_upload_p010
    pc_hdr_upload_p010.argtypes = [
        _CTX_PTR,
        POINTER(c_uint16),
        POINTER(c_uint16),
        c_int,
        c_int,
    ]
    pc_hdr_upload_p010.restype = None

    pc_hdr_present = _dll.pc_hdr_present
    pc_hdr_present.argtypes = [_CTX_PTR]
    pc_hdr_present.restype = None

    pc_hdr_shutdown = _dll.pc_hdr_shutdown
    pc_hdr_shutdown.argtypes = [_CTX_PTR]
    pc_hdr_shutdown.restype = None

    return (
        pc_hdr_init,
        pc_hdr_resize,
        pc_hdr_upload_p010,
        pc_hdr_present,
        pc_hdr_shutdown,
    )


(
    _pc_hdr_init,
    _pc_hdr_resize,
    _pc_hdr_upload_p010,
    _pc_hdr_present,
    _pc_hdr_shutdown,
) = _setup_prototypes()


def hdr_passthrough_available() -> bool:
    """Return True when the HDR DLL loaded and exported functions are present."""

    return (
        _dll is not None
        and _pc_hdr_init is not None
        and _pc_hdr_resize is not None
        and _pc_hdr_upload_p010 is not None
        and _pc_hdr_present is not None
        and _pc_hdr_shutdown is not None
    )


class HDRPreviewWidget(QtWidgets.QWidget):
    """Native HDR preview host widget."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._ctx: Optional[_CTX_PTR] = None
        self._frame_w: int = 0
        self._frame_h: int = 0
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NativeWindow, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_PaintOnScreen, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)

    def init_hdr(self, width: int, height: int) -> None:
        if not hdr_passthrough_available():
            return

        width = int(width)
        height = int(height)

        # Recreate the Vulkan context if the video resolution changes between runs.
        if self._ctx and (width != self._frame_w or height != self._frame_h):
            if _pc_hdr_shutdown is not None:
                _pc_hdr_shutdown(self._ctx)
            self._ctx = None

        if self._ctx:
            return

        self._frame_w = width
        self._frame_h = height
        hwnd_c = c_void_p(int(self.winId()))
        if _pc_hdr_init is None:
            return
        ctx = _pc_hdr_init(hwnd_c, self._frame_w, self._frame_h)
        if ctx:
            self._ctx = ctx

    def feed_p010(
        self,
        y_ptr: int,
        uv_ptr: int,
        stride_y_bytes: int,
        stride_uv_bytes: int,
    ) -> None:
        if not self._ctx:
            return

        y_c = ctypes.cast(y_ptr, POINTER(c_uint16))
        uv_c = ctypes.cast(uv_ptr, POINTER(c_uint16))
        _pc_hdr_upload_p010(
            self._ctx,
            y_c,
            uv_c,
            int(stride_y_bytes),
            int(stride_uv_bytes),
        )
        _pc_hdr_present(self._ctx)

    def upload_p010_frame(self, frame: Sequence[object]) -> None:
        """Convenience wrapper that accepts the normalized preview tuple."""

        if frame is None or len(frame) < 6:
            return
        try:
            width = int(frame[0])
            height = int(frame[1])
            y_plane = np.asarray(frame[2])
            uv_plane = np.asarray(frame[3])
            stride_y = int(frame[4]) or int(y_plane.strides[0])
            stride_uv = int(frame[5]) or int(uv_plane.strides[0])
        except Exception:
            return

        if y_plane.ndim != 2 or uv_plane.ndim != 2:
            return

        self.init_hdr(width, height)
        if not self._ctx:
            return

        try:
            y_ptr = int(y_plane.ctypes.data)
            uv_ptr = int(uv_plane.ctypes.data)
        except Exception:
            return

        self.feed_p010(y_ptr, uv_ptr, stride_y, stride_uv)

    def resizeEvent(self, ev: QtGui.QResizeEvent) -> None:
        super().resizeEvent(ev)
        if self._ctx and _pc_hdr_resize is not None:
            _pc_hdr_resize(self._ctx, self.width(), self.height())

    def closeEvent(self, ev: QtGui.QCloseEvent) -> None:
        if self._ctx and _pc_hdr_shutdown is not None:
            _pc_hdr_shutdown(self._ctx)
            self._ctx = None
        super().closeEvent(ev)
