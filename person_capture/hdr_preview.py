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
            # Already have a context with the right size.
            return

        self._frame_w = width
        self._frame_h = height
        hwnd_c = c_void_p(int(self.winId()))
        if _pc_hdr_init is None:
            _log.error("HDR: _pc_hdr_init missing despite hdr_passthrough_available()")
            return

        _log.info(
            "HDR: init ctx with video=%dx%d, widget=%dx%d",
            self._frame_w,
            self._frame_h,
            self.width(),
            self.height(),
        )

        ctx = _pc_hdr_init(hwnd_c, self._frame_w, self._frame_h)
        if not ctx:
            _log.error(
                "HDR: pc_hdr_init() returned NULL for hwnd=%s, size=%dx%d "
                "(Vulkan/HDR init failed in pc_hdr_vulkan.dll)",
                hwnd_c, self._frame_w, self._frame_h,
            )
            return

        _log.info(
            "HDR: pc_hdr_init() OK, ctx=%r, size=%dx%d",
            ctx, self._frame_w, self._frame_h,
        )
        self._ctx = ctx

        # Force one resize so the swapchain matches the current widget size.
        if _pc_hdr_resize is not None:
            w = max(1, self.width())
            h = max(1, self.height())
            _log.info("HDR: initial resize to widget=%dx%d", w, h)
            _pc_hdr_resize(self._ctx, w, h)

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

        # Sanity-check Y plane against the **logical** frame size without
        # treating padded strides as resolution.
        y_h, y_w = y_plane.shape
        if y_h <= 0 or y_w <= 0:
            return
        # Plane must be at least as large as the logical frame; larger is OK
        # when the decoder aligns rows (e.g. 1920×1080 stored as (1080, 2048)).
        if y_h < height or y_w < width:
            _log.error(
                "HDR: Y plane shape %s smaller than expected (%s, %s); dropping frame",
                y_plane.shape,
                height,
                width,
            )
            return
        if y_h != height or y_w != width:
            _log.debug(
                "HDR: using padded Y plane shape %s for logical frame (%s, %s); relying on stride",
                y_plane.shape,
                height,
                width,
            )

        uv_h, uv_w = uv_plane.shape
        expected_uv_h = max(1, height // 2)
        if uv_h < expected_uv_h or uv_w < width:
            _log.error(
                "HDR: UV plane shape %s smaller than expected (%s, %s); dropping frame",
                uv_plane.shape,
                expected_uv_h,
                width,
            )
            return

        if _log.isEnabledFor(logging.DEBUG):
            try:
                y_min = int(np.min(y_plane))
                y_max = int(np.max(y_plane))
                uv_min = int(np.min(uv_plane))
                uv_max = int(np.max(uv_plane))
                _log.debug(
                    "HDR: P010 Y[%s] min/max=%s/%s, UV[%s] min/max=%s/%s, strides=%s/%s",
                    y_plane.shape,
                    y_min,
                    y_max,
                    uv_plane.shape,
                    uv_min,
                    uv_max,
                    stride_y,
                    stride_uv,
                )
                if y_min == 0 and y_max == 0:
                    _log.warning("HDR: Y plane is zero; passthrough may not be streaming P010 data")
                if uv_min == 0 and uv_max == 0:
                    _log.warning("HDR: UV plane is zero; passthrough may not be streaming P010 data")
            except Exception:
                pass

        self.init_hdr(width, height)
        if not self._ctx:
            # Vulkan/HDR path is dead; let the caller fall back to SDR/libplacebo.
            _log.warning(
                "HDR: context not available after init_hdr(%d, %d); "
                "skipping upload and relying on SDR preview.",
                width, height,
            )
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
            w = max(1, self.width())
            h = max(1, self.height())
            _pc_hdr_resize(self._ctx, w, h)

    def closeEvent(self, ev: QtGui.QCloseEvent) -> None:
        if self._ctx and _pc_hdr_shutdown is not None:
            _pc_hdr_shutdown(self._ctx)
            self._ctx = None
        super().closeEvent(ev)

    def has_valid_ctx(self) -> bool:
        """Return True when the Vulkan HDR context is alive."""
        return bool(self._ctx)
