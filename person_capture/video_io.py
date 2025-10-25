#!/usr/bin/env python3
from __future__ import annotations
import subprocess, json
from dataclasses import dataclass
from typing import Optional

import av
import numpy as np
import logging

try:
    import cv2  # type: ignore
    CV_CAP_PROP_POS_FRAMES = int(cv2.CAP_PROP_POS_FRAMES)
    CV_CAP_PROP_FPS = int(cv2.CAP_PROP_FPS)
    CV_CAP_PROP_FRAME_COUNT = int(cv2.CAP_PROP_FRAME_COUNT)
    CV_CAP_PROP_FRAME_WIDTH = int(cv2.CAP_PROP_FRAME_WIDTH)
    CV_CAP_PROP_FRAME_HEIGHT = int(cv2.CAP_PROP_FRAME_HEIGHT)
except Exception:
    CV_CAP_PROP_POS_FRAMES = 1
    CV_CAP_PROP_FPS = 5
    CV_CAP_PROP_FRAME_COUNT = 7
    CV_CAP_PROP_FRAME_WIDTH = 3
    CV_CAP_PROP_FRAME_HEIGHT = 4


def _ffprobe_json(path: str) -> dict:
    p = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=color_space,color_transfer,color_primaries,color_range,avg_frame_rate,nb_frames,duration,time_base",
            "-of",
            "json",
            path,
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    try:
        return json.loads(p.stdout or "{}")
    except Exception:
        return {}


def _probe_colors(path: str) -> tuple[str, str]:
    meta = _ffprobe_json(path)
    s = (meta.get("streams") or [{}])[0]
    return str(s.get("color_transfer") or "").lower(), str(s.get("color_primaries") or "").lower()


def _probe_range(path: str) -> str:
    """Return 'limited' or 'full' (default limited when unknown)."""
    meta = _ffprobe_json(path)
    s = (meta.get("streams") or [{}])[0]
    r = str(s.get("color_range") or "").lower()
    return "full" if r in ("pc", "jpeg", "full") else "limited"


def is_hdr_stream(path: str) -> bool:
    meta = _ffprobe_json(path)
    try:
        s = (meta.get("streams") or [])[0]
    except Exception:
        return False
    t = str(s.get("color_transfer") or "").lower()
    p_ = str(s.get("color_primaries") or "").lower()
    return (t in ("smpte2084", "arib-std-b67")) or (p_ == "bt2020")


def _fps_from_stream(vs) -> float:
    try:
        num = getattr(vs.average_rate, "numerator", None)
        den = getattr(vs.average_rate, "denominator", None)
        if num and den:
            return float(num) / float(den)
    except Exception:
        pass
    try:
        return float(vs.rate)
    except Exception:
        return 30.0


def _nb_frames_guess(ct, vs) -> int:
    try:
        n = int(vs.frames)
        if n > 0:
            return n
    except Exception:
        pass
    try:
        n = int(getattr(vs, "nb_frames", 0))
        if n > 0:
            return n
    except Exception:
        pass
    try:
        dur = float(vs.duration * vs.time_base)
        fps = _fps_from_stream(vs)
        if dur > 0 and fps > 0:
            return int(dur * fps + 0.5)
    except Exception:
        pass
    return 0


def _index_from_pts(vs, pts: Optional[int], fps: float) -> Optional[int]:
    """Map stream pts to an integer frame index using stream time_base and fps."""
    if pts is None:
        return None
    tb = vs.time_base
    return int(round((pts * tb) * fps))


@dataclass
class _FrameBuf:
    arr: Optional[np.ndarray] = None


class _BaseAvReader:
    def __init__(self, path: str):
        self._path = path
        self._ct = av.open(path)
        self._vs = self._ct.streams.video[0]
        self._fps = _fps_from_stream(self._vs)
        self._total = _nb_frames_guess(self._ct, self._vs)
        self._buf = _FrameBuf()
        self._pos = 0
        self._graph = None
        self._src = None
        self._sink = None
        self._drop_until: Optional[int] = None

    def isOpened(self) -> bool:
        return True

    def release(self):
        try:
            if self._ct:
                self._ct.close()
        except Exception:
            pass

    def get(self, prop_id: int) -> float:
        if prop_id == CV_CAP_PROP_FPS:
            return float(self._fps)
        if prop_id == CV_CAP_PROP_FRAME_WIDTH:
            return float(self._vs.width)
        if prop_id == CV_CAP_PROP_FRAME_HEIGHT:
            return float(self._vs.height)
        if prop_id == CV_CAP_PROP_FRAME_COUNT:
            return float(self._total)
        if prop_id == CV_CAP_PROP_POS_FRAMES:
            return float(self._pos)
        return 0.0

    def set(self, prop_id: int, value: float) -> bool:
        if prop_id == CV_CAP_PROP_POS_FRAMES:
            self._seek_to(int(value))
            return True
        return False

    def _seek_to(self, index: int):
        # Clamp to valid range if total is known
        if self._total and self._total > 0:
            index = min(max(0, index), self._total - 1)
        else:
            index = max(0, index)
        tb = self._vs.time_base
        ts = int(round((index / max(self._fps, 1e-6)) / tb))
        try:
            self._ct.seek(ts, stream=self._vs, any_frame=False)
        except Exception:
            try:
                self._ct.seek(ts, any_frame=True)
            except Exception:
                pass
        self._drop_until = index
        self._pos = max(0, index - 1)
        try:
            if self._graph:
                self._configure_graph(reset=True)
        except Exception:
            pass

    def _configure_graph(self, reset: bool = False):
        raise NotImplementedError

    def grab(self) -> bool:
        try:
            for pkt in self._ct.demux(self._vs):
                for frame in pkt.decode():
                    if self._drop_until is not None:
                        # Use PTS when available, else fall back to sequential counting.
                        idx = _index_from_pts(self._vs, frame.pts, self._fps)
                        if idx is None:
                            idx = self._pos + 1
                        # Skip until we reach the requested target index.
                        if idx < self._drop_until:
                            # Keep the counter in sync while skipping frames with no timestamps.
                            self._pos = idx
                            continue
                        self._drop_until = None
                    if self._graph is None or self._src is None:
                        self._configure_graph(reset=False)
                    self._graph.push(frame)
                    for of in self._graph:
                        arr = of.to_ndarray(format="bgr24")
                        idx = _index_from_pts(self._vs, getattr(of, "pts", None), self._fps)
                        if idx is None:
                            idx = self._pos + 1  # sequential fallback for filters without PTS
                        self._pos = idx
                        self._buf.arr = arr
                        return True
            return False
        except Exception:
            return False

    def retrieve(self) -> tuple[bool, Optional[np.ndarray]]:
        arr = self._buf.arr
        self._buf.arr = None
        return (arr is not None), arr

    # OpenCV-compat convenience
    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        if not self.grab():
            return False, None
        return self.retrieve()


class AvLibplaceboReader(_BaseAvReader):
    """HDR->SDR via vf_libplacebo on GPU; falls back to CPU zscale+tonemap."""

    def __init__(self, path: str):
        super().__init__(path)
        self._use_fallback = False
        self._log = logging.getLogger(__name__)
        self._transfer, self._primaries = _probe_colors(path)
        self._range_in = _probe_range(path)
        try:
            self._configure_graph(reset=False)
        except Exception:
            self._use_fallback = True
            self._configure_graph(reset=False)
        self._log.info(
            "HDR preview path: %s",
            "libplacebo" if not self._use_fallback else "CPU fallback",
        )
        self._log.info(
            "Source transfer=%s primaries=%s",
            self._transfer or "unknown",
            self._primaries or "unknown",
        )
        self._log.info("Input range=%s (expanding to full for preview)", self._range_in)

    def _configure_graph(self, reset: bool = False):
        if reset or self._graph is None:
            g = av.filter.Graph()
            tb = self._vs.time_base
            sar = self._vs.sample_aspect_ratio
            pix_fmt = self._vs.format.name
            src = g.add(
                "buffer",
                f"video_size={self._vs.width}x{self._vs.height}:"
                f"pix_fmt={pix_fmt}:"
                f"time_base={tb.numerator}/{tb.denominator}:"
                f"pixel_aspect={(sar.numerator if sar else 1)}/{(sar.denominator if sar else 1)}",
            )
            if not self._use_fallback:
                # Let libplacebo tone-map into RGB; then convert directly to BGR24.
                # This avoids YUV range ambiguities entirely.
                f1 = g.add(
                    "libplacebo",
                    "tonemapping=auto:gamut_mode=perceptual:target_trc=bt709:target_primaries=bt709:deband=yes:dither=yes",
                )
                f2 = g.add("format", "bgr24")
                sink = g.add("buffersink")
                src.link_to(f1)
                f1.link_to(f2)
                f2.link_to(sink)
            else:
                src_tr = "arib-std-b67" if self._transfer == "arib-std-b67" else "smpte2084"
                src_pr = "bt2020" if self._primaries in ("bt2020", "smpte432") else self._primaries or "bt2020"
                f1 = g.add("zscale", f"primaries={src_pr}:transfer={src_tr}:matrix=bt2020nc")
                f2 = g.add("zscale", "transfer=linear:npl=1000")
                f3 = g.add("format", "gbrpf32le")
                f4 = g.add("tonemap", "tonemap=mobius:param=0.5:desat=0.5:peak=200")
                f5 = g.add(
                    "zscale",
                    "transfer=bt709:primaries=bt709:matrix=bt709:dither=error_diffusion",
                )
                # Expand to full-range before RGB conversion
                f6 = g.add("zscale", f"rangein={self._range_in}:range=full")
                f7 = g.add("format", "bgr24")
                sink = g.add("buffersink")
                src.link_to(f1)
                f1.link_to(f2)
                f2.link_to(f3)
                f3.link_to(f4)
                f4.link_to(f5)
                f5.link_to(f6)
                f6.link_to(f7)
                f7.link_to(sink)
            g.configure()
            self._graph = g
            self._src = src
            self._sink = sink

        path = (
            "libplacebo→BGR24"
            if not self._use_fallback
            else "zscale→tonemap(mobius)→zscale(709+dither)→zscale(range expand)→BGR24"
        )
        self._log.info(
            "HDR pipeline: %s | src: tr=%s prim=%s range=%s | out: BT.709 full-range BGR",
            path,
            self._transfer or "unknown",
            self._primaries or "unknown",
            self._range_in,
        )


def open_video_with_tonemap(path: str):
    """Return a cv2.VideoCapture-like object for HDR streams, else None."""

    if not is_hdr_stream(path):
        return None
    return AvLibplaceboReader(path)
