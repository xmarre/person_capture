#!/usr/bin/env python3
from __future__ import annotations
import subprocess, json, os, sys, math
from dataclasses import dataclass
from typing import Optional

import av
import numpy as np
import logging
import imageio_ffmpeg as iioff

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


def _ffprobe_path() -> Optional[str]:
    try:
        return iioff.get_ffprobe_exe()
    except Exception:
        return None


def _ffmpeg_path() -> Optional[str]:
    try:
        return iioff.get_ffmpeg_exe()
    except Exception:
        return None


def _ffprobe_json(path: str) -> dict:
    probe = _ffprobe_path()
    if not probe:
        return {}
    p = subprocess.run(
        [
            probe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=color_space,color_transfer,color_primaries,color_range,avg_frame_rate,nb_frames,duration,time_base,width,height",
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


def _probe_pixfmt_av(path: str) -> tuple[Optional[str], Optional[int], Optional[int]]:
    """Return (pix_fmt, width, height) via PyAV, or (None,None,None) on failure."""
    try:
        ct = av.open(path)
        vs = ct.streams.video[0]
        fmt = getattr(vs.format, "name", None)
        w, h = vs.width, vs.height
        ct.close()
        return fmt, w, h
    except Exception:
        return None, None, None


def _detect_hdr(path: str) -> tuple[bool, str]:
    t, p = _probe_colors(path)
    pixfmt, w, h = _probe_pixfmt_av(path)
    is_10bit = bool(
        pixfmt
        and any(
            k in pixfmt
            for k in (
                "p10",
                "p12",
                "p14",
                "p16",
                "yuv420p10",
                "yuv420p12",
                "yuv420p16",
            )
        )
    )
    if t in ("smpte2084", "arib-std-b67"):
        return True, f"transfer={t}"
    if is_10bit and p == "bt2020":
        return True, f"primaries=bt2020 & {pixfmt}"
    if is_10bit:
        return True, f">=10-bit {pixfmt}"
    return False, "no HDR transfer and no >=10-bit pix_fmt"


def hdr_detect_reason(path: str) -> str:
    try:
        _, reason = _detect_hdr(path)
        return reason
    except Exception as exc:
        return f"error: {exc}"


def _probe_range(path: str) -> str:
    """Return 'limited' or 'full' (default limited when unknown)."""
    meta = _ffprobe_json(path)
    s = (meta.get("streams") or [{}])[0]
    r = str(s.get("color_range") or "").lower()
    return "full" if r in ("pc", "jpeg", "full") else "limited"


def is_hdr_stream(path: str) -> bool:
    ok, _ = _detect_hdr(path)
    return ok


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
        meta_stream: Optional[dict] = None

        def _stream_meta() -> dict:
            nonlocal meta_stream
            if meta_stream is None:
                meta = _ffprobe_json(self._path)
                meta_stream = (meta.get("streams") or [{}])[0]
            return meta_stream

        try:
            if not self._fps or self._fps <= 0 or not math.isfinite(self._fps):
                s = _stream_meta()
                self._fps = _safe_fps(s.get("avg_frame_rate") or "0/1")
                if not self._fps or not math.isfinite(self._fps):
                    self._fps = 30.0
        except Exception:
            pass
        try:
            if not self._total or self._total <= 0:
                s = _stream_meta()
                dur = float(s.get("duration") or 0.0)
                if dur > 0.0 and self._fps and self._fps > 0.0 and math.isfinite(self._fps):
                    self._total = int(dur * self._fps + 0.5)
        except Exception:
            pass
        self._log.info(
            "HDR detect: transfer=%s primaries=%s range=%s",
            self._transfer or "unknown",
            self._primaries or "unknown",
            self._range_in,
        )
        try:
            self._configure_graph(reset=False)
        except Exception:
            self._use_fallback = True
            self._configure_graph(reset=True)
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
    log = logging.getLogger(__name__)
    # Env override
    force = os.getenv("PERSON_CAPTURE_FORCE_HDR", "").lower() in ("1","true","yes")
    if force:
        log.info("HDR detect: forced by PERSON_CAPTURE_FORCE_HDR=1")
        try:
            return AvLibplaceboReader(path)
        except Exception as e:
            log.warning("PyAV/libplacebo unavailable, falling back to bundled ffmpeg pipe: %s", e)
            fmpeg = _ffmpeg_path()
            if fmpeg:
                return FfmpegPipeReader(path, fmpeg)
            return None
    # Robust detect
    is_hdr, reason = _detect_hdr(path)
    if is_hdr:
        log.info("HDR detect: %s → enabling tone-map path", reason)
        try:
            return AvLibplaceboReader(path)
        except Exception as e:
            log.warning("PyAV/libplacebo unavailable, falling back to bundled ffmpeg pipe: %s", e)
            fmpeg = _ffmpeg_path()
            if fmpeg:
                return FfmpegPipeReader(path, fmpeg)
            return None
    else:
        log.info("HDR detect: %s → using OpenCV SDR path", reason)
        return None


# ---------- Minimal external ffmpeg pipe using imageio-ffmpeg binary ----------
class FfmpegPipeReader:
    """HDR→SDR using bundled ffmpeg. Streams bgr24. OpenCV-like API."""

    def __init__(self, path: str, ffmpeg_exe: str):
        self._log = logging.getLogger(__name__)
        self._path = path
        self._ffmpeg = ffmpeg_exe
        meta = _ffprobe_json(path)
        s = (meta.get("streams") or [{}])[0]
        self._w = int(s.get("width") or 0)
        self._h = int(s.get("height") or 0)
        self._fps = _safe_fps(s.get("avg_frame_rate") or "0/1")
        self._nb = int(s.get("nb_frames") or 0)
        if self._nb <= 0:
            # mkv often omits nb_frames; estimate from duration * fps
            try:
                dur = float(s.get("duration") or 0.0)
                if dur > 0 and self._fps > 0:
                    self._nb = int(dur * self._fps + 0.5)
            except Exception:
                pass
        self._frame_bytes = self._w * self._h * 3
        self._pos = -1
        self._proc = None
        self._arr: Optional[np.ndarray] = None
        self._use_libplacebo = self._ffmpeg_has("libplacebo")
        self._range_in = _probe_range(path)
        self._start(0)
        self._log.info(
            "HDR preview path: bundled ffmpeg pipe (%s)",
            "libplacebo" if self._use_libplacebo else "zscale+tonemap",
        )

    def _ffmpeg_has(self, flt: str) -> bool:
        try:
            out = subprocess.run(
                [self._ffmpeg, "-hide_banner", "-v", "error", "-filters"],
                text=True,
                capture_output=True,
                check=False,
            ).stdout or ""
            return flt in out
        except Exception:
            return False

    def _chain(self) -> str:
        if self._use_libplacebo:
            return (
                "libplacebo=tonemapping=auto:target_primaries=bt709:target_trc=bt709:"
                "dither=yes:deband=yes"
            )
        # conservative fallback chain
        return (
            "zscale=primaries=bt2020:transfer=smpte2084:matrix=bt2020nc,"
            "zscale=transfer=linear:npl=1000,format=gbrpf32le,"
            "tonemap=tonemap=mobius:param=0.5:desat=0.5:peak=200,"
            "zscale=transfer=bt709:primaries=bt709:matrix=bt709:dither=error_diffusion,"
            f"zscale=rangein={self._range_in}:range=full,format=bgr24"
        )

    def _start(self, idx: int):
        if self._proc:
            try:
                self._proc.kill()
            except Exception:
                pass
        t = 0.0 if self._fps <= 0 else max(0.0, idx / float(self._fps))
        cmd = [
            self._ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-ss",
            f"{t:.6f}",
            "-i",
            self._path,
            "-vf",
            self._chain(),
            "-pix_fmt",
            "bgr24",
            "-f",
            "rawvideo",
            "-",
        ]
        flags = 0
        if os.name == "nt":
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            creationflags=flags,
        )
        self._pos = idx - 1
        self._arr = None

    # OpenCV-like API
    def get(self, prop: int) -> float:
        if prop == CV_CAP_PROP_FPS:
            return float(self._fps)
        if prop == CV_CAP_PROP_FRAME_COUNT:
            return float(self._nb)
        if prop == CV_CAP_PROP_POS_FRAMES:
            return float(max(self._pos, 0))
        if prop == CV_CAP_PROP_FRAME_WIDTH:
            return float(self._w)
        if prop == CV_CAP_PROP_FRAME_HEIGHT:
            return float(self._h)
        return 0.0

    def set(self, prop: int, value: float) -> bool:
        if prop == CV_CAP_PROP_POS_FRAMES:
            self._start(max(0, int(value)))
            return True
        return False

    def grab(self) -> bool:
        if not self._proc or not self._proc.stdout:
            return False
        buf = self._proc.stdout.read(self._frame_bytes)
        if not buf or len(buf) < self._frame_bytes:
            return False
        self._arr = np.frombuffer(buf, dtype=np.uint8).reshape(self._h, self._w, 3)
        self._pos += 1
        return True

    def retrieve(self):
        arr, self._arr = self._arr, None
        return (arr is not None), arr

    def read(self):
        return (False, None) if not self.grab() else self.retrieve()

    def release(self):
        try:
            if self._proc:
                self._proc.kill()
        except Exception:
            pass
        self._proc = None

    def __del__(self):
        self.release()

    def isOpened(self) -> bool:
        return self._proc is not None and self._proc.poll() is None


def _safe_fps(frac: str) -> float:
    try:
        a, b = frac.split("/")
        a = float(a)
        b = float(b) if float(b) != 0 else 1.0
        return a / b
    except Exception:
        return 30.0
