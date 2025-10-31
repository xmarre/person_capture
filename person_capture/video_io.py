#!/usr/bin/env python3
from __future__ import annotations
import subprocess, json, os, sys, math, functools, shutil, threading, re
from dataclasses import dataclass
from typing import Optional, Tuple
from pathlib import Path

import av
try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore
import numpy as np
import logging
import imageio_ffmpeg as iioff

# keep FFmpeg chatter down (must be set before cv2 loads the plugin)
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "quiet")

try:
    # keep OpenCV/FFmpeg chatter down
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except Exception:
        pass
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
    """
    Locate ffprobe. Order: explicit env → imageio bundle → PATH.
    """
    log = logging.getLogger(__name__)
    # 0) Explicit env override
    for env in ("PERSON_CAPTURE_FFPROBE", "FFPROBE", "FFPROBE_BIN"):
        p = os.environ.get(env)
        if p and os.path.exists(p):
            return p
    # 1) Ask imageio-ffmpeg directly
    try:
        p = iioff.get_ffprobe_exe()
        if p and os.path.exists(p):
            return p
    except Exception as e:
        log.warning("imageio-ffmpeg.get_ffprobe_exe() failed: %s", e)

    # 2) Derive sibling from the bundled ffmpeg directory (still within bundle)
    try:
        ffm = iioff.get_ffmpeg_exe()
        if ffm and os.path.exists(ffm):
            d = Path(ffm).parent
            # common exact name
            exact = d / ("ffprobe.exe" if os.name == "nt" else "ffprobe")
            if exact.is_file():
                return str(exact)
            # any ffprobe* in the same folder (imageio cache sometimes uses versioned names)
            for cand in d.glob("ffprobe*"):
                if cand.is_file():
                    return str(cand)
            try:
                # Helpful debug: what’s actually in that dir?
                listing = [p.name for p in d.iterdir()]
                log.info("imageio ffmpeg dir had no ffprobe: %s (contents=%s)", d, listing)
            except Exception:
                log.info("imageio ffmpeg dir had no ffprobe: %s", d)
    except Exception as e:
        log.warning("imageio-ffmpeg ffprobe sibling scan failed: %s", e)
    # 3) PATH fallback
    which = shutil.which("ffprobe.exe" if os.name == "nt" else "ffprobe")
    return which


def _ffprobe_version(path: Optional[str] = None) -> str:
    p = path or _ffprobe_path()
    if not p:
        return "missing"
    try:
        out = subprocess.run(
            [p, "-v", "error", "-hide_banner", "-version"],
            text=True,
            capture_output=True,
            check=False,
        )
        line = (out.stdout or out.stderr or "").splitlines()[0:1]
        return line[0] if line else "unknown"
    except Exception:
        return "unknown"


def _ffmpeg_path() -> Optional[str]:
    """
    Prefer an ffmpeg with HDR-capable filters. Order: explicit env → PATH with filters → imageio bundle.
    """

    def _has_filters(bin_path: str, names: tuple[str, ...]) -> bool:
        try:
            out = subprocess.run(
                [bin_path, "-hide_banner", "-v", "error", "-filters"],
                text=True,
                capture_output=True,
                check=False,
            ).stdout or ""
            present = {line.split()[1] for line in out.splitlines() if line and not line.startswith("-")}
            return all(n in present for n in names)
        except Exception:
            return False

    # 0) Explicit env override
    for env in ("PERSON_CAPTURE_FFMPEG", "FFMPEG", "FFMPEG_BIN"):
        p = os.environ.get(env)
        if p and os.path.exists(p):
            return p
    # 1) PATH candidates with filters
    cand = shutil.which("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    if cand:
        if _has_filters(cand, ("libplacebo",)) or _has_filters(cand, ("zscale", "tonemap")):
            return cand
    # 2) imageio bundle
    try:
        return iioff.get_ffmpeg_exe()
    except Exception:
        return None


def _probe_cache_key(path: str) -> tuple[str, int, int]:
    try:
        st = os.stat(path)
        return (path, int(st.st_mtime), int(st.st_size))
    except Exception:
        return (path, 0, 0)


@functools.lru_cache(maxsize=64)
def _ffprobe_json_cached(key: tuple[str, int, int]) -> dict:
    path, *_ = key
    probe = _ffprobe_path()
    if not probe:
        return {}
    base = [
        probe,
        "-v",
        "error",
        "-show_entries",
        "stream=index,codec_name,codec_tag_string,profile,color_space,matrix_coefficients,color_transfer,color_primaries,color_range,avg_frame_rate,nb_frames,duration,time_base,width,height,pix_fmt,bits_per_raw_sample,side_data_list:format=duration",
        "-of",
        "json",
        "-analyzeduration",
        "200M",
        "-probesize",
        "200M",
    ]
    # Try with v:0, then v, then all streams (no selector)
    for sel in (["-select_streams", "v:0"], ["-select_streams", "v"], []):
        args = base + sel + [path]
        p = subprocess.run(args, text=True, capture_output=True, check=False)
        try:
            meta = json.loads(p.stdout or "{}")
            if meta.get("streams"):
                return meta
        except Exception:
            pass
    return {}


def _ffprobe_json(path: str) -> dict:
    return _ffprobe_json_cached(_probe_cache_key(path))


@functools.lru_cache(maxsize=64)
def _ffprobe_first_frame_side_data_cached(key: tuple[str, int, int]) -> list[dict]:
    """Return side_data from the first few frames for DV/HDR metadata."""
    path, *_ = key
    probe = _ffprobe_path()
    if not probe:
        return []
    p = subprocess.run(
        [
            probe,
            "-v",
            "error",
            "-read_intervals",
            "0%+3",  # probe fewer frames; enough for DV/HDR10+ metadata
            "-select_streams",
            "v",
            "-show_frames",  # ensure per-frame metadata (side_data_list) is emitted
            "-show_entries",
            "frame=stream_index,side_data_list,color_transfer,color_primaries,color_space",
            "-analyzeduration",
            "200M",
            "-probesize",
            "200M",
            "-of",
            "json",
            path,
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    try:
        frames = (json.loads(p.stdout or "{}").get("frames") or [])
    except Exception:
        frames = []
    side: list[dict] = []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        idx = frame.get("stream_index")
        sdl = frame.get("side_data_list")
        if isinstance(sdl, list):
            for item in sdl:
                if isinstance(item, dict):
                    enriched = dict(item)
                    enriched["_stream_index"] = idx
                else:
                    enriched = {"_stream_index": idx, "value": item}
                side.append(enriched)
        # also honor frame-level HDR signals even when side_data_list is absent
        ct = str(frame.get("color_transfer") or "").lower()
        if ct in ("smpte2084", "arib-std-b67", "hlg", "bt2020-10", "bt2020-12"):
            side.append({"_stream_index": idx, "side_data_type": f"frame color_transfer {ct}"})
    return side


def _ffprobe_first_frame_side_data(path: str) -> list[dict]:
    return _ffprobe_first_frame_side_data_cached(_probe_cache_key(path))


def _ffprobe_duration(meta: dict) -> float:
    try:
        s = (meta.get("streams") or [{}])[0]
    except Exception:
        s = {}
    # prefer stream duration, else container duration
    for key in ("duration",):
        v = s.get(key)
        if v not in (None, "", "N/A"):
            try:
                return float(v)
            except Exception:
                pass
    try:
        fmt = meta.get("format") or {}
        v = fmt.get("duration")
        return float(v) if v not in (None, "", "N/A") else 0.0
    except Exception:
        return 0.0


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


def _probe_pixfmt_wh_pyav(path: str) -> tuple[Optional[str], Optional[int], Optional[int]]:
    try:
        ct = av.open(path)
        vs = ct.streams.video[0]
        fmt = getattr(vs.format, "name", None) or getattr(getattr(vs, "codec_context", None), "pix_fmt", None)
        w, h = vs.width, vs.height
        ct.close()
        return fmt, w, h
    except Exception:
        return None, None, None


def _str_lower_nonempty(x) -> str:
    if x in (None, ""):
        return ""

    if hasattr(x, "name") and isinstance(getattr(x, "name"), str):
        s = x.name
    else:
        s = str(x)
        if "." in s:
            s = s.rsplit(".", 1)[-1]

    s = s.strip()
    if not s:
        return ""

    s = s.lower()
    return s if s != "unknown" else ""


def _looks_bt2020(s: str) -> bool:
    s = s.lower()
    return "2020" in s or "bt2020" in s


def _is_10bit_pixfmt(pix: str, bprs: int) -> bool:
    pix_l = (pix or "").lower()
    return any(t in pix_l for t in ("p10", "yuv420p10", "yuv422p10", "yuv444p10", "p012", "p016")) or (bprs >= 10)


def _stream_says_hdr_pyav(stream) -> tuple[bool, str]:
    """Stream-level HDR hints from PyAV (no ffprobe)."""
    cc = getattr(stream, "codec_context", stream)
    trc = _str_lower_nonempty(getattr(cc, "color_trc", None) or getattr(cc, "color_transfer", None))
    prim = _str_lower_nonempty(getattr(cc, "color_primaries", None))
    csp = _str_lower_nonempty(getattr(cc, "colorspace", None) or getattr(cc, "color_space", None))
    pix = _str_lower_nonempty(getattr(stream, "format", None) or getattr(cc, "pix_fmt", None))
    try:
        bprs = int(getattr(cc, "bits_per_raw_sample", 0) or 0)
    except Exception:
        bprs = 0
    tag = _str_lower_nonempty(getattr(cc, "codec_tag_string", None) or getattr(cc, "codec_tag", None))
    meta = {}
    try:
        meta = {str(k).lower(): str(v).lower() for k, v in (getattr(stream, "metadata", {}) or {}).items()}
    except Exception:
        pass
    if tag.startswith("dvh") or "dvhe" in tag:
        return True, f"pyav stream tag={tag}"
    if any("dolby" in (k + v) or "dvhe" in (k + v) or "dvh1" in (k + v) for k, v in meta.items()):
        return True, "pyav stream metadata: dolby-vision"

    if trc in ("smpte2084", "arib-std-b67", "hlg", "bt2020-10", "bt2020-12"):
        return True, f"pyav stream transfer={trc or 'unknown'}"

    if _is_10bit_pixfmt(pix, bprs) and (_looks_bt2020(prim) or _looks_bt2020(csp)):
        return True, f"pyav stream bt2020 {pix or bprs}"

    assume_10 = os.getenv("PERSON_CAPTURE_HDR_ASSUME_10BIT", "0").lower() in ("1", "true", "yes", "on")
    name = _str_lower_nonempty(getattr(cc, "name", None) or getattr(stream, "name", None))
    prof = _str_lower_nonempty(getattr(cc, "profile", None) or getattr(stream, "profile", None))
    if assume_10 and ("hevc" in name or "h265" in name) and ("main 10" in prof or _is_10bit_pixfmt(pix, bprs)):
        return True, "pyav stream forced main10"

    return False, "pyav stream: no hdr flags"


def _frames_say_hdr_pyav(container, stream_index: int) -> tuple[bool, str]:
    """
    Decode a handful of frames and look for frame-level HDR cues:
      - frame.color_trc = smpte2084/hlg
      - frame.side_data contains MasteringDisplayMetadata / ContentLightLevel / HDR dynamic metadata
    """
    try:
        vstream = container.streams.video[stream_index]
    except Exception:
        return False, "pyav frame: no stream"

    max_probe = 12
    count = 0
    for frame in container.decode(vstream):
        count += 1
        trc = _str_lower_nonempty(getattr(frame, "color_trc", None))
        if trc in ("smpte2084", "arib-std-b67", "hlg", "bt2020-10", "bt2020-12"):
            return True, f"pyav frame transfer={trc}"
        try:
            sds = list(getattr(frame, "side_data", []) or [])
        except Exception:
            sds = []
        for sd in sds:
            t = _str_lower_nonempty(getattr(sd, "type", None) or str(sd))
            if any(
                key in t
                for key in (
                    "mastering",
                    "contentlight",
                    "content_light",
                    "dynamic",
                    "hdr",
                    "dolby",
                    "dv",
                )
            ):
                return True, f"pyav frame side_data={t or 'hdr'}"
        if count >= max_probe:
            break
    return False, "pyav frame: no hdr cues"


def _detect_hdr_pyav(path: str) -> tuple[bool, str]:
    """
    ffprobe-free HDR detection using PyAV only.
    Signals:
      - transfer: smpte2084 (PQ), arib-std-b67/hlg
      - Dolby Vision hints via codec tags or stream metadata
      - BT.2020 + >=10-bit pixfmt/bprs as fallback
      - Frame-level HDR side-data / transfer (MasteringDisplayMetadata, ContentLightLevel, etc.)
    """
    try:
        ct = av.open(path)
    except Exception as e:
        return False, f"pyav open failed: {e}"
    try:
        vids = [s for s in ct.streams if getattr(s, "type", None) == "video"] or list(ct.streams.video)
    except Exception:
        vids = list(ct.streams.video) if hasattr(ct.streams, "video") else []
    video_streams = list(vids)
    if not video_streams:
        ct.close()
        return False, "pyav: no video streams"

    for i, s in enumerate(video_streams):
        ok, why = _stream_says_hdr_pyav(s)
        if ok:
            ct.close()
            return True, f"pyav v:{i} {why}"

    try:
        ct.close()
    except Exception:
        pass

    for i in range(len(video_streams)):
        try:
            ct_frame = av.open(path)
        except Exception as e:
            ok, why = False, f"pyav frame open failed: {e}"
        else:
            try:
                ok, why = _frames_say_hdr_pyav(ct_frame, i)
            finally:
                try:
                    ct_frame.close()
                except Exception:
                    pass
        if ok:
            return True, f"pyav v:{i} {why}"

    return False, "pyav: no HDR signal"

# -------------------- Robust fps/total probing (no ffprobe needed) --------------------
def _probe_fps_total_cv2(path: str) -> Tuple[Optional[float], Optional[int]]:
    try:
        cap = cv2.VideoCapture(path)
        f = cap.get(CV_CAP_PROP_FPS)
        n = cap.get(CV_CAP_PROP_FRAME_COUNT)
        cap.release()
        fps = float(f) if f and f > 1e-3 else None
        total = int(n) if n and n > 0 else None
        return fps, total
    except Exception:
        return None, None


def _probe_fps_total_pyav(path: str) -> Tuple[Optional[float], Optional[int]]:
    try:
        ct = av.open(path)
    except Exception:
        return None, None
    try:
        vs = ct.streams.video[0]
    except Exception:
        try:
            ct.close()
        except Exception:
            pass
        return None, None

    # fps
    fps = None
    try:
        r = getattr(vs, "average_rate", None) or getattr(vs, "base_rate", None)
        if r:
            fps = float(r)
    except Exception:
        pass

    # total frames
    total = None
    try:
        if getattr(vs, "frames", 0):
            total = int(vs.frames)
    except Exception:
        pass
    if total is None or total <= 0:
        dur_s = 0.0
        try:
            if getattr(vs, "duration", None) and getattr(vs, "time_base", None):
                dur_s = float(vs.duration * vs.time_base)
            elif getattr(ct, "duration", None) and getattr(ct, "time_base", None):
                dur_s = float(ct.duration * ct.time_base)
        except Exception:
            dur_s = 0.0
        if dur_s > 0 and fps and fps > 0:
            total = int(dur_s * fps + 0.5)
    try:
        ct.close()
    except Exception:
        pass
    return fps, total


def probe_fps_total(path: str, fps_hint: Optional[float] = None) -> Tuple[float, int]:
    """Return ``(fps, total_frames)`` without requiring ffprobe."""

    fps, total = _probe_fps_total_cv2(path)
    if (fps is None or fps <= 0) or (total is None or total <= 0):
        f2, t2 = _probe_fps_total_pyav(path)
        fps = fps if (fps and fps > 0) else f2
        total = total if (total and total > 0) else t2
    if fps is None or fps <= 0:
        fps = float(fps_hint or 30.0)
    if total is None:
        total = 0
    return fps, int(total)


def _detect_hdr(path: str) -> tuple[bool, str]:
    probe = _ffprobe_path()
    log = logging.getLogger(__name__)
    # Extra diagnostics so we can see what imageio-ffmpeg is doing on this machine
    try:
        iio_ver = getattr(iioff, "__version__", "unknown")
        ffm = iioff.get_ffmpeg_exe()
    except Exception:
        iio_ver, ffm = "unknown", None
    log.info("HDR: imageio-ffmpeg=%s ffmpeg=%s", iio_ver, ffm or "None")
    log.info("HDR: ffprobe=%s (%s)", probe or "None", _ffprobe_version(probe))
    if not probe:
        # No ffprobe allowed/available → pure PyAV path
        ok, why = _detect_hdr_pyav(path)
        return ok, f"{why} (no ffprobe)"
    meta = _ffprobe_json(path)
    streams = list(meta.get("streams") or [])
    if not streams:
        # ffprobe present but didn’t find streams → try PyAV before giving up
        ok, why = _detect_hdr_pyav(path)
        if ok:
            return True, f"{why} (ffprobe-empty)"
        return False, "no streams (ffprobe saw none)"
    side_frames_all = _ffprobe_first_frame_side_data_cached(_probe_cache_key(path))
    if not isinstance(side_frames_all, list):
        side_frames_all = []
    # check each video stream; early return on first HDR hit
    for idx, s in enumerate(streams):
        stream_idx = s.get("index")
        if stream_idx is None:
            stream_idx = idx
        t = str(s.get("color_transfer") or "").lower()
        p = str(s.get("color_primaries") or "").lower()
        csp = str(s.get("color_space") or "").lower()
        mc = str(s.get("matrix_coefficients") or "").lower()
        pixfmt = str(s.get("pix_fmt") or "")
        tag = str(s.get("codec_tag_string") or "").lower()
        if tag.startswith("dvh") or "dvhe" in tag:
            return True, f"v:{stream_idx} dolby-vision tag={tag}"
        side_stream = s.get("side_data_list") or []
        if not isinstance(side_stream, list):
            side_stream = [side_stream]
        bits = str(s.get("bits_per_raw_sample") or "").strip()
        is_10bit = any(
            token in pixfmt
            for token in (
                "p10",
                "p12",
                "p14",
                "p16",
                "yuv420p10",
                "yuv420p12",
                "yuv420p16",
            )
        )
        if not is_10bit:
            try:
                is_10bit = int(bits) >= 10
            except Exception:
                is_10bit = False
        if t in ("smpte2084", "arib-std-b67", "hlg", "bt2020-10", "bt2020-12"):
            return True, f"v:{stream_idx} transfer={t}"
        side_frames = [
            sd
            for sd in side_frames_all
            if (
                isinstance(sd, dict)
                and (
                    sd.get("_stream_index") == stream_idx
                    or sd.get("_stream_index") is None
                )
            )
        ]
        side_all = side_stream + side_frames
        for sd in side_all:
            try:
                typ = str(sd.get("side_data_type") or "").lower()
            except AttributeError:
                typ = ""
            if any(
                keyword in typ
                for keyword in (
                    "dolby vision",
                    "dovi",
                    "dovi configuration record",
                    "hdr10+",
                    "mastering display metadata",
                    "content light level",
                    "cta 861-3",
                )
            ):
                return True, f"v:{stream_idx} hdr side-data ({typ})"
        if is_10bit and ("2020" in p or "2020" in csp or "2020" in mc):
            return True, f"v:{stream_idx} bt2020 {pixfmt or bits}"
    return False, "no HDR signal across video streams"


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
            return float(getattr(self, "_w", self._vs.width))
        if prop_id == CV_CAP_PROP_FRAME_HEIGHT:
            return float(getattr(self, "_h", self._vs.height))
        if prop_id == CV_CAP_PROP_FRAME_COUNT:
            total = getattr(self, "_total", 0) or getattr(self, "_nb", 0)
            return float(total)
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
        known_total = getattr(self, "_total", 0) or getattr(self, "_nb", 0)
        if known_total and known_total > 0:
            index = min(max(0, index), int(known_total) - 1)
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
                meta = _ffprobe_json(self._path)
                dur = _ffprobe_duration(meta)
                if dur > 0.0 and self._fps and self._fps > 0.0 and math.isfinite(self._fps):
                    n = int(dur * self._fps + 0.5)
                    self._nb = n
                    self._total = n
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
                f_down = None
                try:
                    maxw = int(os.getenv("PC_DECODE_MAX_W", "0"))
                except Exception:
                    maxw = 0
                if maxw > 0:
                    f_down = g.add("scale", f"w=min(iw\\,{maxw}):h=-2")
                f2 = g.add("format", "bgr24")
                sink = g.add("buffersink")
                src.link_to(f1)
                if f_down is not None:
                    f1.link_to(f_down)
                    f_down.link_to(f2)
                else:
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
    # Respect explicit GUI/backend override only for HDR content: skip PyAV when forcing zscale/scale.
    pref = (os.getenv("PC_FORCE_TONEMAP", "") or "").strip().lower()
    if is_hdr and pref in ("zscale", "scale"):
        log.info(
            "HDR detect: %s → forcing external ffmpeg pipe (%s)",
            reason or "unknown", pref
        )
        fmpeg = _ffmpeg_path()
        if fmpeg:
            return FfmpegPipeReader(path, fmpeg)
        return None
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
        # If ffprobe is missing, recover dimensions and fps via PyAV/OpenCV.
        if (self._w <= 0) or (self._h <= 0) or (self._fps <= 0) or (self._nb <= 0):
            _fmt, w_pyav, h_pyav = _probe_pixfmt_wh_pyav(path)
            if w_pyav and h_pyav:
                if self._w <= 0:
                    self._w = int(w_pyav)
                if self._h <= 0:
                    self._h = int(h_pyav)
            f_pyav, n_pyav = _probe_fps_total_pyav(path)
            if f_pyav and f_pyav > 0 and self._fps <= 0:
                self._fps = float(f_pyav)
            if n_pyav and n_pyav > 0 and self._nb <= 0:
                self._nb = int(n_pyav)
        if ((self._w <= 0) or (self._h <= 0) or (self._fps <= 0) or (self._nb <= 0)) and cv2 is not None:
            try:
                cap = cv2.VideoCapture(path)
                try:
                    if self._w <= 0:
                        self._w = int(cap.get(CV_CAP_PROP_FRAME_WIDTH) or 0)
                    if self._h <= 0:
                        self._h = int(cap.get(CV_CAP_PROP_FRAME_HEIGHT) or 0)
                    if self._fps <= 0:
                        self._fps = float(cap.get(CV_CAP_PROP_FPS) or 0.0)
                    if self._nb <= 0:
                        total = cap.get(CV_CAP_PROP_FRAME_COUNT) or 0
                        if total:
                            self._nb = int(total)
                finally:
                    cap.release()
            except Exception:
                pass
        if self._nb <= 0:
            # estimate from container/stream duration × fps
            meta = meta or {}
            dur = _ffprobe_duration(meta)
            if dur > 0 and self._fps > 0:
                self._nb = int(dur * self._fps + 0.5)
        # Abort early if we still lack geometry
        if self._w <= 0 or self._h <= 0:
            raise RuntimeError("HDR pipe init failed: unknown frame size; supply ffprobe or use PATH ffmpeg")
        # expose both counters; some callers only read _total
        self._total = self._nb
        self._pos = -1
        self._proc = None
        self._arr: Optional[np.ndarray] = None
        self._stderr_thread = None
        self._stderr_tail: list[str] = []
        self._lp_opts: dict[str, bool] = {}
        self._filters = self._list_filters()
        self._use_libplacebo = ("libplacebo" in self._filters)
        self._has_zscale = ("zscale" in self._filters)
        self._has_tonemap = ("tonemap" in self._filters)
        self._has_scale = ("scale" in self._filters)
        self._has_scale_vulkan = ("scale_vulkan" in self._filters)
        if not (self._use_libplacebo or (self._has_zscale and self._has_tonemap) or self._has_scale):
            raise RuntimeError("Bundled ffmpeg lacks libplacebo, zscale+tonemap, and scale")
        # Helpful logging for debugging external ffmpeg selection
        self._log.info("HDR ffmpeg binary: %s", self._ffmpeg)
        self._log.info(
            "HDR filters: libplacebo=%s zscale=%s tonemap=%s scale_vulkan=%s",
            self._use_libplacebo,
            self._has_zscale,
            self._has_tonemap,
            self._has_scale_vulkan,
        )
        self._range_in = _probe_range(path)
        self._transfer, self._primaries = _probe_colors(path)
        # Probe Vulkan availability for libplacebo
        self._range_in_tag = "pc" if (self._range_in or "").lower() in ("pc", "full") else "tv"
        self._has_vulkan = self._probe_vulkan()
        # Probe which libplacebo options exist on this ffmpeg build
        if self._use_libplacebo:
            self._lp_opts = self._probe_libplacebo_opts()
            self._log.debug(
                "libplacebo opts enabled: %s",
                [k for k, v in self._lp_opts.items() if v],
            )
        # If decoder-level downscale is active, adjust output dims to match the filter graph.
        try:
            _maxw_req = int(os.getenv("PC_DECODE_MAX_W", "0"))
        except Exception:
            _maxw_req = 0
        _hdr = (self._transfer or "").lower() in ("smpte2084", "arib-std-b67")
        try:
            _minw_hdr = int(os.getenv("PC_TONEMAP_MIN_W", "960"))
        except Exception:
            _minw_hdr = 960
        _maxw = _maxw_req
        if _hdr and _maxw_req > 0 and _maxw_req < _minw_hdr:
            _maxw = _minw_hdr
            self._log.info("HDR clamp: PC_DECODE_MAX_W=%s raised to PC_TONEMAP_MIN_W=%s", _maxw_req, _minw_hdr)
        # Persist the effective cap so the filter chain doesn't re-read env and diverge.
        # This makes prescan/preview and processing use the exact same width policy.
        self._maxw = int(_maxw)
        # Helpful visibility into what we'll actually use
        self._log.info(
            "Downscale cap: req=%s • hdr=%s • used=%s • scale_vulkan=%s",
            _maxw_req,
            _hdr,
            _maxw,
            self._has_scale_vulkan,
        )
        if _maxw > 0 and self._w > _maxw:
            ratio = float(_maxw) / float(self._w)
            self._w = int(_maxw)
            scaled_h = int(math.floor(self._h * ratio))
            self._h = max(2, scaled_h & ~1)  # even height
        self._frame_bytes_u8 = self._w * self._h * 3
        self._frame_bytes_f32 = self._w * self._h * 3 * 4
        self._pix_fmt = "bgr24"
        # Tunables (env overrides). Defaults align with MPC VR feel.
        self._sdr_nits = float(os.getenv("PC_SDR_NITS", "125"))       # 80–200 typical
        self._tm_desat = float(os.getenv("PC_TM_DESAT", "0.25"))      # 0=keep chroma, 1=desaturate more
        self._tm_param = float(os.getenv("PC_TM_PARAM", "0.40"))      # Mobius curve softness
        self._force_mode = (os.getenv("PC_FORCE_TONEMAP", "") or "").strip().lower()
        if not self._force_mode:
            if os.getenv("PC_FORCE_ZSCALE", ""):
                self._force_mode = "zscale"
            elif os.getenv("PC_FORCE_SCALE", ""):
                self._force_mode = "scale"
        # Track the currently active filter mode so we can fall back if needed.
        self._mode = (
            "libplacebo"
            if self._use_libplacebo
            else ("zscale" if (self._has_zscale and self._has_tonemap) else "scale")
        )
        if self._force_mode == "libplacebo" and self._use_libplacebo:
            self._mode = "libplacebo"
        elif self._force_mode == "zscale" and (self._has_zscale and self._has_tonemap):
            self._mode = "zscale"
        elif self._force_mode == "scale" and self._has_scale:
            self._mode = "scale"
        # Stage-specific fallback flags so we can attempt zscale, then scale.
        self._tried_zscale = False
        self._tried_scale = False
        self._start(0)
        mode = (
            "libplacebo"
            if self._mode == "libplacebo"
            else ("zscale+tonemap" if self._mode == "zscale" else "linear+python-tonemap")
        )
        self._log.info("HDR preview path: bundled ffmpeg pipe (%s)", mode)

    def _list_filters(self) -> set[str]:
        try:
            out = subprocess.run(
                [self._ffmpeg, "-hide_banner", "-v", "error", "-filters"],
                text=True,
                capture_output=True,
                check=False,
            ).stdout or ""
            names = []
            for line in out.splitlines():
                if not line or line.startswith("-"):
                    continue
                flag = line[:4].strip()
                if not flag or any(c not in "TSC.PAR" and c != "." for c in flag):
                    continue
                name = line[4:20].strip()
                if name and name != "=":
                    names.append(name)
            return set(names)
        except Exception:
            return set()

    # --- Fallback helper used if a chain produces no frames (e.g., Vulkan init issues) ---
    def try_fallback_chain(self) -> bool:
        """
        Attempt staged fallback:
        libplacebo → zscale+tonemap → linear scale.
        Returns True if we switched chains; False if no further fallback exists.
        """
        # From libplacebo → zscale (once), then → scale (once)
        if self._mode == "libplacebo":
            def _log_tail() -> None:
                tail = getattr(self, "_stderr_tail", [])[-5:]
                if tail:
                    self._log.warning(
                        "HDR: libplacebo yielded no frames; stderr tail: %s",
                        " | ".join(tail),
                    )

            if (self._has_zscale and self._has_tonemap) and not self._tried_zscale:
                _log_tail()
                self._log.warning("HDR: libplacebo yielded no frames; falling back to zscale+tonemap.")
                self._stop()
                self._mode = "zscale"
                self._tried_zscale = True
                self._start(max(self._pos + 1, 0))
                return True
            if self._has_scale and not self._tried_scale:
                _log_tail()
                self._log.warning("HDR: libplacebo yielded no frames; falling back to linear scale.")
                self._stop()
                self._mode = "scale"
                self._tried_scale = True
                self._start(max(self._pos + 1, 0))
                return True
            return False
        # From zscale → scale (once)
        if self._mode == "zscale":
            if self._has_scale and not self._tried_scale:
                self._log.warning("HDR: zscale+tonemap yielded no frames; falling back to linear scale.")
                self._stop()
                self._mode = "scale"
                self._tried_scale = True
                self._start(max(self._pos + 1, 0))
                return True
            return False
        # Already on scale: no more fallbacks
        return False

    def _probe_libplacebo_opts(self) -> dict[str, bool]:
        """
        Ask ffmpeg which options the libplacebo filter supports.
        Some Windows builds lack target_primaries/target_trc/gamut_mode.
        """
        out = ""
        try:
            r = subprocess.run(
                [self._ffmpeg, "-hide_banner", "-h", "filter=libplacebo"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=3.0,
            )
            out = r.stdout or ""
        except Exception:
            return {}
        # Make a quick lookup table of option names present in help output
        found: dict[str, bool] = {}
        for name in (
            "tonemapping",
            "gamut_mode",
            "target_primaries",
            "target_trc",
            "dither",
            "deband",
        ):
            # match whole words like "  target_primaries  " in the help text
            found[name] = bool(re.search(rf"(^|\s){re.escape(name)}(\s|=)", out))
        # Always assume tonemapping is there; older builds still have it
        if not found.get("tonemapping"):
            found["tonemapping"] = True
        return found

    def _chain(self) -> str:
        maxw = int(getattr(self, "_maxw", 0))
        if self._mode == "libplacebo" and self._use_libplacebo:
            if self._has_vulkan:
                # Normalize HDR signaling + bit-depth BEFORE uploading (some builds balk on odd formats).
                tr = (self._transfer or "").lower()
                lp_opts = ["tonemapping=auto"]
                if self._lp_opts.get("gamut_mode"):
                    lp_opts.append("gamut_mode=clip")
                if self._lp_opts.get("target_primaries"):
                    lp_opts.append("target_primaries=bt709")
                if self._lp_opts.get("target_trc"):
                    lp_opts.append("target_trc=bt709")
                if self._lp_opts.get("dither"):
                    lp_opts.append("dither=auto")
                if self._lp_opts.get("deband"):
                    lp_opts.append("deband=yes")
                lp = "libplacebo=" + ":".join(lp_opts)
                filters: list[str] = []
                # Always expand to full-range before upload so libplacebo/Vulkan gets PC range.
                # Do this with zscale first to avoid range-mismatch EINVAL from libplacebo.
                if tr in ("smpte2084", "arib-std-b67"):
                    filters.append(
                        f"zscale=primaries=bt2020:transfer={tr}:matrix=bt2020nc:rangein={self._range_in_tag}:range=pc"
                    )
                    filters.extend(["format=p010le", "hwupload=extra_hw_frames=8"])
                else:
                    if self._range_in_tag != "pc":
                        filters.append("zscale=rangein=tv:range=pc")
                    filters.extend(["format=nv12", "hwupload=extra_hw_frames=8"])
                filters.append(lp)
                scaled_gpu = False
                if maxw > 0 and self._has_scale_vulkan:
                    # NV12/p010 Vulkan scalers require even widths; clamp odd caps here.
                    gpuw = maxw & ~1
                    if gpuw <= 0:
                        gpuw = 2
                    filters.append(f"scale_vulkan=w=min(iw\\,{gpuw}):h=-2")
                    scaled_gpu = True
                filters.append("hwdownload")

                # Build post-download color pipeline first (to land in bgr24),
                # then (optionally) apply CPU scale so odd widths keep working.
                post = []
                if not (self._lp_opts.get("target_primaries") and self._lp_opts.get("target_trc")):
                    post.extend([
                        "format=gbrpf32le",
                        "zscale=transfer=bt709:primaries=bt709:matrix=bt709:rangein=pc:range=pc:dither=error_diffusion",
                        "format=bgr24",
                    ])
                else:
                    post.append("format=bgr24")
                if maxw > 0 and not scaled_gpu:
                    post.append(f"scale=w=min(iw\\,{maxw}):h=-2")
                post.append("setsar=1")

                filters.extend(post)
                return ",".join(filters)
            else:
                # No Vulkan: try CPU libplacebo once; fallback code will switch to zscale if it yields no frames.
                lp_opts = ["tonemapping=auto"]
                if self._lp_opts.get("gamut_mode"):
                    lp_opts.append("gamut_mode=clip")
                if self._lp_opts.get("target_primaries"):
                    lp_opts.append("target_primaries=bt709")
                if self._lp_opts.get("target_trc"):
                    lp_opts.append("target_trc=bt709")
                if self._lp_opts.get("dither"):
                    lp_opts.append("dither=auto")
                if self._lp_opts.get("deband"):
                    lp_opts.append("deband=yes")
                # Keep any CPU downscale after we’ve converted to bgr24.
                parts: list[str] = []
                # If SDR limited, expand to PC before running CPU libplacebo to keep behavior consistent.
                if self._range_in_tag != "pc":
                    parts.append("zscale=rangein=tv:range=pc")
                parts.append("libplacebo=" + ":".join(lp_opts))
                post = []
                if not (self._lp_opts.get("target_primaries") and self._lp_opts.get("target_trc")):
                    post.extend([
                        "format=gbrpf32le",
                        "zscale=transfer=bt709:primaries=bt709:matrix=bt709:rangein=pc:range=pc:dither=error_diffusion",
                        "format=bgr24",
                    ])
                else:
                    post.append("format=bgr24")
                if maxw > 0:
                    post.append(f"scale=w=min(iw\\,{maxw}):h=-2")
                post.append("setsar=1")
                return ",".join(parts + post)
        if self._mode in ("libplacebo", "zscale") and (self._has_zscale and self._has_tonemap):
            # Full HDR→SDR tonemap tuned for SDR target brightness and optional desaturation.
            s = (
                "zscale=primaries=bt2020:transfer=smpte2084:matrix=bt2020nc,"
                "zscale=transfer=linear:npl=1000,format=gbrpf32le,"
                f"tonemap=tonemap=mobius:param={self._tm_param}:desat={self._tm_desat}:peak={self._sdr_nits},"
                "zscale=transfer=bt709:primaries=bt709:matrix=bt709:dither=error_diffusion,"
                f"zscale=rangein={'pc' if (self._range_in or '').lower() in ('pc','full') else 'tv'}:range=pc"
            )
            if maxw > 0:
                s += f",scale=w=min(iw\\,{maxw}):h=-2"
            s += ",format=bgr24,setsar=1"
            return s
        # Fallback path: decode to float RGB (planar), expand to full-range, tonemap in Python.
        range_in = "pc" if (self._range_in or "").lower() in ("pc", "full") else "tv"
        mat = "bt2020nc" if self._primaries == "bt2020" else "bt709"
        s = f"scale=in_color_matrix={mat}:in_range={range_in}:out_range=pc"
        if maxw > 0:
            s += f",scale=w=min(iw\\,{maxw}):h=-2"
        s += ",format=gbrpf32le,setsar=1"
        return s

    def _start(self, idx: int):
        if self._proc:
            self._stop()
        t = 0.0 if self._fps <= 0 else max(0.0, idx / float(self._fps))
        # If libplacebo was selected but there is no Vulkan, use zscale+tonemap immediately.
        if self._mode == "libplacebo" and self._use_libplacebo and not self._has_vulkan:
            self._log.warning("libplacebo: Vulkan not available; switching to zscale+tonemap.")
            self._mode = "zscale"
        pix_fmt = "bgr24" if self._mode in ("libplacebo", "zscale") else "gbrpf32le"
        # Turn up logging while attempting libplacebo so stderr_tail shows the true cause.
        ll = "warning" if self._mode == "libplacebo" else "error"
        cmd = [self._ffmpeg, "-hide_banner", "-loglevel", ll, "-nostdin"]
        # Ensure a Vulkan device exists for libplacebo
        if self._mode == "libplacebo" and self._use_libplacebo and self._has_vulkan:
            cmd += ["-init_hw_device", "vulkan=pl", "-filter_hw_device", "pl"]
        cmd += [
            "-ss",
            f"{t:.6f}",
            "-i",
            self._path,
            "-map",
            "0:v:0",
            "-vsync",
            "0",
            "-vf",
            self._chain(),
            "-f",
            "rawvideo",
            "-pix_fmt",
            pix_fmt,
            "pipe:1",
        ]
        flags = 0
        if os.name == "nt":
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._log.info(
            "Pipe mode=%s vulkan=%s • out=%dx%d • fmt=%s • t=%.3fs",
            self._mode,
            ("on" if (self._mode == "libplacebo" and self._has_vulkan) else "off"),
            self._w,
            self._h,
            pix_fmt,
            t,
        )
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,  # drain & capture errors for fallback logging
            creationflags=flags,
        )
        self._stderr_tail = []

        def _drain() -> None:
            try:
                if not self._proc or not self._proc.stderr:
                    return
                for line in iter(self._proc.stderr.readline, b""):
                    if not line:
                        break
                    ln = line.decode("utf-8", "ignore").strip()
                    if ln:
                        self._stderr_tail.append(ln)
                        if len(self._stderr_tail) > 50:
                            self._stderr_tail.pop(0)
            except Exception:
                pass

        self._stderr_thread = threading.Thread(target=_drain, daemon=True)
        self._stderr_thread.start()
        self._pos = idx - 1
        self._arr = None
        self._pix_fmt = pix_fmt

    def _probe_vulkan(self) -> bool:
        """Return True if this ffmpeg can init a Vulkan device for filters."""
        try:
            r = subprocess.run(
                [
                    self._ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-init_hw_device",
                    "vulkan=pl",
                    "-filters",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=3.0,
            )
            return r.returncode == 0
        except Exception:
            return False

    def _stop(self):
        # Idempotent; callers may invoke multiple times during fallback.
        try:
            if self._proc:
                self._proc.kill()
        except Exception:
            pass
        try:
            if self._proc and self._proc.stdout:
                self._proc.stdout.close()
        except Exception:
            pass
        try:
            if self._proc and self._proc.stderr:
                self._proc.stderr.close()
        except Exception:
            pass
        try:
            t = getattr(self, "_stderr_thread", None)
            if t and t.is_alive():
                t.join(timeout=0.1)
        except Exception:
            pass
        self._stderr_thread = None
        self._proc = None
        self._arr = None

    # OpenCV-like API
    def get(self, prop: int) -> float:
        if prop == CV_CAP_PROP_FPS:
            return float(self._fps)
        if prop == CV_CAP_PROP_FRAME_COUNT:
            total = getattr(self, "_nb", 0) or getattr(self, "_total", 0)
            return float(total)
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
        if self._pix_fmt == "bgr24":
            buf = self._proc.stdout.read(self._frame_bytes_u8)
            if not buf or len(buf) < self._frame_bytes_u8:
                if self.try_fallback_chain():
                    return self.grab()
                return False
            self._arr = np.frombuffer(buf, dtype=np.uint8).reshape(self._h, self._w, 3)
            self._pos += 1
            return True
        # Linear+python-tonemap path: read float32 RGB (planar G,B,R)
        fbytes = self._frame_bytes_f32
        buf = self._proc.stdout.read(fbytes)
        if not buf or len(buf) < fbytes:
            return False
        planar = np.frombuffer(buf, dtype=np.float32)
        if planar.size != self._w * self._h * 3:
            return False
        planar = planar.reshape(3, self._h, self._w)
        rgb = np.stack((planar[2], planar[0], planar[1]), axis=-1)
        if getattr(self, "_transfer", "") == "arib-std-b67":
            rgb_linear = _eotf_hlg(rgb)
        else:
            rgb_linear = _eotf_pq(rgb)
        self._arr = rgb_linear
        self._pos += 1
        return True

    def retrieve(self):
        arr, self._arr = self._arr, None
        if arr is None:
            return False, None
        if self._pix_fmt == "bgr24":
            return True, arr
        # Python tonemap on linear RGB → BT.709 BGR8
        out = _python_tonemap_to_bgr8(
            arr,
            peak_nits=1000.0,
            target_nits=float(getattr(self, "_sdr_nits", 200.0)),
        )
        return True, out

    def read(self):
        # If the process died prematurely (e.g., libplacebo failed to init),
        # attempt a one-time filter-chain fallback before giving up.
        if self._proc is not None and self._proc.poll() is not None:
            if not self.try_fallback_chain():
                return False, None

        if not self.grab():
            # grab() failed (end-of-stream or short read). If we haven't tried fallback yet,
            # switch to a safer chain once.
            if self.try_fallback_chain():
                if not self.grab():
                    return False, None
            else:
                return False, None
        return self.retrieve()

    def release(self):
        self._stop()

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


# ---------- Pure-Python tonemap (Hable) on linear RGB ----------
def _eotf_pq(v: np.ndarray) -> np.ndarray:
    """Apply ST2084 (PQ) EOTF. Input/Output normalized 0..1 where 1.0≈10,000 nits."""
    m1 = 2610.0 / 16384.0
    m2 = 2523.0 / 32.0
    c1 = 3424.0 / 4096.0
    c2 = 2413.0 / 128.0
    c3 = 2392.0 / 128.0
    v = np.clip(v, 0.0, 1.0)
    vp = np.maximum(np.power(v, 1.0 / m2) - c1, 0.0)
    denom = c2 - c3 * np.power(v, 1.0 / m2)
    denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
    out = np.power(vp / denom, 1.0 / m1)
    return np.clip(out, 0.0, 1.0)


def _eotf_hlg(v: np.ndarray) -> np.ndarray:
    """Apply ITU-R BT.2100 HLG EOTF. Input/Output normalized 0..1 where 1.0≈10,000 nits."""
    a, b, c = 0.17883277, 0.28466892, 0.55991073
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.5, (v * v) / 3.0, (np.exp((v - c) / a) + b) / 12.0)


def _oetf_bt709(v: np.ndarray) -> np.ndarray:
    # Rec.709 OETF
    thr = 0.018
    out = np.where(v < thr, 4.5 * v, 1.099 * np.power(np.clip(v, 0.0, None), 0.45) - 0.099)
    return np.clip(out, 0.0, 1.0)


def _hable_filmic(x: np.ndarray, A=0.15, B=0.50, C=0.10, D=0.20, E=0.02, F=0.30, W=11.2) -> np.ndarray:
    def h(y):
        return ((y * (A * y + C * B) + D * E) / (y * (A * y + B) + D * F)) - E / F

    white = h(W)
    return np.clip(h(x) / white, 0.0, 1.0)


def _python_tonemap_to_bgr8(
    rgb_linear: np.ndarray, peak_nits: float = 1000.0, target_nits: float = 200.0
) -> np.ndarray:
    # rgb_linear is in 0..1 where 1.0≈10,000 nits.
    rgb_linear = np.clip(rgb_linear, 0.0, 1.0)
    L_nits = rgb_linear * 10000.0
    Y = 0.2627 * L_nits[..., 0] + 0.6780 * L_nits[..., 1] + 0.0593 * L_nits[..., 2]
    peak = max(peak_nits, 1e-3)
    Y_t = _hable_filmic(Y / peak) * target_nits
    denom = np.maximum(Y, 1e-6)
    s = (Y_t / denom)[..., None]
    RGB_t = np.clip(L_nits * s, 0.0, target_nits)
    RGB_norm = np.clip(RGB_t / 100.0, 0.0, 1.0)
    RGB_709 = _oetf_bt709(RGB_norm)
    out = (np.clip(RGB_709, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return out[..., ::-1]
