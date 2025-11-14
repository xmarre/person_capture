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
        except Exception as e:
            # No CPU fallback: fail loudly if libplacebo cannot be configured.
            raise RuntimeError(f"HDR preview libplacebo graph failed: {e!r}")
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

    # --- Windows long-path helper ---
    @staticmethod
    def _win_longpath(p: str) -> str:
        if os.name != "nt":
            return p
        # Normalize and add \\?\ or \\?\UNC\ for network paths
        try:
            ap = os.path.abspath(p)
        except Exception:
            ap = p
        if ap.startswith("\\\\?\\"):
            return ap
        if ap.startswith("\\\\"):
            return "\\\\?\\UNC" + ap[1:]
        return "\\\\?\\" + ap

    def __init__(self, path: str, ffmpeg_exe: str):
        self._log = logging.getLogger(__name__)
        self._path = path
        self._ffmpeg = ffmpeg_exe
        # Use long-path for ffprobe too, or the probe can fail independently of ffmpeg open.
        try:
            if os.name == "nt" and os.getenv("PC_WIN_LONGPATH", "1").lower() not in ("0", "false", "no"):
                _probe_path = self._win_longpath(path)
            else:
                _probe_path = path
        except Exception:
            _probe_path = path
        meta = _ffprobe_json(_probe_path)
        s = (meta.get("streams") or [{}])[0]
        self._w = int(s.get("width") or 0)
        self._h = int(s.get("height") or 0)
        self._fps = _safe_fps(s.get("avg_frame_rate") or "0/1")
        self._nb = int(s.get("nb_frames") or 0)
        self._src_pixfmt = _str_lower_nonempty(s.get("pix_fmt"))
        try:
            self._bits_per_raw_sample = int(s.get("bits_per_raw_sample") or 0)
        except Exception:
            self._bits_per_raw_sample = 0
        # Hints for robust LP(Vulkan) init and logging
        self._lp_tm_alt = False        # toggle bt2390 ↔ bt.2390 if needed
        self._lp_surf_alt = False      # toggle yuv420p10le/nv12 ↔ p010le/yuv420p if needed
        self._last_cmdline = None
        # Tonemap algo fallback chain: bt.2390 → mobius → hable → clip
        # Start at index 0, but allow env override while debugging.
        self._tm_algos = ["bt.2390", "mobius", "hable", "clip"]
        self._tm_ai = int(os.getenv("PC_LP_TM_INDEX", "0"))
        # Tonemap selection: 'auto' enables fallback rotation via _tm_ai
        self._tm_algo = (os.getenv("PC_TM_ALGO", "auto") or "auto")
        # Strict libplacebo: never allow CPU/zscale fallback and avoid multi-probe churn.
        self._strict_lp = (os.getenv("PC_LP_STRICT", "1").lower() not in ("0", "false", "no"))
        # Pipe pixel format: keep pipe light to avoid ENOMEM on pipe:1
        # Default nv12 (1.5 B/px). Change to bgr24 if you explicitly want RGB pipe.
        self._pipe_pixfmt = (os.getenv("PC_PIPE_PIXFMT", "nv12") or "nv12").lower()
        # Input/demux probe budgets
        try:
            self._probe_m = max(1, int(os.getenv("PC_FF_PROBE_M", "48")))
        except Exception:
            self._probe_m = 48
        try:
            self._analyze_m = max(1, int(os.getenv("PC_FF_ANALYZE_M", "48")))
        except Exception:
            self._analyze_m = 48
        try:
            self._max_probe_pkts = max(64, int(os.getenv("PC_FF_MAX_PROBE_PKTS", "1024")))
        except Exception:
            self._max_probe_pkts = 1024
        self._reduced_probe = False
        # Identify as HDR pipe and always “open” for cv2-style guards.
        self._is_hdr_pipe = True
        self.isOpened = lambda: True
        # If libplacebo lacks out_format/out_pfmt, we insert explicit hwdownload+format
        # for readback (tonemap/colorspace remain on GPU). Default ON to avoid regressions.
        self._allow_dl_fallback = (
            os.getenv("PC_LP_ALLOW_DL_FALLBACK", "1").lower() not in ("0", "false", "no")
        )

        # ---- derived sizes for pipe reads ----
        self._frame_bytes_bgr24 = lambda w, h: (w * h * 3)
        self._frame_bytes_nv12 = lambda w, h: (w * h * 3) // 2
        self._pipe_frame_bytes = (
            self._frame_bytes_nv12(self._w, self._h)
            if self._pipe_pixfmt == "nv12"
            else self._frame_bytes_bgr24(self._w, self._h)
        )
        # Optional override if a chosen out_format fails; toggled by fallback logic.
        self._sw_fmt_override: Optional[str] = None

        # --- Vulkan readback sw-format for hwdownload (Vulkan accepts RGBA-family only) ---
        # Canonical pipeline: CPU → format=p010le → hwupload → libplacebo(Vulkan)
        # → hwdownload → format={bgra,rgba} → (optional) format=nv12/bgr24 for the pipe.
        _dl_default = "bgra" if os.name == "nt" else "rgba"
        self._dl_sw = (os.getenv("PC_LP_DL_SW", _dl_default) or _dl_default).strip().lower()
        if self._dl_sw not in ("rgba", "bgra"):
            self._dl_sw = _dl_default
        # track if we’ve already flipped once (rgba <-> bgra) on hwdownload failures
        self._dl_sw_flipped = False

        # --- quoting helper for libplacebo string enums (bt.2390/mobius/hable/clip) ---
        # Some ffmpeg builds require quotes, otherwise "bt2390" is parsed as an expression → EINVAL.
        def _q(s: str) -> str:
            # Always quote tonemapping enums for libplacebo; some builds try to eval unquoted tokens.
            s = (s or "").strip()
            if not s or s[0] in "'\"":
                return s
            return "'" + s.replace("'", r"\'") + "'"

        self._q = _q
        # Vulkan device pinning (e.g. "0"); if unset, ffmpeg picks default enumerated device.
        self._vk_device = (os.getenv("PC_LP_VK_DEVICE") or "").strip()
        self._vulkan_index: Optional[int] = None
        # When a filter hardware device is provided, deriving a new device inside hwupload can
        # cause a second context with different permissions/ICD → allocations fail on Windows.
        # Default off; enable explicitly for experiments.
        self._hwupload_derive = (
            os.getenv("PC_LP_HWUPLOAD_DERIVE", "0").lower() not in ("0", "false", "no")
        )
        # Vulkan probing modes. Pinned device FIRST to avoid device-churn OOM at init.
        self._vk_probe_modes = [
            {"derive": False, "bind": True,  "dev": "0", "noinit": False},  # pinned :0, single context
            {"derive": True,  "bind": True,  "dev": "0", "noinit": False},  # pinned + derive hwupload
            {"derive": False, "bind": True,  "dev": "",  "noinit": False},  # init default GPU
            # fallback/diagnostic modes only if not strict:
            {"derive": True,  "bind": False, "dev": "",  "noinit": True},   # derive-only (no global init)
            {"derive": False, "bind": False, "dev": "",  "noinit": True},   # fully no-init
        ]
        self._vk_probe_i = 0
        self._vk_bind = True
        self._vk_noinit = False
        # Under strict LP, disable the churny noinit/derive fallbacks entirely.
        if self._strict_lp:
            self._vk_probe_modes = [
                {"derive": False, "bind": True, "dev": "0", "noinit": False}
            ]
            # Also force a concrete device index for the first attempt.
            if not self._vk_device:
                self._vk_device = "0"
        # GPU queue / memory controls
        self._extra_hw_frames = max(1, int(os.getenv("PC_LP_EXTRA_FRAMES", "1")))
        # staged memory relief (0=none, 1=queue=1, 2=queue=2, 3=cap 2560, 4=cap 1920, 5=cap 1280)
        self._mem_relief_stage = 0
        self._fallback_hops = 0
        self._fallback_hops_max = int(
            os.getenv("PC_LP_MAX_HOPS", str(self._calc_fallback_budget()))
        )
        # If ffprobe is missing, recover dimensions and fps via PyAV/OpenCV.
        if (self._w <= 0) or (self._h <= 0) or (self._fps <= 0) or (self._nb <= 0):
            _fmt, w_pyav, h_pyav = _probe_pixfmt_wh_pyav(path)
            if w_pyav and h_pyav:
                if self._w <= 0:
                    self._w = int(w_pyav)
                if self._h <= 0:
                    self._h = int(h_pyav)
            if _fmt:
                self._src_pixfmt = _str_lower_nonempty(_fmt)
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
        self._pipe_buf: Optional[bytes] = None
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
        # Cache a normalized tag for range decisions in filters
        self._range_in_tag = "pc" if (self._range_in or "").lower() in ("pc", "full") else "tv"
        # Probe Vulkan availability for libplacebo
        self._has_vulkan = self._use_libplacebo and self._has_scale_vulkan and self._probe_vulkan()
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
        self._frame_bytes_u8 = self._w * self._h * 3
        self._frame_bytes_f32 = self._w * self._h * 3 * 4
        self._apply_cap_dims()
        self._pix_fmt = "bgr24"
        # Tunables (env overrides). Defaults align with MPC VR feel.
        # Strict LP mode = no CPU/zscale fallback. We'll try one minimal LP chain retry, then error.
        self._lp_minimal = False
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
        if self._strict_lp:
            if not self._use_libplacebo:
                raise RuntimeError(
                    "Strict libplacebo mode is enabled but ffmpeg/libplacebo is unavailable"
                )
            self._mode = "libplacebo"
        else:
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
        # Stage-specific fallback flags so we can attempt zscale, then scale (non-strict mode).
        self._tried_zscale = False
        self._tried_scale = False
        self._start(0)
        mode = (
            "libplacebo"
            if self._mode == "libplacebo"
            else ("zscale+tonemap" if self._mode == "zscale" else "linear+python-tonemap")
        )
        self._log.info("HDR preview path: bundled ffmpeg pipe (%s)", mode)

    def _calc_fallback_budget(self) -> int:
        n = len(getattr(self, "_vk_probe_modes", []))         # Vulkan probe modes
        n += 5                                                # mem-relief stages (queue=1, queue=2, 2560, 1920, 1280)
        n += 1                                                # CPU diagnostic probe (disabled in strict mode)
        n += 1                                                # 2390 alias flip
        n += 1                                                # surface alt
        n += 1                                                # minimal LP chain
        n += max(0, len(getattr(self, "_tm_algos", [])) - 1)  # algo rotations
        # removed: hwdownload/hwmap paths are no longer used
        # n += max(0, len(getattr(self, "_dl_fmts", [])) - 1)
        # n += 1
        n += 2                                                # zscale + scale fallbacks
        return n + 4                                          # headroom

    # ---- Vulkan probe mode helpers ----
    def _apply_vk_mode(self, m: dict) -> None:
        """Apply a single probe-mode dict to current flags."""
        self._hwupload_derive = bool(m.get("derive", False))
        self._vk_bind = bool(m.get("bind", False))
        self._vk_noinit = bool(m.get("noinit", False))
        if "dev" in m:
            dev = m.get("dev", "")
            if dev or m.get("force_dev", False) or not getattr(self, "_vk_device", ""):
                self._vk_device = dev or ""

    def _apply_cap_dims(self) -> None:
        maxw = int(getattr(self, "_maxw", 0))
        if maxw > 0 and self._w > maxw:
            ratio = float(maxw) / float(self._w or 1)
            self._w = int(maxw)
            self._w &= ~1  # NV12 requires even width
            self._h = max(2, int(math.floor(self._h * ratio)) & ~1)
            self._frame_bytes_u8 = self._w * self._h * 3
            self._frame_bytes_f32 = self._w * self._h * 3 * 4
            self._pipe_frame_bytes = (
                self._frame_bytes_nv12(self._w, self._h)
                if self._pipe_pixfmt == "nv12"
                else self._frame_bytes_bgr24(self._w, self._h)
            )

    # ---- libplacebo capability helpers ----
    def _ff_input_path(self) -> str:
        return self._win_longpath(self._path) if (os.name == "nt" and os.getenv("PC_WIN_LONGPATH", "1").lower() not in ("0","false","no")) else self._path

    def _lp_supports_any(self, *names: str) -> str | None:
        """Return the first supported option name from names, or None."""
        opts = getattr(self, "_lp_opts", None) or {}
        for n in names:
            if opts.get(n):
                return n
        return None

    def _lp_add_colorspace_args(self, lp_args: list[str]) -> None:
        """
        Append BT.709 + full-range args using whichever libplacebo option names
        this ffmpeg build supports (modern vs legacy).
        """
        # Single modern umbrella first if present
        if self._lp_supports_any("colorspace"):
            lp_args.append("colorspace=bt709")
        else:
            p = self._lp_supports_any("target_primaries", "color_primaries")
            t = self._lp_supports_any("target_trc", "color_trc")
            if p:
                lp_args.append(f"{p}=bt709")
            if t:
                lp_args.append(f"{t}=bt709")
        if self._lp_supports_any("range"):
            lp_args.append("range=full")
        if self._lp_supports_any("gamut_mode"):
            lp_args.append("gamut_mode=clip")

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

    def _supports_fps_mode(self) -> bool:
        """Return True if this ffmpeg understands -fps_mode (output option)."""
        cached = getattr(self, "_fps_mode_cap", None)
        if cached is not None:
            return bool(cached)
        result = False
        try:
            out = subprocess.run(
                [self._ffmpeg, "-h"],
                text=True,
                capture_output=True,
                check=False,
            ).stdout or ""
            if "fps_mode" in out:
                result = True
            else:
                out = subprocess.run(
                    [self._ffmpeg, "-h", "full"],
                    text=True,
                    capture_output=True,
                    check=False,
                ).stdout or ""
                result = "fps_mode" in out
        except Exception:
            result = False
        setattr(self, "_fps_mode_cap", result)
        return result

    def _apply_current_vk_mode(self) -> None:
        if not self._vk_probe_modes:
            return
        if self._vk_probe_i >= len(self._vk_probe_modes):
            self._vk_probe_i = len(self._vk_probe_modes) - 1
        m = self._vk_probe_modes[self._vk_probe_i]
        self._apply_vk_mode(m)
        dev = str(self._vk_device or "").strip()
        self._log.info(
            "LP(Vulkan): priming mode derive=%s bind=%s noinit=%s dev=%s",
            "True" if m.get("derive", False) else "False",
            "True" if self._vk_bind else "False",
            "True" if self._vk_noinit else "False",
            (dev if dev else "<auto>")
        )

    # --- Fallback helper used if a chain produces no frames (e.g., Vulkan init issues) ---
    def try_fallback_chain(self) -> bool:
        """
        Attempt staged fallback.
        In non-strict mode, also try a one-shot diagnostic (CPU libplacebo) to
        distinguish Vulkan/ICD issues from general libplacebo failures.
        In strict LP mode stay on libplacebo: vary algo alias, surface, then minimal, then alternative algos.
        Non-strict mode continues to fall back to zscale and linear scale.
        Returns True if we switched chains; False if no further fallback exists.
        """
        if self._fallback_hops >= getattr(self, "_fallback_hops_max", 12):
            self._log.error("LP fallback exhausted after %s hops.", self._fallback_hops)
            return False

        def _restart(idx: int) -> None:
            self._fallback_hops += 1
            self._start(idx)

        # Inspect stderr for OOM-type failures first and relieve memory while staying on LP/Vulkan.
        def _stderr_contains(s: str) -> bool:
            tail = " | ".join(getattr(self, "_stderr_tail", [])[-50:]).lower()
            return s.lower() in tail

        # --- Vulkan readback recovery (no hwmap juggling, just RGBA↔BGRA) ---
        # If explicit hwdownload fails with “Invalid output format <fmt> for hwframe download”
        # or similar, try the other RGBA-family format once while staying on Vulkan.
        if self._mode == "libplacebo" and (
            (
                _stderr_contains("invalid output format")
                and _stderr_contains("hwdownload")
            )
            or _stderr_contains("filter format does not support hardware pixel formats")
        ):
            if not getattr(self, "_dl_sw_flipped", False):
                prev = self._dl_sw
                self._dl_sw = "bgra" if prev == "rgba" else "rgba"
                self._dl_sw_flipped = True
                self._log.warning(
                    "LP(Vulkan): hwdownload rejected %s → trying %s",
                    prev,
                    self._dl_sw,
                )
                _restart(max(self._pos + 1, 0))
                return True

            # Both RGBA and BGRA failed → in strict mode abort, otherwise fall back to CPU tonemap.
            msg = "libplacebo(Vulkan): no supported RGBA-family readback format (hwdownload)."
            if self._strict_lp:
                # You asked for strict: no CPU fallback, so fail loudly here.
                raise RuntimeError(msg)

            self._log.warning("%s Falling back to zscale/scale CPU tonemapping.", msg)
            self._stop()
            self._mode = "zscale" if (self._has_zscale and self._has_tonemap) else "scale"
            self._fallback_hops = 0
            _restart(max(self._pos + 1, 0))
            return True

        # If libplacebo refuses the chosen software output pixfmt, flip between nv12 and bgra.
        if self._mode == "libplacebo" and (
            _stderr_contains("out_format")
            or _stderr_contains("pixel format")
            or _stderr_contains("not a suitable output")
        ):
            cur = (self._sw_fmt_override or os.getenv("PC_LP_SW_FMT", "auto")).strip().lower() or "auto"
            nxt = "bgra" if cur in ("auto", "nv12") else "nv12"
            self._sw_fmt_override = nxt
            self._log.warning("LP(Vulkan): out_format refusal → switching to %s", nxt)
            _restart(max(self._pos + 1, 0))
            return True

        if self._mode == "libplacebo":
            # Normalize common “no frame yet” hard failures
            def _is_open_or_pipe_oom() -> bool:
                return (
                    (
                        _stderr_contains("error opening output file pipe:1")
                        or _stderr_contains("error opening output files")
                    )
                    and (
                        _stderr_contains("cannot allocate memory")
                        or _stderr_contains("error initializing filters")
                    )
                )

            # Input-open ENOMEM: shrink probe/analyze/packets and retry, then force long-path.
            _open_err = (
                _stderr_contains("error opening input file")
                or _stderr_contains("error opening input files")
            )
            if _open_err and _stderr_contains("cannot allocate memory"):
                if not getattr(self, "_reduced_probe", False):
                    self._reduced_probe = True
                    old_p, old_a = self._probe_m, self._analyze_m
                    old_pk = getattr(self, "_max_probe_pkts", 1024)
                    self._probe_m = max(4, old_p // 3)
                    self._analyze_m = max(4, old_a // 3)
                    self._max_probe_pkts = max(64, old_pk // 2)
                    self._log.warning(
                        "ffmpeg: input open ENOMEM → reduce probe/analyze/packets "
                        "%sM→%sM / %sM→%sM / %spkts→%spkts",
                        old_p,
                        self._probe_m,
                        old_a,
                        self._analyze_m,
                        old_pk,
                        self._max_probe_pkts,
                    )
                    _restart(max(self._pos + 1, 0))
                    return True
                if os.name == "nt" and not getattr(self, "_forced_longpath", False):
                    self._forced_longpath = True
                    self._log.warning("ffmpeg: input open ENOMEM → retry with Windows long-path prefix")
                    _restart(max(self._pos + 1, 0))
                    return True

            # Pipe ENOMEM before first frame: tighten pipe format/bandwidth while staying on LP/Vulkan.
            if _stderr_contains("error opening output file pipe:1") and _stderr_contains("cannot allocate memory"):
                if getattr(self, "_pipe_tightened", False) is not True:
                    self._pipe_tightened = True
                    if self._pipe_pixfmt != "nv12":
                        self._pipe_pixfmt = "nv12"
                        self._pipe_frame_bytes = self._frame_bytes_nv12(self._w, self._h)
                        self._log.warning("ffmpeg: pipe ENOMEM → forcing nv12 pipe to cut bandwidth")
                        _restart(max(self._pos + 1, 0))
                        return True
                if getattr(self, "_mem_relief_stage", 0) < 6:
                    self._mem_relief_stage = 6
                    new_w = max(640, int(self._w // 2))
                    self._maxw = min(self._w, new_w)
                    self._apply_cap_dims()
                    self._log.warning("ffmpeg: pipe ENOMEM → halve GPU output width to %d", self._w)
                    _restart(max(self._pos + 1, 0))
                    return True

            # Classify common faults
            mem_fault = any(_stderr_contains(k) for k in (
                "cannot allocate memory",
                "out of memory",
                "not enough memory resources",
                "std::bad_alloc",
                "Device creation failed: -12",        # VK_ERROR_OUT_OF_DEVICE_MEMORY/VK mem fail
                "Failed to set value 'vulkan=",        # init_hw_device path
            ))
            arg_fault = (
                _stderr_contains("error reinitializing filters")
                or _stderr_contains("return code -22")
                or _stderr_contains("invalid argument")
            )
            if (mem_fault or arg_fault) and self._use_libplacebo and self._has_vulkan:
                # First: advance Vulkan probe mode (derive/bind/device) before scaling caps.
                modes = getattr(self, "_vk_probe_modes", [])
                if self._vk_probe_i < len(modes) - 1:
                    self._vk_probe_i += 1
                    self._apply_current_vk_mode()
                    m = modes[self._vk_probe_i]
                    self._log.warning(
                        "LP(Vulkan): fault → trying mode derive=%s bind=%s noinit=%s dev=%s",
                        "True" if m.get("derive", False) else "False",
                        "True" if m.get("bind", False) else "False",
                        "True" if m.get("noinit", False) else "False",
                        m.get("dev") or "<auto>",
                    )
                    _restart(max(self._pos + 1, 0))
                    return True
                # If the error explicitly names init_hw_device, force immediate no-init for the next hop.
                if _stderr_contains("init_hw_device") or _stderr_contains("Failed to set value 'vulkan="):
                    self._apply_vk_mode({"derive": True, "bind": False, "noinit": True, "dev": "", "force_dev": True})
                    self._log.warning("LP(Vulkan): init_hw_device error → forcing NO-INIT + derive")
                    _restart(max(self._pos + 1, 0)); return True
                # Stage 0: shrink GPU queue to 1
                if self._mem_relief_stage < 1 and self._extra_hw_frames > 1:
                    self._mem_relief_stage = 1
                    old = self._extra_hw_frames
                    self._extra_hw_frames = 1
                    self._log.warning("LP(Vulkan): memory relief • extra_hw_frames %s→%s", old, self._extra_hw_frames)
                    _restart(max(self._pos + 1, 0))
                    return True
                # Stage 1: shrink GPU queue to 2
                if self._mem_relief_stage < 2 and self._extra_hw_frames > 2:
                    self._mem_relief_stage = 2
                    old = self._extra_hw_frames
                    self._extra_hw_frames = 2
                    self._log.warning("LP(Vulkan): memory relief • extra_hw_frames %s→%s", old, self._extra_hw_frames)
                    _restart(max(self._pos + 1, 0))
                    return True
                # Stage 2: cap width to 2560 on GPU
                if self._mem_relief_stage < 3 and (getattr(self, "_maxw", 0) == 0 or getattr(self, "_maxw", 0) > 2560):
                    self._mem_relief_stage = 3
                    self._maxw = 2560
                    self._log.warning("LP(Vulkan): memory relief • apply GPU downscale cap to w=2560")
                    self._apply_cap_dims()
                    _restart(max(self._pos + 1, 0))
                    return True
                # Stage 3: cap width to 1920
                if self._mem_relief_stage < 4 and getattr(self, "_maxw", 0) > 1920:
                    self._mem_relief_stage = 4
                    self._maxw = 1920
                    self._log.warning("LP(Vulkan): memory relief • apply GPU downscale cap to w=1920")
                    self._apply_cap_dims()
                    _restart(max(self._pos + 1, 0))
                    return True
                # Stage 4: cap width to 1280
                if self._mem_relief_stage < 5 and getattr(self, "_maxw", 0) > 1280:
                    self._mem_relief_stage = 5
                    self._maxw = 1280
                    self._log.warning("LP(Vulkan): memory relief • apply GPU downscale cap to w=1280")
                    self._apply_cap_dims()
                    _restart(max(self._pos + 1, 0))
                    return True

                # If this is a memory-type fault and we've exhausted GPU-side mitigations,
                # skip tonemap algo rotations and go straight to CPU diag (once) or zscale.
                if mem_fault:
                    if not self._lp_surf_alt:
                        self._lp_surf_alt = True
                        self._log.warning("libplacebo: mem fault → trying alternate upload surface.")
                        _restart(max(self._pos + 1, 0))
                        return True
                    if not getattr(self, "_lp_minimal", False):
                        self._lp_minimal = True
                        self._log.warning("libplacebo: mem fault → trying MINIMAL LP chain.")
                        _restart(max(self._pos + 1, 0))
                        return True
                    if self._strict_lp:
                        err = " | ".join(getattr(self, "_stderr_tail", [])[-15:]) or "n/a"
                        raise RuntimeError(
                            f"libplacebo(Vulkan) produced no frames; strict mode forbids CPU fallback. tail: {err}"
                        )
                    if not getattr(self, "_force_cpu_lp_once", False):
                        self._force_cpu_lp_once = True
                        self._log.warning("libplacebo: mem fault persists → forcing one CPU LP diagnostic.")
                        self._has_vulkan = False
                        _restart(max(self._pos + 1, 0))
                        return True
                    if (self._has_zscale and self._has_tonemap) and not self._tried_zscale:
                        self._log.warning("HDR: LP mem fault persists → falling back to zscale+tonemap.")
                        self._stop()
                        self._mode = "zscale"
                        self._fallback_hops = 0
                        self._tried_zscale = True
                        _restart(max(self._pos + 1, 0))
                        return True
                    return False

        if self._mode == "libplacebo":
            def _log_tail(prefix: str) -> None:
                tail = getattr(self, "_stderr_tail", [])[-5:]
                if tail:
                    self._log.warning("%s stderr tail: %s", prefix, " | ".join(tail))

            if self._use_libplacebo:
                if self._has_vulkan:
                    if _stderr_contains("Error reinitializing filters") or _stderr_contains("return code -22"):
                        if self._strict_lp:
                            self._log.warning(
                                "libplacebo: strict mode active → skipping CPU diagnostic probe. "
                                "Set PC_LP_STRICT=0 to allow a one-shot CPU libplacebo test."
                            )
                        elif not getattr(self, "_force_cpu_lp_once", False):
                            self._force_cpu_lp_once = True
                            self._log.warning("libplacebo: forcing CPU path once to diagnose Vulkan issues.")
                            self._has_vulkan = False
                            _restart(max(self._pos + 1, 0))
                            return True
                    base_algo = (self._tm_algo or "").strip().lower()
                    if not base_algo:
                        try:
                            base_algo = (self._tm_algos[self._tm_ai] or "").lower()
                        except Exception:
                            base_algo = ""
                    if base_algo in {"bt2390", "bt_2390", "bt.2390"} and not self._lp_tm_alt:
                        _log_tail("libplacebo emitted no frames;")
                        self._lp_tm_alt = True
                        self._log.warning(
                            "libplacebo produced no frames; retrying with alternate tonemap alias."
                        )
                        _restart(max(self._pos + 1, 0))
                        return True
                    if not self._lp_surf_alt:
                        _log_tail("libplacebo emitted no frames;")
                        self._lp_surf_alt = True
                        self._log.warning(
                            "libplacebo produced no frames; retrying with alternate upload surface."
                        )
                        _restart(max(self._pos + 1, 0))
                        return True
                    if not getattr(self, "_lp_minimal", False):
                        _log_tail("libplacebo emitted no frames;")
                        self._log.warning("libplacebo produced no frames; retrying with MINIMAL LP chain.")
                        self._lp_minimal = True
                        _restart(max(self._pos + 1, 0))
                        return True
                    if self._tm_ai + 1 < len(self._tm_algos):
                        self._tm_ai += 1
                        self._lp_minimal = False
                        self._lp_tm_alt = False
                        self._lp_surf_alt = False
                        self._log.warning(
                            "libplacebo produced no frames; retrying with different tonemap algo: %s",
                            self._tm_algos[self._tm_ai],
                        )
                        _restart(max(self._pos + 1, 0))
                        return True
                if not getattr(self, "_lp_minimal", False):
                    _log_tail("libplacebo emitted no frames;")
                    self._log.warning("libplacebo produced no frames; retrying with MINIMAL LP chain.")
                    self._lp_minimal = True
                    _restart(max(self._pos + 1, 0))
                    return True
            if self._strict_lp:
                err = " | ".join(getattr(self, "_stderr_tail", [])[-15:]) or "n/a"
                raise RuntimeError(
                    f"libplacebo(Vulkan) produced no frames; strict mode forbids CPU fallback. tail: {err}"
                )
            if (self._has_zscale and self._has_tonemap) and not self._tried_zscale:
                _log_tail("HDR: libplacebo yielded no frames;")
                self._log.warning("HDR: libplacebo yielded no frames; falling back to zscale+tonemap.")
                self._stop()
                self._mode = "zscale"
                self._fallback_hops = 0
                self._tried_zscale = True
                _restart(max(self._pos + 1, 0))
                return True
            if self._has_scale and not self._tried_scale:
                _log_tail("HDR: libplacebo yielded no frames;")
                self._log.warning("HDR: libplacebo yielded no frames; falling back to linear scale.")
                self._stop()
                self._mode = "scale"
                self._fallback_hops = 0
                self._tried_scale = True
                _restart(max(self._pos + 1, 0))
                return True
            return False
        if self._mode == "zscale":
            if self._has_scale and not self._tried_scale:
                self._log.warning("HDR: zscale+tonemap yielded no frames; falling back to linear scale.")
                self._stop()
                self._mode = "scale"
                self._fallback_hops = 0
                self._tried_scale = True
                _restart(max(self._pos + 1, 0))
                return True
            return False
        return False

    def _probe_libplacebo_opts(self) -> dict[str, bool]:
        """
        Ask ffmpeg which libplacebo options are available and cache the result.
        We detect both modern and legacy color knobs, dithering, readback aliases,
        and the presence of scale_vulkan.
        """
        # Probe once per process
        cache = getattr(self, "_lp_probe_cache", None)
        if cache is not None:
            return cache
        opts: dict[str, bool] = {}
        ff = getattr(self, "_ffmpeg", None) or shutil.which("ffmpeg") or "ffmpeg"
        try:
            # Example-compatible across many builds; prints option list for vf=libplacebo
            out = subprocess.check_output(
                [ff, "-hide_banner", "-h", "filter=libplacebo"],
                stderr=subprocess.STDOUT,
                universal_newlines=True,
            )
        except Exception:
            out = ""
        low = out.lower()
        # Color knobs (modern & legacy)
        opts["colorspace"]       = (" colorspace " in low) or (" colorspace=" in low)
        opts["color_primaries"]  = (" color_primaries " in low) or (" color_primaries=" in low)
        opts["color_trc"]        = (" color_trc " in low) or (" color_trc=" in low)
        opts["range"]            = (" range " in low) or (" range=" in low)
        opts["target_primaries"] = (" target_primaries " in low) or (" target_primaries=" in low)
        opts["target_trc"]       = (" target_trc " in low) or (" target_trc=" in low)
        # Dither/dithering variants
        opts["dithering"] = (" dithering " in low) or (" dithering=" in low)
        opts["dither"]    = (" dither " in low) or (" dither=" in low)
        # CPU readback aliases (name varies by build)
        opts["out_format"] = (" out_format " in low) or (" out_format=" in low)
        opts["out_pfmt"]   = (" out_pfmt " in low) or (" out_pfmt=" in low)
        # Scale-vulkan availability (optional)
        try:
            flt = subprocess.check_output(
                [ff, "-hide_banner", "-filters"], stderr=subprocess.STDOUT, universal_newlines=True
            ).lower()
        except Exception:
            flt = ""
        opts["scale_vulkan"] = " scale_vulkan " in flt
        self._lp_probe_cache = opts
        return opts

    def _tm_value(self) -> str:
        """
        Resolve the libplacebo tonemapping algorithm.
        - 'auto' (default): use fallback list driven by _tm_ai.
        - explicit algo: honor it (except alias handling for 2390).
        - if explicit is 2390 and we've advanced _tm_ai, switch to the indexed fallback.
        """
        algo = (getattr(self, "_tm_algo", "") or "").strip().lower()
        # auto → use the staged list
        if algo in ("", "auto"):
            try:
                cand = self._tm_algos[self._tm_ai]
            except Exception:
                cand = "bt.2390"
            if cand in {"bt2390", "bt_2390", "bt.2390"}:
                return "bt.2390" if not self._lp_tm_alt else "bt2390"
            return cand
        # explicit 2390 → allow alias & staged fallback when _tm_ai advanced
        if algo in {"bt2390", "bt_2390", "bt.2390"}:
            if self._tm_ai > 0:
                try:
                    cand = self._tm_algos[self._tm_ai]
                except Exception:
                    cand = None
                else:
                    if cand in {"bt2390", "bt_2390", "bt.2390"}:
                        return "bt.2390" if not self._lp_tm_alt else "bt2390"
                    return cand
            return "bt.2390" if not self._lp_tm_alt else "bt2390"
        # any other explicit algo
        return algo

    def _choose_surf(self) -> str:
        """Pick upload surface for hwupload→Vulkan."""
        src10 = _is_10bit_pixfmt(getattr(self, "_src_pixfmt", ""), getattr(self, "_bits_per_raw_sample", 0))
        # Prefer p010le for 10-bit uploads on Vulkan; many Windows builds expect this when hwuploading.
        if self._has_vulkan:
            if not self._lp_surf_alt:
                return "p010le" if src10 else "nv12"
            return "yuv420p10le" if src10 else "yuv420p"
        # CPU path swaps back to the legacy ordering.
        if not self._lp_surf_alt:
            return "yuv420p10le" if src10 else "nv12"
        return "p010le" if src10 else "yuv420p"

    def _vk_init_args(self) -> list[str]:
        """Build -init_hw_device / -filter_hw_device according to probe mode."""
        if getattr(self, "_vk_noinit", False):
            self._log.info("libplacebo: Vulkan no-init mode • relying on derive_device path")
            return []
        dev = getattr(self, "_vk_device", "") or ""
        if not dev:
            idx = getattr(self, "_vulkan_index", None)
            try:
                dev = str(int(idx))
            except (TypeError, ValueError):
                dev = "0"
        tag = "vk"
        init = f"vulkan={tag}:{dev}"
        # Optional: reduce driver allocations
        if (os.getenv("PC_VK_DISABLE_RBA", "0").lower() not in ("0", "false", "no")):
            init += ":disable_robust_buffer_access=1"
        args = ["-init_hw_device", init]
        if getattr(self, "_vk_bind", True):
            args += ["-filter_hw_device", tag]
        return args

    def _chain(self) -> str:
        maxw = int(getattr(self, "_maxw", 0))
        tail_fmt = self._pipe_pixfmt if self._pipe_pixfmt in ("bgr24", "nv12") else "bgr24"
        lp_out_fmt = os.getenv("PC_LP_OUT_FMT", "gbrp10le").strip().lower() or "gbrp10le"
        # Choose software output pixfmt that libplacebo should produce itself.
        # GPU tonemaps; vf=libplacebo performs the readback. Internal override beats env.
        sw_pref = (self._sw_fmt_override or os.getenv("PC_LP_SW_FMT", "auto")).strip().lower() or "auto"
        out_sw = ("nv12" if tail_fmt == "nv12" else "bgra") if sw_pref == "auto" else sw_pref
        # guard against unsupported values
        _safe_sw = {"nv12", "bgra", "bgr0", "rgb0", "rgba"}
        if out_sw not in _safe_sw:
            _auto = "nv12" if tail_fmt == "nv12" else "bgra"
            self._log.warning("PC_LP_SW_FMT=%s unsupported for libplacebo readback; using %s",
                              sw_pref, _auto)
            out_sw = _auto
        # hwdownload-compatible software formats (Vulkan): RGBA-family only
        dl_sw = self._dl_sw
        if dl_sw not in ("rgba", "bgra"):
            dl_sw = "bgra" if os.name == "nt" else "rgba"
        # if pipe wants nv12, we’ll read back to RGBA-family, then format→nv12 on CPU
        if self._mode == "libplacebo" and self._use_libplacebo:
            # ensure libplacebo capabilities are available when building args
            self._lp_opts = getattr(self, "_lp_opts", None) or self._probe_libplacebo_opts()
            # prefer libplacebo's own CPU readback; alias can be out_format or out_pfmt depending on build
            outfmt_key = "out_format" if self._lp_opts.get("out_format") else (
                "out_pfmt" if self._lp_opts.get("out_pfmt") else None
            )
            # If neither key exists, insert explicit hwdownload+format for readback (allowed by default).
            need_explicit_download = (outfmt_key is None)
            if self._has_vulkan:
                surf = self._choose_surf()
                _tm = self._q(self._tm_value())
                if getattr(self, "_lp_minimal", False):
                    # minimal = tonemap + colorspace/range on-GPU, then emit software frames
                    lp_args = [f"tonemapping={_tm}"]
                    self._lp_add_colorspace_args(lp_args)
                    if outfmt_key:
                        lp_args.append(f"{outfmt_key}={out_sw}")
                    filters = [
                        f"format={surf}",
                        (
                            f"hwupload=derive_device=1:extra_hw_frames={self._extra_hw_frames}"
                            if self._hwupload_derive
                            else f"hwupload=extra_hw_frames={self._extra_hw_frames}"
                        ),
                        # optional GPU downscale before readback
                        *(
                            [f"scale_vulkan=w=min(iw\\,{maxw & ~1}):h=-2"]
                            if (maxw > 0 and self._has_scale_vulkan)
                            else []
                        ),
                        "libplacebo=" + ":".join(lp_args),
                    ]
                    if need_explicit_download:
                        if not self._allow_dl_fallback and self._strict_lp:
                            raise RuntimeError("libplacebo lacks out_format/out_pfmt and PC_LP_ALLOW_DL_FALLBACK=0.")
                        # Vulkan hwdownload → RGBA-family only
                        filters += ["hwdownload", f"format={dl_sw}"]
                        if tail_fmt == "nv12":
                            filters.append("format=nv12")
                    if maxw > 0 and not self._has_scale_vulkan:
                        filters.append(f"scale=w=min(iw\\,{maxw}):h=-2")
                    # ensure pipe pixfmt
                    if need_explicit_download and tail_fmt not in ("nv12", dl_sw):
                        filters.append(f"format={tail_fmt}")
                    else:
                        if tail_fmt != out_sw:
                            filters.append(f"format={tail_fmt}")
                    filters.append("setsar=1")
                    return ",".join(filters)
                # Vulkan path: do all colorspace/range in libplacebo to avoid huge CPU float buffers.
                lp_args = [f"tonemapping={_tm}"]
                if self._lp_opts.get("dithering"):
                    lp_args.append("dithering=ordered")
                elif self._lp_opts.get("dither"):
                    lp_args.append("dither=yes")
                # Add BT.709/full-range using whatever names this build supports.
                self._lp_add_colorspace_args(lp_args)
                if outfmt_key:
                    lp_args.append(f"{outfmt_key}={out_sw}")
                filters = [
                    f"format={surf}",
                    (
                        f"hwupload=derive_device=1:extra_hw_frames={self._extra_hw_frames}"
                        if self._hwupload_derive
                        else f"hwupload=extra_hw_frames={self._extra_hw_frames}"
                    ),
                    *(
                        [
                            f"scale_vulkan=w=min(iw\\,{max(maxw & ~1, 2)}):h=-2"
                        ]
                        if (maxw > 0 and self._has_scale_vulkan)
                        else []
                    ),
                    "libplacebo=" + ":".join(lp_args),
                ]
                if need_explicit_download:
                    if not self._allow_dl_fallback and self._strict_lp:
                        raise RuntimeError("libplacebo lacks out_format/out_pfmt and PC_LP_ALLOW_DL_FALLBACK=0.")
                    # Vulkan hwdownload → RGBA-family only
                    filters += ["hwdownload", f"format={dl_sw}"]
                    if tail_fmt == "nv12":
                        filters.append("format=nv12")
                if maxw > 0 and not self._has_scale_vulkan:
                    filters.append(f"scale=w=min(iw\\,{maxw}):h=-2")
                # ensure pipe pixfmt
                if need_explicit_download and tail_fmt not in ("nv12", dl_sw):
                    filters.append(f"format={tail_fmt}")
                else:
                    if tail_fmt != out_sw:
                        filters.append(f"format={tail_fmt}")
                filters.append("setsar=1")
                return ",".join(filters)
            if self._strict_lp:
                raise RuntimeError(
                    "Strict libplacebo mode is enabled but Vulkan/libplacebo is unavailable in ffmpeg."
                )
            _tm = self._tm_value()
            if getattr(self, "_lp_minimal", False):
                # minimal = tonemap + colorspace/range on CPU, then emit bgr24
                lp_args = [f"tonemapping={self._q(_tm)}"]
                self._lp_add_colorspace_args(lp_args)
                parts = ["libplacebo=" + ":".join(lp_args), f"format={lp_out_fmt}"]
                if maxw > 0:
                    parts.append(f"scale=w=min(iw\\,{maxw}):h=-2")
                if tail_fmt == "nv12":
                    parts.append("format=nv12")
                elif tail_fmt != lp_out_fmt:
                    parts.append(f"format={tail_fmt}")
                parts.append("setsar=1")
                return ",".join(parts)
            lp_opts = [f"tonemapping={self._q(self._tm_value())}"]
            self._lp_add_colorspace_args(lp_opts)
            if self._lp_opts.get("dithering"):
                lp_opts.append("dithering=ordered")
            parts: list[str] = []
            if self._range_in_tag != "pc":
                parts.append("zscale=rangein=tv:range=pc")
            parts.append("libplacebo=" + ":".join(lp_opts))
            if self._lp_opts.get("colorspace") and self._lp_opts.get("color_trc") and self._lp_opts.get("range"):
                post = [f"format={lp_out_fmt}"]
            else:
                post = [
                    "format=gbrpf32le",
                    "zscale=transfer=bt709:primaries=bt709:matrix=bt709:rangein=pc:range=pc:dither=error_diffusion",
                    f"format={lp_out_fmt}",
                ]
            parts.extend(post)
            if maxw > 0:
                parts.append(f"scale=w=min(iw\\,{maxw}):h=-2")
            if tail_fmt == "nv12":
                parts.append("format=nv12")
            elif tail_fmt != lp_out_fmt:
                parts.append(f"format={tail_fmt}")
            parts.append("setsar=1")
            return ",".join(parts)
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
            s += f",format={tail_fmt},setsar=1"
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
        # Apply the very first probe mode before constructing the command.
        if self._mode == "libplacebo" and self._use_libplacebo and self._has_vulkan:
            self._apply_current_vk_mode()
        t = 0.0 if self._fps <= 0 else max(0.0, idx / float(self._fps))
        # If libplacebo was selected but there is no Vulkan, handle according to strict mode.
        if self._mode == "libplacebo" and self._use_libplacebo and not self._has_vulkan:
            if self._strict_lp:
                raise RuntimeError(
                    "Strict libplacebo mode requires Vulkan-capable ffmpeg build."
                )
            # Keep libplacebo active so CPU-only builds can still tonemap; just note the missing Vulkan path.
            self._log.warning("libplacebo: Vulkan not available; using CPU libplacebo chain.")
        pix_fmt = self._pipe_pixfmt if self._mode in ("libplacebo", "zscale") else "gbrpf32le"
        # Turn up logging while attempting libplacebo so stderr_tail shows the true cause.
        ll = "info" if self._mode == "libplacebo" else "error"
        cmd = [
            self._ffmpeg,
            "-hide_banner",
            "-loglevel", ll,
            "-nostdin",
            "-ignore_unknown",
            # smaller demux-side footprints
            "-fflags",
            "+genpts",
            "-analyzeduration",
            f"{self._analyze_m}M",
            "-probesize",
            f"{self._probe_m}M",
            # cap the number of packets probes scan to keep allocations in check
            "-max_probe_packets",
            str(self._max_probe_pkts),
            "-err_detect",
            "ignore_err",
            "-threads",
            "1",
        ]
        # Required for hwupload/scale_vulkan to bind to a Vulkan context.
        if self._mode == "libplacebo" and self._use_libplacebo and self._has_vulkan:
            vk_args = self._vk_init_args()
            cmd += vk_args
            if self._vk_device:
                self._log.info("libplacebo: using Vulkan device index :%s", self._vk_device)
        # Only include -ss for t>0. Some HEVC streams + libplacebo + "-ss 0" never emit a frame.
        if t > 1e-6:
            cmd += ["-ss", f"{t:.6f}"]
        vf = self._chain()
        if self._mode == "libplacebo" and self._use_libplacebo:
            self._log.info("LP vf: %s", vf)
        # Plain file input; no '-safe 0' (that option is for certain demuxers only).
        cmd += ["-i", self._ff_input_path()]
        # keep only the primary video; drop audio/subs/data/attachments
        # also be robust if streams are missing via '?'.
        cmd += ["-map", "0:v:0", "-an", "-sn", "-dn", "-map", "-0:t?", "-map", "-0:s?", "-map", "-0:d?"]
        cmd += ["-map_metadata", "-1", "-map_chapters", "-1", "-vf", vf]
        # Keep filter graph single-threaded to bound buffer queues at creation time.
        cmd += ["-filter_threads", os.getenv("PC_FF_FILTER_THREADS", "1")]
        try:
            fps_passthrough = self._supports_fps_mode()
        except Exception:
            fps_passthrough = False
        if fps_passthrough:
            cmd += ["-fps_mode", "passthrough"]
        else:
            cmd += ["-vsync", "0"]
        cmd += [
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
            "Pipe mode=%s vulkan=%s • out=%dx%d • fmt=%s • t=%s",
            self._mode,
            ("on" if (self._mode == "libplacebo" and self._has_vulkan) else "off"),
            self._w,
            self._h,
            pix_fmt,
            (f"{t:.3f}s" if t > 0 else "start"),
        )
        try:
            import shlex

            self._last_cmdline = " ".join(shlex.quote(part) for part in cmd)
            self._log.debug("LP cmd: %s", self._last_cmdline)
        except Exception:
            self._last_cmdline = None
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
                        if len(self._stderr_tail) > 200:
                            self._stderr_tail.pop(0)
            except Exception:
                pass

        self._stderr_thread = threading.Thread(target=_drain, daemon=True)
        self._stderr_thread.start()
        self._pos = idx - 1
        self._arr = None
        self._pipe_buf = None
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
        self._pipe_buf = None

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

    def _nv12_to_bgr(self, raw: bytes) -> np.ndarray | None:
        try:
            import numpy as _np
        except Exception:
            return None
        h, w = int(self._h), int(self._w)
        need = (w * h * 3) // 2
        if len(raw) < need:
            return None
        y_size = w * h
        y = _np.frombuffer(raw, dtype=_np.uint8, count=y_size).reshape(h, w).astype(_np.float32)
        uv = _np.frombuffer(raw, dtype=_np.uint8, offset=y_size, count=y_size // 2).reshape(h // 2, w).astype(_np.float32)
        u = uv[:, 0::2]
        v = uv[:, 1::2]
        u = _np.repeat(_np.repeat(u, 2, axis=0), 2, axis=1)
        v = _np.repeat(_np.repeat(v, 2, axis=0), 2, axis=1)
        C = y - 16.0
        D = u - 128.0
        E = v - 128.0
        r = 1.16438356 * C + 1.79274107 * E
        g = 1.16438356 * C - 0.21324861 * D - 0.53290933 * E
        b = 1.16438356 * C + 2.11240179 * D
        out = _np.stack((b, g, r), axis=-1)
        out = _np.clip(out, 0.0, 255.0).astype(_np.uint8)
        return out

    def _convert_pipe_frame(self, raw: bytes) -> np.ndarray | None:
        if self._pipe_pixfmt == "nv12":
            return self._nv12_to_bgr(raw)
        try:
            import numpy as _np
            return _np.frombuffer(raw, dtype=_np.uint8).reshape(self._h, self._w, 3)
        except Exception:
            return None

    def grab(self) -> bool:
        if not self._proc or not self._proc.stdout:
            return False
        if self._pix_fmt in {"bgr24", "nv12"}:
            need = self._pipe_frame_bytes
            buf = self._proc.stdout.read(need)
            if not buf or len(buf) < need:
                if self.try_fallback_chain():
                    return self.grab()
                return False
            self._pipe_buf = buf
            self._arr = None
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
        self._pipe_buf = None
        self._pos += 1
        return True

    def retrieve(self):
        if self._pix_fmt in {"bgr24", "nv12"}:
            raw, self._pipe_buf = self._pipe_buf, None
            if raw is None:
                return False, None
            frame = self._convert_pipe_frame(raw)
            if frame is None:
                return False, None
            return True, frame
        arr, self._arr = self._arr, None
        if arr is None:
            return False, None
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
