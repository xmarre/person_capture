import os
import subprocess
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np


def _exe_name(base: str) -> str:
    return f"{base}.exe" if os.name == "nt" else base


def resolve_ffmpeg_bins(ffmpeg_dir: str | os.PathLike) -> Tuple[Optional[str], Optional[str]]:
    """Given a folder, return absolute paths to (ffmpeg, ffprobe) if found.

    Accepts either the root directory or its ``bin`` subfolder.
    """

    if not ffmpeg_dir:
        return None, None
    try:
        d = Path(ffmpeg_dir).expanduser()
    except Exception:
        return None, None
    try:
        d = d.resolve()
    except Exception:
        # Resolving can fail on some network paths; fall back to raw path
        pass
    cand_dirs = [d, d / "bin"]
    ffmpeg: Optional[str] = None
    ffprobe: Optional[str] = None
    for root in cand_dirs:
        p = root / _exe_name("ffmpeg")
        q = root / _exe_name("ffprobe")
        if ffmpeg is None and p.is_file():
            ffmpeg = str(p)
        if ffprobe is None and q.is_file():
            ffprobe = str(q)
        if ffmpeg and ffprobe:
            break
    return ffmpeg, ffprobe


def ffmpeg_has_hdr_filters(ffmpeg_path: str) -> Dict[str, bool]:
    """Probe filter availability for HDR tone mapping helpers."""

    try:
        cp = subprocess.run(
            [ffmpeg_path, "-hide_banner", "-v", "error", "-filters"],
            text=True,
            capture_output=True,
            check=False,
        )
        out = cp.stdout or ""
    except Exception:
        return {"libplacebo": False, "zscale": False, "tonemap": False}
    names = {
        line.split()[1]
        for line in out.splitlines()
        if line and not line.startswith("-") and len(line.split()) >= 2
    }
    return {
        "libplacebo": "libplacebo" in names,
        "zscale": "zscale" in names,
        "tonemap": "tonemap" in names,
    }


def set_ffmpeg_env(ffmpeg_dir: str | os.PathLike) -> Dict[str, str]:
    """Resolve binaries and set env vars used by video I/O helpers."""

    ffmpeg, ffprobe = resolve_ffmpeg_bins(ffmpeg_dir)
    applied: Dict[str, str] = {}
    if ffmpeg:
        os.environ["PERSON_CAPTURE_FFMPEG"] = ffmpeg
        applied["PERSON_CAPTURE_FFMPEG"] = ffmpeg
        if not os.environ.get("FFMPEG"):
            os.environ["FFMPEG"] = ffmpeg
            applied["FFMPEG"] = ffmpeg
    if ffprobe:
        os.environ["PERSON_CAPTURE_FFPROBE"] = ffprobe
        applied["PERSON_CAPTURE_FFPROBE"] = ffprobe
        if not os.environ.get("FFPROBE"):
            os.environ["FFPROBE"] = ffprobe
            applied["FFPROBE"] = ffprobe
    # Clear ffprobe JSON cache so new binaries take effect immediately
    try:
        try:
            from . import video_io as _vio  # type: ignore
        except Exception:
            import video_io as _vio  # type: ignore
        cache = getattr(_vio, "_ffprobe_json_cached", None)
        if cache is not None and hasattr(cache, "cache_clear"):
            cache.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        pass
    return applied

def parse_ratio(s: str):
    w, h = s.split(':')
    return float(w), float(h)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def l2_normalize(x, eps=1e-10):
    n = np.linalg.norm(x) + eps
    return x / n

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def crop_img(frame, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    return frame[y1:y2, x1:x2]


def _phash_bits(img: np.ndarray, hash_size: int = 8) -> int:
    """Compute a DCT-based perceptual hash for a BGR image."""

    if img is None or img.size == 0:
        return 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(gray))
    block = dct[:hash_size, :hash_size]
    median = float(np.median(block))
    bits = 0
    bit_idx = 0
    for r in range(hash_size):
        for c in range(hash_size):
            if block[r, c] > median:
                bits |= 1 << bit_idx
            bit_idx += 1
    return int(bits)


def phash_similarity(h1: int, h2: int, nbits: int = 64) -> float:
    """Return similarity score in [0, 1] based on Hamming distance."""

    total_bits = max(1, int(nbits))
    diff = int(h1) ^ int(h2)
    try:
        distance = diff.bit_count()
    except AttributeError:  # pragma: no cover
        distance = bin(diff).count("1")
    return 1.0 - float(distance) / float(total_bits)

def detect_black_borders(bgr, thr=10, max_scan=None):
    """
    Detect constant black borders and return ROI (x1,y1,x2,y2).
    thr: pixel intensity threshold (0-255) below which we consider black.
    max_scan: optional limit for scanning depth from edges.
    """
    if bgr is None or bgr.size == 0:
        return (0,0,0,0)
    H, W = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if max_scan is None:
        max_scan = max(64, min(H, W) // 8)

    # top
    top = 0
    for r in range(min(H, max_scan)):
        if gray[r, :].mean() > thr:
            break
        top = r + 1
    # bottom
    bottom = H
    for r in range(H - 1, max(H - max_scan - 1, -1), -1):
        if gray[r, :].mean() > thr:
            break
        bottom = r
    # left
    left = 0
    for c in range(min(W, max_scan)):
        if gray[:, c].mean() > thr:
            break
        left = c + 1
    # right
    right = W
    for c in range(W - 1, max(W - max_scan - 1, -1), -1):
        if gray[:, c].mean() > thr:
            break
        right = c

    # sanity
    left = _clamp(left, 0, right - 1)
    top = _clamp(top, 0, bottom - 1)
    right = _clamp(right, left + 1, W)
    bottom = _clamp(bottom, top + 1, H)

    return int(left), int(top), int(right), int(bottom)

def expand_box_to_ratio(x1, y1, x2, y2, ratio_w, ratio_h, frame_w, frame_h, anchor=None, head_bias=0.0):
    """
    Return a box with EXACT ratio_w:ratio_h that contains the input box and fits inside the frame.
    Strategy: expand to target ratio around center/anchor, clamp, then if clamping broke the ratio,
    shrink inside the frame while keeping center to hit the exact ratio.
    """
    x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    target = float(ratio_w) / float(ratio_h)

    # center
    if anchor is not None:
        cx, cy = float(anchor[0]), float(anchor[1])
    else:
        cx = x1 + bw * 0.5
        cy = y1 + bh * 0.5
    cy = cy - head_bias * bh  # head bias

    # expand to target ratio minimally
    cur = bw / bh
    if cur < target:
        new_w, new_h = target * bh, bh
    else:
        new_w, new_h = bw, bw / target

    nx1, ny1 = cx - new_w * 0.5, cy - new_h * 0.5
    nx2, ny2 = cx + new_w * 0.5, cy + new_h * 0.5

    # clamp
    nx1 = _clamp(nx1, 0, frame_w - 1)
    ny1 = _clamp(ny1, 0, frame_h - 1)
    nx2 = _clamp(nx2, 0, frame_w - 1)
    ny2 = _clamp(ny2, 0, frame_h - 1)

    # enforce exact ratio by shrinking inside if necessary
    cw, ch = nx2 - nx1, ny2 - ny1
    if cw <= 1 or ch <= 1:
        return int(nx1), int(ny1), int(nx2), int(ny2)

    cur = cw / ch
    if abs(cur - target) > 1e-4:
        if cur < target:
            # width too small relative to height -> reduce height
            ch2 = cw / target
            dy = (ch - ch2) * 0.5
            ny1 += dy; ny2 -= dy
        else:
            # width too large -> reduce width
            cw2 = ch * target
            dx = (cw - cw2) * 0.5
            nx1 += dx; nx2 -= dx

        # ensure inside frame (minor corrections)
        nx1 = _clamp(nx1, 0, frame_w - 1)
        ny1 = _clamp(ny1, 0, frame_h - 1)
        nx2 = _clamp(nx2, 0, frame_w - 1)
        ny2 = _clamp(ny2, 0, frame_h - 1)

    return int(round(nx1)), int(round(ny1)), int(round(nx2)), int(round(ny2))


def cosine_distance(a: Iterable[float], b: Iterable[float]) -> float:
    """Return cosine distance (1 - cosine similarity) for the two vectors."""

    vec_a = np.asarray(list(a), dtype=np.float32).reshape(-1)
    vec_b = np.asarray(list(b), dtype=np.float32).reshape(-1)
    norm_a = float(np.linalg.norm(vec_a)) + 1e-9
    norm_b = float(np.linalg.norm(vec_b)) + 1e-9
    similarity = float(np.dot(vec_a / norm_a, vec_b / norm_b))
    return 1.0 - similarity
