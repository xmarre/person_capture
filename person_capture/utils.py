import os
import math
import cv2
import numpy as np

__all__ = [
    "parse_ratio",
    "ensure_dir",
    "l2_normalize",
    "cosine_distance",
    "expand_box_to_ratio",
    "crop_img",
    "detect_black_borders",
]

def parse_ratio(s: str) -> tuple[float, float]:
    s = str(s).strip().lower().replace(" ", "")
    if ":" not in s:
        raise ValueError(f"Invalid ratio '{s}'. Use W:H, e.g., '2:3'.")
    w, h = s.split(":")
    w = float(w); h = float(h)
    if w <= 0 or h <= 0:
        raise ValueError("Ratio components must be > 0.")
    return w, h

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def l2_normalize(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = float(np.linalg.norm(x))
    return x if n < eps else (x / (n + eps))

def cosine_distance(a: np.ndarray | None, b: np.ndarray | None, eps: float = 1e-9) -> float | None:
    if a is None or b is None:
        return None
    va = l2_normalize(np.asarray(a, dtype=np.float32), eps)
    vb = l2_normalize(np.asarray(b, dtype=np.float32), eps)
    return float(1.0 - np.dot(va, vb))

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def expand_box_to_ratio(x1: float, y1: float, x2: float, y2: float,
                        ratio_w: float, ratio_h: float,
                        frame_w: int, frame_h: int,
                        anchor: tuple[float,float] | None = None,
                        head_bias: float = 0.0) -> tuple[int,int,int,int]:
    """Expand an input box to an exact W:H ratio inside frame bounds.

    Strategy:
      1) Choose center (anchor if provided else box center) with optional head bias.
      2) Expand minimally to reach target ratio.
      3) Clamp to frame.
      4) If clamping broke the ratio, shrink inside to exact ratio.
    """
    x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    target = float(ratio_w) / float(ratio_h)

    # center
    if anchor is not None:
        cx, cy = float(anchor[0]), float(anchor[1])
    else:
        cx = x1 + 0.5 * bw
        cy = y1 + 0.5 * bh
    cy -= head_bias * bh  # bias upwards

    # minimal expansion to target
    cur = bw / bh
    if cur < target:
        new_w, new_h = target * bh, bh
    else:
        new_w, new_h = bw, bw / target

    nx1, ny1 = cx - 0.5 * new_w, cy - 0.5 * new_h
    nx2, ny2 = cx + 0.5 * new_w, cy + 0.5 * new_h

    # clamp to frame
    nx1 = _clamp(nx1, 0, frame_w - 1)
    ny1 = _clamp(ny1, 0, frame_h - 1)
    nx2 = _clamp(nx2, 0, frame_w - 1)
    ny2 = _clamp(ny2, 0, frame_h - 1)

    # enforce exact ratio by shrinking if needed
    cw, ch = nx2 - nx1, ny2 - ny1
    if cw <= 1 or ch <= 1:
        return int(round(nx1)), int(round(ny1)), int(round(nx2)), int(round(ny2))

    cur = cw / ch
    if abs(cur - target) > 1e-4:
        if cur < target:
            # width too small -> shrink height
            new_h = cw / target
            dy = 0.5 * (ch - new_h)
            ny1 += dy; ny2 -= dy
        else:
            # height too small -> shrink width
            new_w = ch * target
            dx = 0.5 * (cw - new_w)
            nx1 += dx; nx2 -= dx

    return int(round(nx1)), int(round(ny1)), int(round(nx2)), int(round(ny2))

def crop_img(frame: np.ndarray, box: tuple[int,int,int,int]) -> np.ndarray:
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, x1); y1 = max(0, y1)
    return frame[y1:y2, x1:x2]

def detect_black_borders(bgr: np.ndarray, thr: int = 10, max_scan: int | None = None) -> tuple[int,int,int,int]:
    """Detect constant black borders. Return ROI (x1,y1,x2,y2)."""
    if bgr is None or bgr.size == 0:
        return (0, 0, 0, 0)
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
    left = max(0, min(left, right - 1))
    top = max(0, min(top, bottom - 1))
    right = max(left + 1, min(right, W))
    bottom = max(top + 1, min(bottom, H))

    return int(left), int(top), int(right), int(bottom)
