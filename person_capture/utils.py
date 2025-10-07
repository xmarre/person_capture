
import math
import os
import cv2
import numpy as np

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

def expand_box_to_ratio(x1, y1, x2, y2, ratio_w, ratio_h, frame_w, frame_h, anchor=None, head_bias=0.0):
    """
    Expand a box to the exact ratio (ratio_w:ratio_h) while clamping to frame.
    Optionally bias the crop upward (head_bias in [0..0.5]) and anchor around a point.
    Returns integer (nx1, ny1, nx2, ny2).
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

    # bias upward for heads
    cy = cy - head_bias * bh

    # compute new size that encloses the box at target ratio
    cur = bw / bh
    if cur < target:
        # need more width
        new_w = target * bh
        new_h = bh
    else:
        new_w = bw
        new_h = bw / target

    nx1 = cx - new_w * 0.5
    ny1 = cy - new_h * 0.5
    nx2 = cx + new_w * 0.5
    ny2 = cy + new_h * 0.5

    # clamp
    nx1 = _clamp(nx1, 0, frame_w - 1)
    ny1 = _clamp(ny1, 0, frame_h - 1)
    nx2 = _clamp(nx2, 0, frame_w - 1)
    ny2 = _clamp(ny2, 0, frame_h - 1)

    # ensure exact ratio after clamping by expanding towards inside if needed
    new_w = nx2 - nx1
    new_h = ny2 - ny1
    if new_w <= 1 or new_h <= 1:
        return int(nx1), int(ny1), int(nx2), int(ny2)

    cur = new_w / new_h
    if abs(cur - target) > 1e-4:
        if cur < target:
            # grow width if possible
            pad = (target * new_h - new_w) * 0.5
            nx1 = _clamp(nx1 - pad, 0, frame_w - 1)
            nx2 = _clamp(nx2 + pad, 0, frame_w - 1)
        else:
            pad = (new_w / target - new_h) * 0.5
            ny1 = _clamp(ny1 - pad, 0, frame_h - 1)
            ny2 = _clamp(ny2 + pad, 0, frame_h - 1)

    return int(round(nx1)), int(round(ny1)), int(round(nx2)), int(round(ny2))

def crop_img(frame, box):
    x1,y1,x2,y2 = [int(v) for v in box]
    return frame[y1:y2, x1:x2]

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
