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

def cosine_distance(a, b):
    # a,b are 1D arrays
    an = l2_normalize(a)
    bn = l2_normalize(b)
    sim = float(np.dot(an, bn))
    # convert to distance in [0,2]
    return 1.0 - sim  # smaller is better

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def expand_box_to_ratio(x1, y1, x2, y2, ratio_w, ratio_h, frame_w, frame_h, anchor=None, head_bias=0.12):
    # Expand a box to target aspect ratio while containing the original box.
    bw = x2 - x1
    bh = y2 - y1
    cx = x1 + bw/2.0
    cy = y1 + bh/2.0

    target = ratio_w / ratio_h
    cur = bw / max(bh, 1e-6)

    if cur < target:
        # widen
        new_w = target * bh
        new_h = bh
    else:
        # taller
        new_w = bw
        new_h = bw / target

    # Optional anchor (e.g., face box center) and head bias for portraits
    if anchor is not None:
        acx, acy = anchor
        cx = 0.7*cx + 0.3*acx  # pull center toward anchor
        cy = 0.6*cy + 0.4*acy - head_bias*new_h  # bias upward to leave headroom
    else:
        cy -= head_bias*new_h

    nx1 = cx - new_w/2.0
    ny1 = cy - new_h/2.0
    nx2 = cx + new_w/2.0
    ny2 = cy + new_h/2.0

    # Clamp to frame
    nx1 = clamp(nx1, 0, frame_w-1)
    ny1 = clamp(ny1, 0, frame_h-1)
    nx2 = clamp(nx2, 0, frame_w-1)
    ny2 = clamp(ny2, 0, frame_h-1)

    # If clamping shrank below target, re-center within frame
    new_w = nx2 - nx1
    new_h = ny2 - ny1
    cur = new_w / max(new_h, 1e-6)
    if abs(cur - target) > 1e-3:
        # Try to adjust width or height minimally
        if cur < target:
            # need more width
            pad = (target*new_h - new_w)/2.0
            nx1 = clamp(nx1 - pad, 0, frame_w-1)
            nx2 = clamp(nx2 + pad, 0, frame_w-1)
        else:
            # need more height
            pad = (new_w/target - new_h)/2.0
            ny1 = clamp(ny1 - pad, 0, frame_h-1)
            ny2 = clamp(ny2 + pad, 0, frame_h-1)

    return int(nx1), int(ny1), int(nx2), int(ny2)

def crop_img(frame, box):
    x1,y1,x2,y2 = [int(v) for v in box]
    return frame[y1:y2, x1:x2]
