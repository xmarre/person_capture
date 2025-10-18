
#!/usr/bin/env python3
"""
Dataset Curator for PersonCapture
- Scores candidate images for identity, quality, and composition.
- Selects up to N images that maximize quality and diversity under dataset quotas.
- Can run headless (CLI) or be embedded as a Qt tab.

Dependencies: numpy, opencv-python, Pillow
Reuses project modules: face_embedder.FaceEmbedder, reid_embedder.ReIDEmbedder, utils
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import cv2

# Robust local imports (package or flat files)
def _imp():
    try:
        from .face_embedder import FaceEmbedder  # type: ignore
        from .reid_embedder import ReIDEmbedder  # type: ignore
        from .utils import detect_black_borders  # type: ignore
        return FaceEmbedder, ReIDEmbedder, detect_black_borders
    except Exception:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from face_embedder import FaceEmbedder  # type: ignore
        from reid_embedder import ReIDEmbedder  # type: ignore
        from utils import detect_black_borders  # type: ignore
        return FaceEmbedder, ReIDEmbedder, detect_black_borders

FaceEmbedder, ReIDEmbedder, detect_black_borders = _imp()


# --------- metrics ---------

def phash64(bgr: np.ndarray) -> int:
    """Simple perceptual hash (64-bit) via DCT of 32x32 gray."""
    if bgr is None or bgr.size == 0:
        return 0
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA)
    g = np.float32(g)
    d = cv2.dct(g)
    # take 8x8 low freq block (skip DC at [0,0] to reduce exposure bias)
    block = d[:8, :8].copy()
    block[0,0] = 0.0
    med = np.median(block)
    bits = (block > med).astype(np.uint8).flatten()
    out = 0
    for i, b in enumerate(bits):
        out |= int(b) << i
    return out


def hamming64(a: int, b: int) -> int:
    x = (a ^ b) & ((1 << 64) - 1)
    if hasattr(int, "bit_count"):
        return x.bit_count()
    return int(bin(x).count("1"))


def sharpness_norm(bgr: np.ndarray) -> float:
    """Normalized variance-of-Laplacian, scaled for 0..1-ish range."""
    if bgr is None or bgr.size == 0:
        return 0.0
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = g.shape[:2]
    m = max(h, w)
    if m > 256:
        s = 256.0 / float(m)
        g = cv2.resize(g, (int(round(w*s)), int(round(h*s))), interpolation=cv2.INTER_AREA)
    lap = cv2.Laplacian(g, cv2.CV_32F)
    var = float(np.var(lap))
    mean = float(np.mean(g))
    v = var / (mean*mean + 1e-6)
    # squash with log for stability
    return float(np.tanh(np.log1p(v)))


def exposure_score(bgr: np.ndarray) -> float:
    """1.0 if well-exposed, lower if crushed or blown."""
    if bgr is None or bgr.size == 0:
        return 0.0
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([g],[0],None,[256],[0,256]).flatten()
    hist = hist / max(1.0, hist.sum())
    low = hist[:8].sum()
    high = hist[-8:].sum()
    mid = hist[16:240].sum()
    # penalize too much in extremes
    s = mid - 0.5*(low+high)
    return float(max(0.0, min(1.0, s)))


def face_fraction(face_xyxy: Optional[Tuple[int,int,int,int]], crop_w: int, crop_h: int) -> float:
    if face_xyxy is None:
        return 0.0
    x1, y1, x2, y2 = face_xyxy
    fh = max(1, y2 - y1)
    return float(fh) / float(max(1, crop_h))  # height fraction is more stable for portrait


def yaw_roll_from_5pts(pts5: np.ndarray) -> Tuple[float,float]:
    """Approximate yaw and roll from 5 landmarks (le,re,nose,lm,rm). Degrees."""
    if pts5 is None or pts5.shape != (5,2):
        return 0.0, 0.0
    le, re, nose, lm, rm = pts5
    # roll: angle of eye line
    roll = float(np.degrees(np.arctan2(re[1]-le[1], re[0]-le[0])))
    # yaw proxy: horizontal eyes-nose symmetry
    eye_mid = (le + re) * 0.5
    dx = (nose[0] - eye_mid[0])
    # normalize by inter-ocular distance
    iod = np.linalg.norm(re - le) + 1e-6
    yaw = float(np.degrees(np.arctan2(dx, iod)))
    return float(yaw), float(roll)


def textlike_corners_score(bgr: np.ndarray) -> float:
    """Heuristic watermark score using MSER regions near corners."""
    if bgr is None or bgr.size == 0:
        return 0.0
    H, W = bgr.shape[:2]
    region = int(0.22 * min(H, W))
    mask = np.zeros((H,W), np.uint8)
    # corners
    mask[:region,:region] = 255
    mask[:region,W-region:] = 255
    mask[H-region:,:region] = 255
    mask[H-region:,W-region:] = 255
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    try:
        mser = cv2.MSER_create(_delta=5, _min_area=60, _max_area=5000)
    except TypeError:
        mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)
    cnt = 0
    for rr in regions:
        x,y,w,h = cv2.boundingRect(rr)
        if mask[y:y+h, x:x+w].max() > 0:
            ar = w/float(h+1e-6)
            if 1.5 <= ar <= 12.0 and w*h >= 80:
                cnt += 1
    # normalize
    return float(min(1.0, cnt/25.0))


def _bb_frac_from_tuple(bb: Any, W: int, H: int) -> float:
    """Derive black-border fraction from either (x,y,w,h) or (x1,y1,x2,y2)."""
    if bb is None:
        return 0.0
    if not isinstance(bb, (tuple, list)) or len(bb) != 4:
        return 0.0
    x, y, a, b = [float(t) for t in bb]

    # Interpret as (x1, y1, x2, y2)
    area_xyxy = 0.0
    if a > x and b > y:
        x1 = max(0.0, min(float(W), x))
        y1 = max(0.0, min(float(H), y))
        x2 = max(0.0, min(float(W), a))
        y2 = max(0.0, min(float(H), b))
        w_xyxy = max(0.0, x2 - x1)
        h_xyxy = max(0.0, y2 - y1)
        if w_xyxy > 0.0 and h_xyxy > 0.0:
            area_xyxy = w_xyxy * h_xyxy

    # Interpret as (x, y, w, h)
    area_xywh = 0.0
    if a > 0.0 and b > 0.0:
        x1 = max(0.0, min(float(W), x))
        y1 = max(0.0, min(float(H), y))
        x2 = max(0.0, min(float(W), x + a))
        y2 = max(0.0, min(float(H), y + b))
        w_xywh = max(0.0, x2 - x1)
        h_xywh = max(0.0, y2 - y1)
        if w_xywh > 0.0 and h_xywh > 0.0:
            area_xywh = w_xywh * h_xywh

    keep = max(area_xyxy, area_xywh)
    total = max(1.0, float(W) * float(H))
    frac = 1.0 - (keep / total)
    return float(max(0.0, min(1.0, frac)))


def mmr_select(items: List["Item"], k: int, sim_mat: Optional[np.ndarray], alpha: float=0.75) -> List[int]:
    """
    Greedy Maximal Marginal Relevance.
    score(i) = alpha*qual(i) - (1-alpha)*max_j sim(i,j) for j in selected
    """
    if not items:
        return []
    k = min(k, len(items))
    selected: List[int] = []
    avail = set(range(len(items)))
    # precompute normalized quality 0..1
    q = np.array([max(0.0, min(1.0, it.quality_score)) for it in items], dtype=np.float32)
    for _ in range(k):
        best_i = None
        best_s = -1e9
        for i in list(avail):
            red = 0.0
            if selected and sim_mat is not None:
                red = float(sim_mat[i, selected].max())
                if red < 0.0:
                    red = 0.0
            s = float(alpha*q[i] - (1.0-alpha)*red)
            if s > best_s:
                best_s, best_i = s, i
        if best_i is None:
            break
        selected.append(best_i)
        avail.remove(best_i)
    return selected


# --------- data ---------

@dataclass
class Item:
    path: str
    face_fd: float          # ArcFace cosine distance to ref (lower is better)
    face_quality: float     # ArcFace chip quality (Laplacian var)
    sharpness: float        # normalized global sharpness
    exposure: float         # 0..1 exposure quality
    face_frac: float        # face height fraction in crop
    yaw: float              # degrees
    roll: float             # degrees
    ratio: str              # e.g., "2:3","1:1","3:2"
    phash: int              # 64-bit perceptual hash
    bg_clip: Optional[np.ndarray]  # CLIP embedding of full crop for diversity
    kps5: Optional[np.ndarray]     # 5pt landmarks if available
    wmark: float            # watermark likelihood 0..1
    bbox: Optional[Tuple[int,int,int,int]]  # face bbox
    meta: Dict[str, float]  # extra fields for UI
    ts: float = 0.0         # inferred timestamp / ordering key
    scene: int = -1         # scene/shot cluster id

    @property
    def quality_score(self) -> float:
        # weighted quality: identity gate + image quality + exposure + watermark penalty + border penalty
        # Map fd in [0, 0.5] to [1..0]; values beyond 0.5 collapse to 0 via clipping
        fd = max(0.0, float(self.face_fd))
        idq = float(np.clip(1.0 - (fd/0.5), 0.0, 1.0))
        q = 0.45*idq + 0.30*self.sharpness + 0.20*self.exposure + 0.05*min(1.0, self.face_quality/1200.0)
        # penalties
        q *= float(max(0.0, 1.0 - 0.6*self.wmark))
        bb = float(self.meta.get("black_border_frac", 0.0))
        bb = float(min(max(bb, 0.0), 0.4))  # clamp to avoid over-penalizing mild pillarboxing
        q *= float(max(0.0, 1.0 - 0.6*bb))
        return float(max(0.0, min(1.0, q)))


# --------- curator ---------

class Curator:
    _re_frame = re.compile(r"\b(?:frame|f|img|i)[_\-]?\s*(\d{3,})(?!\d)", re.IGNORECASE)
    _re_time_s = re.compile(r"(?:t|time)[_\-:]?(\d+(?:\.\d+)?)\s*s", re.IGNORECASE)
    _re_num = re.compile(r"(\d{3,})")

    def __init__(self,
                 ref_image: Optional[str] = None,
                 device: str = "cuda",
                 trt_lib_dir: Optional[str] = None,
                 progress: Optional[Callable[[str,int,int], None]] = None,
                 face_model: str = "scrfd_10g_bnkps",
                 face_det_conf: float = 0.50,
                 assume_identity: bool = False):
        self.device = device
        self._progress = progress
        self.face_model = face_model
        self.face_det_conf = float(face_det_conf)
        self.id_already_passed = bool(assume_identity)
        # identify the module in logs and show immediate heartbeat
        if self._progress:
            mod_path = getattr(sys.modules.get(__name__), "__file__", "<frozen>")
            self._progress(f"init: dataset_curator → {mod_path}", 0, 0)
        # Bridge FaceEmbedder's textual progress into our (phase, done, total) channel.
        def _p(msg: str) -> None:
            try:
                if self._progress:
                    self._progress(f"init: {msg}", 0, 0)
            except Exception:
                pass
        # Let init show up immediately in UI
        if self._progress:
            self._progress("init: loading models", 0, 0)

        self.face = FaceEmbedder(
            ctx=device,
            yolo_model=self.face_model,
            conf=self.face_det_conf,
            use_arcface=not self.id_already_passed,
            progress=_p,
            trt_lib_dir=trt_lib_dir,
        )
        # CLIP used for diversity (background/pose). Reuse ReIDEmbedder for whole image embeddings.
        self.reid = ReIDEmbedder(device=device)
        if self._progress:
            self._progress("init: models ready", 0, 0)

        # Build a small bank from the ref image (optional)
        self.ref_feat = None
        if ref_image:
            ref_bgr = cv2.imread(ref_image, cv2.IMREAD_COLOR)
            if ref_bgr is None:
                if self._progress:
                    self._progress(f"warn: cannot read ref image '{ref_image}'", 0, 0)
            else:
                rfaces = self.face.extract(ref_bgr)
                if rfaces:
                    # take top-1 quality face
                    rbest = max(rfaces, key=lambda f: f.get("quality", 0.0))
                    self.ref_feat = rbest.get("feat")
        else:
            if self._progress:
                mode = "assumed" if self.id_already_passed else "disabled"
                self._progress(f"init: identity gating {mode} (no ref)", 0, 0)

        if self._progress:
            if self.id_already_passed:
                mode = "assumed(no-ref)"
            elif self.ref_feat is not None:
                mode = "ref-gated"
            elif ref_image:
                mode = "ref-missing"
            else:
                mode = "disabled(no-ref)"
            self._progress(
                f"init: {mode}; detector={self.face_model} conf={self.face_det_conf:.2f}",
                0,
                0,
            )

    @staticmethod
    def _infer_ts_from_name(path: str) -> float:
        """Best-effort timestamp from filename or mtime."""
        name = os.path.basename(path)
        for rx in (Curator._re_time_s, Curator._re_frame):
            match = rx.search(name)
            if match:
                try:
                    return float(match.group(1))
                except Exception:
                    continue
        hits = Curator._re_num.findall(name)
        if hits:
            try:
                return float(hits[-1])
            except Exception:
                pass
        try:
            return float(os.path.getmtime(path))
        except Exception:
            return time.time()

    @staticmethod
    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        na = float(np.linalg.norm(a)) + 1e-6
        nb = float(np.linalg.norm(b)) + 1e-6
        return float(np.dot(a, b) / (na * nb))

    def _cluster_scenes(
        self,
        items: List["Item"],
        *,
        sim_thresh: float = 0.92,
        hamm_thresh: int = 7,
        time_gap: float = 4.0,
        nn_window: int = 64,
    ) -> List[int]:
        """Lightweight scene clustering using embeddings, phash, and timestamps."""

        if not items:
            return []

        order = sorted(range(len(items)), key=lambda i: (items[i].ts, items[i].path))
        clusters: List[List[int]] = []
        medoids: List[int] = []

        def same_scene(idx: int, med_idx: int) -> bool:
            a, b = items[idx], items[med_idx]
            if hamming64(a.phash, b.phash) <= hamm_thresh:
                return True
            if a.bg_clip is None or b.bg_clip is None:
                return False
            return self._cos(a.bg_clip, b.bg_clip) >= sim_thresh

        for idx in order:
            it = items[idx]
            assigned = False
            start = max(0, len(clusters) - max(1, nn_window))
            for cidx in range(len(clusters) - 1, start - 1, -1):
                last_idx = clusters[cidx][-1]
                dt = abs(it.ts - items[last_idx].ts)
                if dt > time_gap and not same_scene(idx, medoids[cidx]):
                    continue
                if same_scene(idx, medoids[cidx]):
                    clusters[cidx].append(idx)
                    if items[idx].quality_score > items[medoids[cidx]].quality_score:
                        medoids[cidx] = idx
                    assigned = True
                    break
            if not assigned:
                clusters.append([idx])
                medoids.append(idx)

        if nn_window > 0 and len(clusters) > 1:
            merged: List[List[int]] = []
            for group in clusters:
                if not merged:
                    merged.append(group)
                    continue
                prev = merged[-1]
                ia, ib = items[prev[-1]], items[group[0]]
                if (
                    abs(ib.ts - ia.ts) <= time_gap
                    and (
                        hamming64(ia.phash, ib.phash) <= hamm_thresh
                        or (
                            ia.bg_clip is not None
                            and ib.bg_clip is not None
                            and self._cos(ia.bg_clip, ib.bg_clip) >= sim_thresh
                        )
                    )
                ):
                    prev.extend(group)
                else:
                    merged.append(group)
            clusters = merged

        scene_ids = [-1] * len(items)
        for sid, group in enumerate(clusters):
            for idx in group:
                scene_ids[idx] = sid
        return scene_ids

    @staticmethod
    def _categorize(it: Item) -> str:
        """Assign an item to one of the selection buckets."""
        f = it.face_frac
        ratio = it.ratio
        if ratio == "2:3":
            if f >= 0.33:
                return "closeup"
            if 0.22 <= f < 0.33:
                return "portrait"
            if 0.12 <= f < 0.22:
                return "cowboy"
            return "full"
        if ratio in ("3:2", "wide"):
            return "wide"
        if ratio == "1:1":
            if f >= 0.30:
                return "closeup"
            return "portrait"
        return "portrait"

    def _emit_progress(self, phase: str, done: int, total: int, *, force: bool=False) -> None:
        if self._progress is not None:
            self._progress(phase, done, total)
        elif force or total > 0:
            print(f"[curator] {phase}: {done}/{total}")

    def _fd_min(self, feat: Optional[np.ndarray]) -> float:
        if self.id_already_passed:
            # Identity already validated upstream (PersonCapture crops).
            return 0.0
        if feat is None or self.ref_feat is None:
            return 9.0
        v = np.asarray(feat, dtype=np.float32)
        v = v / max(1e-6, float(np.linalg.norm(v)))
        r = np.asarray(self.ref_feat, dtype=np.float32)
        r = r / max(1e-6, float(np.linalg.norm(r)))
        return float(1.0 - float(np.dot(v, r)))

    def describe(self, path: str) -> Optional[Item]:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        H, W = bgr.shape[:2]
        # face detect in crop
        faces = self.face.extract(bgr)
        best = FaceEmbedder.best_face(faces) if faces else None
        bbox = None
        kps5 = None
        fd = (0.0 if self.id_already_passed else 9.0)
        fq = 0.0
        if best is not None:
            bbox = tuple(int(x) for x in best["bbox"])
            # In identity-passed mode, fd stays 0.0; else compute vs ref bank.
            if not self.id_already_passed:
                fd = self._fd_min(best.get("feat"))
            fq = float(best.get("quality", 0.0))
        # yaw/roll from kps if available
        # FaceEmbedder internally aligns by 5pts; but we can try to recover via _try_keypoints by re-calling detector
        # Expose best chip points is not public; approximate from bbox center to classify crop types
        # Face fraction
        ffrac = face_fraction(bbox, W, H)
        # landmarks: re-run the detector with keypoints if present
        yaw, roll = 0.0, 0.0
        try:
            # Try to get keypoints via private helper by forcing a predict() call here
            res = self.face.det.predict(bgr, conf=max(0.15, self.face.conf), verbose=False, device=self.face.device)[0]
            kps = getattr(res, "keypoints", None)
            if kps is not None and len(res.boxes) > 0:
                # pick the face box with max IoU to bbox
                i_best, iou_best = -1, -1.0
                for i, bb in enumerate(res.boxes.xyxy.cpu().numpy().astype(int)):
                    if bbox is None:
                        i_best = 0; break
                    x1,y1,x2,y2 = [int(t) for t in bb]
                    bx1,by1,bx2,by2 = bbox
                    ix1,iy1 = max(x1,bx1), max(y1,by1)
                    ix2,iy2 = min(x2,bx2), min(y2,by2)
                    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
                    area = (x2-x1)*(y2-y1) + (bx2-bx1)*(by2-by1) - inter + 1e-6
                    iou = inter/area
                    if iou > iou_best:
                        i_best, iou_best = i, iou
                if i_best >= 0:
                    pts = kps[i_best].cpu().numpy().astype(np.float32)
                    # reorder to ArcFace 5pts if at least 5 provided
                    if pts.shape[0] >= 5:
                        # sort by y then map as in FaceEmbedder._canon_5pts
                        order_y = np.argsort(pts[:,1])
                        eyes_idx = order_y[:2]
                        nose_idx = order_y[2]
                        mouth_idx = order_y[3:]
                        if mouth_idx.size == 2:
                            eyes = pts[eyes_idx]
                            mouth = pts[mouth_idx]
                            le, re = eyes[np.argsort(eyes[:,0])]
                            lm, rm = mouth[np.argsort(mouth[:,0])]
                            nose = pts[nose_idx]
                            kps5 = np.stack([le, re, nose, lm, rm], axis=0)
                            yaw, roll = yaw_roll_from_5pts(kps5)
        except Exception:
            pass

        # CLIP embedding of full crop for diversity
        clip_vec = self.reid.extract([bgr])
        clip_vec = clip_vec[0] if clip_vec else None

        # quality
        sharp = sharpness_norm(bgr)
        expo = exposure_score(bgr)
        wmark = textlike_corners_score(bgr)
        # detect black borders (supports: fraction, (x,y,w,h), or (x1,y1,x2,y2); with/without thr kwarg)
        try:
            bb = detect_black_borders(bgr, thr=10)
        except TypeError:
            bb = detect_black_borders(bgr)
        if isinstance(bb, (int, float, np.floating)):
            bb_frac = float(bb)
        elif isinstance(bb, (tuple, list)) and len(bb) == 4:
            bb_frac = _bb_frac_from_tuple(bb, W, H)
        else:
            bb_frac = 0.0

        # reduce to common classes
        def norm_ratio(W,H):
            asp = W/float(H)
            if 0.60 <= asp <= 0.70: return "2:3"
            if 0.95 <= asp <= 1.05: return "1:1"
            if 1.40 <= asp <= 1.70: return "3:2"
            if asp < 0.60: return "narrow"
            if asp > 1.70: return "wide"
            return "other"
        ratio = norm_ratio(W,H)

        meta = {
            "black_border_frac": bb_frac,
        }
        ts = self._infer_ts_from_name(path)

        return Item(
            path=str(path),
            face_fd=float(fd),
            face_quality=float(fq),
            sharpness=float(sharp),
            exposure=float(expo),
            face_frac=float(ffrac),
            yaw=float(yaw),
            roll=float(roll),
            ratio=ratio,
            phash=phash64(bgr),
            bg_clip=(clip_vec.astype(np.float32) if clip_vec is not None else None),
            kps5=(kps5 if kps5 is not None else None),
            wmark=float(wmark),
            bbox=(bbox if bbox is not None else None),
            meta=meta,
            ts=float(ts),
            scene=-1,
        )

    # ---- Selection with quotas ----

    def select(
        self,
        items: List[Item],
        max_images: int = 200,
        fd_max: float = 0.45,
        sharp_min: float = 0.10,
        dedup_hamm: int = 7,
        quotas: Optional[Dict[str, Tuple[int, int]]] = None,
        alpha: float = 0.75,
        *,
        scene_aware: bool = True,
        scene_sim: float = 0.92,
        scene_nn_window: int = 64,
        scene_time_gap: float = 4.0,
        dedup_hamm_scene: int = 4,
        scene_soft_cap: int = 0,
        scene_soft_penalty: float = 0.08,
        profile_yaw_thresh: float = 50.0,
    ) -> List[Item]:
        """
        quotas: dict of category -> (min, max) counts
        categories: 'portrait', 'closeup', 'cowboy', 'full', 'wide', 'profile'
        """
        if not items:
            return []

        # 1) filter identity + basic quality
        pool = [it for it in items if it.face_fd <= fd_max and it.sharpness >= sharp_min]
        if not pool:
            return []

        scene_members: Dict[int, List[int]] = {}
        scene_original_counts: Dict[int, int] = {}
        scene_kept_counts: Dict[int, int] = {}
        if scene_aware:
            scene_ids = self._cluster_scenes(
                pool,
                sim_thresh=scene_sim,
                hamm_thresh=dedup_hamm,
                time_gap=scene_time_gap,
                nn_window=scene_nn_window,
            )
            for idx, sid in enumerate(scene_ids):
                pool[idx].scene = int(sid)
                scene_original_counts[sid] = scene_original_counts.get(sid, 0) + 1
            dedup_pool: List[Item] = []
            for sid in sorted(set(scene_ids) or {-1}):
                idxs = [i for i, sc in enumerate(scene_ids) if sc == sid]
                idxs.sort(key=lambda k: (-pool[k].quality_score, pool[k].face_fd, pool[k].ts, pool[k].path))
                seen_ph: List[int] = []
                for k in idxs:
                    ph = pool[k].phash
                    if any(hamming64(ph, sph) <= dedup_hamm_scene for sph in seen_ph):
                        continue
                    seen_ph.append(ph)
                    dedup_pool.append(pool[k])
                    scene_kept_counts[sid] = scene_kept_counts.get(sid, 0) + 1
            pool = dedup_pool
        else:
            pool.sort(key=lambda it: (-it.quality_score, it.face_fd, it.ts, it.path))
            keep = []
            seen: List[int] = []
            for it in pool:
                ph = it.phash
                if any(hamming64(ph, sph) <= dedup_hamm for sph in seen):
                    continue
                keep.append(it)
                seen.append(ph)
            pool = keep

        if not pool:
            return []

        # rebuild scene membership after deduplication
        if scene_aware:
            for idx, it in enumerate(pool):
                scene_members.setdefault(int(it.scene), []).append(idx)
        else:
            scene_members[0] = list(range(len(pool)))

        if self._progress and scene_aware:
            sizes = sorted((len(v) for v in scene_members.values()), reverse=True)
            if sizes:
                arr = np.asarray(sizes, dtype=np.float32)
                p50 = int(round(float(np.percentile(arr, 50))))
                p95 = int(round(float(np.percentile(arr, 95))))
                biggest_sid = max(scene_members, key=lambda s: len(scene_members[s]))
                kept = sum(len(v) for v in scene_members.values())
                self._emit_progress(
                    f"scenes: {len(sizes)} groups; kept={kept} after in-scene dedup; "
                    f"max={len(scene_members[biggest_sid])} (sid={biggest_sid}) p50={p50} p95={p95}",
                    0,
                    0,
                    force=True,
                )
                debug_flag = os.getenv("PC_DEBUG_SCENES", "0").lower()
                if debug_flag not in {"0", "false", "no"} and scene_original_counts:
                    removed_msgs = []
                    for sid, total in sorted(scene_original_counts.items()):
                        kept_count = scene_kept_counts.get(sid, 0)
                        removed = total - kept_count
                        if removed > 0:
                            removed_msgs.append(f"sid={sid}: -{removed}")
                    if removed_msgs:
                        self._emit_progress(
                            "scene dedup trimmed " + ", ".join(removed_msgs),
                            0,
                            0,
                            force=True,
                        )

        # 3) categorize
        cats = [self._categorize(it) for it in pool]

        # 4) per-category MMR, then global fill with quotas
        # default quotas based on guide:
        # portrait majority, some closeups, some cowboy, few full, few wide, profiles cap low.
        if quotas is None:
            quotas = {
                "portrait": (80, 120),
                "closeup": (25, 45),
                "cowboy": (20, 35),
                "full": (8, 20),
                "wide": (5, 20),
                "profile": (0, 20),  # enforced as a cap below
            }
        # Normalize per-item features once; per-scene similarity built on demand
        feats = [it.bg_clip for it in pool]
        Fn: List[Optional[np.ndarray]] = []
        for f in feats:
            if f is None:
                Fn.append(None)
            else:
                v = np.asarray(f, dtype=np.float32)
                v /= (np.linalg.norm(v) + 1e-6)
                Fn.append(v)

        # build per-scene ranked lists using MMR for variety
        scene_lists: Dict[int, List[int]] = {}
        if scene_members:
            for sid, idxs in scene_members.items():
                if not idxs:
                    continue
                if all(Fn[i] is not None for i in idxs):
                    Fs = np.stack([Fn[i] for i in idxs]).astype(np.float32)
                    S = Fs @ Fs.T
                else:
                    S = None
                sub = [pool[i] for i in idxs]
                order = mmr_select(sub, k=len(sub), sim_mat=S, alpha=alpha)
                scene_lists[sid] = [idxs[j] for j in order]

        # fill respecting quotas
        out_idx: List[int] = []
        counts = {k: 0 for k in quotas.keys()}
        chosen_ph: List[int] = []
        base_vecs: List[np.ndarray] = []
        scene_counts: Dict[int, int] = {}

        def is_profile(it: Item) -> bool:
            return abs(it.yaw) >= profile_yaw_thresh

        def append_base(idx: int) -> None:
            vec = Fn[idx]
            if vec is None:
                return
            base_vecs.append(vec)

        def red_score(idx: int) -> float:
            if not base_vecs:
                return 0.0
            vec = Fn[idx]
            if vec is None:
                return 0.0
            sims = [float(np.dot(vec, b)) for b in base_vecs]
            if not sims:
                return 0.0
            return float(max(0.0, max(sims)))

        def peek_scene_candidate(sid: int, category: str) -> Optional[Tuple[int, int]]:
            lst = scene_lists.get(sid)
            if not lst:
                return None
            pos = 0
            while pos < len(lst):
                idx = lst[pos]
                if cats[idx] != category:
                    pos += 1
                    continue
                _, cmax = quotas.get(category, (0, max_images))
                if counts.get(category, 0) >= cmax:
                    return None
                if any(hamming64(pool[idx].phash, sph) <= dedup_hamm for sph in chosen_ph):
                    lst.pop(pos)
                    continue
                if is_profile(pool[idx]):
                    _, pmax = quotas.get("profile", (0, 0))
                    if counts.get("profile", 0) >= pmax:
                        lst.pop(pos)
                        continue
                return idx, pos
            return None

        # First pass: meet minimums while spreading across scenes
        for cat, (cmin, cmax) in quotas.items():
            if cat == "profile":
                cmin = 0  # profiles are a cap, not a target
            if cmin <= 0:
                continue
            need = min(cmin, max_images)
            while need > 0:
                best_idx = None
                best_sid = None
                best_pos = None
                best_score = -1e9
                for sid in list(scene_lists.keys()):
                    peek = peek_scene_candidate(sid, cat)
                    if peek is None:
                        continue
                    idx, pos = peek
                    score = float(alpha * pool[idx].quality_score - (1.0 - alpha) * red_score(idx))
                    if score > best_score:
                        best_score = score
                        best_idx = idx
                        best_sid = sid
                        best_pos = pos
                if best_idx is None:
                    break
                out_idx.append(best_idx)
                if best_sid is not None:
                    scene_counts[best_sid] = scene_counts.get(best_sid, 0) + 1
                counts[cat] = counts.get(cat, 0) + 1
                if is_profile(pool[best_idx]):
                    counts["profile"] = counts.get("profile", 0) + 1
                chosen_ph.append(pool[best_idx].phash)
                append_base(best_idx)
                if best_sid is not None and scene_lists.get(best_sid):
                    scene_lists[best_sid].pop(best_pos)
                need -= 1

        def pop_head_candidate(sid: int) -> Optional[int]:
            lst = scene_lists.get(sid)
            if not lst:
                return None
            while lst:
                idx = lst[0]
                cat = cats[idx]
                _, cmax = quotas.get(cat, (0, max_images))
                if counts.get(cat, 0) >= cmax:
                    lst.pop(0)
                    continue
                if any(hamming64(pool[idx].phash, sph) <= dedup_hamm for sph in chosen_ph):
                    lst.pop(0)
                    continue
                if is_profile(pool[idx]):
                    _, pmax = quotas.get("profile", (0, 0))
                    if counts.get("profile", 0) >= pmax:
                        lst.pop(0)
                        continue
                return idx
            return None

        # Second pass: water-fill by scenes until quotas or max_images reached
        while len(out_idx) < min(max_images, len(pool)):
            candidates: List[Tuple[int, int]] = []
            for sid in list(scene_lists.keys()):
                idx = pop_head_candidate(sid)
                if idx is not None:
                    candidates.append((sid, idx))
            if not candidates:
                break
            best_sid = None
            best_idx = None
            best_score = -1e9
            for sid, idx in candidates:
                score = float(alpha * pool[idx].quality_score - (1.0 - alpha) * red_score(idx))
                if int(scene_soft_cap) > 0 and scene_counts.get(sid, 0) >= int(scene_soft_cap):
                    score -= scene_soft_penalty
                if score > best_score:
                    best_score = score
                    best_sid = sid
                    best_idx = idx
            if best_idx is None:
                break
            cat = cats[best_idx]
            out_idx.append(best_idx)
            if best_sid is not None:
                scene_counts[best_sid] = scene_counts.get(best_sid, 0) + 1
            counts[cat] = counts.get(cat, 0) + 1
            if is_profile(pool[best_idx]):
                counts["profile"] = counts.get("profile", 0) + 1
            chosen_ph.append(pool[best_idx].phash)
            append_base(best_idx)
            if best_sid is not None and scene_lists.get(best_sid):
                scene_lists[best_sid].pop(0)

        sel = [pool[i] for i in out_idx]
        # final trim if we exceeded max due to minimum pass
        if len(sel) > max_images:
            sel = sel[:max_images]
        return sel

    # ---- run end-to-end on a folder ----

    def run(
        self,
        pool_dir: str,
        out_dir: str,
        max_images: int = 200,
        quotas: Optional[Dict[str, Tuple[int, int]]] = None,
        *,
        scene_aware_override: Optional[bool] = None,
        scene_sim_override: Optional[float] = None,
        scene_time_gap_override: Optional[float] = None,
        scene_nn_window_override: Optional[int] = None,
        dedup_hamm_override: Optional[int] = None,
        scene_dedup_override: Optional[int] = None,
        scene_soft_cap_override: Optional[int] = None,
        scene_soft_penalty_override: Optional[float] = None,
        alpha_override: Optional[float] = None,
        profile_yaw_override: Optional[float] = None,
    ) -> str:
        paths: List[str] = []
        for ext in ("*.jpg","*.jpeg","*.png","*.webp","*.bmp",
                    "*.JPG","*.JPEG","*.PNG","*.WEBP","*.BMP"):
            paths.extend([str(p) for p in Path(pool_dir).glob(ext)])
        paths.sort()
        total = len(paths)
        self._emit_progress("scan", 0, total, force=True)
        if total == 0:
            self._emit_progress("scan: empty folder", 0, 0, force=True)
        items: List[Item] = []
        for i, p in enumerate(paths, 1):
            it = self.describe(p)
            if it is not None:
                items.append(it)
            if (i % 25 == 0) or (i == total):
                self._emit_progress("scan", i, total, force=True)
        def _float_env(name: str, default: float) -> float:
            raw = os.getenv(name)
            if raw is None:
                return default
            raw = raw.strip()
            if not raw:
                return default
            try:
                return float(raw)
            except Exception:
                return default

        def _int_env(name: str, default: int) -> int:
            raw = os.getenv(name)
            if raw is None:
                return default
            raw = raw.strip()
            if not raw:
                return default
            try:
                return int(float(raw))
            except Exception:
                return default

        def _first_float_env(keys: Tuple[str, ...], default: float) -> float:
            for key in keys:
                raw = os.getenv(key)
                if raw is None:
                    continue
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    return float(raw)
                except Exception:
                    continue
            return default

        scene_aware_env = os.getenv("PC_SCENE_AWARE", "1").lower()
        scene_aware = scene_aware_env not in {"0", "false", "no"}
        if scene_aware_override is not None:
            scene_aware = bool(scene_aware_override)

        scene_sim = _float_env("PC_SCENE_SIM", 0.92)
        if scene_sim_override is not None:
            scene_sim = float(scene_sim_override)

        scene_time_gap = _float_env("PC_SCENE_TIME_GAP", 4.0)
        if scene_time_gap_override is not None:
            scene_time_gap = float(scene_time_gap_override)

        scene_nn_window = max(0, _int_env("PC_SCENE_NN_WINDOW", 64))
        if scene_nn_window_override is not None:
            scene_nn_window = max(0, int(scene_nn_window_override))

        dedup_hamm_global = max(0, _int_env("PC_DEDUP_HAMM", 7))
        if dedup_hamm_override is not None:
            dedup_hamm_global = max(0, int(dedup_hamm_override))

        dedup_hamm_scene = max(0, _int_env("PC_SCENE_DEDUP", 4))
        if scene_dedup_override is not None:
            dedup_hamm_scene = max(0, int(scene_dedup_override))

        scene_soft_cap = max(0, _int_env("PC_SCENE_SOFT_CAP", 0))
        if scene_soft_cap_override is not None:
            scene_soft_cap = max(0, int(scene_soft_cap_override))
        scene_soft_penalty = _first_float_env(("PC_SCENE_SOFT_PENALTY", "PC_SCENE_SOFT_BONUS"), 0.08)
        if scene_soft_penalty_override is not None:
            scene_soft_penalty = float(scene_soft_penalty_override)
        scene_soft_penalty = float(max(0.0, scene_soft_penalty))

        alpha = _float_env("PC_MMR_ALPHA", 0.75)
        if alpha_override is not None:
            alpha = float(alpha_override)
        alpha = float(min(1.0, max(0.0, alpha)))

        profile_yaw_thresh = _float_env("PC_PROFILE_YAW", 50.0)
        if profile_yaw_override is not None:
            profile_yaw_thresh = float(profile_yaw_override)
        profile_yaw_thresh = float(max(0.0, profile_yaw_thresh))

        sel = self.select(
            items,
            max_images=max_images,
            quotas=quotas,
            dedup_hamm=dedup_hamm_global,
            scene_aware=scene_aware,
            scene_sim=scene_sim,
            scene_nn_window=scene_nn_window,
            scene_time_gap=scene_time_gap,
            dedup_hamm_scene=dedup_hamm_scene,
            scene_soft_cap=scene_soft_cap,
            scene_soft_penalty=scene_soft_penalty,
            profile_yaw_thresh=profile_yaw_thresh,
            alpha=alpha,
        )
        if not items:
            self._emit_progress("select", 0, 0, force=True)
        elif not sel:
            self._emit_progress("select", 0, len(items), force=True)
        else:
            self._emit_progress("select", len(sel), len(items), force=True)
            if scene_aware:
                scene_ids = {it.scene for it in sel if it.scene >= 0}
                scene_total = max(1, len(scene_ids))
            else:
                scene_total = 1
            summary = (
                f"select summary: picked {len(sel)} of {len(items)} across "
                f"{scene_total} scene{'s' if scene_total != 1 else ''}"
            )
            self._emit_progress(summary, len(sel), len(items), force=True)
            cat_counts = Counter(self._categorize(it) for it in sel)
            profile_count = sum(1 for it in sel if abs(it.yaw) >= profile_yaw_thresh)
            if cat_counts or profile_count:
                order = ["closeup", "portrait", "cowboy", "full", "wide"]
                parts = [f"{cat}={cat_counts[cat]}" for cat in order if cat_counts.get(cat)]
                extra = [c for c in sorted(cat_counts.keys()) if c not in order]
                parts.extend(f"{cat}={cat_counts[cat]}" for cat in extra)
                if profile_count:
                    parts.append(f"profile={profile_count}")
                if parts:
                    self._emit_progress(
                        "categories: " + ", ".join(parts),
                        len(sel),
                        len(items),
                        force=True,
                    )
        # write manifest and copy hardlinks
        os.makedirs(out_dir, exist_ok=True)
        man_csv = os.path.join(out_dir, "dataset_manifest.csv")
        with open(man_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "rank",
                "file",
                "face_fd",
                "sharpness",
                "exposure",
                "face_frac",
                "yaw",
                "roll",
                "ratio",
                "quality",
                "category",
                "is_profile",
            ])
            for i, it in enumerate(sel):
                w.writerow([
                    i + 1,
                    it.path,
                    it.face_fd,
                    it.sharpness,
                    it.exposure,
                    it.face_frac,
                    it.yaw,
                    it.roll,
                    it.ratio,
                    it.quality_score,
                    self._categorize(it),
                    1 if abs(it.yaw) >= profile_yaw_thresh else 0,
                ])

        debug_flag = os.getenv("PC_DEBUG_SCENES", "0").lower()
        if scene_aware and debug_flag not in {"0", "false", "no"}:
            debug_csv = os.path.join(out_dir, "scenes_debug.csv")
            with open(debug_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["file", "scene", "ts", "quality", "category", "is_profile"])
                for it in sel:
                    w.writerow([
                        it.path,
                        it.scene,
                        it.ts,
                        it.quality_score,
                        self._categorize(it),
                        1 if abs(it.yaw) >= profile_yaw_thresh else 0,
                    ])
        # materialize numbered copies for LoRA trainer
        self._emit_progress("write", 0, len(sel), force=True)
        for i, it in enumerate(sel, start=1):
            name = f"{i:04d}.jpg"
            dst = os.path.join(out_dir, name)
            try:
                # hardlink if possible for speed
                if os.name == "nt":
                    import ctypes
                    if not ctypes.windll.kernel32.CreateHardLinkW(str(dst), str(it.path), None):
                        raise OSError("CreateHardLinkW failed")
                else:
                    os.link(it.path, dst)
            except Exception:
                # fallback to copy
                img = cv2.imread(it.path, cv2.IMREAD_COLOR)
                cv2.imwrite(dst, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if (i % 20 == 0) or (i == len(sel)):
                self._emit_progress("write", i, len(sel), force=True)
        # also write JSON with per-image metrics for UI
        metrics_json = os.path.join(out_dir, "metrics.json")

        def _serial(it: Item) -> Dict[str, Any]:
            d = asdict(it)
            v = d.get("bg_clip")
            if isinstance(v, np.ndarray):
                d["bg_clip"] = v.tolist()
            v = d.get("kps5")
            if isinstance(v, np.ndarray):
                d["kps5"] = v.tolist()
            m = d.get("meta")
            if isinstance(m, dict):
                d["meta"] = {k: (float(x) if isinstance(x, (np.generic, int, float)) else x)
                             for k, x in m.items()}
            return d

        legacy_items = [_serial(it) for it in sel]
        with open(metrics_json, "w", encoding="utf-8") as f:
            json.dump(legacy_items, f, indent=2)

        # Temporary parallel export with richer metadata (new shape); keep legacy list above for back-compat.
        metrics_v2 = os.path.join(out_dir, "metrics_v2.json")
        payload = {
            "identity_mode": (
                "assumed"
                if self.id_already_passed
                else ("ref" if self.ref_feat is not None else "disabled")
            ),
            "detector": self.face_model,
            "det_conf": round(self.face_det_conf, 3),
            "scene_aware": bool(scene_aware),
            "scene_sim": float(scene_sim),
            "scene_nn_window": int(scene_nn_window),
            "scene_time_gap": float(scene_time_gap),
            "dedup_hamm": int(dedup_hamm_global),
            "dedup_hamm_scene": int(dedup_hamm_scene),
            "scene_soft_cap": int(scene_soft_cap),
            # keep both keys so older builds reading *_bonus continue to work
            "scene_soft_penalty": float(scene_soft_penalty),
            "scene_soft_bonus": float(scene_soft_penalty),
            "mmr_alpha": float(alpha),
            "profile_yaw_thresh": float(profile_yaw_thresh),
            "items": legacy_items,
        }
        with open(metrics_v2, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return out_dir


# --------- CLI ---------

def _main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, help="folder with candidate images (e.g., output/crops)")
    ap.add_argument("--ref", default="", help="optional reference face image (omit if pool already identity-filtered)")
    ap.add_argument("--out", required=True, help="output folder for curated dataset")
    ap.add_argument("--max", type=int, default=200, help="max images")
    ap.add_argument("--device", default="cuda", choices=["cuda","cpu"], help="device")
    ap.add_argument("--trt-lib-dir", default="", help="TensorRT lib folder if using ArcFace ONNX TRT EP")
    ap.add_argument(
        "--assume-identity",
        action="store_true",
        help="assume all images already passed identity (skip identity gate; set fd=0.0). Defaults to on when --ref omitted.",
    )
    ap.add_argument(
        "--scene-aware",
        type=int,
        choices=[0, 1],
        default=None,
        help="override scene-aware selection (1=on, 0=off)",
    )
    ap.add_argument("--scene-sim", type=float, default=None, help="override scene similarity threshold")
    ap.add_argument("--scene-time-gap", type=float, default=None, help="override max time gap when clustering scenes")
    ap.add_argument("--scene-nn-window", type=int, default=None, help="override scene stitching window")
    ap.add_argument(
        "--dedup-hamm",
        type=int,
        default=None,
        help="override global dedup hamming threshold (default 7; keep ≥ in-scene)",
    )
    ap.add_argument(
        "--scene-dedup",
        type=int,
        default=None,
        help="override in-scene dedup hamming threshold (default 4)",
    )
    ap.add_argument("--scene-soft-cap", type=int, default=None, help="soft cap per scene (0 disables)")
    ap.add_argument(
        "--scene-soft-penalty",
        dest="scene_soft_penalty",
        type=float,
        default=None,
        help="penalty applied when soft cap exceeded (alias: --scene-soft-bonus)",
    )
    ap.add_argument(
        "--scene-soft-bonus",
        dest="scene_soft_penalty",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    ap.add_argument("--mmr-alpha", type=float, default=None, help="MMR trade-off between quality and diversity (0..1)")
    ap.add_argument("--profile-yaw", type=float, default=None, help="Yaw threshold in degrees for treating a face as profile")
    args = ap.parse_args()

    cur = Curator(
        ref_image=(args.ref or None),
        device=args.device,
        trt_lib_dir=(args.trt_lib_dir or None),
        assume_identity=bool(args.assume_identity or not args.ref),
    )
    scene_aware_override = None if args.scene_aware is None else bool(args.scene_aware)
    out = cur.run(
        args.pool,
        args.out,
        max_images=int(args.max),
        scene_aware_override=scene_aware_override,
        scene_sim_override=args.scene_sim,
        scene_time_gap_override=args.scene_time_gap,
        scene_nn_window_override=args.scene_nn_window,
        dedup_hamm_override=args.dedup_hamm,
        scene_dedup_override=args.scene_dedup,
        scene_soft_cap_override=args.scene_soft_cap,
        scene_soft_penalty_override=args.scene_soft_penalty,
        alpha_override=args.mmr_alpha,
        profile_yaw_override=args.profile_yaw,
    )
    print(out)


if __name__ == "__main__":
    _main()
