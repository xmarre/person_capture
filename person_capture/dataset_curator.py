
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

import os, sys, math, csv, json, hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import cv2
from PIL import Image

# Robust local imports (package or flat files)
def _imp():
    try:
        from .face_embedder import FaceEmbedder  # type: ignore
        from .reid_embedder import ReIDEmbedder  # type: ignore
        from .utils import detect_black_borders, parse_ratio  # type: ignore
        return FaceEmbedder, ReIDEmbedder, detect_black_borders, parse_ratio
    except Exception:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from face_embedder import FaceEmbedder  # type: ignore
        from reid_embedder import ReIDEmbedder  # type: ignore
        from utils import detect_black_borders, parse_ratio  # type: ignore
        return FaceEmbedder, ReIDEmbedder, detect_black_borders, parse_ratio

FaceEmbedder, ReIDEmbedder, detect_black_borders, parse_ratio = _imp()


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
    return int(bin((a ^ b) & ((1<<64)-1)).count("1"))


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


def black_border_frac(bgr: np.ndarray) -> float:
    try:
        x1,y1,x2,y2 = detect_black_borders(bgr, thr=10)
    except Exception:
        return 0.0
    H, W = (bgr.shape[:2] if bgr is not None else (1,1))
    keep = max(1, (x2-x1)*(y2-y1))
    frac = 1.0 - keep/float(max(1, W*H))
    return float(max(0.0, min(1.0, frac)))


def face_fraction(face_xyxy: Optional[Tuple[int,int,int,int]], crop_w: int, crop_h: int) -> float:
    if face_xyxy is None:
        return 0.0
    x1,y1,x2,y2 = face_xyxy
    fw, fh = max(1, x2-x1), max(1, y2-y1)
    return float(fh) / float(max(1, crop_h))  # height fraction is more stable for portrait


def yaw_roll_from_5pts(pts5: np.ndarray) -> Tuple[float,float]:
    """Approximate yaw and roll from 5 landmarks (le,re,nose,lm,rm). Degrees."""
    if pts5 is None or pts5.shape != (5,2):
        return 0.0, 0.0
    le, re, nose, lm, rm = pts5
    # roll: angle of eye line
    roll = math.degrees(math.atan2(re[1]-le[1], re[0]-le[0]))
    # yaw proxy: horizontal eyes-nose symmetry
    eye_mid = (le + re) * 0.5
    dx = (nose[0] - eye_mid[0])
    # normalize by inter-ocular distance
    iod = np.linalg.norm(re - le) + 1e-6
    yaw = math.degrees(math.atan(dx / iod))
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

    @property
    def quality_score(self) -> float:
        # weighted quality: identity gate + image quality + exposure + watermark penalty + border penalty
        # Map fd in [0, 0.5] to [1..0] clamp; use soft gate
        fd = max(0.0, min(0.8, float(self.face_fd)))
        idq = float(np.clip(1.0 - (fd/0.5), 0.0, 1.0))
        q = 0.45*idq + 0.30*self.sharpness + 0.20*self.exposure + 0.05*min(1.0, self.face_quality/1200.0)
        # penalties
        q *= float(max(0.0, 1.0 - 0.6*self.wmark))
        return float(max(0.0, min(1.0, q)))


# --------- curator ---------

class Curator:
    def __init__(self, ref_image: str, device: str="cuda",
                 trt_lib_dir: Optional[str]=None,
                 progress: Optional[Callable[[str,int,int], None]]=None):
        self.device = device
        self._progress = progress
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

        self.face = FaceEmbedder(ctx=device, use_arcface=True, progress=_p, trt_lib_dir=trt_lib_dir)
        # CLIP used for diversity (background/pose). Reuse ReIDEmbedder for whole image embeddings.
        self.reid = ReIDEmbedder(device=device)
        if self._progress:
            self._progress("init: models ready", 0, 0)

        # Build a small bank from the ref image
        ref_bgr = cv2.imread(ref_image, cv2.IMREAD_COLOR)
        assert ref_bgr is not None, f"Cannot read ref image: {ref_image}"
        rfaces = self.face.extract(ref_bgr)
        if not rfaces:
            self.ref_feat = None
        else:
            # take top-1 quality face
            rbest = max(rfaces, key=lambda f: f.get("quality",0.0))
            self.ref_feat = rbest.get("feat")

    def _emit_progress(self, phase: str, done: int, total: int, *, force: bool=False) -> None:
        if self._progress is not None:
            self._progress(phase, done, total)
        elif force or total > 0:
            print(f"[curator] {phase}: {done}/{total}")

    def _fd_min(self, feat: Optional[np.ndarray]) -> float:
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
        fd = 9.0
        fq = 0.0
        if best is not None:
            bbox = tuple(int(x) for x in best["bbox"])
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
        # detect black borders
        bb_frac = black_border_frac(bgr)

        # ratio string
        r = f"{W}:{H}"
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
        )

    # ---- Selection with quotas ----

    def select(self,
               items: List[Item],
               max_images: int = 200,
               fd_max: float = 0.45,
               sharp_min: float = 0.10,
               dedup_hamm: int = 6,
               quotas: Optional[Dict[str, Tuple[int,int]]] = None,
               alpha: float = 0.75) -> List[Item]:
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

        # 2) near-dup removal by phash clustering
        pool.sort(key=lambda it: (-it.quality_score, it.face_fd))
        keep = []
        seen: List[int] = []
        for it in pool:
            ph = it.phash
            dup = False
            for sph in seen:
                if hamming64(ph, sph) <= dedup_hamm:
                    dup = True
                    break
            if not dup:
                keep.append(it)
                seen.append(ph)
        pool = keep

        # 3) categorize
        def categorize(it: Item) -> str:
            # face fraction based classes
            f = it.face_frac
            ratio = it.ratio
            # profile by yaw
            if abs(it.yaw) >= 50.0:
                prof = True
            else:
                prof = False
            if ratio == "2:3":
                if f >= 0.33:
                    return "closeup"
                if 0.22 <= f < 0.33:
                    return "portrait"
                if 0.12 <= f < 0.22:
                    return "cowboy"
                return "full"
            elif ratio == "3:2" or ratio == "wide":
                return "wide"
            elif ratio == "1:1":
                if f >= 0.30: return "closeup"
                return "portrait"
            else:
                return "portrait"
        cats = [categorize(it) for it in pool]
        # maintain an index per category
        by_cat: Dict[str, List[int]] = {}
        for i, c in enumerate(cats):
            by_cat.setdefault(c, []).append(i)

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
        # build sim matrix for diversity using bg_clip cosine
        feats = [it.bg_clip for it in pool]
        if any(f is None for f in feats):
            sim = None
        else:
            F = np.stack(feats).astype(np.float32)
            F /= (np.linalg.norm(F, axis=1, keepdims=True) + 1e-6)
            sim = (F @ F.T)

        # helper to get ranked indices within a subset
        def ranked(indices: List[int]) -> List[int]:
            sub = [pool[i] for i in indices]
            S = sim[np.ix_(indices, indices)] if sim is not None else None
            sel_local = mmr_select(sub, k=len(sub), sim_mat=S, alpha=alpha)
            return [indices[j] for j in sel_local]

        ranked_by_cat: Dict[str, List[int]] = {c: ranked(idx) for c, idx in by_cat.items()}

        # fill respecting quotas
        out_idx: List[int] = []
        counts = {k: 0 for k in quotas.keys()}
        # First pass meet minimums
        for c, (cmin, cmax) in quotas.items():
            candidates = ranked_by_cat.get(c, [])
            take = min(cmin, len(candidates))
            out_idx.extend(candidates[:take])
            counts[c] += take
            ranked_by_cat[c] = candidates[take:]
        # Second pass fill to max_images without exceeding per-cat max
        # Also enforce profile cap by skipping items with |yaw|>=50 into "profile" budget.
        def is_profile(it: Item) -> bool:
            return abs(it.yaw) >= 50.0
        # build a flattened ranked list from remaining categories by MMR against already chosen
        chosen_vecs = [pool[i].bg_clip for i in out_idx if pool[i].bg_clip is not None]
        if chosen_vecs and sim is not None:
            base = np.stack(chosen_vecs).astype(np.float32)
            base /= (np.linalg.norm(base, axis=1, keepdims=True) + 1e-6)
        else:
            base = None

        # function to compute reduction vs chosen set
        def red_score(idx: int) -> float:
            if base is None or pool[idx].bg_clip is None:
                return 0.0
            v = pool[idx].bg_clip
            v = v / (np.linalg.norm(v) + 1e-6)
            return float((v @ base.T).max())

        # global heap-like greedy selection
        while len(out_idx) < min(max_images, len(pool)):
            # candidate pool across all categories
            cand = []
            for c, lst in ranked_by_cat.items():
                if not lst: continue
                # skip if category already at max
                cmin, cmax = quotas.get(c, (0, max_images))
                if counts.get(c, 0) >= cmax:
                    continue
                # consider head of list
                cand.append(lst[0])
            if not cand:
                break
            # pick the one with best MMR score against chosen
            best_i = None
            best_s = -1e9
            for i in cand:
                it = pool[i]
                # if profile, ensure we have budget
                if is_profile(it):
                    pmin, pmax = quotas.get("profile", (0, 0))
                    if counts.get("profile", 0) >= pmax:
                        # try to skip profile if over cap
                        continue
                q = it.quality_score
                r = red_score(i)
                s = float(alpha*q - (1.0-alpha)*r)
                if s > best_s:
                    best_s, best_i = s, i
            if best_i is None:
                break
            # accept
            c = cats[best_i]
            out_idx.append(best_i)
            counts[c] = counts.get(c, 0) + 1
            if is_profile(pool[best_i]):
                counts["profile"] = counts.get("profile", 0) + 1
            # pop from its category list
            ranked_by_cat[c].pop(0)

        sel = [pool[i] for i in out_idx]
        # final trim if we exceeded max due to minimum pass
        if len(sel) > max_images:
            sel = sel[:max_images]
        return sel

    # ---- run end-to-end on a folder ----

    def run(self, pool_dir: str, out_dir: str, max_images: int = 200,
            quotas: Optional[Dict[str, Tuple[int,int]]] = None) -> str:
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
        sel = self.select(items, max_images=max_images, quotas=quotas)
        if not items:
            self._emit_progress("select", 0, 0, force=True)
        elif not sel:
            self._emit_progress("select", 0, len(items), force=True)
        else:
            self._emit_progress("select", len(sel), len(items), force=True)
        # write manifest and copy hardlinks
        os.makedirs(out_dir, exist_ok=True)
        man_csv = os.path.join(out_dir, "dataset_manifest.csv")
        with open(man_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["rank","file","face_fd","sharpness","exposure","face_frac","yaw","roll","ratio","quality","category"])
            for i, it in enumerate(sel):
                w.writerow([i+1, it.path, it.face_fd, it.sharpness, it.exposure, it.face_frac, it.yaw, it.roll, it.ratio, it.quality_score,
                            "profile" if abs(it.yaw)>=50.0 else ""])
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

        with open(metrics_json, "w", encoding="utf-8") as f:
            json.dump([_serial(it) for it in sel], f, indent=2)
        return out_dir


# --------- CLI ---------

def _main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, help="folder with candidate images (e.g., output/crops)")
    ap.add_argument("--ref", required=True, help="reference face image")
    ap.add_argument("--out", required=True, help="output folder for curated dataset")
    ap.add_argument("--max", type=int, default=200, help="max images")
    ap.add_argument("--device", default="cuda", choices=["cuda","cpu"], help="device")
    ap.add_argument("--trt-lib-dir", default="", help="TensorRT lib folder if using ArcFace ONNX TRT EP")
    args = ap.parse_args()

    cur = Curator(ref_image=args.ref, device=args.device, trt_lib_dir=(args.trt_lib_dir or None))
    out = cur.run(args.pool, args.out, max_images=int(args.max))
    print(out)


if __name__ == "__main__":
    _main()
