import argparse
import os
import cv2
import csv
import numpy as np
from tqdm import tqdm

from .detectors import PersonDetector
from .face_embedder import FaceEmbedder
from .reid_embedder import ReIDEmbedder
from .utils import ensure_dir, parse_ratio, expand_box_to_ratio, crop_img, cosine_distance

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img

def pick_anchor_from_face(face, person_box):
    if face is None:
        return None
    x1,y1,x2,y2 = person_box
    fb = face['bbox']
    # anchor within person box if overlapping
    fx = (fb[0]+fb[2])/2.0
    fy = (fb[1]+fb[3])/2.0
    if fx>=x1 and fx<=x2 and fy>=y1 and fy<=y2:
        return (fx, fy)
    return None

def combine_scores(face_dist, reid_dist, mode='min'):
    vals = []
    if face_dist is not None:
        vals.append(face_dist)
    if reid_dist is not None:
        vals.append(reid_dist)
    if not vals:
        return None
    if mode == 'min':
        return min(vals)  # require at least one strong signal
    if mode == 'avg':
        return sum(vals)/len(vals)
    if mode == 'face_priority':
        if face_dist is not None:
            return 0.7*face_dist + 0.3*(reid_dist if reid_dist is not None else 0.5)
        else:
            return reid_dist
    return min(vals)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True, help='path to video file')
    ap.add_argument('--ref', required=True, help='reference image of the target person')
    ap.add_argument('--out', required=True, help='output directory')
    ap.add_argument('--ratio', default='2:3', help='crop aspect ratio W:H (e.g., 2:3)')
    ap.add_argument('--frame-stride', type=int, default=2, help='analyze every Nth frame')
    ap.add_argument('--min-det-conf', type=float, default=0.35, help='YOLO min confidence')
    ap.add_argument('--face-thresh', type=float, default=0.32, help='max cosine distance for face match')
    ap.add_argument('--reid-thresh', type=float, default=0.38, help='max cosine distance for reid match')
    ap.add_argument('--combine', default='min', choices=['min','avg','face_priority'])
    ap.add_argument('--device', default='cuda', choices=['cuda','cpu'])
    ap.add_argument('--save-annot', action='store_true', help='save annotated frames')
    ap.add_argument('--yolo', default='yolov8n.pt', help='ultralytics model name or path')
    args = ap.parse_args()

    ensure_dir(args.out)
    crops_dir = os.path.join(args.out, 'crops')
    ann_dir = os.path.join(args.out, 'annot') if args.save_annot else None
    ensure_dir(crops_dir)
    if ann_dir:
        ensure_dir(ann_dir)

    # Init models
    det = PersonDetector(model_name=args.yolo, device=args.device)
    face = FaceEmbedder(ctx=args.device)
    reid = ReIDEmbedder(device=args.device)

    # Reference embeddings
    ref_img = load_image(args.ref)
    ref_faces = face.extract(ref_img)
    ref_face = FaceEmbedder.best_face(ref_faces)
    ref_face_feat = ref_face['feat'] if ref_face else None

    # ReID reference: detect person regions in reference; pick the largest
    ref_persons = det.detect(ref_img, conf=0.1)
    ref_reid_feat = None
    if ref_persons:
        ref_persons.sort(key=lambda d: (d['xyxy'][2]-d['xyxy'][0])*(d['xyxy'][3]-d['xyxy'][1]), reverse=True)
        rx1,ry1,rx2,ry2 = [int(v) for v in ref_persons[0]['xyxy']]
        ref_crop = ref_img[ry1:ry2, rx1:rx2]
        ref_reid_feat = reid.extract([ref_crop])[0]
    else:
        # fallback: whole image
        ref_reid_feat = reid.extract([ref_img])[0]

    # Video loop
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = 0
    hit_count = 0
    ratio_w, ratio_h = parse_ratio(args.ratio)

    # CSV output
    csv_path = os.path.join(args.out, 'index.csv')
    csv_f = open(csv_path, 'w', newline='')
    writer = csv.writer(csv_f)
    writer.writerow(['frame','time_secs','score','face_dist','reid_dist','x1','y1','x2','y2','crop_path'])

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    pbar = tqdm(total=total_frames if total_frames>0 else None, desc='processing')

    while True:
        ret = cap.grab()
        if not ret:
            break
        if frame_idx % max(1, args.frame_stride) != 0:
            frame_idx += 1
            pbar.update(1)
            continue
        # retrieve frame
        ret, frame = cap.retrieve()
        if not ret:
            break
        H, W = frame.shape[:2]

        persons = det.detect(frame, conf=args.min_det_conf)
        if persons:
            # Prepare reid crops batch
            crops = []
            boxes = []
            for p in persons:
                x1,y1,x2,y2 = [int(v) for v in p['xyxy']]
                x1 = max(0, x1); y1 = max(0, y1); x2 = min(W-1, x2); y2 = min(H-1, y2)
                if x2<=x1+2 or y2<=y1+2:
                    continue
                crop = frame[y1:y2, x1:x2]
                crops.append(crop); boxes.append((x1,y1,x2,y2))

            reid_feats = reid.extract(crops) if crops else []

            # Face features per person (optional)
            face_map = {}  # idx -> (face, dist)
            for i, crop in enumerate(crops):
                ffaces = face.extract(crop)
                bestf = FaceEmbedder.best_face(ffaces)
                if bestf and ref_face_feat is not None:
                    fd = 1.0 - float(np.dot(bestf['feat'], ref_face_feat))  # cosine distance on normed features
                    face_map[i] = (bestf, fd)

            # Score and save matches
            for i, feat in enumerate(reid_feats):
                rd = None
                if ref_reid_feat is not None:
                    # cosine distance
                    sim = float(np.dot(feat/np.linalg.norm(feat), ref_reid_feat/np.linalg.norm(ref_reid_feat)))
                    rd = 1.0 - sim
                fd = face_map.get(i, (None, None))[1]
                score = combine_scores(fd, rd, mode=args.combine)

                accept = False
                if score is not None:
                    # acceptance rule: pass if either signal is under its threshold,
                    # and combined score is reasonable.
                    face_ok = (fd is not None and fd <= args.face_thresh)
                    reid_ok = (rd is not None and rd <= args.reid_thresh)
                    accept = face_ok or reid_ok

                if accept:
                    x1,y1,x2,y2 = boxes[i]
                    anchor = None
                    head_bias = 0.0
                    bf = face_map.get(i, (None,None))[0]
                    if bf is not None:
                        # face bbox in crop -> map to frame coords
                        fb = bf['bbox']
                        acx = x1 + (fb[0]+fb[2])/2.0
                        acy = y1 + (fb[1]+fb[3])/2.0
                        anchor = (acx, acy)
                        # dynamic downward shift to include torso: move center down by ~0.9 * face_h
                        face_h = max(1.0, fb[3]-fb[1])
                        box_h  = max(1.0, y2-y1)
                        head_bias = -(0.9 * (face_h / box_h))
                    ex1,ey1,ex2,ey2 = expand_box_to_ratio(
                        x1,y1,x2,y2,
                        ratio_w, ratio_h, W, H,
                        anchor=anchor,
                        head_bias=head_bias
                    )
                    crop_img_path = os.path.join(crops_dir, f"f{frame_idx:08d}.jpg")
                    crop = frame[ey1:ey2, ex1:ex2]
                    cv2.imwrite(crop_img_path, crop)
                    hit_count += 1

                    if args.save_annot:
                        vis = frame.copy()
                        # original person box
                        cv2.rectangle(vis, (x1,y1),(x2,y2),(0,255,0),2)
                        # expanded crop box
                        cv2.rectangle(vis, (ex1,ey1),(ex2,ey2),(255,0,0),2)
                        # face box if any
                        if bf is not None:
                            fb = bf['bbox']
                            cv2.rectangle(vis, (x1+fb[0], y1+fb[1]), (x1+fb[2], y1+fb[3]), (0,0,255), 2)
                        cv2.putText(vis, f"score={score:.3f} fd={fd if fd is not None else -1:.3f} rd={rd if rd is not None else -1:.3f}",
                                    (15,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                        ann_path = os.path.join(ann_dir, f"f{frame_idx:08d}.jpg")
                        cv2.imwrite(ann_path, vis)

                    t = frame_idx / fps
                    writer.writerow([frame_idx, f"{t:.3f}", f"{score:.4f}" if score is not None else "", 
                                     f"{fd:.4f}" if fd is not None else "", f"{rd:.4f}" if rd is not None else "",
                                     ex1,ey1,ex2,ey2, os.path.basename(crop_img_path)])

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    csv_f.close()
    cap.release()
    print(f"Done. Hits: {hit_count}. Index: {csv_path}")

if __name__ == '__main__':
    main()
