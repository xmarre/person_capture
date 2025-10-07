import argparse
import os
import csv
import cv2
import numpy as np

from detectors import PersonDetector
from face_embedder import FaceEmbedder
from reid_embedder import ReIDEmbedder
from utils import ensure_dir, parse_ratio, expand_box_to_ratio, cosine_distance

def load_image(path: str):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img

def combine_scores(face_dist, reid_dist, mode: str = 'min'):
    vals = [v for v in (face_dist, reid_dist) if v is not None]
    if not vals:
        return None
    if mode == 'avg':
        return sum(vals)/len(vals)
    if mode == 'face_priority':
        if face_dist is not None:
            return 0.7*face_dist + 0.3*(reid_dist if reid_dist is not None else 0.5)
        return reid_dist
    # default min: require at least one strong signal
    return min(vals)

def main():
    ap = argparse.ArgumentParser(description="PersonCapture CLI: find target person and save ratio-true crops.")
    ap.add_argument('--video', required=True, help='path to video file')
    ap.add_argument('--ref', required=True, help='reference image of the target person')
    ap.add_argument('--out', required=True, help='output directory')
    ap.add_argument('--ratio', default='2:3', help='crop aspect ratio W:H or list separated by comma')
    ap.add_argument('--frame-stride', type=int, default=2, help='analyze every Nth frame')
    ap.add_argument('--min-det-conf', type=float, default=0.35, help='YOLO min confidence')
    ap.add_argument('--face-thresh', type=float, default=0.36, help='max cosine distance for face match')
    ap.add_argument('--reid-thresh', type=float, default=0.42, help='max cosine distance for reid match')
    ap.add_argument('--combine', default='min', choices=['min','avg','face_priority'])
    ap.add_argument('--device', default='cuda', choices=['cuda','cpu'])
    ap.add_argument('--yolo', default='yolov8n.pt', help='ultralytics model name or path')
    ap.add_argument('--save-annot', action='store_true', help='save annotated frames beside crops')
    args = ap.parse_args()

    ensure_dir(args.out)
    crops_dir = os.path.join(args.out, 'crops')
    ann_dir = os.path.join(args.out, 'annot') if args.save_annot else None
    ensure_dir(crops_dir)
    if ann_dir:
        ensure_dir(ann_dir)

    det = PersonDetector(model_name=args.yolo, device=args.device)
    face = FaceEmbedder(ctx=args.device)
    reid = ReIDEmbedder(device=args.device)

    # Reference features
    ref_img = load_image(args.ref)
    ref_faces = face.extract(ref_img)
    ref_face = FaceEmbedder.best_face(ref_faces)
    ref_face_feat = ref_face['feat'] if ref_face else None

    # ReID reference: largest detected person else whole image
    ref_persons = det.detect(ref_img, conf=0.1)
    if ref_persons:
        ref_persons.sort(key=lambda d: (d['xyxy'][2]-d['xyxy'][0])*(d['xyxy'][3]-d['xyxy'][1]), reverse=True)
        rx1, ry1, rx2, ry2 = [int(v) for v in ref_persons[0]['xyxy']]
        ref_crop = ref_img[ry1:ry2, rx1:rx2]
        ref_reid_feat = reid.extract([ref_crop])[0]
    else:
        ref_reid_feat = reid.extract([ref_img])[0]

    # Video loop
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_idx = 0

    # CSV
    csv_path = os.path.join(args.out, 'index.csv')
    writer = csv.writer(open(csv_path, 'w', newline=''))
    writer.writerow(['frame','time_secs','score','face_dist','reid_dist','x1','y1','x2','y2','crop_path','ratio'])

    ratios = [r.strip() for r in str(args.ratio).split(',') if r.strip()]
    if not ratios:
        ratios = ['2:3']

    last_hit_time = -1e9
    min_gap_sec = 1.5  # conservative default

    while True:
        ret = cap.grab()
        if not ret:
            break
        if frame_idx % max(1, int(args.frame_stride)) != 0:
            frame_idx += 1
            continue

        ok, frame = cap.retrieve()
        if not ok:
            break
        H, W = frame.shape[:2]

        persons = det.detect(frame, conf=args.min_det_conf)
        if not persons:
            frame_idx += 1
            continue

        # Build batch crops for ReID and faces
        crops, boxes = [], []
        for p in persons:
            x1,y1,x2,y2 = [int(v) for v in p['xyxy']]
            x1 = max(0, x1); y1 = max(0, y1); x2 = min(W-1, x2); y2 = min(H-1, y2)
            if x2 <= x1+2 or y2 <= y1+2:
                continue
            crop = frame[y1:y2, x1:x2]
            crops.append(crop); boxes.append((x1,y1,x2,y2))

        if not boxes:
            frame_idx += 1
            continue

        reid_feats = reid.extract(crops)
        face_map = {}  # i -> (best_face, fd)
        for i, crop in enumerate(crops):
            ffaces = face.extract(crop)
            bf = FaceEmbedder.best_face(ffaces)
            if bf is not None and ref_face_feat is not None and bf.get('feat') is not None:
                fd = cosine_distance(bf['feat'], ref_face_feat)
                face_map[i] = (bf, fd)

        candidates = []
        for i, feat in enumerate(reid_feats):
            rd = cosine_distance(feat, ref_reid_feat)
            bf, fd = face_map.get(i, (None, None))
            score = combine_scores(fd, rd, mode=args.combine)
            if score is None:
                continue
            # Apply basic thresholds
            if fd is not None and fd > float(args.face_thresh):
                fd_ok = False
            else:
                fd_ok = True
            if rd is not None and rd > float(args.reid_thresh):
                rd_ok = False
            else:
                rd_ok = True
            if not (fd_ok or rd_ok):
                continue

            # Expand to chosen ratio that minimally enlarges
            best_box, best_ratio = None, None
            bx1,by1,bx2,by2 = boxes[i]
            det_area = max(1, (bx2-bx1)*(by2-by1))
            best_factor = 1e9
            for rs in ratios:
                rw, rh = parse_ratio(rs)
                ex1,ey1,ex2,ey2 = expand_box_to_ratio(bx1,by1,bx2,by2,rw,rh,W,H,anchor=None,head_bias=0.12)
                area = max(1, (ex2-ex1)*(ey2-ey1))
                factor = area/det_area
                if factor < best_factor:
                    best_factor = factor
                    best_box = (ex1,ey1,ex2,ey2)
                    best_ratio = rs

            candidates.append((score, fd, rd, best_box, best_ratio))

        if not candidates:
            frame_idx += 1
            continue

        candidates.sort(key=lambda t: (t[0], -(t[3][2]-t[3][0])*(t[3][3]-t[3][1])))
        chosen = candidates[0]
        score, fd, rd, box, ratio_str = chosen

        # cadence
        now_t = frame_idx/float(fps)
        if now_t - last_hit_time < min_gap_sec:
            frame_idx += 1
            continue

        # save crop
        cx1,cy1,cx2,cy2 = box
        crop_img = frame[cy1:cy2, cx1:cx2]
        out_path = os.path.join(crops_dir, f"f{frame_idx:08d}.jpg")
        cv2.imwrite(out_path, crop_img)
        writer.writerow([frame_idx, now_t, score, fd, rd, cx1, cy1, cx2, cy2, out_path, ratio_str])

        if ann_dir:
            ann = frame.copy()
            cv2.rectangle(ann, (cx1,cy1), (cx2,cy2), (0,255,0), 2)
            cv2.imwrite(os.path.join(ann_dir, f"f{frame_idx:08d}.jpg"), ann)

        last_hit_time = now_t
        frame_idx += 1

    cap.release()
    print(f"Done. Index: {csv_path}")

if __name__ == '__main__':
    main()
