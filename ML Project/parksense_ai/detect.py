import os
import sys
import csv
import glob
import argparse
import cv2

from config import OUTPUT_DIR, SPACES_FILE
from dataset import label_path_for, read_yolo_labels, IMG_EXT
from predictor import ParkingPredictor, annotate
from spaces import load_spaces


def resolve_spaces(img, image_path, args):
    if args.spaces:
        spaces = load_spaces(args.spaces)
        if not spaces:
            sys.exit(f"[ERROR] No spaces in {args.spaces}")
        return spaces, None
    label = args.labels or label_path_for(image_path)
    if os.path.exists(label):
        spaces, truth = read_yolo_labels(label, img.shape[1], img.shape[0])
        if spaces:
            return spaces, truth
    spaces = load_spaces(SPACES_FILE)
    if spaces:
        return spaces, None
    return None, None


def process_image(predictor, path, args):
    img = cv2.imread(path)
    if img is None:
        print(f"[SKIP] cannot read {path}")
        return None
    spaces, truth = resolve_spaces(img, path, args)
    if not spaces:
        print(f"[SKIP] {os.path.basename(path)}: no parking spaces (pass --labels/--spaces or create spaces.json)")
        return None
    labels, conf = predictor.predict(img, spaces)
    out, s = annotate(img, spaces, labels, truth if args.compare else None, args.ids)
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, os.path.splitext(os.path.basename(path))[0] + "_result.jpg")
    cv2.imwrite(out_path, out)
    s["image"] = os.path.basename(path)
    s["low_confidence_spaces"] = [i + 1 for i, c in enumerate(conf) if c < 0.75]
    if truth is not None:
        s["gt_occupied"] = int(sum(truth))
        s["space_accuracy"] = round(sum(int(a == b) for a, b in zip(truth, labels)) / len(truth) * 100, 2)
    print(f"{s['image'][:40]:40s} total={s['total']:3d} free={s['free']:3d} occupied={s['occupied']:3d} "
          f"({s['occupancy_pct']:5.1f}%)" + (f"  labels_occ={s['gt_occupied']:3d} acc={s['space_accuracy']:6.2f}%" if truth is not None else ""))
    if args.show:
        cv2.imshow("Park Sense ", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return s, out_path


def run_folder(predictor, folder, args):
    paths = sorted(p for p in glob.glob(os.path.join(folder, "*")) if p.lower().endswith(IMG_EXT))
    rows = [r[0] for r in (process_image(predictor, p, args) for p in paths) if r]
    csv_path = os.path.join(args.out, "summary.csv")
    keys = ["image", "total", "free", "occupied", "occupancy_pct", "gt_occupied", "space_accuracy"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n[OK] {len(rows)} images -> {args.out}\n[OK] Summary -> {csv_path}")


def run_video(predictor, path, args):
    spaces = load_spaces(args.spaces or SPACES_FILE)
    if not spaces:
        sys.exit("[ERROR] Video mode needs fixed spaces. Create spaces.json with define_spaces.py")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, os.path.splitext(os.path.basename(path))[0] + "_result.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    step = max(1, int(fps * args.every))
    idx, labels = 0, None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if labels is None or idx % step == 0:
            labels, _ = predictor.predict(frame, spaces)
        out, s = annotate(frame, spaces, labels)
        writer.write(out)
        if args.show:
            cv2.imshow("Park Sense  (q to quit)", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        idx += 1
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"[OK] {idx} frames -> {out_path}")


def main():
    p = argparse.ArgumentParser(description="Park Sense  - parking space occupancy detection")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image")
    src.add_argument("--folder")
    src.add_argument("--video")
    p.add_argument("--labels", help="YOLO label .txt giving space boxes for --image")
    p.add_argument("--spaces", help="spaces.json with fixed space boxes")
    p.add_argument("--compare", action="store_true", help="highlight spaces where prediction differs from labels")
    p.add_argument("--ids", action="store_true", help="draw space numbers")
    p.add_argument("--every", type=float, default=1.0, help="video: seconds between re-classification")
    p.add_argument("--out", default=OUTPUT_DIR)
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    predictor = ParkingPredictor()
    print(f"[OK] Model: {predictor.name}")
    if args.video:
        run_video(predictor, args.video, args)
    elif args.folder:
        run_folder(predictor, args.folder, args)
    else:
        process_image(predictor, args.image, args)


if __name__ == "__main__":
    main()
