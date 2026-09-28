import os
import glob
import cv2

from config import DATASET_DIR

IMG_EXT = (".jpg", ".jpeg", ".png")


def list_images(split):
    folder = os.path.join(DATASET_DIR, split, "images")
    return sorted(p for p in glob.glob(os.path.join(folder, "*")) if p.lower().endswith(IMG_EXT))


def label_path_for(image_path):
    folder = os.path.dirname(os.path.dirname(image_path))
    name = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
    return os.path.join(folder, "labels", name)


def read_yolo_labels(label_path, img_w, img_h):
    spaces, classes = [], []
    if not os.path.exists(label_path):
        return spaces, classes
    with open(label_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 5:
                continue
            c, xc, yc, w, h = int(parts[0]), *map(float, parts[1:])
            bw, bh = w * img_w, h * img_h
            spaces.append((int(round(xc * img_w - bw / 2)), int(round(yc * img_h - bh / 2)), int(round(bw)), int(round(bh))))
            classes.append(c)
    return spaces, classes


def load_image_with_labels(image_path, label_path=None):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    spaces, classes = read_yolo_labels(label_path or label_path_for(image_path), img.shape[1], img.shape[0])
    return img, spaces, classes


def camera_of(image_path):
    name = os.path.basename(image_path)
    return name[:10]
