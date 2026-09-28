import os
import sys
import time
import numpy as np

from config import FEATURES_DIR, SPLITS
from dataset import list_images, load_image_with_labels
from features import features_for_spaces


def build(split):
    images = list_images(split)
    if not images:
        sys.exit(f"[ERROR] No images found for split '{split}'. Set PARKSENSE_DATA to the ML-DataSet folder.")
    X, y, img_idx = [], [], []
    t0 = time.time()
    for i, path in enumerate(images):
        img, spaces, classes = load_image_with_labels(path)
        if not spaces:
            continue
        X.append(features_for_spaces(img, spaces))
        y.extend(classes)
        img_idx.extend([i] * len(classes))
        if (i + 1) % 100 == 0 or i + 1 == len(images):
            print(f"  {split:5s} {i + 1:4d}/{len(images)} images  {len(y):6d} patches  {time.time() - t0:5.1f}s")
    X = np.vstack(X)
    y = np.array(y, dtype=np.int8)
    img_idx = np.array(img_idx, dtype=np.int32)
    os.makedirs(FEATURES_DIR, exist_ok=True)
    out = os.path.join(FEATURES_DIR, f"{split}.npz")
    np.savez_compressed(out, X=X, y=y, img_idx=img_idx, images=np.array([os.path.basename(p) for p in images]))
    print(f"[OK] {split}: X={X.shape}  free={(y == 0).sum()}  occupied={(y == 1).sum()}  -> {out}")


if __name__ == "__main__":
    for s in (sys.argv[1:] or SPLITS):
        build(s)
