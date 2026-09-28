import cv2
import numpy as np
from skimage.feature import hog
from config import PATCH_W, PATCH_H, HOG_ORIENTATIONS, HOG_PIXELS_PER_CELL, HOG_CELLS_PER_BLOCK, HSV_BINS


def extract_patch(img, x, y, w, h):
    ih, iw = img.shape[:2]
    x1, y1 = max(0, int(x)), max(0, int(y))
    x2, y2 = min(iw, int(x + w)), min(ih, int(y + h))
    patch = img[y1:y2, x1:x2]
    if patch.size == 0:
        return np.zeros((PATCH_H, PATCH_W, 3), dtype=np.uint8)
    return cv2.resize(patch, (PATCH_W, PATCH_H), interpolation=cv2.INTER_AREA)


def hog_features(gray):
    return hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        feature_vector=True,
    )


def color_features(patch):
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    feats = []
    for ch, rng in zip(range(3), (180, 256, 256)):
        hist = cv2.calcHist([hsv], [ch], None, [HSV_BINS], [0, rng]).ravel()
        feats.append(hist / (hist.sum() + 1e-6))
    return np.concatenate(feats)


def texture_features(gray):
    edges = cv2.Canny(gray, 50, 150)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return np.array([
        gray.mean() / 255.0,
        gray.std() / 255.0,
        (edges > 0).mean(),
        np.log1p(lap.var()) / 10.0,
    ])


def extract_features(patch):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return np.concatenate([hog_features(gray), color_features(patch), texture_features(gray)]).astype(np.float32)


def features_for_spaces(img, spaces):
    return np.array([extract_features(extract_patch(img, *s)) for s in spaces], dtype=np.float32)
