import os
import pickle
import numpy as np
import cv2

from config import MODEL_FILE, COLOR_FREE, COLOR_OCCUPIED, COLOR_WRONG, COLOR_TEXT, COLOR_HUD_BG, BOX_THICK
from features import features_for_spaces


class ParkingPredictor:
    def __init__(self, model_file=MODEL_FILE):
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"{model_file} not found. Run: python train_model.py")
        with open(model_file, "rb") as f:
            bundle = pickle.load(f)
        self.model = bundle["model"]
        self.name = bundle["name"]

    def predict(self, img, spaces):
        if not spaces:
            return np.array([], dtype=int), np.array([])
        X = features_for_spaces(img, spaces)
        labels = self.model.predict(X).astype(int)
        score = self.model.decision_function(X)
        confidence = 1 / (1 + np.exp(-np.abs(score)))
        return labels, confidence


def summarize(labels, truth=None):
    total = len(labels)
    occupied = int(np.sum(labels))
    out = {"total": total, "occupied": occupied, "free": total - occupied, "occupancy_pct": round(occupied / total * 100, 1) if total else 0.0}
    if truth is not None and len(truth) == total and total:
        truth = np.asarray(truth)
        out["gt_occupied"] = int(truth.sum())
        out["space_accuracy"] = round(float((truth == labels).mean()) * 100, 2)
        out["wrong_spaces"] = [int(i) + 1 for i in np.where(truth != labels)[0]]
    return out


def annotate(img, spaces, labels, truth=None, show_ids=False):
    out = img.copy()
    overlay = img.copy()
    for (x, y, w, h), lab in zip(spaces, labels):
        cv2.rectangle(overlay, (x, y), (x + w, y + h), COLOR_OCCUPIED if lab else COLOR_FREE, -1)
    cv2.addWeighted(overlay, 0.25, out, 0.75, 0, out)
    for i, ((x, y, w, h), lab) in enumerate(zip(spaces, labels)):
        cv2.rectangle(out, (x, y), (x + w, y + h), COLOR_OCCUPIED if lab else COLOR_FREE, BOX_THICK)
        if truth is not None and truth[i] != lab:
            cv2.rectangle(out, (x - 3, y - 3), (x + w + 3, y + h + 3), COLOR_WRONG, BOX_THICK)
        if show_ids:
            cv2.putText(out, str(i + 1), (x + 3, y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT, 1)

    s = summarize(labels, truth)
    H, W = out.shape[:2]
    hud_h = 64
    cv2.rectangle(out, (0, H - hud_h), (W, H), COLOR_HUD_BG, -1)
    cv2.putText(out, "PARK SENSE AI", (12, H - hud_h + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 255), 2)
    line = f"Total: {s['total']}   Free: {s['free']}   Occupied: {s['occupied']}   Occupancy: {s['occupancy_pct']:.0f}%"
    if "space_accuracy" in s:
        line += f"   vs labels: {s['space_accuracy']:.1f}%"
    cv2.putText(out, line, (12, H - hud_h + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1)
    bx, by, bw, bh = W - 230, H - hud_h + 12, 200, 16
    pct = s["occupancy_pct"]
    col = COLOR_OCCUPIED if pct > 80 else ((0, 165, 255) if pct > 50 else COLOR_FREE)
    cv2.rectangle(out, (bx, by), (bx + bw, by + bh), (80, 80, 80), -1)
    cv2.rectangle(out, (bx, by), (bx + int(bw * pct / 100), by + bh), col, -1)
    cv2.rectangle(out, (bx, by), (bx + bw, by + bh), (150, 150, 150), 1)
    return out, s
