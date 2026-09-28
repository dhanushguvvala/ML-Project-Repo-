import os
import sys
import json
import numpy as np
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from config import FEATURES_DIR, REPORTS_DIR, PCA_COMPONENTS, RANDOM_STATE, SPLITS

AUDIT_FILE = os.path.join(REPORTS_DIR, "label_audit.json")
DISAGREE_THRESHOLD = 0.30
AUDIT_SAMPLE = 12000


def make_model():
    return Pipeline([("scaler", StandardScaler()), ("pca", PCA(PCA_COMPONENTS, random_state=RANDOM_STATE)), ("clf", SVC(C=3, gamma="scale"))])


def grouped_predictions(X, y, idx, excluded):
    pred = np.zeros_like(y)
    keep = ~np.isin(idx, list(excluded))
    for tr, te in GroupKFold(3).split(X, y, idx):
        tr = tr[keep[tr]]
        if len(tr) > AUDIT_SAMPLE:
            tr = np.random.RandomState(RANDOM_STATE).choice(tr, AUDIT_SAMPLE, replace=False)
        pred[te] = make_model().fit(X[tr], y[tr]).predict(X[te])
    return pred


def audit_split(split, max_rounds=3):
    d = np.load(os.path.join(FEATURES_DIR, f"{split}.npz"))
    X, y, idx, names = d["X"], d["y"].astype(int), d["img_idx"], d["images"]
    excluded, rows = set(), {}
    for r in range(max_rounds):
        pred = grouped_predictions(X, y, idx, excluded)
        new = 0
        for i in np.unique(idx):
            m = idx == i
            rate = float((pred[m] != y[m]).mean())
            if rate > DISAGREE_THRESHOLD:
                rows[int(i)] = {
                    "image": str(names[i]),
                    "spaces": int(m.sum()),
                    "label_occupied": int(y[m].sum()),
                    "cv_predicted_occupied": int(pred[m].sum()),
                    "disagreement": round(rate, 3),
                }
                if i not in excluded:
                    excluded.add(int(i))
                    new += 1
        print(f"{split:5s} round {r + 1}: +{new} flagged (total {len(excluded)})")
        if new == 0:
            break
    flagged = [rows[i] for i in sorted(rows)]
    for f in flagged:
        print(f"   {f['image'][:19]}  labels occ={f['label_occupied']:3d}  model occ={f['cv_predicted_occupied']:3d}  / {f['spaces']}")
    return flagged


def load_flagged():
    if not os.path.exists(AUDIT_FILE):
        return {}
    with open(AUDIT_FILE) as f:
        data = json.load(f)
    return {s: {r["image"] for r in rows} for s, rows in data.items()}


if __name__ == "__main__":
    splits = sys.argv[1:] or SPLITS
    report = {s: audit_split(s) for s in splits}
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(AUDIT_FILE, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[OK] Audit -> {AUDIT_FILE}")
