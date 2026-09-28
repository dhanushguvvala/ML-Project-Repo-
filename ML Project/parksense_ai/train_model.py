import os
import sys
import json
import time
import pickle
import numpy as np
import cv2
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

from audit_labels import load_flagged
from config import FEATURES_DIR, MODEL_FILE, METRICS_FILE, REPORTS_DIR, PCA_COMPONENTS, RANDOM_STATE, CLASS_NAMES


def load(split, drop=()):
    path = os.path.join(FEATURES_DIR, f"{split}.npz")
    if not os.path.exists(path):
        sys.exit(f"[ERROR] {path} missing. Run: python build_features.py")
    d = np.load(path)
    bad = [i for i, n in enumerate(d["images"]) if n in drop]
    keep = ~np.isin(d["img_idx"], bad)
    return d["X"][keep], d["y"][keep].astype(int), d["img_idx"][keep]


def evaluate(model, X, y, idx):
    t0 = time.time()
    pred = model.predict(X)
    ms = (time.time() - t0) / len(y) * 1000
    return pred, scores(y, pred), confusion_matrix(y, pred, labels=[0, 1]), occupancy_error(y, pred, idx), ms


def print_block(title, y, pred, cm, occ):
    print("\n" + "-" * 60)
    print(f" {title}: {len(y)} spaces from {occ['images']} images")
    print("-" * 60)
    print(classification_report(y, pred, target_names=[CLASS_NAMES[0], CLASS_NAMES[1]], digits=4))
    print(f"Confusion matrix [rows=actual FREE,OCC]:\n{cm}")
    print(f"Per-image occupied-count error: mean {occ['mean_abs_count_error']}  max {occ['max_abs_count_error']}  exact {occ['exact_count_images']}/{occ['images']}")


def candidates():
    return {
        "logreg": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=0.01, max_iter=2000))]),
        "rbf_svm_C3": Pipeline([("scaler", StandardScaler()), ("pca", PCA(PCA_COMPONENTS, random_state=RANDOM_STATE)), ("clf", SVC(C=3, gamma="scale"))]),
        "rbf_svm_C10": Pipeline([("scaler", StandardScaler()), ("pca", PCA(PCA_COMPONENTS, random_state=RANDOM_STATE)), ("clf", SVC(C=10, gamma="scale"))]),
    }


def scores(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    return {"accuracy": round(accuracy_score(y_true, y_pred), 4), "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}


def occupancy_error(y_true, y_pred, img_idx):
    errs = [abs(int(y_pred[img_idx == i].sum()) - int(y_true[img_idx == i].sum())) for i in np.unique(img_idx)]
    totals = [int((img_idx == i).sum()) for i in np.unique(img_idx)]
    return {
        "images": len(errs),
        "mean_abs_count_error": round(float(np.mean(errs)), 3),
        "max_abs_count_error": int(np.max(errs)),
        "exact_count_images": int(sum(e == 0 for e in errs)),
        "mean_spaces_per_image": round(float(np.mean(totals)), 1),
    }


def save_confusion(cm, path):
    cell, pad = 150, 110
    img = np.full((pad + 2 * cell + 40, pad + 2 * cell + 20, 3), 255, np.uint8)
    names = [CLASS_NAMES[0], CLASS_NAMES[1]]
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            v = cm[i, j]
            shade = int(255 - 200 * v / cm.max())
            x0, y0 = pad + j * cell, pad + i * cell
            cv2.rectangle(img, (x0, y0), (x0 + cell, y0 + cell), (255, shade, shade), -1)
            cv2.rectangle(img, (x0, y0), (x0 + cell, y0 + cell), (60, 60, 60), 1)
            col = (255, 255, 255) if shade < 130 else (20, 20, 20)
            cv2.putText(img, str(v), (x0 + 40, y0 + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
            cv2.putText(img, f"{v / total * 100:.1f}%", (x0 + 45, y0 + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)
        cv2.putText(img, names[i], (5, pad + i * cell + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1)
        cv2.putText(img, names[i], (pad + i * cell + 30, pad - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1)
    cv2.putText(img, "Predicted", (pad + cell - 45, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
    cv2.putText(img, "Actual", (5, pad - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
    cv2.putText(img, "Test set confusion matrix", (pad - 20, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1)
    cv2.imwrite(path, img)


def main():
    keep_flagged = "--keep-flagged" in sys.argv
    flagged = load_flagged()
    print("=" * 60)
    print(" PARK SENSE AI - Training on PKLot subset")
    print("=" * 60)
    drop_train = set() if keep_flagged else flagged.get("train", set())
    X_tr, y_tr, _ = load("train", drop_train)
    X_va, y_va, _ = load("valid")
    print(f"train {X_tr.shape} (excluded {len(drop_train)} mislabeled images)  valid {X_va.shape}\n")

    results, models = {}, {}
    for name, model in candidates().items():
        t0 = time.time()
        model.fit(X_tr, y_tr)
        fit_s = time.time() - t0
        s = scores(y_va, model.predict(X_va))
        s["fit_seconds"] = round(fit_s, 1)
        results[name] = s
        models[name] = model
        print(f"{name:12s} valid acc={s['accuracy']:.4f} f1={s['f1']:.4f}  ({fit_s:.1f}s)")

    best = max(results, key=lambda n: (results[n]["f1"], results[n]["accuracy"]))
    model = models[best]
    print(f"\n[SELECTED] {best} (best validation F1)")

    X_te, y_te, idx_te = load("test")
    pred, test_all, cm_all, occ_all, ms = evaluate(model, X_te, y_te, idx_te)
    print_block("TEST - all labels as given", y_te, pred, cm_all, occ_all)

    drop_test = flagged.get("test", set())
    X_tc, y_tc, idx_tc = load("test", drop_test)
    pred_c, test_clean, cm_clean, occ_clean, _ = evaluate(model, X_tc, y_tc, idx_tc)
    print_block(f"TEST - excluding {len(drop_test)} audited mislabeled images", y_tc, pred_c, cm_clean, occ_clean)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump({"model": model, "name": best}, f)
    metrics = {
        "selected_model": best,
        "train_spaces": int(len(y_tr)),
        "train_images_excluded": sorted(drop_train),
        "feature_dim": int(X_tr.shape[1]),
        "validation": results,
        "test_all_labels": {**test_all, "confusion_matrix": cm_all.tolist(), "occupancy_count": occ_all},
        "test_clean": {**test_clean, "excluded_images": sorted(drop_test), "confusion_matrix": cm_clean.tolist(), "occupancy_count": occ_clean},
        "inference_ms_per_space": round(ms, 3),
    }
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
    save_confusion(cm_clean, os.path.join(REPORTS_DIR, "confusion_matrix.png"))
    print(f"\n[OK] Model   -> {MODEL_FILE}\n[OK] Metrics -> {METRICS_FILE}")


if __name__ == "__main__":
    main()
