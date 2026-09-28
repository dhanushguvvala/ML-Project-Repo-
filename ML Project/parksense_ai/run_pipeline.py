import subprocess
import sys

STEPS = [
    ["build_features.py"],
    ["audit_labels.py"],
    ["train_model.py"] + sys.argv[1:],
    ["detect.py", "--folder", None, "--compare"],
]

if __name__ == "__main__":
    from config import DATASET_DIR
    import os
    STEPS[3][2] = os.path.join(DATASET_DIR, "test", "images")
    for step in STEPS:
        print(f"\n>>> python {' '.join(step)}")
        if subprocess.call([sys.executable] + step) != 0:
            sys.exit(f"[FAILED] {step[0]}")
    print("\n[DONE] Pipeline complete. Launch dashboard: streamlit run app.py")
