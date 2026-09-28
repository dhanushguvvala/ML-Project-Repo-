import os
import json
import tempfile
import numpy as np
import cv2
import streamlit as st

from config import METRICS_FILE, SPACES_FILE, REPORTS_DIR
from dataset import read_yolo_labels
from predictor import ParkingPredictor, annotate
from spaces import load_spaces

st.set_page_config(page_title="Park Sense AI", page_icon="🅿️", layout="wide")


@st.cache_resource
def get_predictor():
    return ParkingPredictor()


st.title("🅿️ Park Sense ")
st.caption("Parking space occupancy detection · HOG + colour features · SVM · trained on PKLot")

tab_detect, tab_model = st.tabs(["Detect", "Model report"])

with tab_detect:
    c1, c2 = st.columns(2)
    img_file = c1.file_uploader("Parking lot image", type=["jpg", "jpeg", "png"])
    lbl_file = c2.file_uploader("Space boxes (YOLO .txt) — optional if spaces.json exists", type=["txt"])
    compare = st.checkbox("Highlight disagreements with the label file", value=False)

    if img_file:
        img = cv2.imdecode(np.frombuffer(img_file.read(), np.uint8), cv2.IMREAD_COLOR)
        truth = None
        if lbl_file:
            with tempfile.NamedTemporaryFile("wb", suffix=".txt", delete=False) as f:
                f.write(lbl_file.read())
            spaces, truth = read_yolo_labels(f.name, img.shape[1], img.shape[0])
            os.remove(f.name)
        else:
            spaces = load_spaces(SPACES_FILE)
        if not spaces:
            st.warning("No parking spaces defined. Upload a YOLO label file or create spaces.json with define_spaces.py.")
        else:
            labels, conf = get_predictor().predict(img, spaces)
            out, s = annotate(img, spaces, labels, truth if compare else None)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total", s["total"])
            m2.metric("Free", s["free"])
            m3.metric("Occupied", s["occupied"])
            m4.metric("Occupancy", f"{s['occupancy_pct']:.0f}%")
            if truth is not None and len(truth) == len(labels):
                acc = float((np.array(truth) == labels).mean() * 100)
                st.info(f"Agreement with label file: {acc:.2f}%  ({int((np.array(truth) != labels).sum())} spaces differ)")
            st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), width="stretch")
            low = [i + 1 for i, c in enumerate(conf) if c < 0.75]
            if low:
                st.caption(f"Low-confidence spaces: {low}")
            ok, buf = cv2.imencode(".jpg", out)
            st.download_button("Download result", buf.tobytes(), "parksense_result.jpg", "image/jpeg")

with tab_model:
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE) as f:
            m = json.load(f)
        st.subheader(f"Selected model: {m['selected_model']}")
        a, b = st.columns(2)
        a.markdown("**Test set (audited, clean labels)**")
        a.json({k: m["test_clean"][k] for k in ("accuracy", "precision", "recall", "f1")})
        b.markdown("**Test set (all labels as given)**")
        b.json({k: m["test_all_labels"][k] for k in ("accuracy", "precision", "recall", "f1")})
        st.markdown("**Validation comparison**")
        st.table(m["validation"])
        cm_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion matrix (clean test set)", width=380)
    else:
        st.warning("Run train_model.py to generate the model report.")
