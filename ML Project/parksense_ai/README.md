# Park Sense AI — Parking Space Occupancy Detection

Classifies every parking space in a lot image as **FREE** or **OCCUPIED**, trained on a 1,200-image subset of the PKLot dataset (YOLOv8 label format).

## Pipeline

| Stage | What happens |
|---|---|
| Space ROIs | Boxes from YOLO labels, or a fixed `spaces.json` per camera |
| Patch | Each space cropped and resized to 64×64 |
| Features (1792-d) | HOG (1764) + HSV colour histograms (24) + texture stats (4) |
| Label audit | Grouped cross-validation flags images whose labels contradict the image |
| Model | StandardScaler → PCA(150) → SVM (RBF), chosen on validation F1 |
| Output | Annotated image/video, occupancy HUD, CSV summary, Streamlit dashboard |

## Dataset

| Split | Images | Space patches |
|---|---|---|
| train | 1000 | 56,448 (54,048 used after audit) |
| valid | 100 | 5,876 |
| test | 100 | 5,544 (one image has no labels) |

Classes: `0 = FREE`, `1 = OCCUPIED`.

## Results (held-out test set)

| Evaluation | Accuracy | F1 (occupied) | Mean per-image count error |
|---|---|---|---|
| Test, labels as given | 96.32% | 0.9616 | 2.06 spaces |
| Test, 2 mislabeled images excluded | 99.89% | 0.9988 | 0.06 spaces |

Validation: logistic regression 99.59%, RBF SVM C=3 99.83%, RBF SVM C=10 99.85% (selected).

### Label noise found in the dataset
24 train images and 2 test images (camera with 100 spaces, 2012-10-30 to 2012-11-07) mark nearly every space FREE while the photos show a full lot. `audit_labels.py` detects them automatically; they are listed in `label_audit.json`. Training excludes the 24 train images by default (`--keep-flagged` to include them). Both test numbers are reported.

### Limitation
PKLot train/valid/test come from the same cameras, so these scores measure performance on known cameras. Accuracy on a new camera angle will be lower until images from it are added to training.

## Project layout

```
ML-DataSet/                     dataset (keep next to parksense_ai/)
parksense_ai/
├── README.md
├── requirements.txt
├── run_pipeline.py             runs the full pipeline end to end
│
├── config.py                   paths, feature and display settings
├── dataset.py                  YOLO label reading
├── features.py                 patch cropping + HOG/colour/texture features
├── build_features.py           step 1: extract features -> features/
├── audit_labels.py             step 2: find mislabeled images -> reports/label_audit.json
├── train_model.py              step 3: compare models, select, test -> models/, reports/
├── predictor.py                model loading, prediction, drawing
├── spaces.py                   spaces.json read/write
├── define_spaces.py            create spaces.json (from labels or by drawing)
├── detect.py                   CLI: image / folder / video
├── app.py                      Streamlit dashboard
│
├── models/
│   └── model.pkl               trained RBF SVM pipeline
├── reports/
│   ├── metrics.json            validation + test scores
│   ├── label_audit.json        flagged mislabeled images
│   ├── confusion_matrix.png
│   └── detection_result.jpg    example output
├── spaces/
│   └── spaces.json             example: 40-space camera
├── samples/
│   ├── lot_camera_100.jpg      sample image + its label file
│   ├── lot_camera_100.txt
│   └── lot_camera_40.jpg       sample for spaces.json mode
├── features/                   created by build_features.py
└── outputs/                    created by detect.py
```

The dataset folder must be named `ML-DataSet` and sit beside `parksense_ai/`, or set the `PARKSENSE_DATA` environment variable to its path.

## Run

```bash
pip install -r requirements.txt

python run_pipeline.py

python detect.py --image samples/lot_camera_100.jpg --labels samples/lot_camera_100.txt --compare
python detect.py --image samples/lot_camera_40.jpg --spaces spaces/spaces.json --ids
python detect.py --folder ../ML-DataSet/test/images --compare
python detect.py --video cam.mp4 --spaces spaces/spaces.json

python define_spaces.py --image lot.jpg --from-labels
python define_spaces.py --image lot.jpg

streamlit run app.py
```

`models/model.pkl` is already trained, so `detect.py` and `app.py` work without retraining. Results go to `outputs/`. `run_pipeline.py` does the full retrain (about 5 minutes); individual steps are `build_features.py`, `audit_labels.py`, `train_model.py`.

In `--compare` mode, yellow outlines mark spaces where the prediction differs from the label file.
