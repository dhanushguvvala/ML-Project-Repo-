import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.environ.get("PARKSENSE_DATA", os.path.join(os.path.dirname(BASE_DIR), "ML-DataSet"))
SPLITS = ("train", "valid", "test")

MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FEATURES_DIR = os.path.join(BASE_DIR, "features")
MODEL_FILE = os.path.join(MODELS_DIR, "model.pkl")
METRICS_FILE = os.path.join(REPORTS_DIR, "metrics.json")
SPACES_FILE = os.path.join(BASE_DIR, "spaces", "spaces.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

CLASS_NAMES = {0: "FREE", 1: "OCCUPIED"}

PATCH_W = 64
PATCH_H = 64
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HSV_BINS = 8

PCA_COMPONENTS = 150
RANDOM_STATE = 42

COLOR_FREE = (0, 200, 0)
COLOR_OCCUPIED = (0, 0, 220)
COLOR_WRONG = (0, 215, 255)
COLOR_TEXT = (255, 255, 255)
COLOR_HUD_BG = (30, 30, 30)
BOX_THICK = 2
