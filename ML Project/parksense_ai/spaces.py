import json
import os

from config import SPACES_FILE
from dataset import load_image_with_labels


def save_spaces(spaces, path=SPACES_FILE, source=None):
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"source": source, "spaces": [list(map(int, s)) for s in spaces]}, f, indent=1)


def load_spaces(path=SPACES_FILE):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [tuple(s) for s in json.load(f)["spaces"]]


def spaces_from_labels(image_path, label_path=None):
    _, spaces, classes = load_image_with_labels(image_path, label_path)
    return spaces, classes
