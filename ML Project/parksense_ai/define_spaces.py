import sys
import argparse
import cv2

from config import SPACES_FILE
from spaces import save_spaces, load_spaces, spaces_from_labels


def from_labels(image, labels, out):
    spaces, _ = spaces_from_labels(image, labels)
    if not spaces:
        sys.exit("[ERROR] No boxes found in label file")
    save_spaces(spaces, out, source=image)
    print(f"[OK] {len(spaces)} spaces from labels -> {out}")


def interactive(image, out):
    img = cv2.imread(image)
    if img is None:
        sys.exit(f"[ERROR] Cannot read {image}")
    spaces = load_spaces(out)
    state = {"start": None, "cur": None}

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["start"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and state["start"]:
            state["cur"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and state["start"]:
            (x0, y0) = state["start"]
            w, h = abs(x - x0), abs(y - y0)
            if w > 5 and h > 5:
                spaces.append((min(x, x0), min(y, y0), w, h))
            state["start"] = state["cur"] = None

    cv2.namedWindow("Define Parking Spaces")
    cv2.setMouseCallback("Define Parking Spaces", on_mouse)
    print("drag = add space | u = undo | c = clear | s = save | q = quit")
    while True:
        frame = img.copy()
        for i, (x, y, w, h) in enumerate(spaces):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 255), 2)
            cv2.putText(frame, str(i + 1), (x + 3, y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
        if state["start"] and state["cur"]:
            cv2.rectangle(frame, state["start"], state["cur"], (0, 255, 255), 1)
        cv2.putText(frame, f"spaces: {len(spaces)}  u=undo c=clear s=save q=quit", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("Define Parking Spaces", frame)
        k = cv2.waitKey(20) & 0xFF
        if k == ord("u") and spaces:
            spaces.pop()
        elif k == ord("c"):
            spaces.clear()
        elif k == ord("s"):
            save_spaces(spaces, out, source=image)
            print(f"[OK] saved {len(spaces)} spaces -> {out}")
        elif k in (ord("q"), 27):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Create spaces.json for a fixed camera")
    p.add_argument("--image", required=True)
    p.add_argument("--from-labels", nargs="?", const="auto", help="build from YOLO labels (auto-finds the label file)")
    p.add_argument("--out", default=SPACES_FILE)
    a = p.parse_args()
    if a.from_labels:
        from_labels(a.image, None if a.from_labels == "auto" else a.from_labels, a.out)
    else:
        interactive(a.image, a.out)
