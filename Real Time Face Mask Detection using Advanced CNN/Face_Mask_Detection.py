import cv2
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import time
import json
import os
import sys

# ─────────────────────────────────────────────
# PRE-RUN CHECK
# Ensures train_model.py was run first
# ─────────────────────────────────────────────
REQUIRED_FILES = {
    "mask_detector.keras": "Run train_model.py first to generate the trained model.",
    "labels.json":         "Run train_model.py first to generate the label mapping."
}

missing = False
for filename, message in REQUIRED_FILES.items():
    if not os.path.exists(filename):
        print(f"❌ Missing: '{filename}' — {message}")
        missing = True

if missing:
    print("\n▶ Correct order:\n  1. python train_model.py\n  2. python Face_Mask_Detection.py")
    sys.exit(1)

print("✅ mask_detector.keras found.")
print("✅ labels.json found.")

# ─────────────────────────────────────────────
# LOAD MODEL & LABELS
# ─────────────────────────────────────────────
model = load_model("mask_detector.keras")

with open("labels.json") as f:
    class_indices = json.load(f)

# index → label name  (e.g. {0: "IncorrectMask", 1: "Mask", 2: "WithoutMask"})
index_to_label = {v: k for k, v in class_indices.items()}
num_classes = len(class_indices)

print(f"\n✅ Labels loaded: {class_indices}\n")

# ─────────────────────────────────────────────
# FIND WHICH INDEX BELONGS TO EACH CATEGORY
# Works regardless of what you named your folders
# ─────────────────────────────────────────────
def find_label_index(keyword):
    """Case-insensitive, underscore/space-insensitive search."""
    keyword_clean = keyword.lower().replace("_", "").replace(" ", "")
    for label, idx in class_indices.items():
        label_clean = label.lower().replace("_", "").replace(" ", "")
        if keyword_clean in label_clean:
            return idx
    return None

INCORRECT_IDX = find_label_index("incorrect")
MASK_IDX      = find_label_index("mask")
NOMASK_IDX    = find_label_index("without")

if NOMASK_IDX is None:
    NOMASK_IDX = find_label_index("nomask") or find_label_index("no")

# Make sure MASK_IDX doesn't point to incorrect-mask class
if MASK_IDX == INCORRECT_IDX:
    for label, idx in class_indices.items():
        lc = label.lower()
        if "mask" in lc and "incorrect" not in lc and "without" not in lc and "no" not in lc:
            MASK_IDX = idx
            break

print(f"  Mask index      : {MASK_IDX}  → {index_to_label.get(MASK_IDX)}")
print(f"  IncorrectMask   : {INCORRECT_IDX}  → {index_to_label.get(INCORRECT_IDX)}")
print(f"  NoMask index    : {NOMASK_IDX}  → {index_to_label.get(NOMASK_IDX)}")

# ─────────────────────────────────────────────
# THRESHOLDS
# ─────────────────────────────────────────────
INCORRECT_THRESHOLD = 0.40
SMOOTHING_FRAMES    = 5

# ─────────────────────────────────────────────
# PREDICTION SMOOTHER
# ─────────────────────────────────────────────
class PredictionSmoother:
    def __init__(self, n=SMOOTHING_FRAMES, num_classes=3):
        self.n = n
        self.num_classes = num_classes
        self.history = []

    def update(self, prediction):
        self.history.append(prediction)
        if len(self.history) > self.n:
            self.history.pop(0)
        return np.mean(self.history, axis=0)

smoother = PredictionSmoother(n=SMOOTHING_FRAMES, num_classes=num_classes)

# ─────────────────────────────────────────────
# PREPROCESS FACE CROP
# ─────────────────────────────────────────────
def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (224, 224))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = preprocess_input(face_img)
    return np.expand_dims(face_img, axis=0)

# ─────────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────────
COLOR_MASK      = (0,   255, 0)
COLOR_NOMASK    = (0,   0,   255)
COLOR_INCORRECT = (0,   200, 255)

def get_color_and_display(label_key):
    lk = label_key.lower().replace("_", "").replace(" ", "")
    if "incorrect" in lk:
        return COLOR_INCORRECT, "Incorrect Mask"
    elif "without" in lk or "nomask" in lk or lk == "nomask":
        return COLOR_NOMASK, "No Mask"
    else:
        return COLOR_MASK, "Mask"

# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
detector = MTCNN()
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open webcam.")
    sys.exit(1)

print("\n▶ Starting real-time detection. Press ESC to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()
    rgb_frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces      = detector.detect_faces(rgb_frame)

    for face in faces:
        x, y, w, h = face['box']
        x, y = max(0, x), max(0, y)

        pad = int(0.1 * max(w, h))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)

        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue

        processed  = preprocess_face(face_crop)
        raw_pred   = model.predict(processed, verbose=0)[0]
        prediction = smoother.update(raw_pred)

        if INCORRECT_IDX is not None and prediction[INCORRECT_IDX] > INCORRECT_THRESHOLD:
            label_key  = index_to_label[INCORRECT_IDX]
            confidence = prediction[INCORRECT_IDX] * 100
        else:
            best_idx   = np.argmax(prediction)
            label_key  = index_to_label[best_idx]
            confidence = prediction[best_idx] * 100

        color, display_name = get_color_and_display(label_key)
        text = f"{display_name}: {confidence:.1f}%"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x1, y1 - th - 14), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, text, (x1 + 3, y1 - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        lk = label_key.lower().replace("_", "").replace(" ", "")

        if "without" in lk or lk in ("nomask", "no"):
            cv2.putText(frame,
                        "WARNING: Please Wear a Mask!",
                        (20, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                        COLOR_NOMASK, 3)

        if "incorrect" in lk:
            cv2.putText(frame,
                        "WARNING: Cover Your Nose Properly!",
                        (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                        COLOR_INCORRECT, 3)

    fps = 1.0 / max(time.time() - start_time, 1e-6)
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)

    cv2.imshow("Real-Time Face Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
