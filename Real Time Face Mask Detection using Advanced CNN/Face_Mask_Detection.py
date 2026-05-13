import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import time
import json
import threading
import tensorflow as tf
from flask import Flask, render_template
from flask_socketio import SocketIO
from collections import deque, defaultdict

# ──────────────────────────────────────────────
# GPU / TF optimizations
# ──────────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

try:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
except Exception:
    pass

# ──────────────────────────────────────────────
# Flask + SocketIO setup
# ──────────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY'] = 'mask_detection_secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

@app.route('/')
def index():
    return render_template('dashboard.html')

def start_flask():
    print("🚀 Starting Flask server...")
    socketio.run(app, host='127.0.0.1', port=5000, debug=False, use_reloader=False)

# ──────────────────────────────────────────────
# Shared state
# ──────────────────────────────────────────────
lock = threading.Lock()
state = {
    "counts"          : {"Mask": 0, "No Mask": 0, "Incorrect Mask": 0},
    "fps_history"     : [],
    "conf_history"    : [],
    "recent_detections": [],
    "session_start"   : time.time(),
    "total_frames"    : 0,
    "faces_detected"  : 0,
}
MAX_HISTORY = 60

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
IMG_SIZE           = 224
SMOOTH_WINDOW      = 5      # frames to average predictions per tracked face
EMIT_INTERVAL      = 0.1   # max dashboard emits per second
CONF_THRESHOLD     = 0.50  # YOLOv8 face detection confidence threshold
IOU_THRESHOLD      = 0.40  # NMS IoU threshold for YOLOv8
last_emit_time     = 0.0

COLOR_MAP = {
    "Mask"          : (0, 255, 0),
    "No Mask"       : (0, 0, 255),
    "Incorrect Mask": (0, 255, 255),
}

# ──────────────────────────────────────────────
# IoU helper
# ──────────────────────────────────────────────
def iou(boxA, boxB):
    """Compute IoU for two [x, y, w, h] boxes."""
    ax2, ay2 = boxA[0] + boxA[2], boxA[1] + boxA[3]
    bx2, by2 = boxB[0] + boxB[2], boxB[1] + boxB[3]
    ix1 = max(boxA[0], boxB[0]);  iy1 = max(boxA[1], boxB[1])
    ix2 = min(ax2,     bx2);      iy2 = min(ay2,     by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = boxA[2]*boxA[3] + boxB[2]*boxB[3] - inter
    return inter / (union + 1e-6)

# ──────────────────────────────────────────────
# Face tracker (IoU-based, per-face smoothing)
# ──────────────────────────────────────────────
class FaceTracker:
    def __init__(self, smooth_window=SMOOTH_WINDOW, iou_threshold=0.30, max_lost=10):
        self.tracks     = {}
        self.next_id    = 0
        self.smooth_win = smooth_window
        self.iou_thresh = iou_threshold
        self.max_lost   = max_lost
        self.lost_count = defaultdict(int)

    def update(self, detected_boxes, predictions):
        matched = set()

        for box, pred in zip(detected_boxes, predictions):
            best_iou, best_tid = self.iou_thresh, None
            for tid, track in self.tracks.items():
                score = iou(box, track["box"])
                if score > best_iou:
                    best_iou, best_tid = score, tid

            if best_tid is not None:
                self.tracks[best_tid]["box"] = box
                self.tracks[best_tid]["preds"].append(pred)
                self.lost_count[best_tid] = 0
                matched.add(best_tid)
            else:
                tid = self.next_id;  self.next_id += 1
                self.tracks[tid] = {
                    "box"  : box,
                    "preds": deque([pred], maxlen=self.smooth_win),
                }
                self.lost_count[tid] = 0
                matched.add(tid)

        stale = [tid for tid in self.tracks
                 if tid not in matched and self.lost_count.__setitem__(tid, self.lost_count[tid] + 1) is None
                 and self.lost_count[tid] > self.max_lost]
        for tid in stale:
            del self.tracks[tid];  del self.lost_count[tid]

        return self.tracks

    def smoothed_prediction(self, tid):
        return np.mean(np.stack(self.tracks[tid]["preds"]), axis=0)

# ──────────────────────────────────────────────
# Batched face crop + preprocess
# ──────────────────────────────────────────────
def preprocess_faces(frame, boxes):
    batch, valid = [], []
    h_frame, w_frame = frame.shape[:2]
    for (x, y, w, h) in boxes:
        x1 = max(0, x);  y1 = max(0, y)
        x2 = min(w_frame, x + w);  y2 = min(h_frame, y + h)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
        batch.append(resized.astype(np.float32) / 255.0)
        valid.append((x1, y1, x2 - x1, y2 - y1))
    if not batch:
        return np.empty((0, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32), []
    return np.stack(batch), valid

# ──────────────────────────────────────────────
# Raw label → display label mapping
# Handles all common label naming conventions:
#   "WithMask" / "with_mask" / "Mask"      → "Mask"
#   "WithoutMask" / "without_mask"         → "No Mask"
#   "IncorrectMask" / "incorrect_mask"     → "Incorrect Mask"
# ──────────────────────────────────────────────
def normalize_label(raw):
    # Strips underscores/spaces and lowercases to handle any naming style:
    #   "with_mask"      → "Mask"
    #   "without_mask"   → "No Mask"
    #   "incorrect_mask" → "Incorrect Mask"
    r = raw.lower().replace("_", "").replace(" ", "")
    if r in ("withmask", "mask"):
        return "Mask"
    if r in ("withoutmask", "nomask"):
        return "No Mask"
    if r in ("incorrectmask", "incorrectly", "incorrect"):
        return "Incorrect Mask"
    # Fallback
    return raw.replace("_", " ").title()

# ──────────────────────────────────────────────
# Decode mask-classifier prediction
# ──────────────────────────────────────────────
def decode_prediction(pred_vec, labels, incorrect_index, incorrect_thresh=0.30):
    if pred_vec[incorrect_index] > incorrect_thresh:
        return "Incorrect Mask", float(pred_vec[incorrect_index]) * 100
    idx   = int(np.argmax(pred_vec))
    label = normalize_label(labels[idx])
    return label, float(pred_vec[idx]) * 100

# ──────────────────────────────────────────────
# Draw fancy bounding box
# ──────────────────────────────────────────────
def draw_box(frame, x, y, w, h, label, conf, color):
    # Filled rectangle behind label
    label_text = f"{label}: {conf:.1f}%"
    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.rectangle(frame, (x, y - th - 12), (x + tw + 6, y), color, -1)
    cv2.putText(frame, label_text, (x + 3, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2, cv2.LINE_AA)

# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == '__main__':
    try:
        # ── Flask ───────────────────────────────────
        flask_thread = threading.Thread(target=start_flask, daemon=True)
        flask_thread.start()
        print("✅ Dashboard running at: http://127.0.0.1:5000")
        print("📷 Starting webcam... Press ESC to quit")
        time.sleep(2)

        # ── Mask classifier (Keras) ──────────────────
        print("🔄 Loading mask classifier...")
        mask_model = load_model("best_model.keras")
        dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        mask_model.predict(dummy, verbose=0)           # warm-up
        print("✅ Mask classifier ready")

        # ── Labels ──────────────────────────────────
        with open("labels.json") as f:
            class_indices = json.load(f)
        labels = [lbl for lbl, _ in sorted(class_indices.items(), key=lambda x: x[1])]
        # Find IncorrectMask index robustly regardless of naming style
        incorrect_index = next(
            (i for i, l in enumerate(labels)
             if l.lower().replace("_", "") in ("incorrectmask", "incorrect")),
            None
        )
        if incorrect_index is None:
            print("\u26a0\ufe0f  WARNING: 'IncorrectMask' class not found \u2014 defaulting to index 0")
            incorrect_index = 0
        print(f"\U0001f4cb Labels loaded: {labels}")
        print(f"\U0001f4cb IncorrectMask index: {incorrect_index}")

        # ── YOLOv8n face detector ────────────────────
        # Uses the pretrained YOLOv8n-face model from Ultralytics.
        # It will be auto-downloaded on first run (~6 MB).
        # If you have a custom weights file, replace the path below:
        #   yolo_detector = YOLO("yolov8n-face.pt")
        print("🔄 Loading YOLOv8n face detector...")
        yolo_detector = YOLO("yolov8n-face.pt")       # auto-downloads if missing
        yolo_detector.fuse()                           # fuse Conv+BN layers → faster inference
        print("✅ YOLOv8n ready")

        # ── Tracker ─────────────────────────────────
        tracker = FaceTracker(smooth_window=SMOOTH_WINDOW)

        # ── Webcam ──────────────────────────────────
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS,          30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

        if not cap.isOpened():
            print("❌ ERROR: Webcam not accessible")
            exit()

        fps_deque = deque(maxlen=30)   # rolling FPS window

        print("▶  Detection loop started — press ESC in the OpenCV window to stop")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t_start = time.time()
            elapsed = round(t_start - state["session_start"], 1)

            # ────────────────────────────────────────
            # Step 1 — YOLOv8n face detection
            #   • Single forward pass, runs at ~60–100 FPS on CPU/GPU
            #   • Returns xyxy bounding boxes directly
            # ────────────────────────────────────────
            results = yolo_detector.predict(
                frame,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                imgsz=640,
                verbose=False,
                stream=False,
            )

            # Parse YOLO output → [x, y, w, h] list
            detected_boxes = []
            if results and results[0].boxes is not None:
                for box in results[0].boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, box[:4])
                    detected_boxes.append([x1, y1, x2 - x1, y2 - y1])

            # ────────────────────────────────────────
            # Step 2 — Batched mask classification
            # ────────────────────────────────────────
            if detected_boxes:
                batch, valid_boxes = preprocess_faces(frame, detected_boxes)
                if len(batch) > 0:
                    predictions = mask_model.predict(batch, verbose=0)
                    tracks = tracker.update(valid_boxes, predictions)
                else:
                    tracks = tracker.update([], [])
            else:
                tracks = tracker.update([], [])

            # ────────────────────────────────────────
            # Step 3 — Draw smoothed results
            # ────────────────────────────────────────
            for tid, track in tracks.items():
                x, y, w, h  = track["box"]
                smooth_pred  = tracker.smoothed_prediction(tid)
                label, conf  = decode_prediction(smooth_pred, labels, incorrect_index)
                color        = COLOR_MAP.get(label, (255, 255, 255))
                draw_box(frame, x, y, w, h, label, conf, color)

                with lock:
                    if label in state["counts"]:
                        state["counts"][label] += 1
                    state["faces_detected"] += 1

                    # ── conf_history: scatter chart data ──────────
                    state["conf_history"].append({
                        "t"         : elapsed,
                        "label"     : label,
                        "confidence": round(conf, 2),
                    })
                    if len(state["conf_history"]) > MAX_HISTORY:
                        state["conf_history"].pop(0)

                    # ── recent_detections: detection log (last 20) ──
                    state["recent_detections"].insert(0, {
                        "time"      : elapsed,
                        "label"     : label,
                        "confidence": round(conf, 2),
                    })
                    state["recent_detections"] = state["recent_detections"][:20]

            # ────────────────────────────────────────
            # FPS overlay (rolling average)
            # ────────────────────────────────────────
            fps_deque.append(time.time() - t_start)
            fps = round(1.0 / (sum(fps_deque) / len(fps_deque)), 1)

            cv2.putText(frame, f"FPS: {fps}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Faces: {len(tracks)}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, "Detector: YOLOv8n",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

            # ────────────────────────────────────────
            # Update shared state (fps_history + elapsed)
            # ────────────────────────────────────────
            with lock:
                state["total_frames"] += 1
                state["fps_history"].append({"t": elapsed, "fps": fps})
                if len(state["fps_history"]) > MAX_HISTORY:
                    state["fps_history"].pop(0)

            # ────────────────────────────────────────
            # Throttled SocketIO emit (max 10/sec)
            # Sends ALL fields the dashboard expects:
            #   counts, fps, total_frames, faces_detected,
            #   elapsed, fps_history, conf_history, recent_detections
            # ────────────────────────────────────────
            now = time.time()
            if now - last_emit_time >= EMIT_INTERVAL:
                with lock:
                    payload = {
                        "counts"           : dict(state["counts"]),
                        "fps"              : fps,
                        "total_frames"     : state["total_frames"],
                        "faces_detected"   : state["faces_detected"],
                        "elapsed"          : elapsed,
                        "fps_history"      : list(state["fps_history"]),
                        "conf_history"     : list(state["conf_history"]),
                        "recent_detections": list(state["recent_detections"]),
                    }
                socketio.emit('detection_update', payload)
                last_emit_time = now

            cv2.imshow("Face Mask Detection [YOLOv8n]", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
        print("🛑 Detection stopped")

    except Exception as e:
        import traceback
        print("❌ ERROR:", e)
        traceback.print_exc()
