import cv2
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
import time
import json

# Load trained model
model = load_model("mask_detector.keras")

# Load label mapping
with open("labels.json") as f:
    class_indices = json.load(f)

# Sort labels correctly
labels = [label for label, index in sorted(class_indices.items(), key=lambda x: x[1])]

# Initialize face detector
detector = MTCNN()

# Start webcam
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()

    faces = detector.detect_faces(frame)

    for face in faces:

        x, y, w, h = face['box']

        x = abs(x)
        y = abs(y)

        face_crop = frame[y:y+h, x:x+w]

        if face_crop.size == 0:
            continue

        # Preprocess
        face_crop = cv2.resize(face_crop, (224,224))
        face_crop = face_crop / 255.0
        face_crop = np.reshape(face_crop, (1,224,224,3))

        # Prediction
        prediction = model.predict(face_crop, verbose=0)[0]

        # 🔥 SOLUTION 1: Prioritize incorrect_mask
        incorrect_index = labels.index("IncorrectMask")

        if prediction[incorrect_index] > 0.30:
            label = "IncorrectMask"
            confidence = prediction[incorrect_index] * 100
        else:
            label_index = np.argmax(prediction)
            label = labels[label_index]
            confidence = prediction[label_index] * 100

        # Color coding
        if label == "mask":
            color = (0,255,0)
        elif label == "no_mask":
            color = (0,0,255)
        else:
            color = (0,255,255)

        display_label = label.replace("_", " ").title()
        text = f"{display_label}: {confidence:.2f}%"

        # Draw box
        cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)

        # Show label
        cv2.putText(frame, text, (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2)

        # Warnings
        if label == "no_mask":
            cv2.putText(frame,
                        "WARNING: Please Wear a Mask",
                        (20,50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0,0,255),
                        3)

        if label == "incorrect_mask":
            cv2.putText(frame,
                        "WARNING: Cover Your Nose Properly",
                        (20,90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0,255,255),
                        3)

    # FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)

    cv2.putText(frame,
                f"FPS: {int(fps)}",
                (20,30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255,255,0),
                2)

    cv2.imshow("Real-Time Face Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
