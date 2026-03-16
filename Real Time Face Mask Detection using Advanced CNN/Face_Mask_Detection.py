import cv2
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
import time

# Load trained model
model = load_model("mask_detector.keras")

# Initialize face detector
detector = MTCNN()

# Start webcam
cap = cv2.VideoCapture(0)

# Updated labels for 3 classes
labels = ["Mask", "No Mask", "Incorrect Mask"]

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

        # Preprocess image
        face_crop = cv2.resize(face_crop, (224,224))
        face_crop = face_crop / 255.0
        face_crop = np.reshape(face_crop, (1,224,224,3))

        # Prediction
        prediction = model.predict(face_crop, verbose=0)[0]

        label_index = np.argmax(prediction)
        label = labels[label_index]

        confidence = prediction[label_index] * 100

        # Color coding
        if label == "Mask":
            color = (0,255,0)       # Green
        elif label == "No Mask":
            color = (0,0,255)       # Red
        else:
            color = (0,255,255)     # Yellow

        text = f"{label}: {confidence:.2f}%"

        # Draw face box
        cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)

        # Display label
        cv2.putText(frame, text, (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2)

        # Warning messages
        if label == "No Mask":
            cv2.putText(frame,
                        "WARNING: Please Wear a Mask",
                        (20,50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0,0,255),
                        3)

        if label == "Incorrect Mask":
            cv2.putText(frame,
                        "WARNING: Cover Your Nose Properly",
                        (20,90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0,255,255),
                        3)

    # FPS calculation
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

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
