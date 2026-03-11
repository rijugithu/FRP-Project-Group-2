import cv2
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.models import load_model

model = load_model("mask_detector.keras")
detector = MTCNN()

cap = cv2.VideoCapture(0)

labels = ["Mask","No Mask"]

while True:
    ret, frame = cap.read()

    faces = detector.detect_faces(frame)

    for face in faces:

        x,y,w,h = face['box']

        face_crop = frame[y:y+h, x:x+w]
        face_crop = cv2.resize(face_crop,(224,224))
        face_crop = face_crop/255.0
        face_crop = np.reshape(face_crop,(1,224,224,3))

        prediction = model.predict(face_crop)
        label = labels[np.argmax(prediction)]

        if label == "Mask":
            color = (0,255,0)
        else:
            color = (0,0,255)

        cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
        cv2.putText(frame,label,(x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)

    cv2.imshow("Mask Detection",frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()