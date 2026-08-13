import cv2
import cvzone
import math
import numpy as np
import imutils
import pickle
import time
import csv
from ultralytics import YOLO
from collections.abc import Iterable


def flatten(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:
            yield item

# Load pre-trained models & files
embeddingFile = "output/embeddings.pickle"
embeddingModel = "openface_nn4.small2.v1.t7"
recognizerFile = "output/recognizer.pickle"
labelEncFile = "output/le.pickle"
conf = 0.5  # Confidence threshold for face detection

print("[INFO] Loading face detector...")
prototxt = "model/deploy.prototxt"
model = "model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt, model)

print("[INFO] Loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(embeddingModel)
recognizer = pickle.loads(open(recognizerFile, "rb").read())
le = pickle.loads(open(labelEncFile, "rb").read())

# Load weapon detection model
weapon_model = YOLO("bestv8n.pt")
classnames = ['AK-47']

# Load video file instead of webcam
video_path = "2.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("[ERROR] Could not open video file.")
    exit()

print("[INFO] Processing video...")

while True:
    ret, frame = cap.read()
    if not ret:
        cap = cv2.VideoCapture(video_path)
        continue  # Restart video when it ends

    frame = cv2.resize(frame, (1200, 750))
    (h, w) = frame.shape[:2]
    
    # Weapon detection
    results = weapon_model(frame)
    for info in results:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            confidence = box.conf[0]
            conf_value = math.ceil(confidence * 100)
            class_detect = int(box.cls[0])
            class_detect = classnames[class_detect]

            if conf_value > 40:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cvzone.putTextRect(frame, f'{class_detect}', [x1 + 8, y1 - 12], thickness=2, scale=1)

    # Face detection and recognition
    imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),
                                      (104.0, 177.0, 123.0), swapRB=False, crop=False)
    detector.setInput(imageBlob)
    detections = detector.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]
            if fW < 20 or fH < 20:
                continue

            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            text = "{} : {:.2f}%".format(name, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Video", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Press ESC to stop
        break

cap.release()
cv2.destroyAllWindows()