import cv2
import cvzone
import math
import numpy as np
import pickle
import time
import streamlit as st
from ultralytics import YOLO
from collections.abc import Iterable

st.set_page_config(layout="wide")

# Add CSS for styling
st.markdown(
    """
    <style>
    .center-text {
        text-align: center;
    }
    .red-text {
        color: red;
        font-size: 24px;
        font-weight: bold;
    }
    .blue-text {
        color: blue;
        font-size: 28px;
        font-weight: bold;
    }
    .criminal-text {
        font-size: 30px;
        font-weight: bold;
        text-align: center;
        display: block;
    }
    .image-container {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit UI Setup
st.markdown("<h1 style='text-align: center; font-size: 50px;color:blue'>Surveillance System for Crime Detection</h1>", unsafe_allow_html=True)

# Create two columns: Video feed on the left, detection panel on the right
col1, col2 = st.columns([2, 1])  # Video feed takes 2x space, detection panel 1x space

with col1:
    st.write("### Real-time Feed")
    frame_holder = st.empty()  # Placeholder for video frame

with col2:
    st.markdown("<h3 class='red-text center-text' >DETECTION</h3>", unsafe_allow_html=True)
    # st.markdown("<p class='center-text' style='color: black; font-size: 20px;'>No detection</p>", unsafe_allow_html=True)
    
    # Police notification message (will be updated when a weapon is detected)
    police_notified_holder = st.empty()
    
    # Weapon warning message
    weapon_warning_holder = st.empty()  # Placeholder for weapon warning
 
    # Center the detected face image
    with st.container():
        col_center = st.columns([1, 2, 1])[1]  # Center column for face image
        with col_center:
            detected_face_holder = st.empty()  # Placeholder for detected face image
        
        # Add blank placeholder image to reserve space before detection starts
        # Create a blank image (white background) to reserve face display space
        empty_face_placeholder = np.ones((200, 200, 3), dtype=np.uint8) * 255  # White square image

        # Show this blank image initially to reserve space for face detection
        detected_face_holder.image(empty_face_placeholder, caption="Waiting for Face...", use_container_width=False)
    
    st.markdown("<h3 class='blue-text center-text'>CRIMINAL NAME & ACCURACY</h3>", unsafe_allow_html=True)
    detected_name_holder = st.empty()  # Placeholder for detected criminal name

# Load pre-trained models & files
embeddingFile = "output/embeddings.pickle"
embeddingModel = "openface_nn4.small2.v1.t7"
recognizerFile = "output/recognizer.pickle"
labelEncFile = "output/le.pickle"
conf = 0.5  # Confidence threshold for face detection

prototxt = "model/deploy.prototxt"
model = "model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt, model)

embedder = cv2.dnn.readNetFromTorch(embeddingModel)
recognizer = pickle.loads(open(recognizerFile, "rb").read())
le = pickle.loads(open(labelEncFile, "rb").read())

# Load weapon detection model
weapon_model = YOLO("bestv8n.pt")
classnames = ['AK-47']

# Load video file
video_path = "2forchecking.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    st.error("Error: Could not open video file.")
    st.stop()

st.write("[INFO] Processing Live Feed...")

# Flag to keep track if a weapon has been detected at least once
weapon_detected_once = False

last_detected_name = "Unknown"  # Store the last detected name & accuracy
detected_name = "Unknown"  # Initialize detected_name
    
while True:
    ret, frame = cap.read()
    if not ret:
        # st.write("[INFO] Video Processing Stopped")
        break  # Stop when video ends

    frame = cv2.resize(frame, (1280, 700))
    (h, w) = frame.shape[:2]
    
    # Weapon detection
    weapon_detected = False
    weapon_name = ""
    
    results = weapon_model(frame)
    for info in results:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            conf_value = math.ceil(confidence * 100)
            class_detect = int(box.cls[0])
            class_detect = classnames[class_detect]

            if conf_value > 40:
                weapon_detected = True
                weapon_name = class_detect
                weapon_detected_once = True
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cvzone.putTextRect(frame, f'{class_detect}', [x1 + 8, y1 - 12], thickness=2, scale=1)

    # Update the weapon warning in the detection panel
    if weapon_detected:
        weapon_warning_holder.markdown(
            f"<h3 class='red-text center-text' style='color:red;'>⚠️ WARNING: {weapon_name} Detected! ⚠️</h3>",
            unsafe_allow_html=True
        )
    else:
        weapon_warning_holder.markdown(f"<h3 style='color:white;'></h3>",
            unsafe_allow_html=True
        )
    # Show police notification permanently once a weapon is detected
    if weapon_detected_once:
        police_notified_holder.markdown(
            "<h3 class='red-text center-text' style='color:red;'>🚨 POLICE NOTIFIED 🚨</h3>",
            unsafe_allow_html=True
        )
    else:
        police_notified_holder.markdown(f"<h3 style='color:white;'></h3>",
            unsafe_allow_html=True
        )
    detected_face = None
    
    

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
            detected_name = f"{name} : {proba * 100:.2f}%"
            last_detected_name = detected_name  # Store the last detected name

            
            text = "{} : {:.2f}%".format(name, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Store the detected face for display
            detected_face = face

    # Convert frame to RGB and update the video feed
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_holder.image(frame, channels="RGB", use_container_width=True)

    # Update detection panel
    if detected_face is not None:
        detected_face = cv2.resize(detected_face, (200, 200))  # Resize detected face
        detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2RGB)
        detected_face_holder.image(detected_face, caption="Face Detected", use_container_width=False)

        # Center the detected face image
        with col_center:
            detected_face_holder.image(detected_face, caption="Face Detected", use_container_width=False)

    # Centered Criminal Details (Larger Font)
    if detected_name == "Unknown" and last_detected_name!="Unknown":  
        detected_name = last_detected_name  # Use the last detected name if no new detection

    detected_name_holder.markdown(f"<p class='criminal-text'>{last_detected_name}</p>", unsafe_allow_html=True)

    time.sleep(0.03)

cap.release()
# st.write("[INFO] Video Stream Ended")
