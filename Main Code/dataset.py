import imutils
import time
import cv2
import csv
import os

# Load Haar Cascade for face detection
cascade = 'haarcascade_frontalface_default.xml'
detector = cv2.CascadeClassifier(cascade)

# User Input
Name = input("Enter your Name: ")
Roll_Number = input("Enter your Roll Number: ")
dataset = 'dataset'
sub_data = Name
path = os.path.join(dataset, sub_data)

# Create directory for saving images
if not os.path.isdir(path):
    os.mkdir(path)
    print(sub_data)

# Save user info in CSV
info = [str(Name), str(Roll_Number)]
with open('student.csv', 'a') as csvFile:
    write = csv.writer(csvFile)
    write.writerow(info)
csvFile.close()

# Choose between webcam or video file
use_webcam = input("Do you want to use a webcam? (yes/no): ").strip().lower()

if use_webcam == "yes":
    cam = cv2.VideoCapture(0)  # Webcam feed
else:
    video_path = "C:\\Users\\wesdf\\Downloads\\research\\weapon\\weapon_detection\\2.mp4"  # Change this to your actual video file path
    cam = cv2.VideoCapture(video_path)  # Load video file

# Face Capture Process
print("Starting video processing...")
time.sleep(2.0)
total = 0



while total < 50:
    _, frame = cam.read()
    img = imutils.resize(frame, width=400)
    rects = detector.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), scaleFactor=1.1,minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        p = os.path.sep.join([path, "{}.png".format(
            str(total).zfill(5))])
        cv2.imwrite(p, img)
        total += 1
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break


# Cleanup
cam.release()
cv2.destroyAllWindows()

