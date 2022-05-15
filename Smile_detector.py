import cv2 as cv
import numpy as np

img = cv.VideoCapture(0)
smile_classifier = cv.CascadeClassifier('smile.xml')

while True:
    successful_read, frame = img.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    if gray is None:
        print("The image was not able to load")

    detect_smile = smile_classifier.detectMultiScale(gray, minNeighbors=20, scaleFactor=1.7)

    for (x, y, w, h) in detect_smile:
        cv.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 4)

    cv.imshow("Smile Detector", gray)

    q = cv.waitKey(1)
