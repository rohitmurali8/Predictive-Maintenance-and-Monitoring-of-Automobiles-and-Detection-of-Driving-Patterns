from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from imutils.video import FPS
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def yawning(mouth):
        D = dist.euclidean(mouth[3], mouth[9])
        return D

EYE_AR_THRESH = 0.4
EYE_AR_CONSEC_FRAMES = 48
COUNTER = 0

YAWN_THRESH = 20.0
YAWN_FRAMES = 40
COUNT = 0

BLINK = 0.0

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('C:/Data/Software/shape_predictor_68_face_landmarks.dat')

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    height,width = frame.shape[:2]
    frame = cv2.resize(frame,(height,450))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]
        yawn = yawning(mouth)
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouth = cv2.convexHull(mouth)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouth], -1, (0, 255, 0), 1)
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            print(COUNTER)
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
                    COUNTER = 0
                    cv2.putText(frame, "EAR:{:.2f}".format(ear),(300,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
