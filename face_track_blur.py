import cv2
import matplotlib.pyplot as plt
import numpy as np

faceCascade = cv2.CascadeClassifier('/DATA/haarcascades/haarcascade_frontalface_default.xml')

def detect_faces(img):
    faceimg = img.copy()
    face_rect = faceCascade.detectMultiScale(faceimg, 1.2, 3)
    
    if len(face_rect)>=1:
        for (x,y,w,h) in face_rect:
            roi = faceimg[y:y+h,x:x+w,:]
            roi = cv2.medianBlur(roi, 37)
            faceimg[y:y+h,x:x+w,:] = roi
        
    return faceimg

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read(0)
    cv2.imshow('face', detect_faces(frame))
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
