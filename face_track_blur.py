import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse

def detect_faces(net, img, model):
    faceimg = img.copy()
    if model == 'dnn':
        conf_threshold = 0.7
        blob = cv2.dnn.blobFromImage(faceimg, 1.0, (300,300), [104,117,123], True, False)
        net.setInput(blob)
        detections = net.forward()
        face_rect = []
        for i in range(detections.shape[2]):
            confidence = detections[0,0,i,2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * faceimg.shape[1]) ## multiplying by frameWidth ##
                y1 = int(detections[0, 0, i, 4] * faceimg.shape[0]) ## multiplying by frameHeight ##
                x2 = int(detections[0, 0, i, 5] * faceimg.shape[1]) ## multiplying by frameWidth ##
                y2 = int(detections[0, 0, i, 6] * faceimg.shape[0]) ## multiplying by frameHeight ##
                face_rect.append([x1, y1, x2-x1, y2-y1])
    else:
        face_rect = faceCascade.detectMultiScale(faceimg, 1.2, 3)
    
    if len(face_rect)>=1:
        for (x,y,w,h) in face_rect:
            roi = faceimg[y:y+h,x:x+w,:]
            if len(roi)>=1:
                roi = cv2.medianBlur(roi, 37)
                faceimg[y:y+h,x:x+w,:] = roi
        
    return faceimg

if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(description='Face Detection')
    parser.add_argument("--model", type=str, default="dnn", help="Type of archtitecture to run")
    args = parser.parse_args()
    model = args.model

    if model == 'haarcascade':
        faceCascade = cv2.CascadeClassifier('DATA/haarcascade_frontalface_default.xml')
        print(faceCascade)
    else: 
        modelFile = 'DATA/opencv_face_detector_uint8.pb'
        configFile = 'DATA/opencv_face_detector.pbtxt'
        net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
        net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)


    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read(0)
        if ret:
            if model == 'dnn':
                cv2.imshow('face', detect_faces(net, frame, 'dnn'))
            else:
                cv2.imshow('face', detect_faces(None, frame, 'haarcascade'))
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
