#-----獲取人臉樣本-----
import cv2
import numpy as np



cap = cv2.VideoCapture(0)

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

count = 0

while True:    
    success,img = cap.read()    
    if success is True: 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    else:   
        break
    faces = face_detector.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+w), (255, 0, 0))
        count += 1
        
        cv2.imwrite("test"+str(count+8)+'.jpg',gray[y:y+h,x:x+w]) 
        cv2.imshow('image',img)       
    k = cv2.waitKey(1)        
    if k == '27':
        break        
    # elif count >= 800:
    #     break
