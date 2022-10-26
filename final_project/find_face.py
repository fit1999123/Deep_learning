#-----獲取人臉樣本-----
import cv2
import numpy as np





face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread("test2.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

faces = face_detector.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in faces:

        cv2.rectangle(img, (x, y), (x+w, y+w), (255, 0, 0))
        
cv2.imwrite("test"+str(8)+".jpg",gray[y:y+h,x:x+w])    
cv2.waitKey(0)        
 