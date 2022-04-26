from cv2 import imshow
import numpy as np
import matplotlib.pyplot as plt 
import cv2


cap = cv2.VideoCapture("video.mp4")

if cap.isOpened() == False:
    print("error")

while cap.isOpened():

    ret,frame = cap.read()
    if ret==True:
        cv2,imshow("frame",frame)
        if cv2.waitKey(100) & 0xFF == ord("q"):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()