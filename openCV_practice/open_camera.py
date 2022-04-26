
import numpy as np
import matplotlib.pyplot as plt 
import cv2


cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"MP4V")
writer = cv2.VideoWriter("mysupervideo.mp4",fourcc,20.0,(width,height))

while cap.isOpened():
        ret,frame = cap.read()
        frame= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        writer.write(frame)
        cv2.imshow("frame",frame)
        if  cv2.waitKey(1) & 0xFF ==ord("q"):
            break



cap.release()
writer.release()
cv2.destroyAllWindows()
