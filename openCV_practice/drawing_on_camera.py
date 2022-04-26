from tkinter import Frame
import cv2

cap = cv2.VideoCapture(0)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

x = w//2
y = h//2

w2 = w//4
h2 = h//4

while True:

    ret,frame = cap.read() 

    cv2.rectangle(frame,(x,y),(x+w2,x+h2),color = (0,0,255),thickness=4)
    cv2.imshow("frame",frame)
    if  cv2.waitKey(1) & 0xFF ==ord("q"):
            break


cap.release()
cv2.destroyAllWindows()
