import PIL
import numpy as np
import matplotlib.pyplot as plt 
import cv2
from PIL import Image

rainbow = cv2.imread("rainbow.jpeg")
rainbow = cv2.cvtColor(rainbow,cv2.COLOR_BGR2RGB)
wall = cv2.imread("wall.jpg")
wall = cv2.cvtColor(wall,cv2.COLOR_BGR2RGB)
girl = cv2.imread("pic.jpg")
girl = cv2.cvtColor(girl,cv2.COLOR_BGR2RGB)
color = ["b","g","r"]

girl_hist=cv2.calcHist([girl],channels=[0],mask=None,histSize=[256],ranges=[0,256])

fig = plt.figure(figsize=(8,6))
ax1= fig.add_subplot(2,1,1)
ax1.plot(girl_hist)
ax2= fig.add_subplot(2,1,2)

for i, col in enumerate(color):
    colorful_girl_hist = cv2.calcHist([girl], [i], None, [256], [0, 256])
    ax2.plot(colorful_girl_hist, color = col)
  

plt.show()
plt.close()