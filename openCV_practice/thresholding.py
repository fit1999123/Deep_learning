import PIL
from matplotlib import image
import numpy as np
import matplotlib.pyplot as plt 
import cv2
from PIL import Image

img = cv2.imread("rainbow.jpg",0)
#影像二值化

ret,thresh1=cv2.threshold(img,127,255,cv2.THRESH_BINARY)

fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax1.imshow(img,cmap="gray")
ax2 = fig.add_subplot(1,2,2)
ax2.imshow(thresh1,cmap="gray")
plt.show()
plt.close()