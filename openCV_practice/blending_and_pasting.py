from calendar import c
import PIL
from matplotlib import image
import numpy as np
import matplotlib.pyplot as plt 
import cv2
from PIL import Image


img = cv2.imread("pic.jpg")
img = cv2.resize(img,(600,380))
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)


img2 = cv2.imread("pic2.jpg")
img2 = cv2.resize(img2,(600,380))
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)

blending_img = cv2.addWeighted(src1=img,alpha=0.5,src2=img2,beta =0.7,gamma=0)


plt.imshow(blending_img)
plt.show()
plt.close()