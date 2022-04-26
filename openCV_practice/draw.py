from calendar import c
import PIL
from matplotlib import image
import numpy as np
import matplotlib.pyplot as plt 
import cv2
from PIL import Image


bg = np.zeros(shape=(480,600,3),dtype=np.uint8)

cv2.rectangle(bg,pt1=(380,200),pt2=(510,100),color=(255,0,0),thickness=5)
cv2.rectangle(bg,pt1=(300,300),pt2=(490,490),color=(0,0,255),thickness=5)

print(bg)

plt.imshow(bg)
plt.show()
plt.close()