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
img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
print(img)

fig = plt.figure(figsize=(8,6))

ax = fig.add_subplot(1,1,1)
ax.imshow(img,cmap = "gray")

plt.show()
plt.close()

# cv2.imshow("result",img)
# cv2.waitKey(0)
# # cv2.waitKey(1)
# print(cv2.waitKey(1))
# # print(type(cv2.waitKey(0)))
