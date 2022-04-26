import scipy
import numpy as np
import matplotlib.pyplot as plt 
import cv2


img = cv2.imread("wall.jpg",0)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
#Laplace
lap1 = cv2.Laplacian(sobelx,cv2.CV_64F)
lap2 = cv2.Laplacian(sobely,cv2.CV_64F)
lap3 = cv2.Laplacian(img,cv2.CV_64F)
#二值化
ret,thresh1=cv2.threshold(sobelx,200,255,cv2.THRESH_BINARY)
ret,thresh2=cv2.threshold(sobely,200,255,cv2.THRESH_BINARY)
ret,thresh3=cv2.threshold(lap3,200,255,cv2.THRESH_BINARY_INV)


fig = plt.figure(figsize=(8,6))
ax1 = fig.add_subplot(2,3,1)
ax1.imshow(img,cmap="gray")
ax2 = fig.add_subplot(2,3,2)
ax2.imshow(sobelx,cmap="gray")
ax3 = fig.add_subplot(2,3,3)
ax3.imshow(sobely,cmap="gray")
ax4 = fig.add_subplot(2,3,4)
ax4.imshow(thresh1,cmap="gray")
ax5 = fig.add_subplot(2,3,5)
ax5.imshow(thresh2,cmap="gray")
ax6 = fig.add_subplot(2,3,6)
ax6.imshow(lap3,cmap="gray")
plt.show()
plt.close()