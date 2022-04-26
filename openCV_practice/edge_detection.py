import numpy as np
import matplotlib.pyplot as plt 
import cv2


img = cv2.imread("pic2.jpg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)


def remove_noise(img):

    med_val = np.median(img)
    lower = int(max(0,0.7*med_val))
    upper = int(min(255,1.3*med_val))
    blur_img = cv2.blur(img,ksize=(5,5))
    edges = cv2.Canny(image=blur_img,threshold1=lower,threshold2=upper)

    return edges

edges = remove_noise(img)


fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.imshow(edges)




plt.show()
plt.close()