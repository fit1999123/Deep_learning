import PIL
from matplotlib import image
import numpy as np
import matplotlib.pyplot as plt 
import cv2
from PIL import Image


pic = Image.open("pic.jpg")
pic = pic.resize((600,480))

pic_arr = np.array(pic)

# print(pic_arr)

pic_red = np.copy(pic_arr)

pic_red[:,:,0] =155
print(pic_red)

plt.imshow(pic_red)
plt.show()
plt.close()



