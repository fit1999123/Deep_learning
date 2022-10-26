import numpy as np
import imgaug.augmenters as iaa
import cv2
import glob
import matplotlib.pyplot as plt

img_path = glob.glob("dataset2/*[imon]*")

img_lst = []

### load dataset

for i in img_path:
    
    img = cv2.imread(i)
    
    img_lst.append(img)
    

### Image Augmentation


augmentation = iaa.Sequential([
    #Flip 
    iaa.Fliplr(0.5),
    #Affine
    iaa.Affine(translate_percent = {
            "x":(-0.2,0.2), 
            "y":(-0.2,0.2)},
            rotate=(-30,30),
            scale= (0.8,1.2)),
    #brightness
    iaa.Multiply((0.8,1.5)),
    #linercontrast
    iaa.LinearContrast((0.6,1.4)),
    iaa.Sometimes(0.5,
    # GaussianBlur
    iaa.GaussianBlur((0,3))
    )
    ])





augmented_img_lst = augmentation(images=img_lst)



### save Image



for img in range(len(augmented_img_lst)):

    cv2.imwrite("dataset2/Simon."+str((img+96542))+'.jpg',augmented_img_lst[img]) 



### show Images

# fig = plt.figure()

# ax1 = fig.add_subplot(1,1,1)

# ax1.imshow(augmented_img_lst[2])

# plt.show()
# plt.close()
