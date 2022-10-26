import matplotlib.pyplot as plt
import cv2
import numpy as np
from keras.models import load_model
import glob



model = load_model("face.h5")

img_path = glob.glob("*.jpg")

img_lst = []

for i in img_path:
    
    img = cv2.imread(i)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(50,50))
    img_lst.append(img)


t = ["Sue","Simon","Who are you?"]

lst = []

for i in img_lst:

    result = model.predict(i.reshape(-1,50,50,1),verbose = 0)[0]
    idx = result.argmax()
    confidence = result.max()*100

    if confidence >=70:

        lst.append(t[idx])
        
    else:

        lst.append(t[2])

    

fig = plt.figure(dpi=100)


for i in range(len(img_lst)):

    ax = fig.add_subplot(3,3,i+1)
    ax.imshow(img_lst[i],cmap="gray")    
    ax.set_title(lst[i])
    ax.set_xticks([])
    ax.set_yticks([])
    
    
plt.show()
plt.close()