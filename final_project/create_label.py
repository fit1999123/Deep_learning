import numpy as np
import os 
import random
import cv2

def my_label(image_name):

    name = image_name.split(".")[-3]

    if name =="Sue":

        return np.array([1,0])
            
    elif name =="Simon":

        return np.array([0,1])
  
  
def my_data():

    data = []

    for img in os.listdir("dataset"):

        path = os.path.join("dataset",img)
        img_data = cv2.imread(path)
        img_data = cv2.cvtColor(img_data,cv2.COLOR_BGR2GRAY)
        img_data = cv2.resize(img_data, (50,50))
        data.append([np.array(img_data),my_label(img)])
        
    random.shuffle(data)

    return data



data = my_data()
train  = data[:int(len(data)*0.8)]
test = data[int(len(data)*0.8):]
x_train = np.array([i[0] for i in train]).reshape(-1,50,50,1)
y_train = np.array([i[1] for i in train])
x_test = np.array([i[0] for i in test]).reshape(-1,50,50,1)
y_test = np.array([i[1] for i in test])


