from pyexpat import model
from statistics import mode
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dropout,Dense,Conv2D,MaxPool2D,Flatten



(x_train,y_train),(x_test,y_test) = mnist.load_data()

# print(x_train.shape)



#to cateorical
#ex : 

#1 = [0,1,0,0,0,0,0,....,0]
#2 = [0,0,1,0,0,0,0,....,0]

y_cateorical_test = to_categorical(y_test)
y_cateorical_train = to_categorical(y_train)

#normalize

x_train_normal = x_train/x_train.max()
x_test_normal = x_test/x_test.max()

x_train_normal = x_train_normal.reshape(60000,28,28,1)
x_test_normal = x_test_normal.reshape(10000,28,28,1)


#creat model

model = Sequential()

#convolutional layer

model.add(Conv2D(filters=16,kernel_size=(5,5),padding="same",input_shape=(28,28,1),activation="relu"))

#pooling layer

model.add(MaxPool2D(pool_size=(2,2)))

# #second convolutional layer

model.add(Conv2D(filters=36,kernel_size=(4,4),padding="same",input_shape=(28,28,1),activation="relu"))

# #second pooling layer

model.add(MaxPool2D(pool_size=(2,2)))

#avoid overfitting

model.add(Dropout(0.25))

#flatten

model.add(Flatten())

model.add(Dense(128,activation="relu"))

model.add(Dropout(0.5))

model.add(Dense(10,activation='softmax'))

#training model

model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy']) 

model.summary()


model.fit(x=x_train_normal,y=y_cateorical_train,epochs=5)



print(model.evaluate(x_test_normal,y_cateorical_test))