from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPool2D,Flatten
from create_label import x_train,x_test,y_train,y_test
from keras import optimizers



LR = 0.001
#creat model
model = Sequential()
#convolutional layer
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="valid",input_shape=(50,50,1),activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="valid",input_shape=(50,50,1),activation="relu"))
#pooling layer
model.add(MaxPool2D(pool_size=(2,2)))
#second convolutional layer
model.add(Conv2D(filters=128,kernel_size=(3,3),padding="valid",activation="relu"))
model.add(Conv2D(filters=128,kernel_size=(3,3),padding="valid",activation="relu"))
#second pooling layer
model.add(MaxPool2D(pool_size=(2,2)))
#flatten
model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dense(64,activation="relu"))
model.add(Dense(2,activation='sigmoid'))
#training model
model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(LR),metrics=['accuracy']) 
model.summary()
model.fit(x=x_train,y=y_train,epochs=8,verbose = 2,batch_size=32)
model.save("face2.h5")
#testing model
print(model.evaluate(x_test,y_test))
