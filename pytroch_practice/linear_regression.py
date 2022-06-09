
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
class LinearRegresson:
    
    def __init__(self,learn_rate = 0.001,n_iters=1000):
        self.learn_rate = learn_rate
        self.n_iters = n_iters
        self.weight = None
        self.bias = None
    
    def fit(self,x,y):
        print(x.shape)
        n_sample,n_features = x.shape
        self.weight = np.zeros(n_features)
        self.bias = 0
        for i in range(self.n_iters):
            y_predicted = np.dot(x,self.weight)+self.bias # y = wx+b
            dw = (1/n_sample) * np.dot(x.T,(y_predicted-y))
            db = (1/n_sample) * np.sum(y_predicted-y)
            self.weight-=self.learn_rate*dw
            self.bias-=self.learn_rate*db
    
    def predict(self,x):
        y_predicted = np.dot(x,self.weight)+self.bias # y = wx+b
        
        return y_predicted


x_np,y_np = datasets.make_regression(n_samples=200,n_features=1,n_targets=1,noise=20)


# print(y_np.shape)
x = torch.from_numpy(x_np.astype(np.float32))
y = torch.from_numpy(y_np.astype(np.float32))
y = y.view(y.shape[0],1)

# print(x)
# print(y)
#creat model
n_sample,n_feature =x.shape
model= nn.Linear(n_feature,n_feature)
w,b = model.parameters()
#loss and optimzer
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD([w],lr=learning_rate)
#training loop
n_iters = 100

for i in range(n_iters):
    #forward pass and loss
    y_predicted = model(x)
    loss = criterion(y_predicted,y)
    #backward
    loss.backward()
    #update
    optimizer.step()
    optimizer.zero_grad()
    if (i+1)%10 ==0:
        w,b = model.parameters()

        print(f"epoch {i+1}: w = {w[0][0].item():3f}, loss ={loss.item():.8f}")
     

# model = LinearRegresson(learn_rate=0.01)
# model.fit(x,y)
predict = model(x).detach().numpy()

plt.plot(x_np,predict,"r")
plt.plot(x_np,y_np,"bo")
plt.show()
plt.close()