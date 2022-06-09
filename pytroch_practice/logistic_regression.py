from pickletools import optimize
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

#prepare data

breast_cancer = datasets.load_breast_cancer()

x,y, = breast_cancer.data,breast_cancer.target

# print(x,y)
n_sample,n_feature = x.shape
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1234)

# print(n_sample,n_feature)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
# print(x_train,x_test)
x_train = torch.from_numpy(x_train.astype(np.float32))
x_test = torch.from_numpy(x_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))
y_train=y_train.view(y_train.shape[0],1)
y_test=y_test.view(y_test.shape[0],1)
# 1.set model
#f = wx +b 
class LogisticRegression(nn.Module):
    
    def __init__(self,n_input_features):
        super(LogisticRegression,self).__init__()
        self.linear = nn.Linear(n_input_features,1)
    def forward(self,x):
        
        y_pred = torch.sigmoid(self.linear(x)) # y = 1/1+e^-x
        return y_pred
    
model = LogisticRegression(n_feature)

# 2.loss and optimizer

criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(),lr = 0.01)

# 3.training loop

n_iters = 100

for i in range(n_iters):
    #forward pass and loss
    y_pred =model(x_train)
    loss = criterion(y_pred,y_train)
    #backward
    loss.backward()
    #update
    optimizer.step()
    
    optimizer.zero_grad()
    if (i+1)%10 ==0:
        

        print(f"epoch {i+1}: loss ={loss.item():.4f}")
        
with torch.no_grad():
    
    y_pred = model(x_test)
    y_pred_cls = y_pred.round()
    acc = y_pred_cls.eq(y_test).sum()/y_test.shape[0]
    print(f"accuracy = {acc:4f}")