from sklearn import datasets
import torch 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler


iris = datasets.load_iris()

feature = pd.DataFrame(iris.data,columns=iris.feature_names)
# print(feature)
target = pd.DataFrame(iris.target, columns=['target'])
# print(target)
iris_data = pd.concat([feature, target], axis=1)
# print(iris_data)
iris_data = iris_data[['sepal length (cm)', 'sepal width (cm)', 'target']]
iris_data = iris_data[iris_data.target <= 1]

print(iris_data)
train_feature, test_feature, train_target, test_target = train_test_split(
    iris_data[['sepal length (cm)', 'sepal width (cm)']], iris_data[['target']], test_size=0.3, random_state=4
)

sc = StandardScaler()
train_feature = sc.fit_transform(train_feature)
test_feature = sc.fit_transform(test_feature)
train_target = np.array(train_target)
test_target = np.array(test_target)
print(train_feature, "\n",test_feature)



class LogisticRegression():
    def __init__(self):
        super(LogisticRegression, self).__init__()

    def linear(self, x, w, b):

        return np.dot(x, w) + b

    def sigmoid(self, x):

        return 1/(1 + np.exp(-x))

    def forward(self, x, w, b):
        y_pred = self.sigmoid(self.linear(x, w, b)).reshape(-1, 1)

        return y_pred



model = LogisticRegression()
learning_rate = 0.01





class BinaryCrossEntropy():
    def __init__(self):
        super(BinaryCrossEntropy, self).__init__()
    
    def cross_entropy(self, y_pred, target):
        x = target*np.log(y_pred) + (1-target)*np.log(1-y_pred)

        return -(np.mean(x))

    def forward(self, y_pred, target):

        return self.cross_entropy(y_pred, target)

# GradientDescent
class GradientDescent():
    def __init__(self, lr=0.1):
        super(GradientDescent, self).__init__()
        self.lr = lr

    def forward(self, w, b, y_pred, target, data):
        w = w - self.lr * np.mean(data * (y_pred - target), axis=0)
        b = b - self.lr * np.mean((y_pred - target), axis=0)

        return w, b







criterion = BinaryCrossEntropy()
optimizer = GradientDescent(lr=learning_rate)



# 3) training loop
w = np.array([0, 0])
b = np.array([0])
num_epochs = 100
for epoch in range(num_epochs):
    for i, data in enumerate(train_feature):
        # forward pass and loss
        y_pred = model.forward(data, w, b)
        loss = criterion.forward(y_pred, train_target[i])

        # update
        w, b = optimizer.forward(w, b, y_pred, train_target[i], data)

    if (epoch+1) % 10 == 0:
        print(f'epoch {epoch + 1}: loss = {loss}')
        
        
        
# checking testing accuracy
y_pred = model.forward(test_feature, w, b)
y_pred_cls = y_pred.round()
acc = np.equal(y_pred_cls, test_target).sum() / float(test_target.shape[0])
print(f'accuracy = {acc: .4f}')