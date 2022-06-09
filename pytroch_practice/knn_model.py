import numpy as np
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

cmap = ListedColormap(["red","green","blue"])


def calculate_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))
class knn():
    
    def __init__(self,k=3):
        self.k = k
    def fit(self,x,y):
        self.x_train = x
        self.y_train = y    
    def predict(self,x):
        predicted_labels = [self._predict(i) for i in x]
        return np.array(predicted_labels)
        
    def _predict(self,x):
        #計算距離
        distance = [calculate_distance(x,i) for i in self.x_train]
        #get k nearst samples,labels
        k_indices = np.argsort(distance)[:self.k]
        # print(k_indices)
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # print(k_nearest_labels)
        #vote,最常見的物件
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    
iris = datasets.load_iris()
x,y = iris.data, iris.target
print(x.shape)
print(y.shape)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1234)

# print(x_train) 鄰居
# print(y_train) 鄰居的種類

# print(x_test)  待分類的樣本
# print(y_test)  待分類的正確答案



plt.figure()
plt.scatter(x[:,0],x[:,1],c=y,cmap=cmap,s=20)
plt.show()
plt.close()

clf = knn(3)
clf.fit(x_train,y_train)
predictions = clf.predict(x_test)

acc = np.sum(predictions==y_test)/len(y_test)

# print(acc)