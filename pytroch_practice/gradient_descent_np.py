import numpy as np


#訓練樣本
x = np.array([1,2,3,4],dtype=np.float32)
y = 2*x**3
w = 0.0
#model prediction 
def forward(x):

    return w*x

#loss = MSE
def loss(y,y_prediction):
    
    return ((y_prediction-y)**2).mean()
#梯度
#MSE = 1/N*(wx-y)**2
#dj/dw = 1/N*2x*(wx-y)

def gradient(x,y,y_prediction):
    
    return np.dot(2*x,y_prediction-y).mean()


#start training

learning_rate = 0.01
n_iters = 10*2

for i in range(n_iters):
    #開始猜
    y_pred = forward(x)
    #計算損失的y
    loss_y = loss(y,y_pred)
    
    dw = gradient(x,y,y_pred)
    
    #update_weight
    
    w -= learning_rate*dw 
    
print(f"prediction before training : f(5) = {forward(5):.3f}")
