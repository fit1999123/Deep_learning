import torch
import torch.nn as nn

#訓練樣本
x = torch.tensor([[1],[2],[3],[4]],dtype=torch.float32)
#實際結果
y = 2*x
x_test = torch.tensor([[5]],dtype=torch.float32)

# w = torch.tensor(0.0,dtype=torch.float32,requires_grad=True)
# #model prediction 
# def forward(x):
#     return w*x

n_sample,n_feature =x.shape

model= nn.Linear(n_feature,n_feature)
print(model)
print(f"prediction before training : f(5) = {model(x_test).item():.3f}")

#loss = MSE
# def loss(y,y_prediction):
w,b = model.parameters()
#     return ((y_prediction-y)**2).mean()
#梯度 = backward pass
#MSE = 1/N*(wx-y)**2
#dj/dw = 1/N*2x*(wx-y)

# def gradient(x,y,y_prediction):
    
#     return np.dot(2*x,y_prediction-y).mean()


#start training

learning_rate = 0.01
n_iters = 10000

loss = nn.MSELoss()
optimizer = torch.optim.SGD([w],lr=learning_rate)

for i in range(n_iters):
    #開始猜
    y_pred = model(x)
    #計算損失的y
    loss_y = loss(y,y_pred)
    
    loss_y.backward() #dloss_y/dw
    
    #update_weight
    optimizer.step()
    optimizer.zero_grad()
        
    if i%1000 ==0:
        w,b = model.parameters()

        print(f"epoch {i+1}: w = {w[0][0].item():3f}, loss ={loss_y:.8f}")
     
    
print(f"prediction after training : f(5) = {model(x_test).item():.3f}")
