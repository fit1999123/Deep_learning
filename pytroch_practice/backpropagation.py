from tkinter import Y
import torch

x = torch.tensor(1.0,requires_grad=True)
w = torch.tensor(1.0,requires_grad=True)
y = torch.tensor(2.0,requires_grad=True)
y_hat = x*w
s = y_hat-y
loss=s**2

print(loss)
loss.backward()
print(w.grad)
