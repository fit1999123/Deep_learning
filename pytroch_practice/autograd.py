import torch


x = torch.tensor([1,2,3],dtype=torch.float32,requires_grad=True)


fx = 2*x**3


fx.backward(fx)

print("grad x = ",x.grad)

print("fx = ",fx)

x.grad.zero_() #empty the x grad
