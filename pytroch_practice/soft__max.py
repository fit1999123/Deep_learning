import torch
import torch.nn as nn
import numpy as np



def softmax_np(x):
    
    return np.exp(x)/np.sum(np.exp(x),axis=0)



x = np.array([1,2,3,4,5,6],dtype=np.float32)

output = softmax_np(x)

print(output)


x = torch.from_numpy(x)

output2 = torch.softmax(x,dim=0)

print(output2)