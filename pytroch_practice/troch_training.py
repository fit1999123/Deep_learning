import torch
import numpy as np 
import cv2


# if torch.cuda.is_available():
    
#     device = torch.device("cuda")
#     x = torch.ones(5,device=device)
#     y = torch.ones(5)
#     y = y.to(device)
#     z = x+y
#     # z = z.to("cpu")
#     print(z)
#     print(type(z))
# else:
#     print("No GPU")





a = [1,2,3,4]

b= iter(a)


print(b)