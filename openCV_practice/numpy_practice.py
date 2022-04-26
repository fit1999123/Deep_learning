from pickletools import uint8
import numpy as np

# my_list = [1,2,3]

# type(np.array(my_list))
# print(np.arange(0,3))

# print(np.zeros((5,5)))
# print(np.ones((5,5)))

arr = np.random.randint(1,101,100)
arr = arr.reshape((10,10))
print(arr)
print(arr[2:5,1:3])