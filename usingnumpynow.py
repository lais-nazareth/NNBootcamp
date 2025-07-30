import torch
import numpy as np

# converting a Torch tensor to a numpy array
a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

# see how the numpy array changed their value
a.add_(1)
print(a)
print(b)

# converting numpy array to torch tensor
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

# move the tensor to the gpu
r2 = torch.tensor([[3,2,1], [4, 5, 6], [3, 4, 2]])
r2 = r2.cuda()
print(r2)

# provide easy switching between CPU and GPU
a = torch.tensor([2, 2, 1])
CUDA = torch.cuda.is_available()
print(CUDA)
if CUDA:
    a = a.cuda()
    print(a)

# you can also convert a list to a tensor
a = [2, 3, 1, 4]
print(a)
toList = torch.tensor(a)
print(toList, toList.dtype)

data = [[1., 2.], [3., 4.],
        [5., 6.], [7., 8.]]
T = torch.tensor(data)
print(T, T.dtype)
