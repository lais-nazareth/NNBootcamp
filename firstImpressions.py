import torch

# this is a 1D tensor
a = torch.tensor([2, 2, 1])
print(a)

# this is a 2D tensor
b = torch.tensor([[2, 2, 1], [3, 5, 4], [7, 8, 9], [1, 3, 2]])
print(b)

# the size of tensors (shape is an attribute and size is a method)
print(a.shape)
print(a.size())

print(b.shape)
print(b.size())

#  get the height/number of rows of b
print(b.shape[0])

c = torch.FloatTensor([[2, 1, 4], [5, 2, 3], [4, 7, 2]])
# or we can do
# c = torch.tensor([2, 2, 1], dtype = torch.float)

d = torch.DoubleTensor([[2, 1, 4], [5, 6, 7], [0, 1 , 9]])
# or we can do
# d = torch.tensor([2, 2, 1], dtype = torch.double)

print(c)
print(c.dtype)

print(c.mean())
print(c.std())

print(d)
print(d.dtype)

print(d.mean())
print(d.std())

print()

# reshape b
# note: if one of the dimensions is -1, the size can be figured out by python
# view is the method to reshape

print(b.view(-1, 1))
print(b.view(12))
print(b.view(-1, 4))
print(b.view(3, 4))

# assign b a new shape
b = b.view(1, -1)
print(b)
print(b.size())

# create a 3D tensor with 2 channels, 3 rows and 4 collumns (channels, rows, collumns) 
threeDim = torch.randn(2, 3, 4)
print('\n')
print(threeDim)
print(threeDim.view(2, 12))
print(threeDim.view(2, -1))

# create a matrix with random numbers between 0 and 1
r = torch.rand(4, 4)
print(r)

# create a matrix with random numbers taken from a normal distribution with mean 0 and variance 1
r2 = torch.randn(4, 4)
print(r2)
print(r2.dtype)

# create an array of 5 random integers from values between 6 and 9 (exclusive of 10)
inArray = torch.randint(6, 10, (5,))
print(inArray)
print(inArray.dtype)

# create a matrix of size 3x3 filled with random integers between 6 and 9 (exclusive of 10)
inMatrix = torch.randint(6, 10, (3,3))
print(inMatrix)

# get the number of elements in inArray
print(torch.numel(inArray))
# get the number of elements in inMatrix
print(torch.numel(inMatrix))

# construct a matrix of 0s and of dtype long
z = torch.zeros(3, 3, dtype=torch.long)
print(z)

# construct a matrix of 1s
o = torch.ones(3, 3)
print(o)
print(o.dtype)

# convert the data type of the tensor
r2_like = torch.rand_like(r2, dtype=torch.double)
print(r2_like)

# add two tensors (must be same size and data type)
addResult = torch.add(r, r2)
print(addResult)

# in-place addition (change the value of r2) WHATCH FOR _
r2.add_(r) # r2 = torch.add(r, r2)
print(r2)

print(r2[:,1]) # all rows in first collumn
print(r2[:,:2]) # all rows from collumn 0 until 2 (exclusive)
print(r2[:3,:]) # 
numTen = r2[2, 3] #returns tensor
print(numTen)
print(numTen.item()) # extract number from tensor

