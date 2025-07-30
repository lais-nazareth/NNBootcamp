import torch

tensor1 = torch.tensor([1, 2, 3, 4])
tensor_a = torch.unsqueeze(tensor1, 0)
print(tensor_a)
print(tensor_a.shape)

tensor_b = torch.unsqueeze(tensor1, 1)
print(tensor_b)
print(tensor_b.shape)
print('\n')
tensor2 = torch.rand(2, 3, 4)
print(tensor2)
print('\n')
tensor_c = tensor2[:,:,2] # all channels all rows in the 2nd column
print(tensor_c)
print(tensor_c.shape)
print('\n')
tensor_d = torch.unsqueeze(tensor_c, 2)
print(tensor_d)
print(tensor_d.shape)