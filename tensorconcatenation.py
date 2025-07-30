import torch
import numpy as np

# tensor concatenation
first1 = torch.randn(2, 5)
print(first1)

second1 = torch.randn(3, 5)
print(second1)

# concatenate along the 0 dimension (concatenate rows)
con1 = torch.cat([first1, second1])
print('\n')
print(con1)
print('\n')

first2 = torch.randn(2, 3)
print(first2)

second2 = torch.randn(2, 5)
print(second2)

# concatenate along the 1 dimension (concatenate columns)
con2 = torch.cat([first2, second2], 1)

print('\n')
print(con2)
print('\n')

