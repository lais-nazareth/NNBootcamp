import torch
import numpy as np

# remember, if requires_grad=True, the Tensor object keeps track of how it was created
x = torch.tensor([1.,2.,3.], requires_grad=True)
y = torch.tensor([4.,5.,6.], requires_grad=True)
# notice that both x and y have their requires_grad set to true, therefore we compute gradients with respect to them
z = x + y
print(z)
# z knows that it was created as a result of addition of x and y. It knows it wasnt read from a file
print(z.grad_fn)
# and if we go further on this
s = z.sum()
print(s)
print(s.grad_fn)

# now if we want to backpropagate on s, we can find the gradients of s with respect to x
s.backward()
print(x.grad)

# by default, tensors have requires_grad=False
x = torch.randn(2, 2)
y = torch.randn(2, 2)
print(x.requires_grad, y.requires_grad)
z = x + y
# so you can't backprop thorugh z
print(z.grad_fn)

# another way to set the requires_grad=True is
x.requires_grad_()
y.requires_grad_()
# z contains enough information to compute gradients, as we saw above
z = x + y
print(z.grad_fn)
# if any input to the operation has 'requires_grad=True', so will the output
print(z.requires_grad)
# now z has the computation history that relates itself to x and y

new_z = z.detach()
print(new_z.grad_fn)
# z.deatch() returns a tensor that shares the same storage as 'z', but with the computational history forgotten
# it doesn't know anything about how it was computed.
# in other words, we have broken the tensor away from its past history

# you can also stop autograd from tracking history on tensors
# this concept is useful when applying Transfer Learning

print(x.requires_grad)
print((x+10).requires_grad)

with torch.no_grad():
    print((x+10).requires_grad)

# one last example
x = torch.ones(2, 2, requires_grad=True)
print(x)
y = x + 2
print(y)
print(y.grad_fn)
z = y * y * 3
out = z.mean()
print(z, out)
out.backward()
print(x.grad)