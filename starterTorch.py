import torch

x = torch.tensor([5.5, 3])
print(x)

x = torch.rand(5, 3)
print(x)

x = torch.zeros(5, 3, dtype=torch.long)
print(x)

x = torch.tensor([5.5, 3])
print(x)

x = x.new_ones(5, 3, dtype=torch.double)  # new_* methods take in sizes
print(x)

x = torch.randn_like(x, dtype=torch.float)  # override dtype!
print(x)  # result has the same size

print(x.size())

# Operations on tensors
# Syntax 1
y = torch.rand(5, 3)
print(x + y)

# Syntax 2
print(torch.add(x, y))

# Using an output tensor as an argument to save the operation
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

# adds x to y in place
y.add_(x)
print(y)

# Use standard slicing to slice the tensors
print(x[:, 1])

# Resizing/reshaping tensors
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())

# Use .item() to get items from the tensors
x = torch.randn(1)
print(x)
print(x.item())

# Using GPU with tensorflow
print(torch.cuda.get_device_name(0))

if torch.cuda.is_available():
    device = torch.device("cuda")  # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)  # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))
