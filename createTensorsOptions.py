import torch
import numpy as np

data = np.array([1, 2, 3])
type(data)

tensor1 = torch.Tensor(data)
tensor2 = torch.tensor(data)
tensor3 = torch.as_tensor(data)
tensor4 = torch.from_numpy(data)

print(tensor1)
print(tensor2)
print(tensor3)
print(tensor4)

# torch.Tensor is a constructor which uses the global default dtype which is float.

print(torch.get_default_dtype())
print(tensor1.dtype == torch.get_default_dtype())

# torch.as_tensor and torch.from_numpy use memory sharing and do not create a new copy of the data.
# This means that if there is any change in the underlying array the tensor itself will also change.
# This feature can be used when optimizing the code.
