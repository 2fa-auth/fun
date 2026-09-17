import torch

x = torch.tensor([[1., 2., 4., 2., 6., 4.],
                  [1., 2., 1., 0., 3., 2.]])

values, indices = torch.sort(x, dim=0)
print(values)
