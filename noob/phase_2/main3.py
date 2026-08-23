#!/home/client/Documents/fun/py/venv/bin/python3

import torch 
import torch.nn as nn
import torch.optim as optim


class NoobModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.w = nn.Parameter(torch.tensor(0., requires_grad=True))
    self.b = nn.Parameter(torch.tensor(0., requires_grad=True))
  def forward(self, X):
    return X * self.w + self.b
  
X = torch.tensor([1., 2., 3., 4.])
target = torch.tensor([7., 9., 11., 13.])

model = NoobModel()
opt = optim.SGD(model.parameters(), lr=0.01)

for s in range(100):
  pred = model(X)
  loss = ((pred - target) ** 2).mean()
  
  opt.zero_grad()
  loss.backward()
  opt.step()


print(f'model.w: {model.w}')
print(f'model.b: {model.b}')
print(f'final loss: {loss}')
