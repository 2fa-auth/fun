#!/home/client/Documents/fun/py/venv/bin/python3
import torch

class LinearModel(torch.nn.Module):
  def __init__(self):
    super().__init__()

    self.w = torch.nn.Parameter(torch.tensor(0.))
    self.b = torch.nn.Parameter(torch.tensor(0.))

  def forward(self, X):
    return X * self.w + self.b

if __name__ == '__main__':
  model = LinearModel()

  X = torch.tensor([1.])

  print(list(model.parameters()))
  print(model(X))










