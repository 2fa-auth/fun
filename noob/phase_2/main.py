#!/home/client/Documents/fun/py/venv/bin/python3
import torch 
import torch.nn as nn
import torch.optim as optim


class Mod(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(1, 8)
    self.fc2 = nn.Linear(8, 4)
    self.fc3 = nn.Linear(4, 1)

    self.relu = nn.ReLU()
  
  def forward(self, X):
    out = self.relu(self.fc1(X))
    out = self.relu(self.fc2(out))
    out = self.fc3(out)
    return out

def main():
  X = torch.linspace(-2, 2, 100).unsqueeze(-1)
  target = X ** 2

  model = Mod()
  opt = optim.SGD(model.parameters(), lr=0.01)
  criterion = nn.MSELoss()
  
  for step in range(1000):
    pred = model(X)
    loss = criterion(pred, target)
    opt.zero_grad()
    loss.backward()
    opt.step()
    
    if step % 100 == 0:
      print(f"step = {step} | loss = {loss}")
  print('train end\n')

  X = torch.linspace(-1, -3, 10).unsqueeze(-1)
  target = X ** 2
  print("real:")
  print(f"X = {X}")
  print(f"target = {target}")
  print(f"\nprediction:")
  with torch.no_grad():
      pred = model(X)
      print(f'X = {X}')
      print(f'pred = {pred}')

if __name__ == "__main__":
  main()
  
