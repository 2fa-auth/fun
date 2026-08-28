#!/home/client/Documents/fun/py/venv/bin/python3
import torch 
import torch.nn as nn
from torch.optim import SGD
import torch.nn.functional as F
import torch.utils.data as data 

class NoobSet(data.Dataset):
  def __init__(self, x, target):
    super().__init__()

    self.x = x
    self.target = target

    self.data = ()
    for num_val in range(x.size(0)):
      self.data += ((x[num_val], target[num_val]), )

    self.len_data = len(self.data)

  def __len__(self):
    return self.len_data

  def __getitem__(self, index):
    return self.data[index]


class NoobModel(nn.Module):
  def __init__(self, in_features, out_features):
    super().__init__()

    self.fc1 = nn.Linear(in_features, 8)
    self.fc2 = nn.Linear(8, 8)
    self.fc3 = nn.Linear(8, out_features)
    
    self.relu = nn.ReLU()

  def forward(self, X):
    out = self.relu(self.fc1(X))
    out = self.relu(self.fc2(out))
    out = self.fc3(out)

    return out

def main():
  x = torch.tensor([[24, 12.5, 5, 1], 
                   [35, 42.1, 12, 0],
                   [18, 2.0, 1, 1],
                   [41, 28.4, 9, 1],
                   [50, 5.3, 2, 0]], dtype=torch.float32)
  target = torch.tensor([0, 1, 0, 1, 0])

  data_set = NoobSet(x, target)
  train_loader = data.DataLoader(dataset=data_set, batch_size=5, shuffle=False)
   
  model = NoobModel(x.shape[1], 2)
  criterion = nn.CrossEntropyLoss()
  optimizer = SGD(model.parameters(), lr=0.01)
  
  for _ep in range(200):
    for x, y in train_loader:
      logits = model(x)
      loss = criterion(logits, y)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if _ep % 10 == 0:
        print(f'[{_ep}] | loss {loss.item()}')

  logits = model(x)
  logits = F.softmax(logits, dim=1)
  print(f'предсказания модели:\n{logits.tolist()}')
  

if __name__ == "__main__":
  main()