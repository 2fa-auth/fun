#!/home/client/Documents/fun/py/venv/bin/python3
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

class PetDataSet: # совместимый с torch.utils.data.Dataset
  """
  PetDataSet :
    принимает объект и метку
    возвращает количество примеров samples
    по индексу возвращает соответствующую пару
  """
  def __init__(self, data, target):
    self.data = data
    self.target = target

  def __getitem__(self, index):
    return (self.data[index], self.target[index])

  def __len__(self):
    return self.data.size(0)

def gen_size(data_size, test_size):
  return int(data_size * test_size / 100)

def gen_sets(train_arr, target_train_arr, target_test_arr, test_arr):
  set_train = PetDataSet(train_arr, target_train_arr)
  set_test = PetDataSet(test_arr, target_test_arr)
  return set_train, set_test

def gen_data(x, target, train_size):
  train_arr = x[:train_size]
  test_arr = x[train_size:]
  target_train_arr = target[:train_size]
  target_test_arr = target[train_size:]

  set_train, set_test = gen_sets(train_arr, target_train_arr, test_arr, target_test_arr)

  data_train = data.DataLoader(set_train, batch_size=32, shuffle=True)
  data_test = data.DataLoader(set_test, batch_size=16, shuffle=False)
  return (data_train, data_test)


# === DROPOUT ===
# class MLP(nn.Module):
#   def __init__(self, out_features_x, out_features_y):
#     super().__init__()

#     self.fc1 = nn.Linear(out_features_x, 16)
#     self.fc2 = nn.Linear(16, 8)
#     self.fc3 = nn.Linear(8, 4)
#     self.fc4 = nn.Linear(4, out_features_y)
#     self.relu = nn.ReLU()
#     self.dropout = nn.Dropout(p=0.5)

#   def forward(self, X):
#     out = self.relu(self.fc1(X))
#     out = self.relu(self.fc2(out))
#     out = self.relu(self.fc3(out))
#     out = self.fc4(out) 
#     return out

class MLP(nn.Module):
  def __init__(self, out_features_x, out_features_y):
    super().__init__()

    self.fc1 = nn.Linear(out_features_x, 16)
    self.fc2 = nn.Linear(16, 8)
    self.fc3 = nn.Linear(8, 4)
    self.fc4 = nn.Linear(4, out_features_y)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(p=0.5)

  def forward(self, X):
    out = self.dropout(self.relu(self.fc1(X)))
    out = self.dropout(self.relu(self.fc2(out)))
    out = self.dropout(self.relu(self.fc3(out)))
    out = self.fc4(out) 
    return out


def main():
  data_size = 100
  
  test_size = gen_size(data_size, 30) # 30% тестовой выборке, а остальное обучающей
  train_size = data_size - test_size

  x = torch.linspace(-5, 5, data_size).unsqueeze(-1)
  target = x ** 2  

  data_train, data_test = gen_data(x, target, train_size)

  model = MLP(1, 1)

  model.train()
  pred1 = model(x)
  pred2 = model(x)
  print(f"режим train: pred1 {pred1[:2]} | pred2 {pred2[:2]}")

  model.eval()
  pred1 = model(x)
  pred2 = model(x)
  print(f"режим eval: pred1 {pred1[:2]} | pred2 {pred2[:2]}")

  exit(0)

  opt = optim.SGD(model.parameters(), lr=0.001)
  criterion = nn.MSELoss()

  num_ep = 1000

  model.train()
  for _ep in range(num_ep):
    loss_mean = 0
    loss_cnt = 0
    for x, y in data_train:
      pred = model(x)
      loss = criterion(pred, y)
      loss_mean += loss.item()
      loss_cnt += 1

      opt.zero_grad()
      loss.backward()
      opt.step()

    if _ep % 50 == 0:
      print(f'_ep = {_ep} | loss mean = {loss_mean}')
     
  model.eval()


if __name__ == "__main__":
  main()
  
  