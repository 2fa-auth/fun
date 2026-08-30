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
#     out = self.dropout(self.relu(self.fc1(X)))
#     out = self.dropout(self.relu(self.fc2(out)))
#     out = self.dropout(self.relu(self.fc3(out)))
#     out = self.fc4(out) 
#     return out


# === BATCH NORMALIZATION ===
# class MLP(nn.Module):
#   def __init__(self, out_features_x, out_features_y):
#     super().__init__()

#     self.fc1 = nn.Linear(out_features_x, 16)
#     self.fc1_batchnorm = nn.BatchNorm1d(16)
#     self.fc2 = nn.Linear(16, 8)
#     self.fc2_batchnorm = nn.BatchNorm1d(8)
#     self.fc3 = nn.Linear(8, 4)
#     self.fc3_batchnorm = nn.BatchNorm1d(4)
#     self.fc4 = nn.Linear(4, out_features_y)

#     self.relu = nn.ReLU()

#   def forward(self, X):
#     out = self.relu(self.fc1_batchnorm(self.fc1(X)))
#     out = self.relu(self.fc2_batchnorm(self.fc2(out)))
#     out = self.relu(self.fc3_batchnorm(self.fc3(out)))
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

  def forward(self, X):
    out = self.relu(self.fc1(X))
    out = self.relu(self.fc2(out))
    out = self.relu(self.fc3(out))
    out = self.fc4(out) 
    return out


def main():
  data_size = 1000
  
  test_size = gen_size(data_size, 30) # 30% тестовой выборке, а остальное обучающей
  train_size = data_size - test_size

  print(f'train size = {train_size}')
  print(f'test size = {test_size}')
  

  x = torch.linspace(-5, 5, data_size).unsqueeze(-1)
  target = x ** 2 
  data_train, data_test = gen_data(x, target, train_size)

  model = MLP(1, 1)

  opt = optim.SGD(model.parameters(), lr=0.001)
  criterion = nn.MSELoss()

  num_ep = 1000

  for _ep in range(num_ep):
    train_loss = 0
    train_cnt = 0
    val_loss = 0
    val_cnt = 0

    model.train()
    for x, y in data_train:
      pred = model(x)
      loss = criterion(pred, y)
      train_loss += loss.item()
      train_cnt += 1

      opt.zero_grad()
      loss.backward()
      opt.step()

    model.eval()
    with torch.no_grad():
      for x, y in data_test:

        pred = model(x)
        val_loss += criterion(pred, y).item()
        val_cnt += 1      

    if _ep % 50 == 0:
      val_loss_mean = val_loss / val_cnt
      train_loss_mean = train_loss / train_cnt
      print(f'ep [{_ep}/{num_ep}] | loss mean (train) {train_loss_mean} | loss mean (val) {val_loss_mean}')



if __name__ == "__main__":
  main()

# вычисление batch normalization при batch = [-3, 3, 4]:
# 1/ batch.mean = 1.3
# 2/ x-mean (new batch) = [-4.3, 1.7, 2.7]
# 3/ variance := new_batch ** 2 = ([18.48, 2.89, 7.29 / 3]).mean = 9.55
# 4/ std = 3.09
# 5/ result: new batch / 3.09 = [-1.39, 0.55, 0.8] 


# вычисление softmax(x) при logits = [1, 2, 3] 
# exp(logit)
  # exp(1) = 2.71828 
  # exp(2) = 7.38905
  # exp(3) = 20.0855
# softmax(logit)
  # softmax(1) = 2.71828 / 30.19283 = 0.900
  # softmax(2) = 7.38905 / 30.19283 = 0.2447
  # softmax(3) = 20.0855 / 30.19283 = 0.6652