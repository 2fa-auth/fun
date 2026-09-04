#!/home/client/Documents/fun/py/venv/bin/python3

import torch
import torch.nn as nn 
import torchvision.models as models

import torch.utils.data as data

"""
pet detector & домашний детектор.
  *пока на псевдо изображениях
"""


class SetClassBoxes(data.Dataset): 
  def __init__(self, images, target):
    super().__init__()

    self.images = images
    self.target = target
    self.len = int((images.size(0) + target.size(0)) / 2)

  def __len__(self):
    return self.len

  def __getitem__(self, index):
    return (self.images[index], self.target[index])
  
class ModelLastLayer(nn.Module): 
  def __init__(self, in_features, out_features):
    super().__init__()

    self.fc1 = nn.Linear(in_features, 512)
    self.fc2 = nn.Linear(512, 256)
    self.fc3 = nn.Linear(256, out_features)
    self.relu = nn.ReLU()
  def forward(self, x):
    x = x.view(x.size(0), -1)
    out = self.relu(self.fc1(x))
    out = self.relu(self.fc2(out))
    out = self.fc3(out)
    return out


def gen_xy(images, target, test_percent, val_percent):
  main_size = images.size(0)

  train_percent = 100 - (test_percent + val_percent)   
  train_size = int(main_size * train_percent / 100)
  val_size = int(main_size * val_percent / 100)

  train_x, train_y = images[:train_size, ...], target[:train_size, ...]
  val_x, val_y = images[train_size:train_size+val_size, ...], target[train_size:train_size+val_size, ...]
  test_x, test_y = images[train_size+val_size:, ...], target[train_size+val_size:,...]
  return (train_x,train_y,val_x, val_y,test_x,test_y)


def main():
  width_image, height_image = (64, 64) # 4096
  low, high = 0, int(width_image + height_image) / 2 
  size_selection = 300
  percent_val = 15
  percent_test = 15
  num_classes = 18

  class_id = torch.round(torch.rand((size_selection, 1)) * num_classes)
  coords = torch.rand((size_selection, 4)) * high
  images = torch.rand((size_selection, 3, width_image, height_image)) * (high - low) + low 
  target = torch.cat([class_id, coords], dim=1)

  train_x, train_y, val_x, val_y, test_x, test_y = gen_xy(images, target, percent_val, percent_test)
  train_set = SetClassBoxes(train_x, train_y)
  val_set = SetClassBoxes(val_x, val_y)
  test_set = SetClassBoxes(test_x, test_y)
  train_loader = data.DataLoader(dataset=train_set, batch_size=64, shuffle=True)
  val_loader = data.DataLoader(dataset=val_set, batch_size=16, shuffle=True)
  test_loader = data.DataLoader(dataset=test_set, batch_size=16, shuffle=False)


  model = ModelLastLayer(images.size(1)*images.size(2)*images.size(3), 5)    
  criterion = nn.MSELoss()
  optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
  num_ep = 200
 


  # обучение / валидация
  for _ep in range(num_ep):
    loss_train, t_cnt = 0,0
    loss_val, v_cnt = 0,0

    model.train()
    for x, y in train_loader:
      pred = model(x)
      loss = criterion(pred, y)
      loss_train += loss.item()
      t_cnt += 1
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    with torch.no_grad():
      model.eval() 
      for x, y in val_loader:
        pred = model(x)
        loss_val += criterion(pred, y).item()
        v_cnt += 1

    if _ep % 10 == 0:
      loss_mean_train = loss_train / t_cnt
      loss_mean_val = loss_val / v_cnt
      print(f'ep [{_ep}/{num_ep}] \t\t loss train {loss_mean_train} \t\t loss val {loss_mean_val}')
    

  # тест




    
if __name__ == "__main__":
  main()
