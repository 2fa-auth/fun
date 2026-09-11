#!/home/client/Documents/fun/py/venv/bin/python3
import torch
import torch.nn as nn 
import torchvision.models as models
import torch.utils.data as data

import math 
import copy
import random

"""
pet detector & домашний детектор.
  *на псевдо изображениях
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

    self.fc1 = nn.Linear(in_features, 128)
    self.bn1 = nn.BatchNorm1d(128)
    self.fc2 = nn.Linear(128, 256)
    self.bn2 = nn.BatchNorm1d(256)
    self.fc3 = nn.Linear(256, 128)
    self.bn3 = nn.BatchNorm1d(128)
    self.fc4 = nn.Linear(128, out_features)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = x.view(x.size(0), -1)
    out = self.relu(self.bn1(self.fc1(x)))
    out = self.relu(self.bn2(self.fc2(out)))
    out = self.relu(self.bn3(self.fc3(out)))
    out = self.fc4(out)
    return out
  
class Bbox:
  def __init__(self, size_selection, coords, images):
    self.size_selection = size_selection
    self.images = images
    self.coords = coords

    #сортировка
    x1 = torch.min(self.coords[:, 0], self.coords[:, 2]) 
    y1 = torch.min(self.coords[:, 1], self.coords[:, 3])
    x2 = torch.max(self.coords[:, 0], coords[:, 2])
    y2 = torch.max(self.coords[:, 1], self.coords[:, 3])

    self.coords[:, 0] = x1 
    self.coords[:, 1] = y1 
    self.coords[:, 2] = x2
    self.coords[:, 3] = y2

  def draw_bbox(self, min_shades=0, max_shades=255):
    for sz in range(self.size_selection):
      x1 = self.coords[sz, 0:1] 
      y1 = self.coords[sz, 1:2]
      x2 = self.coords[sz, 2:3]
      y2 = self.coords[sz, 3:4]

      for color in range(3):
        rgb = random.random() * (max_shades - min_shades) + min_shades #255 оттенков
        self.images[sz, color, y1:y1+1, x1:x2+1] = rgb
        self.images[sz, color, y2:y2+1, x1:x2+1] = rgb
        self.images[sz, color, y1:y2+1, x1:x1+1] = rgb
        self.images[sz, color, y1:y2+1, x2:x2+1] = rgb  

    return self.images


class BboxLoss_withMSE:
  def __init__(self):
    self.num_calls = 0

  def IoU(self, box1, box2):
    x1_box1 = torch.min(box1[..., 1:2], box1[..., 3:4])
    y1_box1 = torch.min(box1[...,2:3], box1[..., 4:5]) 
    x2_box1 = torch.max(box1[..., 3:4], box1[..., 1:2])
    y2_box1 = torch.max(box1[..., 4:5], box1[..., 2:3])

    x1_box2 = torch.min(box2[..., 1:2], box2[..., 3:4])
    y1_box2 = torch.min(box2[...,2:3], box2[..., 4:5]) 
    x2_box2 = torch.max(box2[..., 3:4], box2[..., 1:2])
    y2_box2 = torch.max(box2[..., 4:5], box2[..., 2:3])

    x1_box = torch.max(x1_box1, x1_box2)
    y1_box = torch.max(y1_box1, y1_box2)
    x2_box = torch.min(x2_box1, x2_box2)
    y2_box = torch.min(y2_box1, y2_box2)

    width = torch.clamp(x2_box - x1_box, 0)
    height = torch.clamp(y2_box - y1_box, 0)

    width_box1, height_box1 = x2_box1 - x1_box1, y2_box1 - y1_box1
    width_box2, height_box2 = x2_box2 - x1_box2, y2_box2 - y1_box2
    intersection_area = width * height
    box1_area, box2_area = width_box1 * height_box1, width_box2 * height_box2
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / (union_area + 1e-6)

  def __call__(self, pred, y):
    criterion = nn.MSELoss()

    iou_loss = self.IoU(pred, y)
    # if self.num_calls % 1000 == 0:
      # print(self.num_calls)
      # print(iou_loss)
    # self.num_calls += 1
    
    coords_loss = criterion(pred[:, 1:], y[:, 1:])

    iou_loss = 1 - iou_loss
    loss_class = criterion(y[..., 0], pred[..., 0])

    return (iou_loss + coords_loss).mean() # здесь я специально не добавлял ошибку класса поскольку классов пока не существует



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
  width_image, height_image = (7, 7)  
  low, high = 0, int(width_image + height_image) / 2 
  size_selection = 400
  percent_val = 15 
  percent_test = 15 
  num_classes = 0

  class_id = torch.round(torch.round(torch.rand((size_selection, 1)) * num_classes + 0)).to(torch.int32)
  coords = torch.floor(torch.rand((size_selection, 4)) * (high - low) + low).to(torch.int32)
  target = torch.cat([class_id, coords], dim=1)
  images = torch.randn(size_selection, 3, width_image, height_image)*0+0

  box_image = Bbox(size_selection, coords, images)
  images = box_image.draw_bbox(min_shades=1,max_shades=1)

  target = target.to(torch.float32)

  train_x, train_y, val_x, val_y, test_x, test_y = gen_xy(images, target, percent_val, percent_test)
  train_loader = data.DataLoader(dataset = SetClassBoxes(train_x, train_y), batch_size=64, shuffle=True)
  val_loader = data.DataLoader(dataset = SetClassBoxes(val_x, val_y), batch_size=32, shuffle=True)
  test_loader = data.DataLoader(dataset = SetClassBoxes(test_x, test_y), batch_size=32, shuffle=False)

  model = ModelLastLayer(images.size(1)*images.size(2)*images.size(3), 5)    
  criterion = BboxLoss_withMSE()
  optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)
  num_ep = 1000



  print("ОБУЧЕНИЕ & ВАЛИДАЦИЯ\n")
  for _ep in range(num_ep):
    loss_train, t_cnt = 0,0
    loss_val, v_cnt = 0,0

    model.train()
    for x, y in train_loader:
      pred = model(x)
      loss = criterion(pred, y)
      # if _ep % 10 == 0:
        # print(f'pred:\n{pred}, y:\n{y}\nloss = {loss}')

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

    if _ep % 200 == 0:
      loss_mean_train = loss_train / t_cnt
      loss_mean_val = loss_val / v_cnt
      print(f'ep [{_ep}/{num_ep}] \t\t LOSS TRAIN {loss_mean_train} \t\t LOSS VAL {loss_mean_val}')
    
  print("\nТЕСТ")
  model.eval()
  losses ,l_cnt = 0,0
  with torch.no_grad():
    for x,y in test_loader:
      pred=model(x)
      loss=criterion(pred, y)
      losses += loss.item()
      l_cnt +=1
    print(f"средняя ошибка модели: {losses / l_cnt}")
        
if __name__ == "__main__":
  main()
