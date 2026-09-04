#!/home/client/Documents/fun/py/venv/bin/python3

import torch
import torch.nn as nn 
import torchvision.models as models

import torch.utils.data as data

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
    self.fc1 = nn.Linear(in_features, 1024)
    self.bn1 = nn.BatchNorm1d(1024)
    self.fc2 = nn.Linear(1024, 2048)
    self.bn2 = nn.BatchNorm1d(2048)
    self.fc3 = nn.Linear(2048, 512)
    self.bn3 = nn.BatchNorm1d(512)
    self.fc4 = nn.Linear(512, out_features)
    self.relu = nn.ReLU()
  def forward(self, x):
    x = x.view(x.size(0), -1)
    out = self.relu(self.bn1(self.fc1(x)))
    out = self.relu(self.bn2(self.fc2(out)))
    out = self.relu(self.bn3(self.fc3(out)))
    out = self.fc4(out)
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

def IoU(box1, box2):
  x1_box1, y1_box1 = torch.min(box1[..., 1:2], box1[..., 3:4]), torch.min(box1[...,2:3], box1[..., 4:5]) 
  x2_box1, y2_box1 = torch.max(box1[..., 3:4], box1[..., 1:2]), torch.max(box1[..., 4:5], box1[..., 2:3])
  x1_box2, y1_box2 = torch.min(box2[..., 1:2], box2[..., 3:4]), torch.min(box2[...,2:3], box2[..., 4:5]) 
  x2_box2, y2_box2 = torch.max(box2[..., 3:4], box2[..., 1:2]), torch.max(box2[..., 4:5], box2[..., 2:3])
  x1_box, y1_box = torch.max(x1_box1, x1_box2), torch.max(y1_box1, y1_box2)
  x2_box, y2_box = torch.min(x2_box1, x2_box2), torch.min(y2_box1, y2_box2)
  width, height = torch.clamp(x2_box - x1_box, 0), torch.clamp(y2_box - y1_box, 0)

  width_box1, height_box1 = x2_box1 - x1_box1, y2_box1 - y1_box1
  width_box2, height_box2 = x2_box2 - x1_box2, y2_box2 - y1_box2
  intersection_area = width * height
  box1_area, box2_area = width_box1 * height_box1, width_box2 * height_box2
  union_area = box1_area + box2_area - intersection_area
  return (intersection_area / union_area) + 1e-6

def BOXESLoss(pred, y):
  iou_boxes = IoU(pred, y)

  criterion = nn.MSELoss()
  loss_coords = criterion(pred, y)
  loss_iou = 1 - iou_boxes
  loss_class = criterion(y[..., 0], pred[..., 0])

  return (loss_coords + loss_iou + loss_class).mean()



def main():
  width_image, height_image = (16, 16) # размер изображения  
  low, high = 0, int(width_image + height_image) / 2 
  size_selection = 400 # количество изображений
  percent_val = 15 # 15% от 100%
  percent_test = 15 # 15% от 100% 
  num_classes = 2 # количество классов 

  class_id = torch.round(torch.rand((size_selection, 1)) * num_classes)
  coords = torch.rand((size_selection, 4)) * high

  images = torch.rand((size_selection, 3, width_image, height_image)) * (high - low) + low 
  target = torch.cat([class_id, coords], dim=1)

  train_x, train_y, val_x, val_y, test_x, test_y = gen_xy(images, target, percent_val, percent_test)
  train_set = SetClassBoxes(train_x, train_y)
  val_set = SetClassBoxes(val_x, val_y)
  test_set = SetClassBoxes(test_x, test_y)
  train_loader = data.DataLoader(dataset=train_set, batch_size=32, shuffle=True)
  val_loader = data.DataLoader(dataset=val_set, batch_size=16, shuffle=True)
  test_loader = data.DataLoader(dataset=test_set, batch_size=16, shuffle=False)

  model = ModelLastLayer(images.size(1)*images.size(2)*images.size(3), 5)    
  optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
  num_ep = 400
  print("ОБУЧЕНИЕ & ВАЛИДАЦИЯ\n")
  for _ep in range(num_ep):
    loss_train, t_cnt = 0,0
    loss_val, v_cnt = 0,0

    model.train()
    for x, y in train_loader:
      pred = model(x)
      loss = BOXESLoss(pred, y)
      loss_train += loss.item()
      t_cnt += 1
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    with torch.no_grad():
      model.eval() 
      for x, y in val_loader:
        pred = model(x)
        loss_val += BOXESLoss(pred, y).item()
        v_cnt += 1

    if _ep % 10 == 0:
      loss_mean_train = loss_train / t_cnt
      loss_mean_val = loss_val / v_cnt
      print(f'ep [{_ep}/{num_ep}] \t\t LOSS TRAIN {loss_mean_train} \t\t LOSS VAL {loss_mean_val}')
    
  print("\nТЕСТ")
  model.eval()
  losses = 0
  l_cnt = 0
  with torch.no_grad():
    for x, y in test_loader:
      pred = model(x)
      loss = BOXESLoss(pred, y)
      losses += loss.item()
      l_cnt += 1
    print(f"средняя ошибка модели: {losses / l_cnt}")
        
if __name__ == "__main__":
  main()
