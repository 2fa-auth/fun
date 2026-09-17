#!/home/client/Documents/fun/py/venv/bin/python3
import torch
import torch.nn as nn 
import torchvision.models as models
import torch.utils.data as data

import math 
import copy
import random

class SetClassBoxes(data.Dataset): 
  def __init__(self, images, target):
    super().__init__()
    self.images = images
    self.target = target
    self.len = int(images.size(0))

  def __len__(self):
    return self.len

  def __getitem__(self, index):
    return (self.images[index], self.target[index])


class ModelLastLayer(nn.Module): 
  def __init__(self, in_features, out_features):
    super().__init__()

    self.fc1 = nn.Linear(in_features, 128)
    self.bn1 = nn.BatchNorm1d(128)
    self.fc2 = nn.Linear(128, 512)
    self.bn2 = nn.BatchNorm1d(512)
    self.fc3 = nn.Linear(512, 256)
    self.bn3 = nn.BatchNorm1d(256)
    self.fc4 = nn.Linear(256, out_features)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = x.view(x.size(0), -1)
    out = self.relu(self.bn1(self.fc1(x)))
    out = self.relu(self.bn2(self.fc2(out)))
    out = self.relu(self.bn3(self.fc3(out)))
    out = self.fc4(out)
    return out

class BboxLoss_withMSE:
  def IoU(self, box1, box2):
    x1_box1 = torch.min(box1[..., 2:3], box1[..., 4:5])
    y1_box1 = torch.min(box1[...,3:4], box1[..., 5:6]) 
    x2_box1 = torch.max(box1[..., 4:5], box1[..., 2:3])
    y2_box1 = torch.max(box1[..., 5:6], box1[...,3:4])

    x1_box2 = torch.min(box2[..., 2:3], box2[..., 4:5])
    y1_box2 = torch.min(box2[...,3:4], box2[..., 5:6]) 
    x2_box2 = torch.max(box2[..., 4:5], box2[..., 2:3])
    y2_box2 = torch.max(box2[..., 5:6], box2[..., 3:4])
    """
    x1_box1 = torch.min(box1[2:3], box1[ 4:5])
    y1_box1 = torch.min(box1[3:4], box1[ 5:6]) 
    x2_box1 = torch.max(box1[ 4:5], box1[ 2:3])
    y2_box1 = torch.max(box1[ 5:6], box1[3:4])

    x1_box2 = torch.min(box2[ 2:3], box2[ 4:5])
    y1_box2 = torch.min(box2[3:4], box2[ 5:6]) 
    x2_box2 = torch.max(box2[ 4:5], box2[ 2:3])
    y2_box2 = torch.max(box2[ 5:6], box2[ 3:4])
    """
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

  def __call__(self, pred, y, iou_print=0):
    criterion = nn.MSELoss()

    present = pred[..., 0].unsqueeze(-1)

    present_loss = criterion(pred[..., 0:1], y[..., 0:1])
    class_loss = (criterion(pred[..., 1:2], y[..., 1:2])) * present

    iou_loss = (1 - self.IoU(pred, y)) * present

    if iou_print:
      print(f"IOU LOSS: {iou_loss}")
    coords_loss = (criterion(pred[:, :, 2:], y[:, :, 2:])) * present

    return (iou_loss + coords_loss + class_loss + present_loss).mean()


class Bbox:
  def __init__(self, images):
    self.batch_size = images.size(0)
    self.images = images
    self.h = images.size(-1)
    self.w = images.size(-2)
    self.hbox = 3
    self.wbox = 3

  def draw_boxes(self, labels):
    target = torch.zeros(self.batch_size, labels, 6)

    for n in range(self.batch_size):
      for n_class in range(2):
        label = random.randint(0,labels)
        present = 0 if label == 0 else 1
        y_rand = random.randint(1,self.h-2) * present
        x_rand = random.randint(1,self.w-2) * present
        self.images[n, :, y_rand, x_rand] = label

        x1 = (x_rand-1) * present 
        y1 = (y_rand-1) * present
        x2 = (x_rand+1) * present
        y2 = (y_rand+1) * present

        target[n, n_class, 0] = present
        target[n, n_class, 1] = label
        target[n, n_class, 2] = x1
        target[n, n_class, 3] = y1
        target[n, n_class, 4] = x2
        target[n, n_class, 5] = y2

        if present:
          self.images[n, :, y1:y1+1, x1:x2+1] = 10
          self.images[n, :, y2:y2+1, x1:x2+1] = 10
          self.images[n, :, y1:y2+1, x1:x1+1] = 10
          self.images[n, :, y1:y2+1, x2:x2+1] = 10

    order = torch.argsort(target[:, :, 2], dim=1)
    target = target.gather(1, order.unsqueeze(-1).expand_as(target))
    return self.images, target      

def gen_xy(images, target, test_percent, val_percent):
  main_size = images.size(0)
  train_percent = 100 - (test_percent + val_percent)   

  train_size = int(main_size * train_percent / 100)
  val_size = int(main_size * val_percent / 100)

  train_x = images[:train_size, ...]
  train_y = target[:train_size, ...]
  val_x = images[train_size:train_size+val_size, ...]
  val_y = target[train_size:train_size+val_size, ...]
  test_x = images[train_size+val_size:, ...]
  test_y = target[train_size+val_size:,...]

  return (train_x,train_y,val_x, val_y,test_x,test_y)


def main():
  size_val, size_test = (20, 10)
  h, w = (7, 7)  

  batch_size = 700
  labels = 2

  images = torch.zeros(batch_size, 3, w,h)
  box = Bbox(images)
  images, target = box.draw_boxes(labels)


  train_x, train_y, val_x, val_y, test_x, test_y = gen_xy(images, target, size_val, size_test)
  train_loader = data.DataLoader(dataset = SetClassBoxes(train_x, train_y), batch_size=64, shuffle=True)
  val_loader = data.DataLoader(dataset = SetClassBoxes(val_x, val_y), batch_size=32, shuffle=True)
  test_loader = data.DataLoader(dataset = SetClassBoxes(test_x, test_y), batch_size=32, shuffle=False)

  model = ModelLastLayer(images.size(1)*images.size(2)*images.size(3), 6*labels)    
  criterion = BboxLoss_withMSE()
  optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)
  num_ep = 2000


  print("ОБУЧЕНИЕ & ВАЛИДАЦИЯ\n")
  for _ep in range(num_ep):
    loss_train, t_cnt = 0,0
    loss_val, v_cnt = 0,0

    model.train()
    for x, y in train_loader:
      pred = model(x).reshape(-1, 2, 6)
      order = torch.argsort(pred[:, :, 2], dim=1)
      pred = pred.gather(1, order.unsqueeze(-1).expand_as(pred))

      loss = criterion(pred, y)
      # if _ep % 1000 == 0:
        # print(f'pred:\n{pred}, y:\n{y}\nloss = {loss}')

      loss_train += loss.item()
      t_cnt += 1
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    with torch.no_grad():
      model.eval() 
      for x, y in val_loader:
        pred = model(x).reshape(-1, 2, 6)
        order = torch.argsort(pred[:,:,2], dim=1)
        pred = pred.gather(1,order.unsqueeze(-1).expand_as(pred))
        loss_val += criterion(pred, y).item()
        v_cnt += 1

    if _ep % 200 == 0:
      loss_mean_train = loss_train / t_cnt
      loss_mean_val = loss_val / v_cnt
      print(f'ep [{_ep}/{num_ep}] \t\t LOSS TRAIN {loss_mean_train} \t\t LOSS VAL {loss_mean_val}')
      
  print("\nТЕСТ") 
  model.eval()

  losses = 0 
  l_cnt = 0

  patience = 4
  
  with torch.no_grad():
    for x,y in test_loader:
      pred=model(x).reshape(-1, 2, 6)
      order = torch.argsort(pred[:,:,2], dim=1)
      pred = pred.gather(1, order.unsqueeze(-1).expand_as(pred))

      loss=criterion(pred, y, iou_print=1)

      if patience > 0:
        # print(f'predision:\n{torch.round(pred[0])}')
        # print(f'y:\n{torch.round(y[0])}')
        patience -= 1      

      losses += loss.item()
      l_cnt +=1
    print(f"средняя ошибка модели после теста: {losses / l_cnt}")

  torch.save(model.state_dict(), 'model_params.pth.tar')



if __name__ == "__main__":
  main()