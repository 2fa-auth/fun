#!/home/client/Documents/fun/venv/bin/python3
import os
from copy import deepcopy
from PIL import Image 
from time import sleep
import math 

import torch
from torch.optim import Adam
from torchvision.tv_tensors import BoundingBoxes
import torch.utils.data as data
import torchvision.transforms.v2 as v2
import torch.nn as nn

import cv2
import numpy as np

import utils
from model import BACKBONEModel


class SetNoob(data.Dataset):
  def __init__(self, pathto_dset, namedset, train=True, transforms=None):
    self.images = os.path.join(pathto_dset, namedset, "images", "train" if train else "val")
    self.labels = os.path.join(pathto_dset, namedset, "labels", "train" if train else "val")
    self.imgs_list = os.listdir(self.images)
    self.lbs_list = os.listdir(self.labels)
    self.transforms = transforms

  def __len__(self): 
    return len(self.imgs_list)    

  def __getitem__(self, index):
    target=torch.zeros(7, 7, NUM_BOXES * 5 + NUM_CLASSES)
    image=self.imgs_list[index]
    labels=[]

    # парсинг label*.txt файла
    for label in self.lbs_list:
      if label.split('.')[0] == image.split('.')[0]:
        with open(os.path.join(self.labels, label), "r") as f:
          ts = f.read().split('\n') # массив содержащий строку из labels
          # парсинг
          for t in ts: 
            ar = []
            for val in t.split(' '): 
              if val != '': ar.append(float(val)) # str -> float 
            if ar: labels.append(ar)
    
    labels = torch.tensor(labels)
    image = Image.open(os.path.join(self.images,image)).convert('RGB')
    w, h = image.size
    if self.transforms:
      bbox = BoundingBoxes(data=utils.denorm_coords(labels[:, 1:], w, h), format='CXCYWH', canvas_size=(h,w))
      image, labels[:, 1:] = self.transforms(image, bbox)
      labels[:, 1:] = utils.norm_coords(bbox, w, h)

    # 1. labels: [class, cx,cy,w,h] -> [cx1,cy1,w1,h1, conf1, cx2,cy2,w2,h2, conf2...N classes]
    # 2. добавление labels в сетку
    bbox1 = labels
    bbox2 = torch.zeros_like(bbox1)
    class_id = bbox1[:, 0].to(torch.int32)
    ones = torch.tensor([1 for _ in range(0, bbox1.size(0))]).unsqueeze(0).T
    labels = torch.cat([bbox1, bbox2, ones, torch.zeros(bbox1.size(0), NUM_CLASSES)], dim=1)[:, 1:]
    labels[:, class_id] = 1

    cell_x = torch.floor(labels[:, 0] * W_grid).to(torch.int32) 
    cell_y = torch.floor(labels[:, 1] * H_grid).to(torch.int32)
    target[cell_x, cell_y] = labels

    exit()
    return image, target


class BboxLoss_withMSE:
  def __call__(self, pred, y):
    objectness = y[..., 4].unsqueeze(-1)
    criterion = nn.MSELoss(reduction='none')

    present_loss = criterion(pred[..., 4:5], y[..., 4:5]).mean()
    
    iou_loss = (objectness * (1 - utils.IoU(pred, y))).mean()
    # class_loss = (present * criterion(pred[..., 1:2], y[..., 1:2])).mean()
    coords_loss = (objectness * criterion(pred[..., :4], y[..., :4])).mean()

    return (present_loss)


class ModelTrainer:
  def __init__(self, model, sets, lr=0.001):
    self.dev = 'cuda' if torch.cuda.is_available else 'cpu'
    self.model = model.to(self.dev)
    train_set, val_set, test_set = sets
    self.train_loader = data.DataLoader(dataset=train_set, batch_size=2, shuffle=True)
    self.val_loader = data.DataLoader(dataset=val_set, batch_size=2, shuffle=False)  
    self.test_loader = data.DataLoader(dataset=test_set, batch_size=2, shuffle=False)
    self.criterion = BboxLoss_withMSE()
    self.optimizer = Adam(params=self.model.parameters(), lr=lr)
  
  def train(self):
    losses = 0
    for x, y in self.train_loader:
      x, y = x.to(self.dev), y.to(self.dev)
      pred = self.model(x).reshape(-1, 10, 85)
      pred = utils.sort_by_x1(pred)

      loss = self.criterion(pred, y)
      
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()
      losses += loss.item()

    return losses / len(self.train_loader)

  def valid(self):
    losses = 0
    with torch.no_grad():
      for x, y in self.val_loader:
        x, y = x.to(self.dev), y.to(self.dev)
        pred = self.model(x).reshape(-1, 10, 85)
        pred = utils.sort_by_x1(pred)
        losses += self.criterion(pred, y).item()

    return losses / len(self.val_loader)
  
  def test(self):
    losses = 0
    last_pred = 0
    with torch.no_grad():
      for x,y in self.test_loader:
        x, y = x.to(self.dev), y.to(self.dev)
        pred = self.model(x).reshape(-1, 10, 85)
        pred = utils.sort_by_x1(pred)
        last_pred = pred
        last_y = y


    pred[..., 4] = torch.where(pred[..., 4] >= 0.5, 1, 0)

    # detection_objects(x, last_pred)
    print(y[0])
    print(pred[0])

    losses += self.criterion(pred, y).item()

    return losses / len(self.test_loader)

  def fit(self):
    #for x, y in self.train_loader: # отобразить изображение с рамками
      #detection_objects(x, y)
      #exit()

    for _ep in range(EPOCHS):
      self.model.train()
      train_meanloss = self.train()
      self.model.eval()
      val_meanloss = self.valid()
      if _ep % 10 == 0: print(f'[{_ep}/{EPOCHS}]\tLOSS TRAIN {train_meanloss}\tLOSS VAL {val_meanloss}')
    self.model.eval()

  def __call__(self): self.fit()

NUM_CLASSES = 80
NUM_BOXES = 2
H_image = 448
W_image = 448
H_grid = 7
W_grid = 7 
EPOCHS = 40

def main():
  transforms = v2.Compose([
    v2.Resize((H_image, W_image)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
  ])

  train_set = SetNoob("./", "coco8", train=True, transforms=transforms)
  val_set = SetNoob("./", "coco8", train=False, transforms=transforms)
  test_set = SetNoob("./", "coco8", train=True, transforms=transforms)

  # размер предсказания: 7 * 7 * (2 * 5 + 20) = 1470 чисел
  # каждая из 49 ячеек предсказывает: [cx, cy, w, h, conf1, cx, cy, w, h, conf2] + 20 классов
  out_fs = 7 * 7 * (NUM_BOXES * 5 + NUM_CLASSES)
  model = BACKBONEModel(in_chan=3, out_features=out_fs)
  trainer = ModelTrainer(model, (train_set, val_set, test_set)) ; trainer()

  print(f'общий показатель ошибки модели: {trainer.test()}')

if __name__ == "__main__":
  main()