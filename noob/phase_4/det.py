#!/home/client/Documents/fun/venv/bin/python3
import os
from copy import deepcopy
from PIL import Image 
from time import sleep

import torch
from torch.optim import Adam
from torchvision.tv_tensors import BoundingBoxes
import torch.utils.data as data
import torchvision.transforms.v2 as v2
import torch.nn as nn

import cv2
import numpy as np


NUM_CLASSES = 80
NUM_PREDICTIONS = 10 
H = 224 
W = 224 
CHANS = 3 
EPOCHS = 40

def unscale_coords(w, h, bbox):
  return bbox[:, :4] * torch.tensor([w,h,w,h], device=bbox.device)

def scale_coords(w, h, bbox): 
  return (bbox.reshape(-1, 2) / torch.tensor([w, h], device=bbox.device)).reshape(-1, 4)

def detection_objects(x, y):
  for image, target in zip(x, y):    
    image = image.to('cpu')
    target = target.to('cpu')
    
    image = image.permute(1, 2, 0).numpy()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    target = unscale_coords(W, H, target)

    det_image = image.copy()

    for t in target:
      x1 = int(t[0].item()) #pt1
      y1 = int(t[1].item()) #pt1
      x2 = int(t[2].item()) #pt2
      y2 = int(t[3].item()) #pt2
      
      cv2.rectangle(det_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
      cv2.imshow('picture from COCO8', det_image)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
class SetNoob(data.Dataset):
  def __init__(self, pathto_dset, namedset, train=True, transforms=None):
    self.images = os.path.join(pathto_dset, namedset, "images", "train" if train else "val")
    self.labels = os.path.join(pathto_dset, namedset, "labels", "train" if train else "val")
    self.imgs_list = os.listdir(self.images)
    self.lbs_list = os.listdir(self.labels)
    self.transforms = transforms

  def __len__(self): return len(self.imgs_list)    

  def __getitem__(self, index):
    image_target = self.imgs_list[index]
    target = torch.zeros((NUM_PREDICTIONS, NUM_CLASSES+5))

    num_targs = 0
    for label in self.lbs_list: 
      if label.split('.')[0] == image_target.split('.')[0]:
        with open(os.path.join(self.labels, label), "r") as f:
          ts = f.read().split('\n')
          for t in ts: 
            ar = []
            for val in t.split(' '): 
              if val != '': ar.append(float(val))
            if ar: 
              ar_target = deepcopy(ar)
              for i in range(len(ar)-1): ar_target[i] = ar[i+1]
              ar_target[-1] = 1 if ar[0] > 0 else 0

              for i in range(NUM_CLASSES): ar_target.append( 1 if i == ar[0] else 0 )
              target[num_targs] = torch.tensor(ar_target).unsqueeze(0) ; num_targs += 1              

    image_target = Image.open(os.path.join(self.images, image_target)).convert('RGB')
    w, h = image_target.size

    bbox = target[:, :4]
    x1, y1 = bbox[:, 0] - bbox[:, 2] / 2, bbox[:, 1] - bbox[:, 3] / 2
    x2, y2 = bbox[:, 0] + bbox[:, 2] / 2, bbox[:, 1] + bbox[:, 3] / 2
    bbox[:, 0], bbox[:, 1] = x1, y1
    bbox[:, 2], bbox[:, 3] = x2, y2

    if self.transforms:
      bbox = BoundingBoxes(data=unscale_coords(w, h, bbox), format='XYXY', canvas_size=(h, w))
      image_target, target[:, :4] = self.transforms(image_target, bbox)
      target[:, :4] = scale_coords(w, h, bbox) # нормализация координат

    return (image_target, target)


class NoobDetectionModel(nn.Module): 
  def __init__(self, in_features, out_features):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(in_features,128),  nn.BatchNorm1d(128), nn.ReLU(),
      nn.Linear(128, 512),         nn.BatchNorm1d(512), nn.ReLU(),
      nn.Linear(512, 256),         nn.BatchNorm1d(256), nn.ReLU(),
      nn.Linear(256, out_features)
    )

  def forward(self, x): return self.net(x.view(x.size(0), -1))

class ModelTrainer: 
  def __init__(self, model, sets, lr=0.001):
    self.dev = 'cuda' if torch.cuda.is_available else 'cpu'
    self.model = model.to(self.dev)
    train_set, val_set, test_set = sets
    self.train_loader = data.DataLoader(dataset=train_set, batch_size=2, shuffle=True)
    self.val_loader = data.DataLoader(dataset=val_set, batch_size=2, shuffle=False)  
    self.test_loader = data.DataLoader(dataset=test_set, batch_size=2, shuffle=False)
    self.criterion = nn.MSELoss()
    self.optimizer = Adam(params=self.model.parameters(), lr=lr)
  
  def train(self):
    losses = 0
    for x, y in self.train_loader:
      x, y = x.to(self.dev), y.to(self.dev)
      pred = self.model(x).reshape(-1, 10, 85)
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
        losses += self.criterion(pred, y).item()

    return losses / len(self.val_loader)
  
  def test(self):
    losses = 0
    last_pred = 0
    with torch.no_grad():
      for x,y in self.test_loader:
        x, y = x.to(self.dev), y.to(self.dev)
        pred=self.model(x).reshape(-1, 10, 85)
        last_pred = pred

    detection_objects(x, last_pred)
    losses += self.criterion(pred, y).item()

    return losses / len(self.test_loader)

  def fit(self):
    # расскоментируй и убери нормализацию координат (помечена комментом) если хочешь проверить изображения на bbox
    # for x, y in self.train_loader: 
      # detection_objects(x, y)
      # exit()

    for _ep in range(EPOCHS):
      self.model.train()
      train_meanloss = self.train()
      self.model.eval()
      val_meanloss = self.valid()
      if _ep % 10 == 0: print(f'[{_ep}/{EPOCHS}]\tLOSS TRAIN {train_meanloss}\tLOSS VAL {val_meanloss}')
    self.model.eval()

  def __call__(self): self.fit()

def main():
  transforms = v2.Compose([
    v2.Resize((H, W)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
  ])

  train_set = SetNoob("./", "coco8", train=False, transforms=transforms)
  val_set = SetNoob("./", "coco8", train=False, transforms=transforms)
  test_set = SetNoob("./", "coco8", train=False, transforms=transforms)

  model = NoobDetectionModel(in_features=H * W * CHANS, out_features=NUM_PREDICTIONS * (5 + NUM_CLASSES))
  trainer = ModelTrainer(model, (train_set, val_set, test_set))
  trainer()

  print(f'общий показатель ошибки модели: {trainer.test()}')

if __name__ == "__main__":
  main()