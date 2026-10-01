#!/home/client/Documents/fun/venv/bin/python3
import torch.utils.data as data
import torchvision.transforms.v2 as v2
from torchvision.tv_tensors import BoundingBoxes
import torch.nn as nn
from torch.optim import Adam
import torch

import os
from PIL import Image 
import cv2
import numpy as np
from copy import deepcopy
from time import sleep

# output: 
#   x = [batch, 3, h, w]
#   y = [batch, NUM_PREDICIONS, NUM_CLASSES]
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
              for i in range(len(ar)-1): 
                ar_target[i] = ar[i+1]
              ar_target[-1] = 1 if ar[0] > 0 else 0

              for i in range(NUM_CLASSES): ar_target.append( 1 if i == ar[0] else 0 )
              target[num_targs] = torch.tensor(ar_target).unsqueeze(0) ; num_targs += 1              

    image_target = os.path.join(self.images, image_target)
    image_target = Image.open(image_target).convert('RGB')
    
    h, w = reversed(image_target.size) #for comfortable

    if self.transforms: 
      bbox = target[:, :4].clone().detach() #bbox.shape = [10, 4], target.shape = [10, 85]

      x1 = bbox[:, 0] - bbox[:, 2] / 2 
      y1 = bbox[:, 1] - bbox[:, 3] / 2
      x2 = bbox[:, 0] + bbox[:, 2] / 2
      y2 = bbox[:, 1] + bbox[:, 3] / 2
      bbox[:, 0] = x1 * w
      bbox[:, 1] = y1 * h
      bbox[:, 2] = x2 * w
      bbox[:, 3] = y2 * h

      bbox = BoundingBoxes(data=bbox, format='XYXY', canvas_size=(h, w))
      image_target, target[:, :4] = self.transforms(image_target, bbox)

    return (image_target, target)


""" 
# ручной collate_fn 
def def_call(batch):
  batch = [(image, target) for image, target in batch]
  images = [image for image, _ in batch]
  targets = [target for _, target in batch]
  return images, targets
"""

def view_image(image, name_widow):
  cv2.imshow(name_widow, image)

# нарисовать прямоугольник в принимаемых координатах
def detection_objects(x, y):
  for image, target in zip(x, y):    
    image = image.permute(1, 2, 0).numpy() 

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    det_image = image.copy()

    for t in target:
      x1 = int(t[0].item()) #pt1
      y1 = int(t[1].item()) #pt1
      x2 = int(t[2].item()) #pt2
      y2 = int(t[3].item()) #pt2
      cv2.rectangle(det_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
      view_image(det_image, 'picture from COCO8')
    cv2.waitKey(0)
    cv2.destroyAllWindows()

class NootDetectionModel(nn.Module): 
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
    # x = x.view(x.size(0), -1) # hacks
    out = self.relu(self.bn1(self.fc1(x)))
    out = self.relu(self.bn2(self.fc2(out)))
    out = self.relu(self.bn3(self.fc3(out)))
    return self.fc4(out)

NUM_CLASSES = 30
NUM_PREDICTIONS = 10

def main():
  H, W = (224, 224)
  CHANS = 3
  EPOCHS = 100

  transforms = v2.Compose([
    v2.Resize((H, W)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
  ])

  train_set = SetNoob("./", "coco8", train=True, transforms=transforms)
  # val_set = SetNoob("./", "coco8", train=False, transforms=transforms)

  train_loader = data.DataLoader(dataset=train_set, batch_size=2, shuffle=False)
  # val_loader = data.DataLoader(dataset=val_set, batch_size=1, shuffle=False)  
  
  for x, y in train_loader: # <== вывести изображения с рамками
    detection_objects(x, y)
    break
  
  exit()
  in_dims_model = H * W * CHANS
  out_dims_model = NUM_PREDICTIONS * (5 + NUM_CLASSES) #где 5 => [x1, y1, x2, y2, objectness]
  model = NootDetectionModel(in_features=in_dims_model, out_features=out_dims_model)
  criterion = nn.MSELoss()
  optimizer = Adam(params=model.parameters(), lr=0.001)
  
  print("ОБУЧЕНИЕ && ВАЛИДАЦИЯ\n")
  for _ep in range(EPOCHS):
    train_avg_loss = 0
    train_cnt_loss = 0
    val_avg_loss = 0
    val_cnt_loss = 0

    model.train()
    for images, targets in train_loader:
      print(images)
      exit()
      # pred = model(images).reshape(-1, 10, 85)
      pred = model(images)
      print(pred.shape)
      exit()

      loss = criterion(pred, targets)
      print(loss.shape)
      exit()

      train_avg_loss += loss.item()
      train_cnt_loss += 1
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    with torch.no_grad():
      model.eval()
      for images, targets in val_loader:
        print(images)
        exit()
        pred = model(images)

        val_avg_loss += criterion(pred, targets)
        val_cnt_loss += 1
    
    if _ep % 10 == 0: 
      print(f'ep [{_ep}/{EPOCHS}] \t LOSS TRAIN {train_avg_loss / train_cnt_loss} \t LOSS VAL {val_avg_loss / val_cnt_loss}')

  print("\nТЕСТ")
  model.eval()


  print(train_set[0])


if __name__ == "__main__":
  main()