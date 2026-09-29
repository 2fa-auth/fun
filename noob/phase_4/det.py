#!/home/client/Documents/fun/venv/bin/python3
import torch
import torch.utils.data as data
import torchvision.transforms.v2 as v2
from torchvision.tv_tensors import BoundingBoxes

import os
from PIL import Image 
import cv2
import numpy as np

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
    image = self.imgs_list[index]
    targets = []
    
    for label in self.lbs_list:
      if label.split('.')[0] == image.split('.')[0]:
        with open(os.path.join(self.labels, label), "r") as f:
          ts = f.read().split('\n')
          for t in ts: 
            ar = []
            for val in t.split(' '):
              if val != '': ar.append(float(val))
            if ar: targets.append(ar)
    targets = torch.tensor(targets)

    image = Image.open(os.path.join(self.images, image)).convert('RGB')
    w, h = image.size
    if self.transforms: 
      bbox = targets[:, 1:].clone().detach()

      x1 = bbox[:, 0] - bbox[:, 2] / 2 
      y1 = bbox[:, 1] - bbox[:, 3] / 2
      x2 = bbox[:, 0] + bbox[:, 2] / 2
      y2 = bbox[:, 1] + bbox[:, 3] / 2
      bbox[:, 0] = x1 * w 
      bbox[:, 1] = y1 * h
      bbox[:, 2] = x2 * w
      bbox[:, 3] = y2 * h

      bbox = BoundingBoxes(data=bbox, format='XYXY', canvas_size=(h, w))
      image, bbox = self.transforms(image, bbox)

      targets[:, 1:] = bbox

    return (image, targets)

def def_call(batch):
  batch = [(image, target) for image, target in batch]
  images = [image for image, _ in batch]
  targets = [target for _, target in batch]
  return images, targets

def view_image(image, name_widow):
  cv2.imshow(name_widow, image)
  cv2.waitKey(0) 
  cv2.destroyAllWindows()

def detection_objects(images, targets):
  for image, target in zip(images, targets):    
    image = image.permute(1, 2, 0).numpy() 
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    det_image = image.copy()

    for i in range(len(target)):
      x1 = int(target[i, 1].item()) #pt1
      y1 = int(target[i, 2].item()) #pt1
      x2 = int(target[i, 3].item()) #pt2
      y2 = int(target[i, 4].item()) #pt2
      cv2.rectangle(det_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
    view_image(det_image, 'rect') # показывает всего ОДНУ картинку из 'SetNoob'

def main():
  transforms = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
  ])

  train_set = SetNoob("./", "coco8", train=True, transforms=transforms)
  val_set = SetNoob("./", "coco8", train=False, transforms=transforms)

  train_loader = data.DataLoader(dataset=train_set, batch_size=1, collate_fn=def_call, shuffle=False)
  val_loader = data.DataLoader(dataset=val_set, batch_size=1, shuffle=False)  

  for images, targets in train_loader:
    detection_objects(images, targets)
  

if __name__ == "__main__":
  main()