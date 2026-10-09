#!/home/client/Documents/fun/venv/bin/python3
import os
from PIL import Image 
import torch
from torch.optim import Adam
from torchvision.tv_tensors import BoundingBoxes
import torch.utils.data as data
import torchvision.transforms.v2 as v2
import torch.nn as nn
import utils
from model import BACKBONEModel
from loss import BboxLoss_withMSE

NUM_CLASSES = 80 
NUM_BOXES = 2 
H_image = 448 
W_image = 448 
H_grid = 7 
W_grid = 7 
EPOCHS = 40

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
    image=self.imgs_list[index]
    labels=[]
    # парсинг label*.txt файла
    for label in self.lbs_list:
      if label.split('.')[0] == image.split('.')[0]:
        with open(os.path.join(self.labels, label), "r") as f:
          ts = f.read().split('\n') 
          for t in ts: 
            ar = []
            for val in t.split(' '): 
              if val != '': ar.append(float(val))
            if ar: labels.append(ar)
    labels = torch.tensor(labels)
    image = Image.open(os.path.join(self.images,image)).convert('RGB')
    w, h = image.size
    if self.transforms:
      bbox = BoundingBoxes(data=utils.denorm_coords(labels[:, 1:], w, h), format='CXCYWH', canvas_size=(h,w))
      image, labels[:, 1:] = self.transforms(image, bbox)
      labels[:, 1:] = utils.norm_coords(bbox, w, h)
    # создание сетки 7x7 
    target=torch.zeros(7, 7, NUM_BOXES * 5 + NUM_CLASSES)
    class_ids = labels[:, 0].to(torch.int32)
    labels=torch.cat([
      labels,
      torch.zeros_like(labels),
      torch.ones(labels.size(0), 1),
      torch.zeros(labels.size(0),
      NUM_CLASSES)], dim=1)[:, 1:]
    
    for i, ax in enumerate(labels): ax[10+class_ids[i]] = 1
    # запись в оси X и Y сетки 
    cell_x = torch.floor(labels[:, 0] * W_grid).to(torch.int32) 
    cell_y = torch.floor(labels[:, 1] * H_grid).to(torch.int32)
    target[cell_x, cell_y] = labels
    return image, target




class ModelTrainer:
  def __init__(self, model, train_set, val_set, test_set, lr=0.001):
    self.dev = 'cuda' if torch.cuda.is_available else 'cpu'
    self.model = model.to(self.dev)
    self.train_loader = data.DataLoader(dataset=train_set, batch_size=2, shuffle=True)
    self.val_loader = data.DataLoader(dataset=val_set, batch_size=2, shuffle=False)  
    self.test_loader = data.DataLoader(dataset=test_set, batch_size=2, shuffle=False)
    self.criterion = loss.BboxLoss_withMSE()
    self.optimizer = Adam(params=self.model.parameters(), lr=lr)
  
  def train(self):
    losses = 0
    for x, y in self.train_loader:
      x, y = x.to(self.dev), y.to(self.dev)
      pred = self.model(x)
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
        pred = self.model(x)
        losses += self.criterion(pred, y).item()
    return losses / len(self.val_loader)
  
  def test(self):
    losses = 0
    last_pred = 0
    with torch.no_grad():
      for x,y in self.test_loader:
        x, y = x.to(self.dev), y.to(self.dev)
        pred = self.model(x)
        last_pred = pred
        last_y = y
    losses += self.criterion(pred, y).item()
    return losses / len(self.test_loader)

  def fit(self):
    for x, y in self.train_loader: # отобразить изображение с рамками
      print(utils.extract_labels(y[0], bbox=False))
      # utils.detection_objects(x, y)
      exit()
    for _ep in range(EPOCHS):
      self.model.train()
      train_meanloss = self.train()
      self.model.eval()
      val_meanloss = self.valid()
      if _ep % 10 == 0: print(f'[{_ep}/{EPOCHS}]\tLOSS TRAIN {train_meanloss}\tLOSS VAL {val_meanloss}')
    self.model.eval()

  def __call__(self): 
    self.fit()


def main():
  transforms = v2.Compose([
    v2.Resize((H_image, W_image)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
  ])
  trainer = ModelTrainer(
    model=BACKBONEModel(3, 7*7*(NUM_BOXES*5+NUM_CLASSES)), 
    train_set=SetNoob("./", "coco8",train=True, transforms=transforms),
    val_set=SetNoob("./", "coco8",train=False, transforms=transforms),
    test_set=SetNoob("./", "coco8",train=True, transforms=transforms)); trainer()

  print(f'общий показатель ошибки модели: {trainer.test()}')

if __name__ == "__main__":
  main()