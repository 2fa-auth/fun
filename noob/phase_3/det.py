#!/home/client/Documents/fun/py/venv/bin/python3
import torch
import torch.nn as nn 
import torch.utils.data as data

# detector

class SetClassBoxes(data.Dataset): 
  def __init__(self, images, target):
    super().__init__()
    self.images = images
    self.target = target
    self.len = int(images.size(0))

  def __len__(self): return self.len
  def __getitem__(self, index): return (self.images[index], self.target[index])

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
    x = x.view(x.size(0), -1) # hacks
    out = self.relu(self.bn1(self.fc1(x)))
    out = self.relu(self.bn2(self.fc2(out)))
    out = self.relu(self.bn3(self.fc3(out)))
    return self.fc4(out)

class BboxLoss_withMSE:
  def IoU(self, box1, box2):
    x1_box1, y1_box1 = torch.min(box1[..., 2:3], box1[..., 4:5]), torch.min(box1[...,3:4], box1[..., 5:6]) 
    x2_box1, y2_box1 = torch.max(box1[..., 4:5], box1[..., 2:3]), torch.max(box1[..., 5:6], box1[...,3:4])
    x1_box2, y1_box2 = torch.min(box2[..., 2:3], box2[..., 4:5]), torch.min(box2[...,3:4], box2[..., 5:6]) 
    x2_box2, y2_box2 = torch.max(box2[..., 4:5], box2[..., 2:3]), torch.max(box2[..., 5:6], box2[..., 3:4])
    x1_box, y1_box = torch.max(x1_box1, x1_box2), torch.max(y1_box1, y1_box2)
    x2_box, y2_box = torch.min(x2_box1, x2_box2), torch.min(y2_box1, y2_box2)

    intersection_area = torch.clamp(x2_box - x1_box, 0) * torch.clamp(y2_box - y1_box, 0)
    wbox1, hbox1 = x2_box1 - x1_box1, y2_box1 - y1_box1
    wbox2, hbox2 = x2_box2 - x1_box2, y2_box2 - y1_box2
    area_box1, area_box2 = wbox1 * hbox1, wbox2 * hbox2
    union_area = area_box1 + area_box2 - intersection_area
    return intersection_area / (union_area + 1e-6)

  def __call__(self, pred, y):
    present = y[..., 0].unsqueeze(-1)
    criterion = nn.MSELoss(reduction='none')

    present_loss = criterion(pred[..., 0:1], y[..., 0:1]).mean()
    iou_loss = (present * (1 - self.IoU(pred, y))).mean()
    class_loss = (present * criterion(pred[..., 1:2], y[..., 1:2])).mean()
    coords_loss = (present * criterion(pred[..., 2:], y[..., 2:])).mean()

    zero_coords = criterion(pred[..., 2:], y[..., 2:]).mean()

    print(pred[0])
    print(y[0])


    exit(0)
    return (iou_loss + coords_loss + class_loss + present_loss)

class Bbox:
  def __init__(self, images):
    self.batch_size = images.size(0)
    self.images = images
    self.h = images.size(-1)
    self.w = images.size(-2)
    self.hbox = 3
    self.wbox = 3

  def draw_boxes(self, labels, max_labels=2):
    import random

    target = torch.zeros(self.batch_size, max_labels, 6)
    for b in range(self.batch_size):
      for n_class in range(2):
        label = random.randint(0, labels)
        present = 0 if label == 0 else 1
        y_rand = random.randint(1,self.h-2) * present
        x_rand = random.randint(1,self.w-2) * present
        self.images[b, :, y_rand, x_rand] = label

        x1 = (x_rand-1) * present 
        y1 = (y_rand-1) * present
        x2 = (x_rand+1) * present
        y2 = (y_rand+1) * present

        target[b, n_class, 0] = present
        target[b, n_class, 1] = label
        target[b, n_class, 2] = x1
        target[b, n_class, 3] = y1
        target[b, n_class, 4] = x2
        target[b, n_class, 5] = y2

        if present:
          self.images[b, :, y1:y1+1, x1:x2+1] = 10
          self.images[b, :, y2:y2+1, x1:x2+1] = 10
          self.images[b, :, y1:y2+1, x1:x1+1] = 10
          self.images[b, :, y1:y2+1, x2:x2+1] = 10

    order = torch.argsort(target[:, :, 2], dim=1)
    target = target.gather(1, order.unsqueeze(-1).expand_as(target))
    """ # реализует target в зависящий от конкретного количества объектов на изображении
    [
      [[]],[[]],[[]] shape = batch_size, 3, 7
      [[]]       shape = batch_size, 1, 7
      ..batch_size
    ]


    target = []

    for img in range(self.batch_size):
      target.append([])
      amount_labels = random.randint(min_labels, max_labels)
      if not amount_labels: target[img] = [[el*0 for el in range(0,6)]]
      else:
        for _ in range(amount_labels):
          label = random.randint(1, labels)
          Y_rand = random.randint(1,self.h-2)
          X_rand = random.randint(1,self.w-2)
          self.images[img, :, Y_rand, X_rand] = label
          x1, y1 = X_rand-1, Y_rand-1
          x2, y2 = X_rand+1, Y_rand+1

          target[img].append([1, label, x1, y1, x2, y2])

          self.images[img, :, y1:y1+1, x1:x2+1] = 10
          self.images[img, :, y2:y2+1, x1:x2+1] = 10
          self.images[img, :, y1:y2+1, x1:x1+1] = 10
          self.images[img, :, y1:y2+1, x2:x2+1] = 10

    # print(self.images[0,0])
    # print(target[0])
    """
    
    order = torch.argsort(target[:, :, 2], dim=1)
    return self.images, target.gather(1, order.unsqueeze(-1).expand_as(target))      

def fetch_subset(images, target, test_percent, val_percent):
  main_size = images.size(0)
  train_percent = 100 - (test_percent + val_percent)

  train_size = int(main_size * train_percent / 100)
  val_size = int(main_size * val_percent / 100)

  return (images[:train_size], target[:train_size], # TRAIN
          images[train_size:train_size+val_size], target[train_size:train_size+val_size], # VALIDATION
          images[train_size+val_size:], target[train_size+val_size:]) # TEST

def main():
  size_val, size_test = (20, 10)
  h, w = (7, 7)  
  batch_size = 1000
  labels = 2

  images = torch.zeros(batch_size, 3, w,h)
  box = Bbox(images)
  images, target = box.draw_boxes(labels) # до 2 включительно
  print(images.shape, target.shape)

  X_train, Y_train, X_val, Y_val, X_test, Y_test = fetch_subset(images, target, size_val, size_test)

  train_loader = data.DataLoader(dataset = SetClassBoxes(X_train, Y_train), batch_size=64, shuffle=True)
  val_loader = data.DataLoader(dataset = SetClassBoxes(X_val, Y_val), batch_size=32, shuffle=True)
  test_loader = data.DataLoader(dataset = SetClassBoxes(X_test, Y_test), batch_size=32, shuffle=False)

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
    if _ep % 200 == 0: print(f'ep [{_ep}/{num_ep}] \t LOSS TRAIN {loss_train / t_cnt} \t LOSS VAL {loss_val / v_cnt}')
      
  print("\nТЕСТ") 
  model.eval()
  losses, l_cnt = 0, 0
  patience = 4
  with torch.no_grad():
    for x,y in test_loader:
      pred=model(x).reshape(-1, 2, 6)
      order = torch.argsort(pred[:,:,2], dim=1)
      pred = pred.gather(1, order.unsqueeze(-1).expand_as(pred))
      loss=criterion(pred, y)

      if patience: print(f'predision:\n{torch.round(pred[0])}\ny:\n{torch.round(y[0])}'); patience -= 1      
      losses += loss.item()
      l_cnt +=1
    print(f"средняя ошибка модели после теста: {losses / l_cnt}")

  torch.save(model.state_dict(), 'model_params.pth.tar')

if __name__ == "__main__":
  main()