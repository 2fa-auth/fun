#!/home/client/Documents/fun/venv/bin/python3
import torch
import torch.utils.data as data
import torchvision.transforms.v2 as v2


import os
from PIL import Image 

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
          ts.pop() if not ts[-1] else None
          for t in ts: targets.append([float(val) for val in t.split(' ')])
    
    for t in targets:
      x1 = t[-4] - t[-2] / 2 
      y1 = t[-3] - t[-1] / 2
      x2 = t[-4] + t[-2] / 2
      y2 = t[-3] + t[-1] / 2
      t[-4] = x1
      t[-3] = y1
      t[-2] = x2
      t[-1] = y2

    image = Image.open(os.path.join(self.images, image)).convert('RGB')
    if self.transforms:
      image, targets = self.transforms(image, targets)
    return (image, targets)

def def_call(batch):
  batch = [(image, torch.tensor(target)) for image, target in batch]
  images = [image for image, _ in batch]
  targets = [target for _, target in batch]

  return images, targets



if __name__ == "__main__":
  transforms = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
  ])

  train_set = SetNoob("./", "coco8", train=True, transforms=transforms)
  val_set = SetNoob("./", "coco8", train=False, transforms=transforms)

  train_loader = data.DataLoader(dataset=train_set, batch_size=1, collate_fn=def_call, shuffle=False)
  val_loader = data.DataLoader(dataset=val_set, batch_size=1, shuffle=False)

  for image, bbox in train_loader:
    print(image)
    print(bbox)
    break

  """ парсинг файлов аннотаций//изображний//категорий в формате JSON: 
  with open('images.json', 'r', encoding='utf-8') as f: imgs = json.load(f)['images']
  with open('annotations.json', 'r', encoding='utf-8') as f: annots = json.load(f)['annotations']
  with open('categories.json', 'r', encoding='utf-8') as f: cats = json.load(f)['categories']

  for img in imgs:  
    for annot in annots:
      if annot['image_id'] == img['id']:
        bbox = annot['bbox']
        print(f'BBOX of image {img['file_name']}: xywh: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}] \t',end=' ')
        print(f'x1y1x2y2: [{bbox[0]}, {bbox[1]}, {bbox[0]+bbox[2]}, {bbox[1]+bbox[3]}] \t',end=' ')
        for cat in cats: 
          if cat['id'] == annot['category_id']: print(f'cat: {cat['name']}')                                                       
  """  

