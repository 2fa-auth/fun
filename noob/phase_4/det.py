#!/home/client/Documents/fun/py/venv/bin/python3
import torch.utils.data as data
import os

class SetNoob(data.Dataset):
  def __init__(self, pathto_dset, namedset, train=True):
    self.images = os.path.join(pathto_dset, namedset, "images", "train" if train else "val")
    self.labels = os.path.join(pathto_dset, namedset, "labels", "train" if train else "val")
    self.imgs_list = os.listdir(self.images)
    self.lbs_list = os.listdir(self.labels)

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

    image = os.path.join(self.images, image)
    return (image, targets)

if __name__ == "__main__":
  train_set = SetNoob("./", "coco8", train=True)
  val_set = SetNoob("./", "coco8", train=False)

  train_loader = data.DataLoader(dataset=train_set, batch_size=1, shuffle=False)
  val_loader = data.DataLoader(dataset=val_set, batch_size=1, shuffle=False)

  for image, label in train_loader:
    print(image)
    print(label)
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

