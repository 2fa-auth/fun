#!/home/client/Documents/fun/venv/bin/python3
import torch
import cv2
import det


# вычисляет пересечение над объединением (Intersection over Union)
def IoU(box1, box2):
  x1_box1 = torch.min(box1[..., 2:3], box1[..., 4:5])
  y1_box1 = torch.min(box1[...,3:4], box1[..., 5:6]) 
  x2_box1 = torch.max(box1[..., 4:5], box1[..., 2:3])
  y2_box1 = torch.max(box1[..., 5:6], box1[...,3:4])
  x1_box2 = torch.min(box2[..., 2:3], box2[..., 4:5])
  y1_box2 = torch.min(box2[...,3:4], box2[..., 5:6]) 
  x2_box2 = torch.max(box2[..., 4:5], box2[..., 2:3])
  y2_box2 = torch.max(box2[..., 5:6], box2[..., 3:4])
    
  x1_box = torch.max(x1_box1, x1_box2)
  y1_box = torch.max(y1_box1, y1_box2)
  x2_box = torch.min(x2_box1, x2_box2)
  y2_box = torch.min(y2_box1, y2_box2)

  intersection_area = torch.clamp(x2_box - x1_box, 0) * torch.clamp(y2_box - y1_box, 0)

  wbox1 = x2_box1 - x1_box1
  hbox1 = y2_box1 - y1_box1
  wbox2 = x2_box2 - x1_box2 
  hbox2 = y2_box2 - y1_box2

  area_box1 = wbox1 * hbox1
  area_box2 = wbox2 * hbox2

  union_area = area_box1 + area_box2 - intersection_area
  return intersection_area / (union_area + 1e-6)

# нормализует координаты
def denorm_coords(bbox, w, h):
  return bbox * torch.tensor([w, h, w, h])

# денормализует координаты
def norm_coords(bbox, w, h):
  return bbox / torch.tensor([w, h, w, h])

# меняет стиль
def cxcywh_to_xyxy(coords):
  """
  переводит стиль координат ИЗ [x_center, y_center, w, h] В [x1, y1, x2, y2]:
    x1 = x_center - w / 2
    y1 = y_center - h / 2
    x2 = x_center + w / 2
    y2 = y_center + h / 2
  """
  coords = coords.clone()

  x1 = coords[..., 0] - coords[..., 2] / 2
  y1 = coords[..., 1] - coords[..., 3] / 2
  x2 = coords[..., 0] + coords[..., 2] / 2
  y2 = coords[..., 1] + coords[..., 3] / 2
  coords[..., 0] = x1
  coords[..., 1] = y1
  coords[..., 2] = x2
  coords[..., 3] = y2 
  return coords

# отображает изображение с рамками 
def detection_objects(x, y):
  labels = torch.tensor([])
  for ax1 in y:
    for ax2 in ax1:
      for el in ax2:
        if el[0] != 0: labels = torch.cat([labels, el.unsqueeze(0)], dim=0)
  
  for image, target in zip(x, labels):    
    image = image.to('cpu')
    target = target.to('cpu')

    image = image.permute(1, 2, 0).numpy()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    det_image = image.copy()
    target[:4] = cxcywh_to_xyxy(target[:4])
    target[:4] = denorm_coords(target[:4], det.W_image, det.H_image)

    for i in range(target.size(0)):
      x1 = int(target[0].item()) 
      y1 = int(target[1].item()) 
      x2 = int(target[2].item()) 
      y2 = int(target[3].item()) 

      cv2.rectangle(det_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
      cv2.imshow('picture from COCO8', det_image)
      cv2.waitKey(0)
      cv2.destroyAllWindows()

def sort_by_x1(target):
  indices = torch.argsort(target[..., 0])
  return torch.gather(input=target, dim=1, index=indices.unsqueeze(-1).expand_as(target))