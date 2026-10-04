import torch
import cv2
import det


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
    target = unscale_coords(det.W, det.H, target)

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

def sort_by_x1(target):
  indices = torch.argsort(target[..., 0])
  return torch.gather(input=target, dim=1, index=indices.unsqueeze(-1).expand_as(target))