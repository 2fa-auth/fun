import torch

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