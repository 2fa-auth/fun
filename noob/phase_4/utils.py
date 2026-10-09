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
def denorm_coords(bbox, w, h): return bbox * torch.tensor([w, h, w, h])

# денормализует координаты
def norm_coords(bbox, w, h): return bbox / torch.tensor([w, h, w, h])

# из x_center, y_center, w, h -> x_min, y_min, x_max, y_max
def cxcywh_to_xyxy(coords):
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

# извлечение labels из сетки 
def extract_labels(grid, bbox=True): # `grid` должен иметь форму [X_grid, Y_grid, N_predisions]
  tlist=[]
  for ax1 in grid:
    for preds in ax1:
      if not preds[4]: continue # 4 index is confidence 
    else: tlist.append((preds[:4] if bbox else preds))
  return torch.stack([t for t in tlist if t.numel() > 0], 0)

def view_image(name_img, img):
  cv2.imshow(name_img, img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

# отображает изображение с рамками 
def detection_objects(x, y):
  img = x[0].to('cpu')
  bboxes = extract_labels(y[0]).to('cpu')
  bboxes = cxcywh_to_xyxy(bboxes)
  bboxes = denorm_coords(bboxes, det.W_image, det.H_image)
  img = cv2.cvtColor(img.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)
  
  for bbox in bboxes:  
    x_min = int(bbox[0].item()) 
    y_min = int(bbox[1].item()) 
    x_max = int(bbox[2].item()) 
    y_max = int(bbox[3].item()) 
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
    
  view_image('picture coco8', img)

# сортирует bbox по x1 
def sort_by_x1(target):
  indices = torch.argsort(target[..., 0])
  return torch.gather(input=target, dim=1, index=indices.unsqueeze(-1).expand_as(target))