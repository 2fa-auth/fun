#!/home/client/Documents/fun/py/venv/bin/python3

# разминка

if __name__ == "__main__":
  import random
  import torch

  h,w = 10,10
  batch_size = 1
  
  image = torch.zeros(batch_size, h,w)
  hbox = 3
  wbox = 3
  labels = 2

  target = ()

  num_labels = random.randint(0, 2)
  if not num_labels: target += ((0, 0, 0, 0, 0, 0, 0),)
  else:
    for _ in range(num_labels): # спавн рандомного количества объектов 
      label = random.randint(1,labels)
      y_rand = random.randint(1,h-2)
      x_rand = random.randint(1,w-2)
      image[:, y_rand, x_rand] = label

      x1 = (x_rand-1)
      y1 = (y_rand-1)
      x2 = (x_rand+1)
      y2 = (y_rand+1)

      target += ((label, x_rand, y_rand, x1,y1, x2,y2),)
      image[:, y1:y1+1, x1:x2+1] = 10
      image[:, y2:y2+1, x1:x2+1] = 10
      image[:, y1:y2+1, x1:x1+1] = 10
      image[:, y1:y2+1, x2:x2+1] = 10

  print(image)
  for coord in target: print(coord)