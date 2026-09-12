#!/home/client/Documents/fun/py/venv/bin/python3
import torch 
import random

  
if __name__ == "__main__":
  h,w = 10,10
  
  image = torch.zeros(10, h,w)
  hbox = 3
  wbox = 3
  labels = 2
  for num_img in range(image.size(0)):
    nums_label = random.randint(0,2)
    for _ in range(nums_label):
      label = random.randint(1,labels)
      y_rand = random.randint(1,h-2)
      x_rand = random.randint(1,w-2)
      image[num_img, y_rand, x_rand] = label

      x1 = x_rand-1
      y1 = y_rand-1
      x2 = x_rand+1
      y2 = y_rand+1

      image[num_img, y1:y1+1, x1:x2+1] = 10
      image[num_img, y2:y2+1, x1:x2+1] = 10
      image[num_img, y1:y2+1, x1:x1+1] = 10
      image[num_img, y1:y2+1, x2:x2+1] = 10


  print(image)

