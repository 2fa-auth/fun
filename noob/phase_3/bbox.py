#!/home/client/Documents/fun/py/venv/bin/python3
import torch 
import random

  
if __name__ == "__main__":
  h,w = 6,6
  
  image = torch.randn(h,w) * 0 + 0

  hbox = h-1
  wbox = w-1


  x1 = random.randint(0,wbox-2)
  y1 = random.randint(0,hbox-2)

  x2 = random.randint(x1+2,wbox) 
  y2 = random.randint(y1+2,hbox)

  x1 = min(x1,x2)
  y1 = min(y1,y2)
  x2 = max(x1,x2)
  y2 = max(y1,y2)


  print(x1,y1,x2,y2)
  print(f'исходный\n{image}\n')

  image[y1:y1+1, x1:x2+1] = 1
  image[y2:y2+1, x1:x2+1] = 1
  image[y1:y2+1, x1:x1+1] = 1
  image[y1:y2+1, x2:x2+1] = 1


  print(f'slice method:\n{image}')

