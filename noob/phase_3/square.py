#!/home/client/Documents/fun/py/venv/bin/python3
import torch 


def main():
  image = torch.randn(6, 6) * 0 + 0
  x1,y1 = (2, 1)
  x2,y2 = (4, 4)  

  print(f'исходный\n{image}\n')
  #bounding box (рамки):

  #срезный:
  image[y1:y1+1, x1:x2+1] = 1
  image[y2:y2+1, x1:x2+1] = 1
  image[y1:y2+1, x1:x1+1] = 1
  image[y1:y2+1, x2:x2+1] = 1

  print(f'slice method:\n{image}')

  #цикличный:
  for y in range(image.size(0)):
    for x in range(image.size(1)):
      if (y == y1 or y == y2) and x >= x1 and x <= x2:
        image[y, x] = torch.randn(1,1) * 0 + 1
      if (x == x1 or x == x2) and y >= y1 and y <= y2:
        image[y, x] = torch.randn(1,1) * 0 + 1

  print(f'loop method: \n{image}')



if __name__ == "__main__":
  main()