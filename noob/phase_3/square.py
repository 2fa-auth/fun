#!/home/client/Documents/fun/py/venv/bin/python3
import torch 


def main():
  x1,y1 = (2, 1)
  x2,y2 = (4, 4)
  image = torch.randn(6, 6) * 0 + 0

  #вариант 1
  image[y1, x1] = 1.
  image[y2, x2] = 1.

  for y in range(image.size(0)):
    for x in range(image.size(1)):
      if x <= x2 and x >= x1 and y >= y1 and y <= y2:
        image[y, x] = torch.randn(1,1)

  print(image)
  #вариант 2
  h = (y2-y1)+1
  w = (x2-x1)+1
  image[y1:y2+1, x1:x2+1] = torch.randn(h,w)
  print(image)




if __name__ == "__main__":
  main()