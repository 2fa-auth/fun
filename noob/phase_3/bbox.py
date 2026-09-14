#!/home/client/Documents/fun/py/venv/bin/python3
import torch 
import random

  
if __name__ == "__main__":
  h,w = 10,10
  
  image = torch.zeros(10, h,w)
  hbox = 3
  wbox = 3
  labels = 2

  target = ()

  for num_img in range(image.size(0)):
    for _ in range(2):
      label = random.randint(0,labels)
      present = 0 if label == 0 else 1
      y_rand = random.randint(1,h-2) * present
      x_rand = random.randint(1,w-2) * present
      image[num_img, y_rand, x_rand] = label

      x1 = (x_rand-1) * present
      y1 = (y_rand-1) * present
      x2 = (x_rand+1) * present
      y2 = (y_rand+1) * present

      target += ((label, x_rand, y_rand, x1,y1, x2,y2),)

      if present:
        image[num_img, y1:y1+1, x1:x2+1] = 10
        image[num_img, y2:y2+1, x1:x2+1] = 10
        image[num_img, y1:y2+1, x1:x1+1] = 10
        image[num_img, y1:y2+1, x2:x2+1] = 10


  print(image)
  for coord in target:
    print(coord)

"""
цель модели; научиться распознавть несколько объектов на 
изображении, где:
  число '0' - фон
  число '1' - первый объект
  число '2' - второй объект
  число '10' - рамка, которая находит объект

четыре закодированных правила по которым модель работает:
  1/ составляя рамку (bounding box) из числа '10'. 
  2/ на изображении могут быть всего два объекта (класса):
     '2' и '1'
  3/ они расположены в рандомном порядке и в 
     рандомном количестве.
  4/ координаты объектов генерируется в диапозоне всей 
     длины-2 и ширины-2 изображения
"""