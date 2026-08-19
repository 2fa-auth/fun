#!/home/client/Documents/fun/py/venv/bin/python3
import random

""" 
линейное преобразование 

(вспомнить python :)
"""

class Linear:
  def __init__(self, in_features: int, out_features: int):
    self.in_features = in_features
    self.out_features = out_features
    self.b = [random.randint(1, 3) for _ in range(out_features)]
    self.W = [ [random.randint(1, 6) for _ in range(out_features)] for _ in range(in_features)]
    
  def __call__(self, x: list):
    if self.in_features != len(x):
      raise ValueError(f"in features {self.in_featuresno} equal len of x {len(x)}")
    y = []    
    for index1 in range(self.out_features):
      list_sum = []
      for index2 in range(self.in_features):
        list_sum.append(x[index2] * self.W[index2][index1])
      y.append(sum(list_sum) + self.b[index1])                               
    return y      

if __name__ == '__main__':
  layer = Linear(3, 2)
  x = [1, 2, 3, 3]

  try:
    y = layer(x)
  except ValueError as e:
    print(f"Fatal error: {e}")
  else:
    print(y)
