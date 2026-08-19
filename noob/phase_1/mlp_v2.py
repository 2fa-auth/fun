#!/home/client/Documents/fun/py/venv/bin/python3
import numpy as np

class Linear:
  def __init__(self, in_dim, out_dim):
    self.W = np.random.randn(in_dim, out_dim)    
    self.b = np.random.randn(out_dim)

  def forward(self, X):
    return X @ self.W + self.b

  def backward(self):
    pass
  
  def update(self):
    pass

class ReLU:
  def forward(self, X):
    return np.maximum(X, 0)

  def backward(self, grad_output, X): 
    return grad_output @ (X > 0)  

class MLP:
  def __init__(self):
    self.fc1 = Linear(2, 3)
    self.relu = ReLU()
    self.fc2 = Linear(3, 1)
    
  def forward(self, X):
    return X

  def backward(self):
    pass


def mseloss(pred, target):
    return ((pred - target)**2).mean()

def main():
  X = np.array([ # (8, 2)
    [0., 0.],
    [0., 1.],
    [1., 0.],
    [1., 1.],
    [2., 1.],
    [1., 2.],
    [2., 2.],
    [3., 1.]
  ])
  target = np.array([ # (8,)
    0.,
    3.,
    2.,
    5.,
    7.,
    4.,
    6.,
    7.
  ]) 
  # после обучения предсказания должны быть близки к:
  # [0, 3, 2, 5, 7, 4, 6, 7]

  model = MLP() 
  pred = model.forward(X)
  print(f'pred.shape = {pred.shape}')
  print(pred)












if __name__ == '__main__':
  main()