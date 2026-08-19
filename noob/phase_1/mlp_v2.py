#!/home/client/Documents/fun/py/venv/bin/python3
import numpy as np


class MSELoss:
  def __call__(self, pred, target):
    return ((pred - target)**2).mean()


  
class Linear:
  def __init__(self, in_dim=1, out_dim=1):
    self.W = np.random.randn(in_dim, out_dim)    
    self.b = np.random.randn(out_dim)
    self.grad_W = np.array([])
    self.grad_b = np.array([])

  def forward(self, X):
    return X @ self.W + self.b

  def backward(self, grad_output, X):
    print(f'grad_output = {grad_output.shape}')
    print(f'X = {X.shape}')
    print()
    grad_W = X @ grad_output
    grad_b = grad_output.sum()

    self.grad_W = grad_W
    self.grad_b = grad_b

    return grad_output
     
class OptimMLP(Linear):
  def __init__(self, lr):
    super().__init__()
    self.lr = lr

  def optimizer(self):
    print(f'OLD self.W: \n{self.W}')
    print(f'изменится НА \n{self.grad_W}')
    print(f'OLD self.b = \n{self.b}')
    print(f'изменится НА \n{self.grad_b}')

    self.W -= self.lr * self.grad_W
    self.b -= self.lr * self.grad_b

    print(f'NEW self.W = \n{self.W}')
    print(f'NEW self.b = \n{self.b}')

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
    X = self.relu.forward(self.fc1.forward(X))
    X = self.fc2.forward(X)
    return X # = > prediction of model

  # backpropogation
  def backward(self, grad_loss, X):
    grad_output = self.relu.backward(self.fc2.backward(grad_loss, X), X)
    grad_output = self.fc1.backward(grad_output, X)


def main():
  X = np.array([
    [0., 0.],
    [0., 1.],
    [1., 0.],
    [1., 1.],
    [2., 1.],
    [1., 2.],
    [2., 2.],
    [3., 1.]
  ])
  print(f'X.shape = {X.shape}') # (8, 2)

  target = np.array([
    0.,
    3.,
    2.,
    5.,
    7.,
    4.,
    6.,
    7.
  ])
  print(f'target.shape = {target.shape}') # (8,)
  print()
  # после обучения предсказания должны быть близки к:
  # [0, 3, 2, 5, 7, 4, 6, 7]

  model = MLP()

  criterion = MSELoss()
  optim = OptimMLP(0.01)
 
  pred = model.forward(X)
  
  # получи итоговое лин. преобразование
  loss = criterion(pred, target)
  # среднюю ошибку
  grad_loss = 2 * (pred - target) / len(X)
  # градиент для всех target[i] для каждого X[j] (grad_loss = (8, 8))
  
  model.backward(grad_loss, X)
  # посчитай для всех градиент (а не только для dL/dpred)

  optim.optimizer()
  # обнови значения

  print(f'средняя ошибка = {loss}')
  # выведи среднюю ошибку

if __name__ == '__main__':
  main()