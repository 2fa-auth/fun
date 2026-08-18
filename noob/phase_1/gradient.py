#!/home/client/Documents/fun/py/venv/bin/python3
import numpy as np

""" 
базовая вычислительная архитектура градиента с нуля:
  forward ==> barkward ==> optimization ==> again ...
"""

class Linear:
  def __init__(self, in_features, out_features):
    self.p = np.random.randn(in_features, out_features)
    self.b = np.random.randn(out_features)

  def forward(self, X):
    return X @ self.p + self.b

  def backward(self, grad_output, X):
    X = X.T # (4, 2) => (2, 4) (по правилу broadcasting)
    grad_p = X @ grad_output 
    grad_b = grad_output.sum(axis=0)

    return grad_p, grad_b

  def optimization(self, grad_p, grad_b, lr=0.01):
    self.p -= lr * grad_p
    self.b -= lr * grad_b 


def main():
  X = np.array([
    [2., 3.],
    [1., 4.],
    [3., 1.],
    [5., 2.]
  ])

  target = np.array([
    [10., 5., 2.],
    [9.,  4., 3.],
    [7.,  8., 1.],
    [13., 6., 4.]
  ])

  layer = Linear(2,3)

  for step in range(1000):

    pred = layer.forward(X)

    loss = ((pred - target)**2).mean() 
    grad_output = 2 * (pred - target) / len(X)
    grad_p, grad_b = layer.backward(grad_output, X)
    layer.optimization(grad_p, grad_b, 0.01)

    if step % 100 == 0:
      print(step, loss)

if __name__ == '__main__':
  main()