#!/home/client/Documents/fun/py/venv/bin/python3
import numpy as np

""" forward ==> barkward ==> optimization ==> again ..."""

class Linear:
  def __init__(self, in_features):
    self.p = np.random.randn(in_features)
    self.b = np.random.randn()

  def forward(self, X):
    return X @ self.p + self.b

  def backward(self, grad_output, X):
    grad_p = grad_output @ X
    grad_b = grad_output.sum()

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
    10.,
    9.,
    7.,
    13.
  ])

  layer = Linear(2)

  for step in range(1000):

    y_pred = layer.forward(X)

    loss = ((y_pred - target)**2).mean() 
    grad_output = 2 * (y_pred - target) / len(X)
    grad_p, grad_b = layer.backward(grad_output, X)
    layer.update(grad_p, grad_b, 0.01)

    if step % 100 == 0:
      print(step, loss)

if __name__ == '__main__':
  main()