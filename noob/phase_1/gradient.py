#!/home/client/Documents/fun/py/venv/bin/python3
import numpy as np


def train_step(X, target, p, learning_rate):
  """
  функция вычисляет:
    loss: L = (y-t)**2
      производная: dL/dy = 2(y-t)
    y: y = px
      производная: dy/dp = x
    backprop:
      dL/dp = dL/dy * dy/dp
    optimization:
      p_new = p - lr * dL/dp 
  """

  # FORWARD
  y_pred = X @ p 
  loss = (y_pred - target) ** 2 # shape = (4,)
  # BACKWARD
  grad_p = (2/len(X)) * (y_pred - target) @ X # shape = (2)
  # UPDATE
  p -= learning_rate * grad_p # shape = (2,)

  return p, loss

def main():
  X = np.array([
    [2., 3.],
    [1., 4.],
    [3., 1.],
    [5., 2.]
  ])
  p = np.array([1., 2.])

  target = np.array([
    10.,
    9.,
    7.,
    13.
  ])

  for step in range(100):
    p, loss = train_step(
    X, 
    target,
    p,
    learning_rate=0.05
  )
  print(f'step = {step}')
  print(f'p = {p}')
  print(f'loss={loss}')

if __name__ == '__main__':
  main()