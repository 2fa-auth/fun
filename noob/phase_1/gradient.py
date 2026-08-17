#!/home/client/Documents/fun/py/venv/bin/python3
import numpy as np

def train_step(x, target, p, learning_rate):
  # FORWARD
  y_pred = x @ p 

  loss = (y_pred - target) ** 2 # shape = (4,)
  # BACKWARD
  grad_p = (2/len(x)) * (y_pred - target) @ x # shape = (2)
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
