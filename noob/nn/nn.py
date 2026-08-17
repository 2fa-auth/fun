#!/home/client/Documents/fun/py/venv/bin/python3
import numpy as np

X = np.array([
  [1, 2, 3],
  [4, 5, 6],
  [7, 8, 9],
  [10, 11, 12]
], dtype=float)

W = np.array([
  [1, 2],
  [3, 4],
  [5, 6]
], dtype=float)

b = np.array([10, 20], dtype=float)

Y = X @ W + b



print(Y)

