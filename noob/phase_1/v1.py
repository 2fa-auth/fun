#!/home/client/Documents/fun/py/venv/bin/python3

"""
реализация модели MLP (перцептрон)
"""
import numpy as np

class ReLU:
  def __call__(self, X):
    return np.maximum(X, 0)

class Linear:
  def __init__(self, in_features, out_features):
    self.in_features = in_features
    self.out_features = out_features
    self.W = np.random.randn(self.in_features, self.out_features)
    self.bias = np.random.randn(self.out_features)*0.01

  def __call__(self, X):
    return X @ self.W + self.bias

class MLP:
  def __init__(self):
    self.fc1 = Linear(3, 4)
    self.fc2 = Linear(4, 2)
    self.relu = ReLU()

  def __call__(self, X): 
    X = self.relu(self.fc1(X))  
    X = self.fc2(X) 
    return X
    
def main():
  X = np.random.randn(8,3)

  model = MLP()
  Y = model(X)

  print(X.shape) # (8, 3)
  print(Y.shape) # (8, 2) 

if __name__ == "__main__":
  main()   