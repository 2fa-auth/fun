#!/home/client/Documents/fun/py/venv/bin/python3
import numpy as np

"""
архитектура модели MLP (с backpropogation)
Версия 2
"""

class Parameter:
  def __init__(self, data, grad):
    self.data = data
    self.grad = grad

class Linear:
  def __init__(self, in_features, out_features):
    self.W = Parameter(np.random.randn(in_features, out_features) * 0.1, np.zeros((in_features, out_features)))
    self.b = Parameter(np.zeros(out_features), np.zeros((out_features)))
    self.X = []

  def forward(self, X):
    self.X = X
    return X @ self.W.data + self.b.data

  def backward(self, grad_output):
    """
    вычислить градиенты для dL/dw, dL/db, dL/dx
    """
    # grad_output = dL/dY
    X = self.X
    self.W.grad = X.T @ grad_output # dL/dw = 2a * ... * X
    self.b.grad = grad_output.sum(axis=0) 
    grad_X = grad_output @ self.W.data.T # dL/dx = 2a * ... * W
 
    return grad_X, self.W.grad, self.b.grad


class ReLU:
  def forward(self, X):
    return np.maximum(X, 0)

  def backward(self, grad_output, X):
    # dL/dX = dL/dY * dY/dX
    grad_X = grad_output * (X > 0)

    return grad_X


class SGD:
  def __init__(self, params, lr):
    self.params = params
    self.lr = lr

  def step(self):
    for parameter in self.params:
      parameter.data -= self.lr * parameter.grad

  def zero_grad(self):
    for parameter in self.params:
      parameter.grad.fill(0)


class MLP:
  def __init__(self):
    self.fc1 = Linear(2, 4)
    self.relu = ReLU()
    self.fc2 = Linear(4, 1)
 
    # cache grads
    self.grad_W1 = np.zeros(self.fc1.W.data.shape)
    self.grad_b1 = np.zeros(self.fc1.b.data.shape)
    self.grad_W2 = np.zeros(self.fc2.W.data.shape)
    self.grad_b2 = np.zeros(self.fc2.b.data.shape)
  
  def forward(self, X):
    self.z1 = self.fc1.forward(X)
    self.a1 = self.relu.forward(self.z1)
    self.y = self.fc2.forward(self.a1)

    return self.y

  def backward(self, grad_loss, X):
    grad_a1, grad_W2, grad_b2 = self.fc2.backward(grad_loss)
    grad_z1 = self.relu.backward(grad_a1, self.z1)
    _, grad_W1, grad_b1 = self.fc1.backward(grad_z1)
    
    self.grad_W1[...] = grad_W1
    self.grad_b1[...] = grad_b1
    self.grad_W2[...] = grad_W2
    self.grad_b2[...] = grad_b2

    return grad_W1, grad_b1, grad_W2, grad_b2

  def parameters(self):
    return [W_parameter, b_parameter, W_parameter, b_parameter]

def main():
  X = np.array([
    [2., 3.],
    [1., 4.],
    [3., 1.],
    [5., 2.]
  ])
  target = np.array([
    [10.],
    [9.],
    [7.],
    [13.]
  ])
  
  model = MLP()
  optim = SGD(model.parameters(), lr=0.01)

  for step in range(1000):
    pred = model.forward(X)
    loss = ((pred - target) ** 2).mean()
    grad_loss = 2 * (pred - target) / len(X)

    optim.zero_grad()
    grads = model.backward(grad_loss, X)
  
    optim.step()
    

    if step % 100 == 0:
      print(f"step={step}, loss={loss:.6f}")



if __name__ == "__main__":
    main()
