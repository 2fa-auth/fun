#!/home/client/Documents/fun/py/venv/bin/python3
import numpy as np


class Linear:
  def __init__(self, in_features, out_features):
    self.W = np.random.randn(in_features, out_features) * 0.1
    self.b = np.zeros(out_features)

  def forward(self, X):
    return X @ self.W + self.b

  def backward(self, grad_output, X):
    # grad_output = dL/dY

    # dL/dW
    grad_W = X.T @ grad_output

    # dL/db
    grad_b = grad_output.sum(axis=0)

    # dL/dX
    grad_X = grad_output @ self.W.T

    return grad_X, grad_W, grad_b

  def update(self, grad_W, grad_b, lr):
    self.W -= lr * grad_W
    self.b -= lr * grad_b


class ReLU:
  def forward(self, X):
    return np.maximum(X, 0)

  def backward(self, grad_output, X):
    # dL/dX = dL/dY * dY/dX
    grad_X = grad_output * (X > 0)

    return grad_X


class MLP:
  def __init__(self):
    self.fc1 = Linear(2, 4)
    self.relu = ReLU()
    self.fc2 = Linear(4, 1)

  def forward(self, X):
    self.z1 = self.fc1.forward(X)
    self.a1 = self.relu.forward(self.z1)
    self.y = self.fc2.forward(self.a1)

    return self.y

  def backward(self, grad_loss, X):
    grad_a1, grad_W2, grad_b2 = self.fc2.backward(grad_loss,self.a1)
    grad_z1 = self.relu.backward(grad_a1, self.z1)
    _, grad_W1, grad_b1 = self.fc1.backward(grad_z1,X)

    return grad_W1, grad_b1, grad_W2, grad_b2

  def update(self, grads, lr):
    grad_W1, grad_b1, grad_W2, grad_b2 = grads
    self.fc1.update(grad_W1, grad_b1, lr)
    self.fc2.update(grad_W2, grad_b2, lr)
        


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

  for step in range(1000):
    pred = model.forward(X)
    loss = ((pred - target) ** 2).mean()
    grad_loss = 2 * (pred - target) / len(X)
    grads = model.backward(grad_loss, X)
    model.update(grads, lr=0.01)

    if step % 100 == 0:
      print(f"step={step}, loss={loss:.6f}")


if __name__ == "__main__":
    main()