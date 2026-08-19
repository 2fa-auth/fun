import numpy as np

"""немного поучиться математике""" 

def relu(z):
  return 0 if z < 0 else z

x = 2.
w = 3.
b = 1

def forward():
  z = w*x+b # z = 7
  a = relu(z)
  L = a**2
  return z, a, L

z, a, L = forward()

def backward():
  """цель: вычислить dL / dx, dL / dw, dL / db"""
  grad_a = 2*a               # dL / da
  grad_z = 0 if z < 0 else 1 # da / dz
  grad_w = x                 # dz / dw 
  grad_x = w                 # dz / dx
  grad_b = 1                 # dz / db

  grad_x = grad_a * grad_z * grad_x # dL / dx = dL/da * da/dz * dz/dx
  grad_W = grad_a * grad_z * grad_w # dL / dw = dL/da * da/dz * dz/dw
  grad_bias = grad_a * grad_z * grad_b # dL / db = dL/da * da/dz * dz/db

  return grad_x, grad_W, grad_bias

grad_x, grad_W, grad_bias = backward()
print(grad_x, grad_W, grad_bias) # 42, 28, 14


"""
relu
grad_output:
  dL/da = 2*a

grad_input:
  dL/dz = dL/da * da/dz        
"""


