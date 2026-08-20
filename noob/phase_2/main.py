#!/home/client/Documents/fun/py/venv/bin/python3
import torch 

x = torch.tensor(3., requires_grad=True)
y = 3 * x + 2
y.backward() # dy/dx

print(x.grad) # 3  








