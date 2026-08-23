#!/home/client/Documents/fun/py/venv/bin/python3
import torch 


x = torch.tensor(2., requires_grad=True)
a = x * 3
b = a ** 2
b.backward()


print(x.grad) # 36 


