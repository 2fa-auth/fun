#!/home/client/Documents/fun/py/venv/bin/python3
import numpy as np

""" балуюсь с mse : ) """


def mse(pred, target):
  return ((pred - target)**2).mean()

def main():
  pred = np.array([1, 2, 3])
  target = np.array([1, 4, 2])
  loss = mse(pred, target)

  print(loss.shape) # ()
  print(loss) # [0, 4, 1]

  target = 10
  # поиск минимума, пока втупую; без градиента
  losses = {mse(pred, target): pred for pred in np.arange(0, 20, 0.5)} 
  print(f'минимальный loss достигается при pred равному: {losses[min(losses)]}')

if __name__ == "__main__":
  main()