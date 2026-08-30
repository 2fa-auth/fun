#!/home/client/Documents/fun/py/venv/bin/python3
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F

from math import ceil, floor

"""
многометочная классификация и метрики (КОСТЫЛЬ):
  * в этом файле отдается приоритет вычислению метрик.
    менее приоритетнее сама модель.

  * в дальнейшем файл будет использоваться для эксплуатации 
    вывода метрик для более глубокого анализа качества модели.

  За основу взята матрица ошибок (T*/ F*):
    precision = TP + (TP / FP)
    recall = TP + (TP / FN)
    F1 = 2 * (precision * recall) / (precision + recall)

"""


# -- нубский сет
class LOLSet(data.Dataset):
  def __init__(self, x, target):
    super().__init__()
    self.x = x
    self.target = target
    self.len = int((x.size(0) + target.size(0)) / 2)
  def __getitem__(self, index):
    return (self.x[index], self.target[index])
  def __len__(self):
    return self.len

# -- нубская модель
class NoobMultiLabelModel(nn.Module):
  def __init__(self, in_features, out_features):
    super().__init__()

    self.fc1 = nn.Linear(in_features, 64)
    self.fc2 = nn.Linear(64, 32)
    self.fc3 = nn.Linear(32, out_features)
  def forward(self, X):
    out = F.relu(self.fc1(X))
    out = F.relu(self.fc2(out))
    out = self.fc3(out)
    return out


def main():
  # -- нубская выборка
  x = torch.tensor([
    [1.0, 0.0, 2.0, 0.5],
    [0.0, 2.0, 1.0, 1.5],
    [3.0, 1.0, 0.0, 2.0],
    [1.5, 3.0, 2.0, 0.0],
    [2.0, 0.5, 3.0, 1.0],
    [0.5, 2.5, 1.0, 3.0],
    [3.0, 2.0, 2.0, 1.0],
    [1.0, 3.0, 0.5, 2.5],
    [2.5, 1.0, 1.5, 0.5],
    [0.5, 1.5, 2.5, 2.0],
    [3.5, 0.5, 1.0, 1.5],
    [1.0, 2.5, 3.0, 0.5],
    [2.0, 3.0, 0.5, 1.0],
    [0.5, 0.5, 3.0, 2.5],
    [3.0, 1.5, 2.5, 0.0],
    [1.5, 2.0, 0.0, 3.0],
    [2.5, 2.5, 1.0, 0.5],
    [0.0, 3.0, 2.0, 1.0],
    [3.0, 0.0, 1.5, 2.5],
    [1.0, 1.0, 3.0, 2.0],
    [2.2, 1.1, 2.8, 0.7],
    [0.3, 2.8, 1.7, 2.2],
    [3.2, 0.8, 0.9, 2.3],
    [1.7, 2.7, 2.4, 0.3],
    [2.8, 0.4, 3.1, 1.3],
  ], dtype=torch.float32)


  target = torch.tensor([
    [1., 0., 1., 0., 0.],
    [0., 1., 1., 0., 1.],
    [1., 0., 0., 1., 0.],
    [0., 1., 1., 1., 0.],
    [1., 0., 1., 0., 1.],
    [0., 1., 0., 1., 1.],
    [1., 1., 1., 0., 0.],
    [0., 1., 0., 0., 1.],
    [1., 0., 1., 0., 0.],
    [0., 1., 1., 1., 0.],
    [1., 0., 0., 1., 1.],
    [0., 1., 1., 0., 1.],
    [1., 1., 0., 1., 0.],
    [0., 0., 1., 1., 1.],
    [1., 0., 1., 0., 1.],
    [0., 1., 0., 1., 0.],
    [1., 1., 1., 0., 0.],
    [0., 1., 1., 1., 0.],
    [1., 0., 0., 1., 1.],
    [0., 1., 1., 0., 1.],
    [1., 0., 1., 0., 0.],
    [0., 1., 1., 1., 0.],
    [1., 0., 0., 1., 1.],
    [0., 1., 1., 0., 0.],
    [1., 0., 1., 0., 1.],
  ], dtype=torch.float32)


  """
  100 - 100%
  x   - 15% 

  """


  data_size = x.size(0)
  train_size = floor(data_size * 70 / 100)
  val_size = ceil(data_size * 15 / 100)
  test_size = ceil(data_size * 15 / 100)


  print(f'кол-во выборки: {train_size + val_size + test_size}')
  print(f'кол-во обучающей выборки (70% от всей): {train_size}')
  print(f'кол-во валидационной выборки (15% от всей): {val_size}')
  print(f'кол-во тестовой выборки (15% от всей): {test_size}\n')



  train_set = LOLSet(x[:train_size], target[:train_size])
  val_set = LOLSet(x[train_size:train_size+test_size], target[train_size:train_size+test_size])
  test_set = LOLSet(x[train_size+val_size:], target[train_size+val_size:])

  train_loader = data.DataLoader(dataset=train_set, batch_size=4, shuffle=False) # нубская выборка не нуждается в перемешке
  val_loader = data.DataLoader(dataset=val_set, batch_size=2, shuffle=False)
  test_loader = data.DataLoader(dataset=test_set, batch_size=4, shuffle=False)

  model = NoobMultiLabelModel(4, 5)
  opt = torch.optim.Adam(model.parameters(), lr=0.001)
  criterion = torch.nn.BCEWithLogitsLoss()
  num_ep = 1000
  model.train()

  for ep in range(num_ep):
    train_loss = 0
    tl_cnt = 0

    val_loss = 0
    vl_cnt = 0

    for x, y in train_loader:
      pred = model(x)
      loss = criterion(pred, y)
      opt.zero_grad()
      loss.backward()
      opt.step()

      train_loss += loss.item()
      tl_cnt += 1

    for x, y in val_loader:
      pred = model(x)
      val_loss += criterion(pred, y).item()
      vl_cnt += 1      


    if ep % 50 == 0:
      train_loss_mean = train_loss / tl_cnt
      val_loss_mean = val_loss / vl_cnt
  
      print(f'ep [{ep}/{num_ep}]\tTRAIN loss {train_loss_mean} \t\t VAL loss {val_loss_mean}')


  model.eval()

  with torch.no_grad():
    for x, target in test_loader:
      pred = model(x)
      logit = (F.sigmoid(pred) > 0.5).float()

      label_wise_accuracy = 0
      exact_match_accuracy = 0

      macro_tp = 0
      macro_fp = 0
      macro_tn = 0
      macro_fn = 0
      f1_classes = []
      micro_tp = 0
      micro_fp = 0
      micro_tn = 0
      micro_fn = 0
      micro_precision = 0
      micro_recall = 0
      micro_f1 = 0
      macro_precision = 0
      macro_recall = 0
      macro_f1 = 0

      label_wise_accuracy = (logit == target).sum() / ((logit.numel() + target.numel()) / 2)   
      exact_match_accuracy = 1. if label_wise_accuracy == 1. else 0. # c более чем одним sample потреубется изменить вычисление

      for index_ax2 in range(logit.size(1)):
        macro_tp = macro_fp = macro_tn = macro_fn = 0
        macro_precision = macro_recall = 0

        for index_ax1 in range(logit.size(0)):
          if logit[index_ax1, index_ax2] == 1.:
            if target[index_ax1, index_ax2] == 1.:
              macro_tp += 1
              micro_tp += 1
            if target[index_ax1, index_ax2] == 0:
              macro_fp += 1
              micro_fp += 1
          if logit[index_ax1, index_ax2] == 0:
            if target[index_ax1, index_ax2] == 0:
              macro_tn += 1
              micro_tn += 1
            if target[index_ax1, index_ax2] == 1:
              macro_fn += 1
              micro_fn += 1
            
        macro_precision = macro_tp / (macro_tp + macro_fp)
        macro_recall = macro_tp / (macro_tp + macro_fn)
        f1_classes.append(2 * (macro_precision * macro_recall) / (macro_precision + macro_recall))

      micro_precision = micro_tp / (micro_tp + micro_fp)
      micro_recall = micro_tp / (micro_tp + micro_fn)

      macro_f1 = sum(f1_classes) / len(f1_classes)
      micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)

      print(f'\nточность модели с точки зрения попадания: {label_wise_accuracy}')
      print(f'точность модели с точки зрения совпадения: {exact_match_accuracy}')
      print(f'macro F1 score: {macro_f1}')
      print(f'micro F1 score: {micro_f1}')

if __name__ == "__main__":
  main()