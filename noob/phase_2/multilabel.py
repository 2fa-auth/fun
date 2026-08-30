#!/home/client/Documents/fun/py/venv/bin/python3
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F

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
  ])
  target = torch.tensor([
    [1., 0., 1., 0., 0.],
    [0., 1., 1., 0., 1.],
    [1., 0., 0., 1., 0.],
    [0., 1., 1., 1., 0.],
    [1., 0., 1., 0., 1.],
    [0., 1., 0., 1., 1.],
    [1., 1., 1., 0., 0.],
    [0., 1., 0., 0., 1.],
  ])

  data_set = LOLSet(x, target)
  data_loader = data.DataLoader(dataset=data_set, batch_size=4, shuffle=False)

  model = NoobMultiLabelModel(4, 5)
  opt = torch.optim.Adam(model.parameters(), lr=0.001)
  criterion = torch.nn.BCEWithLogitsLoss()
  num_ep = 1000
  model.train()

  for ep in range(num_ep):
    losses = 0
    l_counter = 0
    for obj, y in data_loader:
      raw_logit = model(obj)
      loss = criterion(raw_logit, y)
      opt.zero_grad()
      loss.backward()
      opt.step()

      losses += loss.item()
      l_counter += 1
    if ep % 200 == 0:
      print(f'ep [{ep}/{num_ep}] loss mean | {losses / l_counter}')
  model.eval()

  with torch.no_grad():
    raw_logit = model(x)
    logit = (F.sigmoid(raw_logit) > 0.5).float()

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