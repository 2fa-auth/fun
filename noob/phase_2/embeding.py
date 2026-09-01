#!/home/client/Documents/fun/py/venv/bin/python3
import torch 
import torch.nn as nn


class EmbeddingsClassifiser(nn.Module):
  def __init__(self, num_cities, num_objects, dim_cities, dim_objects, out_features):
    super().__init__()

    self.emb_cities = nn.Embedding(num_embeddings=num_cities, embedding_dim=dim_cities)
    self.emb_objects = nn.Embedding(num_embeddings=num_objects, embedding_dim=dim_objects)

    self.net = nn.Sequential(
      nn.Linear(dim_cities + dim_objects, 16),
      nn.ReLU(),
      nn.Linear(16, 16),
      nn.ReLU(),
      nn.Linear(16, out_features)
    )


  def forward(self, city_index, object_index):
    emb_city = self.emb_cities(city_index)
    emb_object = self.emb_objects(object_index)

    x = torch.cat([emb_city, emb_object], dim=1)
    return self.net(x)



def main():
  cities_names = ['Москва', 'Сочи', 'Санкт-петербург', 'Ростов-на-Дону', "Шахты"]
  obj_names = ['квартира', 'дом', 'машина', 'человек']

  cities_names_indices = {city: idx for idx, city in enumerate(cities_names)}
  obj_names_indices = {obj: idx for idx, obj in enumerate(obj_names)}

  model = EmbeddingsClassifiser(5, 4, 3, 3, 1)


  city_name = "Москва"
  object_name = "квартира"

  city_index = torch.tensor([cities_names_indices[city_name]])
  object_index = torch.tensor([obj_names_indices[object_name]])

  pred = model(city_index, object_index)
  print(pred.shape)



if __name__ == "__main__":
  main()