#!/home/client/Documents/fun/py/venv/bin/python3
import json

if __name__ == "__main__":
  with open('images.json', 'r', encoding='utf-8') as f: imgs = json.load(f)['images']
  with open('annotations.json', 'r', encoding='utf-8') as f: annots = json.load(f)['annotations']
  with open('categories.json', 'r', encoding='utf-8') as f: cats = json.load(f)['categories']

  for img in imgs:  
    for annot in annots:
      if annot['image_id'] == img['id']:
        bbox = annot['bbox']
        print(f'BBOX of image {img['file_name']}: xywh: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}] \t',end=' ')
        print(f'x1y1x2y2: [{bbox[0]}, {bbox[1]}, {bbox[0]+bbox[2]}, {bbox[1]+bbox[3]}] \t',end=' ')
        for cat in cats: 
          if cat['id'] == annot['category_id']: print(f'cat: {cat['name']}')                                                       
  

