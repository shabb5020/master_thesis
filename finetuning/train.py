logo_images = '../datasets/logo/images/train/'
logo_texts = '../datasets/logo/texts.json'

product_images = '../datasets/products/images/train/'
product_titles = '../datasets/products/texts/product_titles/train.json'




import torch
import clip

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)



import json
def read_text(file):    
    with open(file, 'r') as file_contents:
        texts = []    
        for line in file_contents:
          json_obj = json.loads(line)
          texts.append(json_obj)        
    return texts



def get_image_title(images_path, texts):
    image_list = []
    title_list = []

    for line in texts:
      image_path = images_path + line["image_path"]
      image_list.append(image_path)

      title = line["product_title"]
      title_list.append(title)  
        
    return image_list, title_list



from PIL import Image

class image_title_dataset:
  def __init__(self, images, titles):
    self.images = images
    self.titles = titles

  def __len__(self):
    return len(self.titles)

  def __getitem__(self, idx):
    images = preprocess(Image.open(self.images[idx]))
    titles = self.titles[idx]
    return images, titles


