images_path = '../../datasets/products/images/validation/'
text_file_brands = '../../datasets/products/texts/product_brands/validation.json'
text_file_categories = '../../datasets/products/texts/product_categories/validation.json'
text_file_colors = '../../datasets/products/texts/product_colors/validation.json'
text_file_titles = '../../datasets/products/texts/product_titles/validation.json'



import torch
import clip
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device=device)


categories = ("Sunglasses", "T-Shirt", "Shoes", "Jacket", "Socks", "Track Pant", "Shorts", "Cap", "Bag", "Beanie")
brands = ("Ray-Ban", "Carrera", "Gucci", "Versace", "Prada", "Tommy Hilfiger", "Lacoste", "U.S. Polo Assn.", "DKNY", 
          "Polo Ralph Lauren", "Nike", "Adidas", "Puma", "Calvin Klein", "Reebok", "Under Armour", "Brooks Brothers", 
          "Haimont", "ASICS", "Saucony", "FitVille", "Brooks", "Skechers", "Red Tape", "Little Donkey Andy", "33,000ft", 
          "Columbia", "Carhartt", "MAGCOMSEN", "The North Face", "Darn Tough", "VRD", "G Gradual", "Fila", "BROKIG", 
          "Champion", "NORTHYARD", "Mizuno", "Hurley", "Timberland")
colors = ("Black", "White", "Grey", "Brown", "Red", "Green", "Blue", "Orange", "Yellow", "Pink", "Violet", "Purple")


categories_template = [
    'the product is {}.',
    'the product is called {}.',
    'the item is identified as {}.',
    'the item is sold as {}.'
]

brands_template = [
    'the brand is {}.',
    'the manufacturer is {}.',
    'the item is made by {}.',
    'the product is manufactured by {}.'
]

colors_template = [
    'the color is {}.',
    'the item is {} in hue.',
    'the product is {} in color.',
    'the shade of the product is {}.'
]



import json
def read_text(file):    
    with open(file, 'r') as file_contents:
        texts = []    
        for line in file_contents:
          json_obj = json.loads(line)
          texts.append(json_obj)        
    return texts



def get_image_title(texts):
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


def zeroshot_weight_calculator(attributes):
    
    tokens = clip.tokenize(attributes)
    class_embeddings = model.encode_text(tokens)
    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)    
    zeroshot_weights = class_embeddings.T
    
    return zeroshot_weights


from tqdm.notebook import tqdm

def zeroshot_weight_calculator_tmpl(attributes, templates):
    with torch.no_grad():
        weights = []
        #for element in tqdm(attributes):
        for element in attributes:
            texts = [template.format(element) for template in templates]
            texts = clip.tokenize(texts)
            text_embeddings = model.encode_text(texts)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            text_embedding = text_embeddings.mean(dim=0)
            text_embedding /= text_embedding.norm()
            weights.append(text_embedding)
        nn_weights = torch.stack(weights, dim=1)
    return nn_weights



def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    target = torch.tensor(target)    
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]



def top_elemets(attribute, indexes):
    temp = []
    
    for i in range(len(indexes[0])):
        temp.append(attribute[indexes[0][i]])
    return temp