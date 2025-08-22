sample_directory = "../datasets/sample"
images_directory_validation = 'datasets/product_images/validation/'
text_file_location = 'datasets/product_brands_test.json'



# import and load openai-clip-model

import torch
import clip
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device=device)



# There are 40 brands, 10 categories and 12 colors.

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



# function definition for zero-shot weigh calculation

import torch
import clip
from tqdm.notebook import tqdm
def zeroshot_weight_calculator(classifications, template):
    with torch.no_grad():
        weights = []
        for element in tqdm(classifications):
            text = [line.format(element) for line in template]
            text_tokenized = clip.tokenize(text)
            text_encoded = model.encode_text(text_tokenized)
            text_encoded /= text_encoded.norm(dim=-1, keepdim=True)
            text_encoded = text_encoded.mean(dim=0)
            text_encoded /= text_encoded.norm()
            weights.append(text_encoded)
        nn_weights = torch.stack(weights, dim=1)
    return nn_weights



import os
from PIL import Image
def image_processing(directory):
    images = []
    
    for filename in os.listdir(directory):
        image = Image.open(os.path.join(directory, filename)).convert("RGB")
        images.append(preprocess(image))
        return images



import numpy as np
def image_encoding(images):
    images_tensors = torch.tensor(np.stack(images)) # convert images into tensors [n,3,255,255]
    images_features = model.encode_image(images_tensors).float()
    images_features /= images_features.norm(dim=-1, keepdim=True)
    return images_features



def top_elemets(attribute, indexes):
    temp = []
    
    for i in range(len(indexes[0])):
        temp.append(attribute[indexes[0][i]])
    return temp



def generate_titles(brands, categories, colors):
    
    titles = []
    i=j=k=0
    
    for i in range(len(brands)):
        for j in range(len(categories)):
            for k in range(len(colors)):
                titles.append(brands[i] + " | " + categories[j] + " | " + colors[k])

    return titles



def calculate_title_probability(titles, image_features):
    title_tokenized = clip.tokenize(titles)
    title_encoded = model.encode_text(title_tokenized).float()
    title_encoded /= title_encoded.norm(dim=-1, keepdim=True)
    logits = title_encoded @ image_features.T
    probabilities = (100.0 * image_features @ title_encoded.T).softmax(dim=-1)    
    return probabilities



def top5_titles(titles, probabilities):
    titles_top5 = []
    top5_probabilities, top5_indexes = probabilities.cpu().topk(5, dim=-1)
    for i in range(len(top5_indexes[0])):
        #print(f"Title candidate {i+1}: ", predicted_titles[top5_indexes[0][i]], f' ---> probability = {float("{:.2f}".format(top5_probabilities[0][i]*100))}%')
        titles_top5.append(f'Title candidate {i+1}: ' + 
                           titles[top5_indexes[0][i]] + 
                           f' ---> probability = {float("{:.2f}".format(top5_probabilities[0][i]*100))}%')

    return titles_top5