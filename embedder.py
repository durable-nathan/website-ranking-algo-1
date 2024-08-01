import os
import torch
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from lib.modal import app, cache_path

# TODO: make this embed all images that it is given, and export the embedded data for later to store into db


def get_image_rgb(image_url):
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    return image

def get_single_image_embedding(image_url, processor, model, device):
    print("get_single_image_embedding: 1")
    image_rgb = get_image_rgb(image_url)
    print("get_single_image_embedding: 2")
    image = processor(
        text = None,
        images = image_rgb,
        return_tensors="pt"
        )["pixel_values"].to(device)
    embedding = model.get_image_features(image)
    print("get_single_image_embedding: 3")
    # convert the embeddings to numpy array
    return embedding.cpu().detach().numpy()



@app.function(gpu="A10G")
def get_one_embedding(image_url=None, text=None):
    from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
    
    model = CLIPModel.from_pretrained(cache_path).to("cuda")
    processor = CLIPProcessor.from_pretrained(cache_path)
    tokenizer = CLIPTokenizer.from_pretrained(cache_path)
    device = "cuda"

    if image_url:
        print("is image", image_url)
        image_rgb = get_image_rgb(image_url)
        image = processor(
            text = None,
            images = image_rgb,
            return_tensors="pt"
            )["pixel_values"].to(device)
        embedding = model.get_image_features(image)
        return embedding.cpu().detach().numpy()

    elif text:
        print("is text", text)
        text = processor(
            text=text,
            images=None,
            return_tensors="pt"
            )["input_ids"].to(device)
        embedding = model.get_text_features(text)
        return embedding.cpu().detach().numpy()


    # convert the embeddings to numpy array
