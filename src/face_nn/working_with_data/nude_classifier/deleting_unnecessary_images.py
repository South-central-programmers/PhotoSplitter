import os
import torch
import clip
from PIL import Image
import numpy as np
from tqdm import tqdm

device = "mps"
model, preprocess = clip.load("ViT-B/32", device=device)


def is_anime(image_path, threshold=0.5):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text = clip.tokenize(["a anime porno", "a real life porno"]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    return probs[0][0] > threshold


def remove_anime_images(directory):
    for root, dirs, files in os.walk(directory):
        for file in tqdm(files):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                file_path = os.path.join(root, file)
                try:
                    if is_anime(file_path):
                        print(f"Deleting {file_path}")
                        os.remove(file_path)
                except Exception as e:
                    print(f"Error while deleting {file_path}: {e}")


directories = [
    "data/nude_classification/nude_classification_images/train/nude",
    "data/nude_classification/nude_classification_images/test/nude",
    "data/nude_classification/nude_classification_images/val/nude",
]

for directory in directories:
    remove_anime_images(directory)
