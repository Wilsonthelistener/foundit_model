import clip
from PIL import Image
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model, preprocess = clip.load("ViT-B/32", device=device)

# Example images
image_paths = ["cat.png", "dog.png", "car.png"]
images = [preprocess(Image.open(path)).unsqueeze(0).to(device) for path in image_paths]
image_input = torch.cat(images, dim=0)

# Text descriptions
descriptions = ["a photo of a cat", "a photo of a dog", "a photo of a car"]
text_tokens = clip.tokenize(descriptions).to(device)

# Encode features
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_tokens)

image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)

# Compute similarity
similarity = image_features @ text_features.T
best_match_indices = similarity.argmax(dim=1)

for i, idx in enumerate(best_match_indices):
    print(f"{image_paths[i]} is best described by: '{descriptions[idx]}'")

