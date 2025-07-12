# clip_search_custom.py

import os
import torch
import clip
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

# Modell und Gerät
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Bildordner
image_folder = os.path.join("input", "custom_collection")
image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder)
               if f.lower().endswith(("jpg", "jpeg", "png"))]

# Bilder laden und preprocessen
images = []
image_tensors = []

for path in image_paths:
    image = Image.open(path).convert("RGB")
    images.append(image)
    image_tensors.append(preprocess(image))


# Tensor-Stack und Bild-Embeddings
image_input = torch.stack(image_tensors).to(device)
with torch.no_grad():
    image_features = model.encode_image(image_input)

# Suchphrasen definieren
text_prompts = [
    "Ein Tiger im Dschungel",
    "Abendessen bei Kerzenschein",
    "Surfer",
    "Winterlandschaft mit Tannen",
    "Fußball"
]

# Suche durchführen
for prompt in text_prompts:
    print(f"\n🔍 Suche für Prompt: {prompt}")
    text_tokens = clip.tokenize([prompt]).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)

    similarities = torch.nn.functional.cosine_similarity(text_features, image_features)
    top_indices = similarities.argsort(descending=True)[:5]

    # Visualisierung
    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(top_indices):
        sim = similarities[idx].item()
        plt.subplot(1, 5, i+1)
        plt.imshow(images[idx])
        plt.title(f"{sim:.4f}", fontsize=10) 
        plt.axis("off")

    plt.suptitle(f"Top-5 für: {prompt}", fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.80)
        # Sicherstellen, dass der Ordner existiert
    os.makedirs("results", exist_ok=True)

    # Prompt-Dateinamen säubern
    filename = prompt.replace(" ", "_").replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    save_path = os.path.join("results", f"{filename}.png")

    # Speichern
    plt.savefig(save_path)
    print(f"✅ Ergebnis gespeichert unter: {save_path}")

    plt.close()

