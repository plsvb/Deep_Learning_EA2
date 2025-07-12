import os
import torch
import clip
from PIL import Image
from zipfile import ZipFile
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt

# 📁 Relativer Pfad zur ZIP-Datei
zip_path = "input/travel_images.zip"

# 🔍 Textanfragen
text_prompts = [
    "Sonnenblumenfeld",
    "Kulinarische Köstlichkeiten",
    "London",
    "Die Muschel am Strand",
    "Kühe in den Bergen","

]

# 🧠 CLIP-Modell laden
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 📦 ZIP-Datei öffnen und Bilder laden
images = []
file_names = []
with ZipFile(zip_path, "r") as archive:
    for file_name in archive.namelist():
        if file_name.lower().endswith(("jpg", "jpeg", "png")):
            with archive.open(file_name) as file:
                try:
                    img = Image.open(BytesIO(file.read())).convert("RGB")
                    images.append(preprocess(img).unsqueeze(0))
                    file_names.append(file_name)
                except Exception as e:
                    print(f"⚠️ Fehler bei {file_name}: {e}")

print(f"{len(images)} Bilder geladen.")

# Embeddings berechnen
image_input = torch.cat(images).to(device)
text_tokens = clip.tokenize(text_prompts).to(device)

with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_tokens)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# Ergebnisse speichern
output_dir = "output/clip_results"
os.makedirs(output_dir, exist_ok=True)

for i, prompt in enumerate(text_prompts):
    similarities = (100.0 * image_features @ text_features[i].unsqueeze(0).T).squeeze(1)
    top_indices = similarities.topk(5).indices

    fig, axs = plt.subplots(1, 5, figsize=(15, 4))
    fig.suptitle(f"Top-5 für: {prompt}", fontsize=14)

    for j, idx in enumerate(top_indices):
        file_name = file_names[idx]
        with ZipFile(zip_path, "r") as archive:
            with archive.open(file_name) as file:
                img = Image.open(BytesIO(file.read())).convert("RGB")
                axs[j].imshow(img)
                axs[j].axis("off")
                axs[j].set_title(f"{similarities[idx]:.1f}")

    plt.tight_layout()
    filename = os.path.join(output_dir, f"top5_{i+1}_{prompt.replace(' ', '_')}.png")
    plt.savefig(filename)
    plt.close()
    print(f"Gespeichert: {filename}")

print("Suche abgeschlossen. Ergebnisse unter output/clip_results/")
