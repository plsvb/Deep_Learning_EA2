# clip_image2image.py

import os
import requests
from PIL import Image
from io import BytesIO
import torch
import clip
from torchvision import transforms
import matplotlib.pyplot as plt

# Ordnerpfade
collection_dir = "input/custom_collection"
example_dir = "input/example_images"
result_dir = "results/image2image"
os.makedirs(example_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)

# Beispielbilder herunterladen
API_KEY = ""  # Ersetzen!
queries = ["football", "snow landscape", "pasta"]

print("🔽 Lade Beispielbilder herunter...")
for query in queries:
    url = f"https://pixabay.com/api/?key={API_KEY}&q={query}&image_type=photo&per_page=3"
    r = requests.get(url).json()
    if "hits" in r and r["hits"]:
        img_url = r["hits"][0]["webformatURL"]
        img_data = requests.get(img_url).content
        filename = os.path.join(example_dir, f"{query.replace(' ', '_')}.jpg")
        with open(filename, "wb") as f:
            f.write(img_data)
        print(f"✅ Bild für '{query}' gespeichert.")
    else:
        print(f"❌ Keine Treffer für '{query}'")

# CLIP vorbereiten
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Bilder der Sammlung vorbereiten
print("📂 Lade Bilddatenbank...")
collection_images = []
collection_paths = []

for fname in os.listdir(collection_dir):
    if fname.lower().endswith(("jpg", "jpeg", "png")):
        path = os.path.join(collection_dir, fname)
        try:
            image = Image.open(path).convert("RGB")
            collection_images.append(preprocess(image))
            collection_paths.append(path)
        except:
            print(f"⚠️ Fehler beim Laden von {path}")

collection_tensor = torch.stack(collection_images).to(device)

with torch.no_grad():
    collection_features = model.encode_image(collection_tensor)

# Ähnlichkeitsberechnung & Plot
def plot_similarities(example_path, query_name):
    try:
        image = Image.open(example_path).convert("RGB")
    except:
        print(f"❌ Fehler beim Öffnen von {example_path}")
        return

    example_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        example_feature = model.encode_image(example_tensor)
        sims = torch.nn.functional.cosine_similarity(example_feature, collection_features)

    top_k = min(5, sims.shape[0])  # Schutz gegen topk(>n)
    top_idx = sims.topk(top_k).indices

    plt.figure(figsize=(12, 3))

    # Eingabebild anzeigen
    plt.subplot(1, top_k + 1, 1)
    plt.imshow(image)
    plt.title("Query", fontsize=10)
    plt.axis("off")

    for i, idx in enumerate(top_idx):
        sim_score = sims[idx].item()
        similar_img = Image.open(collection_paths[idx]).convert("RGB")
        plt.subplot(1, top_k + 1, i + 2)
        plt.imshow(similar_img)
        plt.title(f"{sim_score:.4f}", fontsize=8)
        plt.axis("off")

    plt.tight_layout()
    save_path = os.path.join(result_dir, f"similar_{query_name}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"💾 Gespeichert: {save_path}")

# Verarbeitung starten
print("🔎 Starte Bild-zu-Bild-Suche...")
for file in os.listdir(example_dir):
    if file.lower().endswith(("jpg", "jpeg", "png")):
        path = os.path.join(example_dir, file)
        name = os.path.splitext(file)[0]
        plot_similarities(path, name)
