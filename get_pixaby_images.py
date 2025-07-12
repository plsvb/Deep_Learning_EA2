import os
import requests

API_KEY = ""  # 👉 Hier deinen Pixabay API Key einfügen
QUERIES = [
    "nature", "animals", "cities", "sports", "food",
    "people", "technology", "beach", "mountains", "transport"
]

SAVE_DIR = "input/custom_collection"
os.makedirs(SAVE_DIR, exist_ok=True)

img_count = 0
for query in QUERIES:
    print(f"🔍 Lade Bilder für: {query}")
    url = f"https://pixabay.com/api/?key={API_KEY}&q={query}&image_type=photo&per_page=20&page=1"
    r = requests.get(url).json()

    hits = r.get("hits", [])[:20]  # max. 20 Bilder pro Kategorie
    for i, hit in enumerate(hits):
        img_url = hit["webformatURL"]
        img_data = requests.get(img_url).content
        filename = os.path.join(SAVE_DIR, f"{img_count:04d}_{query}.jpg")
        with open(filename, "wb") as f:
            f.write(img_data)
        img_count += 1

print(f"\n✅ Insgesamt {img_count} Bilder gespeichert.")
