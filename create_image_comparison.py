from PIL import Image
import os

# Verzeichnisse
input_dir = "input"
output_dir = "output"
comparison_path = "output/comparison.png"

# Bildnamen (an deine Dateien anpassen)
image_files = ["image1.jpg", "image2.jpg", "image3.jpg", "image4.jpg", "image5.jpg"]

# Liste der Bilder vorbereiten
original_images = []
edited_images = []

for img_name in image_files:
    original = Image.open(os.path.join(input_dir, img_name)).resize((256, 256))
    edited_name = os.path.splitext(img_name)[0] + "_edited.png"
    edited = Image.open(os.path.join(output_dir, edited_name)).resize((256, 256))
    
    original_images.append(original)
    edited_images.append(edited)

# Bildbreite und -höhe berechnen
total_width = 256 * len(image_files)
total_height = 256 * 2  # zwei Reihen

# Neues Bild erstellen
comparison_image = Image.new('RGB', (total_width, total_height))

# Originalbilder oben anordnen
for i, img in enumerate(original_images):
    comparison_image.paste(img, (i * 256, 0))

# Transformierte Bilder unten anordnen
for i, img in enumerate(edited_images):
    comparison_image.paste(img, (i * 256, 256))

# Gesamtes Vergleichsbild speichern
comparison_image.save(comparison_path)
print(f"Vergleichsbild gespeichert unter: {comparison_path}")
