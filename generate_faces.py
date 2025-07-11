import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import os

# Laden des vortrainierten Progressive GAN Modells von TensorFlow Hub (ProGAN-128)
# Hinweis: Beim ersten Ausführen wird das Modell aus dem Internet geladen (TF Hub).
model = hub.load("https://tfhub.dev/google/progan-128/1").signatures['default']

# Parameter für die Bildgenerierung
latent_dim = 512    # Dimensionalität des latenten Vektors
num_faces  = 4      # Anzahl der zu generierenden Gesichter

# Erzeuge vier zufällige latente Vektoren z (Standardnormalverteilung)
latent_vectors = tf.random.normal([num_faces, latent_dim])

# Generiere Bilder aus den latenten Vektoren mit dem GAN-Modell
# Die Ausgabe 'default' des Modells enthält die generierten Bilder (Float32 Tensor)
generated_images = model(latent_vectors)['default']  # Shape: (4, 128, 128, 3)

# Konvertiere die generierten Bilder in das uint8-Format (Pixel 0-255) für die Ausgabe
images_np = generated_images.numpy()  # Tensor -> NumPy-Array
images_uint8 = np.clip(images_np * 255, 0, 255).astype(np.uint8)  # Wertebereich [0,255]

# Erstelle einen Ausgabe-Ordner, falls nicht vorhanden
os.makedirs("output", exist_ok=True)

# Speichere jedes generierte Gesicht als einzelnes Bild
for i in range(num_faces):
    img = Image.fromarray(images_uint8[i])
    img.save(os.path.join("output", f"face_{i+1}.png"))
    print(f"Gesicht {i+1} gespeichert: output/face_{i+1}.png")

# Erstelle eine 2x2 Collage der vier Gesichter
collage_width  = 128 * 2  # zwei Bilder nebeneinander
collage_height = 128 * 2  # zwei Bilder untereinander
collage_image = Image.new('RGB', (collage_width, collage_height))

# Positionen für die 2x2 Anordnung (oben-links, oben-rechts, unten-links, unten-rechts)
positions = [(0, 0), (128, 0), (0, 128), (128, 128)]
for img_array, pos in zip(images_uint8, positions):
    img = Image.fromarray(img_array)
    collage_image.paste(img, pos)

# Speichere die Collage als PNG-Bild
collage_path = os.path.join("output", "faces_collage.png")
collage_image.save(collage_path)
print(f"2x2 Collage gespeichert: {collage_path}")
