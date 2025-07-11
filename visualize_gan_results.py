import numpy as np
from PIL import Image
import os

# Verzeichnisse und Dateinamen der Ergebnisse
output_dir = "output"
collage_file = os.path.join(output_dir, "faces_collage.png")

# Prüfe, ob die Collage-Datei existiert
if not os.path.exists(collage_file):
    raise FileNotFoundError(f"Collage-Bild nicht gefunden: {collage_file}. Bitte zuerst generate_faces.py ausführen.")

# Lade die zuvor generierte 2x2 Collage der zufälligen Gesichter
collage_image = Image.open(collage_file)

# Generiere die interpolierten Gesichter (11 Bilder nebeneinander)
# Wir verwenden die Funktion interpolate_faces aus dem entsprechenden Modul
try:
    from interpolate_faces import interpolate_faces
except ImportError:
    raise ImportError("Das Modul interpolate_faces.py konnte nicht importiert werden.")
    
# Hole 11 interpolierte Bilder als Liste von Arrays
interpolated_images = interpolate_faces()  # list of 11 numpy arrays (128x128x3, dtype=uint8)

# Erstelle ein einzelnes Bild, das alle interpolierten Bilder in einer Reihe zeigt
# Konkatenieren der Bilder horizontal (axis=1)
row_image_array = np.concatenate(interpolated_images, axis=1)  # Ergebnisform: (128, 128*11, 3)
row_image = Image.fromarray(row_image_array)

# Speichere die Collage (zur Sicherheit erneut speichern) und die Interpolations-Reihe als PNG
os.makedirs(output_dir, exist_ok=True)
collage_output_path = os.path.join(output_dir, "faces_collage.png")
row_output_path = os.path.join(output_dir, "interpolation_row.png")

collage_image.save(collage_output_path)
row_image.save(row_output_path)

print(f"Collage der zufälligen Gesichter gespeichert: {collage_output_path}")
print(f"Interpolation-Reihe der Gesichter gespeichert: {row_output_path}")
