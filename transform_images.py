import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from PIL import Image
import os

# Gerät wählen
device = "cuda" if torch.cuda.is_available() else "cpu"

# Pipeline laden
print("Lade InstructPix2Pix-Modell ...")
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    "timbrooks/instruct-pix2pix",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)
print("Modell erfolgreich geladen.\n")

# Verzeichnisse
input_dir = "input"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Prompts zu Bildern (angepasst)
images_and_prompts = {
    "image1.jpg": "Erzeuge eine winterliche Szene mit Schnee auf dem Boden und an den Bäumen.",
    "image2.jpg": "Wandle das Gebäude in eine mittelalterliche Burg mit Türmen und Zinnen um.",
    "image3.jpg": "Gestalte die Bibliothek futuristisch mit leuchtenden blauen Regalen und moderner Architektur.",
    "image4.jpg": "Füge dem Klassenzimmer bunte futuristische Dekorationen hinzu.",
    "image5.jpg": "Verwandle die Aula in ein modernes Bühnenbild mit Lichteffekten."
}

# Für jedes Bild Transformation starten
for filename, prompt in images_and_prompts.items():
    input_path = os.path.join(input_dir, filename)
    print(f"\n🔍 Starte Verarbeitung von: {filename}")

    try:
        img = Image.open(input_path)
        print(f"  Bild geladen – Format: {img.format}, Modus: {img.mode}, Größe: {img.size}")

        if img.mode != "RGB":
            img = img.convert("RGB")
            print("  Konvertiert zu RGB")

        img = img.resize((512, 512))
        print("  Bild auf 512x512 verkleinert")

        print(f"  Starte Inferenz mit Prompt: '{prompt}'")
        result = pipe(
            prompt,
            image=img,
            num_inference_steps=30,
            guidance_scale=7.5,
            image_guidance_scale=1.5
        ).images[0]

        result = result.convert("RGB")
        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_edited.jpg")
        result.save(output_path)
        print(f"  ✅ Gespeichert unter: {output_path}")

    except Exception as e:
        print(f"  ⚠️ Fehler bei {filename}: {e}")

print("\n✔️ Alle Bilder verarbeitet.")
