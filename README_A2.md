# Generierung von Gesichtsbildern mit Progressive GAN (ProGAN-128)

Dieses Projekt demonstriert die Generierung künstlicher Gesichter mit einem vortrainierten **Progressive GAN**. Dazu verwenden wir ein **ProGAN-128** Modell, das über [TensorFlow Hub](https://tfhub.dev/google/progan-128/1) bereitgestellt wird. Anhand dieses Modells werden zufällige Gesichter erzeugt und fließende Übergänge (Interpolation) zwischen zwei generierten Gesichtern visualisiert.

## Anforderungen und Installation

Stellen Sie sicher, dass die folgenden Bibliotheken installiert sind:

- **TensorFlow 2.x** (inkl. GPU-Unterstützung falls verfügbar)  
- **TensorFlow Hub** (`tensorflow_hub`)  
- **Matplotlib** (`matplotlib`)  
- **Pillow** (`PIL` Paket für Python)  
- (Optional: **NumPy** falls nicht bereits durch TensorFlow installiert)

Installation mit `pip`, z.B.:

```bash
pip install tensorflow tensorflow-hub matplotlib Pillow
