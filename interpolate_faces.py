import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

# Laden des vortrainierten Progressive GAN Modells (ProGAN-128) von TensorFlow Hub
model = hub.load("https://tfhub.dev/google/progan-128/1").signatures['default']

latent_dim = 512  # Dimension des latenten Raums

def interpolate_faces(z1=None, z2=None, num_steps=11):
    """
    Interpoliere zwischen zwei latenten Vektoren z1 und z2 im latenten Raum.
    Gibt eine Liste von num_steps Bildern (als NumPy-Arrays) zurück, 
    die die Interpolationsschritte repräsentieren.
    """
    # Falls keine Vektoren übergeben: zwei zufällige latente Vektoren erzeugen
    if z1 is None:
        z1 = tf.random.normal([latent_dim])  # Form: (512,)
    if z2 is None:
        z2 = tf.random.normal([latent_dim])  # Form: (512,)
    
    # Interpolationsparameter lambda von 0 bis 1 (inklusiv) in num_steps Schritten
    alphas = np.linspace(0.0, 1.0, num_steps)
    
    # Liste für die generierten Interpolationsbilder
    interpolated_images = []
    
    # Berechne lineare Interpolation im latenten Raum und generiere jedes Zwischenbild
    for alpha in alphas:
        # Linear interpolierter Vektor: (1-alpha)*z1 + alpha*z2
        z_interpolated = (1.0 - alpha) * z1 + alpha * z2
        # Füge die Dimension [1, latent_dim] hinzu, damit das Modell den Vektor akzeptiert
        z_interpolated = tf.reshape(z_interpolated, (1, latent_dim))
        # Generiere das Bild aus dem interpolierten latenten Vektor
        image = model(z_interpolated)['default'][0]  # generiertes Bild (128x128x3)
        interpolated_images.append(image.numpy())
    
    # Konvertiere alle Bilder zu uint8 (0-255) und gebe sie als Liste von Arrays zurück
    interpolated_images = np.clip(np.array(interpolated_images) * 255, 0, 255).astype(np.uint8)
    image_list = [interpolated_images[i] for i in range(interpolated_images.shape[0])]
    return image_list

# Wenn das Skript direkt ausgeführt wird, generiere eine Beispielliste von interpolierten Bildern
if __name__ == "__main__":
    images = interpolate_faces()
    print(f"{len(images)} interpolierte Bilder erzeugt.")
