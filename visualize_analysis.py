import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from data_preprocessing import x_test, y_test

# Gespeichertes Modell laden
model = load_model('cifar10_cnn.h5')

# Vorhersagen machen
y_pred_probs = model.predict(x_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true = y_test.flatten()

# Confusion Matrix berechnen
cm = confusion_matrix(y_true, y_pred_classes)

# Confusion Matrix anzeigen
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=['airplane','auto','bird','cat','deer','dog','frog','horse','ship','truck'],
            yticklabels=['airplane','auto','bird','cat','deer','dog','frog','horse','ship','truck'])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')
plt.show()

# Fehlklassifizierte Bilder visualisieren
misclassified_idx = np.where(y_pred_classes != y_true)[0]
sample_idx = np.random.choice(misclassified_idx, size=5, replace=False)

plt.figure(figsize=(12,5))
class_names = ['airplane','auto','bird','cat','deer','dog','frog','horse','ship','truck']
for i, idx in enumerate(sample_idx):
    plt.subplot(1,5,i+1)
    plt.imshow(x_test[idx])
    plt.title(f"True: {class_names[y_true[idx]]}\nPred: {class_names[y_pred_classes[idx]]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
