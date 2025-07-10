from data_preprocessing import x_train, y_train_cat, x_test, y_test_cat
from model import create_cifar10_model
from tensorflow.keras.optimizers import Adam

# Modell erzeugen
model = create_cifar10_model()

# Kompilieren des Modells
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Modell trainieren
history = model.fit(x_train, y_train_cat, epochs=15, batch_size=64, validation_split=0.2, verbose=1)

# Evaluation
test_loss, test_accuracy = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Modell speichern
model.save('cifar10_cnn.h5')
