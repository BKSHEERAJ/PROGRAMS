import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Data
(xtr, ytr), (xte, yte) = mnist.load_data()
xtr, xte = xtr / 255.0, xte / 255.0
xtr, xte = xtr[..., None], xte[..., None]
ytr, yte = to_categorical(ytr), to_categorical(yte)

# Model
m = Sequential([
    Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(),
    Dropout(0.25),

    Conv2D(64, 3, activation='relu'),
    MaxPooling2D(),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),

    Dense(10, activation='softmax')
])

m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

h = m.fit(
    xtr, ytr,
    validation_split=0.1,
    epochs=10,
    batch_size=128,
    verbose=2
)

print("Test Accuracy:", m.evaluate(xte, yte, verbose=0)[1])

# Accuracy + Loss Plots
plt.subplot(1, 2, 1)
plt.plot(h.history['accuracy'])
plt.plot(h.history['val_accuracy'])
plt.title("Accuracy")

plt.subplot(1, 2, 2)
plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.title("Loss")
plt.show()

# Prediction Samples
pred = tf.argmax(m.predict(xte[:5]), axis=1)

for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.imshow(xte[i].reshape(28, 28), cmap='gray')
    plt.title(f"P:{pred[i]}, T:{tf.argmax(yte[i])}")
    plt.axis('off')

plt.show()
