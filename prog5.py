import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Dataset generation
features, labels = make_classification(
    n_samples=1000,
    n_features=20,
    n_classes=2,
    random_state=42
)

# Train-test split
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels,
    test_size=0.2,
    random_state=42
)

# Model with dropout regularization
def create_dropout_model(dropout_rate=0.2):
    return Sequential([
        Dense(64, activation="relu", input_shape=(20,)),
        Dropout(dropout_rate),
        Dense(32, activation="relu"),
        Dropout(dropout_rate),
        Dense(1, activation="sigmoid")
    ])

# Model with gradient clipping
def create_clipped_model(clip_norm=1.0):
    model = Sequential([
        Dense(64, activation="relu", input_shape=(20,)),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=Adam(clipnorm=clip_norm),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# Training dropout model
dropout_model_instance = create_dropout_model()
dropout_model_instance.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history_dropout = dropout_model_instance.fit(
    features_train, labels_train,
    epochs=50,
    validation_data=(features_test, labels_test),
    verbose=0
)

# Training clipped model
clipped_model_instance = create_clipped_model()
history_clipped = clipped_model_instance.fit(
    features_train, labels_train,
    epochs=50,
    validation_data=(features_test, labels_test),
    verbose=0
)

# Plot accuracy comparison
plt.plot(history_dropout.history["accuracy"], "--", label="Dropout Train Accuracy")
plt.plot(history_dropout.history["val_accuracy"], "--", label="Dropout Validation Accuracy")

plt.plot(history_clipped.history["accuracy"], "--", label="Clipped Train Accuracy")
plt.plot(history_clipped.history["val_accuracy"], "--", label="Clipped Validation Accuracy")

plt.legend()
plt.title("Training Comparison: Dropout vs Gradient Clipping")
plt.show()
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


program5b:-
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.callbacks import EarlyStopping

# Load & normalize MNIST
(train_images, train_digits), (test_images, test_digits) = tf.keras.datasets.mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

# Parity labels (sum of digits → even = 0, odd = 1)
train_parity_labels = np.array([sum(map(int, str(digit))) % 2 for digit in train_digits])
test_parity_labels = np.array([sum(map(int, str(digit))) % 2 for digit in test_digits])

# Show sample images
plt.figure(figsize=(6, 6))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(train_images[i], cmap="gray")
    plt.axis("off")
plt.show()

# Multi-output model → digit + parity
inp = Input((28, 28))
x = layers.Flatten()(inp)
x = layers.Dense(128, activation='softmax')(x)

digit = layers.Dense(10, activation='softmax')(x)
parity = layers.Dense(1, activation='sigmoid')(x)

model = Model(inp, [digit, parity])

model.compile(
    optimizer='adam',
    loss=['sparse_categorical_crossentropy', 'binary_crossentropy'],
    metrics=[['accuracy'], ['accuracy']]
)

# Early stopping
es = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    train_images,
    [train_digits, train_parity_labels],
    epochs=20,
    validation_split=0.2,
    callbacks=[es]
)

# Early stop epoch
stop_ep = np.argmin(history.history['val_loss'])
print("\nEarly stopped at epoch:", stop_ep)

# Plot
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.axvline(stop_ep, color='red', linestyle='--', label='Early Stop')
plt.legend()
plt.title("Training vs Validation Loss")
plt.show()
