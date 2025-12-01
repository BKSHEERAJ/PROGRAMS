from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt

# Load + preprocess
X, y = load_iris(return_X_y=True)

encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model builder
def build_model():
    return Sequential([
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(3, activation="softmax")
    ])

# Create two models: GD (batch_size = 32) vs SGD (batch_size = 1)
model_gd = build_model()
model_sgd = build_model()

model_gd.compile(
    optimizer=SGD(learning_rate=0.01),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model_sgd.compile(
    optimizer=SGD(learning_rate=0.01),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Training
history_gd = model_gd.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=0
)

history_sgd = model_sgd.fit(
    X_train, y_train,
    epochs=50,
    batch_size=1,
    validation_data=(X_test, y_test),
    verbose=0
)

# Plot Loss Comparison
for history, name in [(history_gd, "GD"), (history_sgd, "SGD")]:
    plt.plot(history.history["loss"], label=f"Train Loss {name}")
    plt.plot(history.history["val_loss"], label=f"Validation Loss {name}")

plt.legend()
plt.title("Loss Comparison")
plt.show()

# Plot Accuracy Comparison
for history, name in [(history_gd, "GD"), (history_sgd, "SGD")]:
    plt.plot(history.history["accuracy"], label=f"Train Accuracy {name}")
    plt.plot(history.history["val_accuracy"], label=f"Validation Accuracy {name}")

plt.legend()
plt.title("Accuracy Comparison")
plt.show()
