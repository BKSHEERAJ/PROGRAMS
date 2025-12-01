import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Layer


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, y_train = x_train[:1000] / 255.0, y_train[:1000]
x_test, y_test = x_test[:200] / 255.0, y_test[:200]


def model():
    m = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    m.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    return m


def adversarial(x, eps=0.1):
    return np.clip(x + eps * np.sign(np.random.randn(*x.shape)), 0, 1)

# ----- Tangent Prop layer -----
class TangentProp(Layer):
    def call(self, x):
        return x + tf.random.normal(tf.shape(x), stddev=0.1)

# ----- Train model w/ adversarial data -----
m1 = model()
x_adv = adversarial(x_train)
x_mix = np.r_[x_train, x_adv]
y_mix = np.r_[y_train, y_train]
h1 = m1.fit(x_mix, y_mix, epochs=5,
            validation_data=(x_test, y_test), verbose=0)

# ----- Train model w/ Tangent Prop -----
m2 = model()
m2.add(TangentProp())
h2 = m2.fit(x_train, y_train, epochs=5,
            validation_data=(x_test, y_test), verbose=0)

print(f"TangentProp Model Accuracy: {m2.evaluate(x_test, y_test, verbose=0)[1] * 100:.2f}%")

# ----- Tangent Distance Classifier -----
flat = lambda x: x.reshape(len(x), -1)

def tangent_nn(train_x, train_y, test_x):
    return np.array([
        train_y[np.argmin(np.linalg.norm(flat(train_x) - i, axis=1))]
        for i in flat(test_x)
    ])

acc_td = np.mean(tangent_nn(x_train, y_train, x_test) == y_test) * 100
print(f"Tangent Distance Classifier Accuracy: {acc_td:.2f}%")

# ----- Plot -----
plt.plot(h1.history['loss'], '--', label='Adv Loss')
plt.plot(h2.history['loss'], '--', label='TangentProp Loss')
plt.legend()
plt.title("Training Loss")
plt.show()
