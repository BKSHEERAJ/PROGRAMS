import numpy as np

data = np.array([[2, 3], [1, 4], [3, 5], [4, 2]])
dataset = data
labels = np.array([0, 0, 1, 1])

x_train = dataset
y_train = labels

class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = None

    def fit(self, x_train, y_train):
        n_features = x_train.shape[1]
        self.w = np.zeros(n_features)
        self.b = 0.0

        for _ in range(self.epochs):
            for x, y in zip(x_train, y_train):
                pred = self.predict(x)
                error = y - pred
                self.w += self.learning_rate * error * x
                self.b += self.learning_rate * error

    def predict(self, x):
        z = np.dot(x, self.w) + self.b
        return int(z >= 0)

# Training the model
model = Perceptron(learning_rate=0.01, epochs=100)
model.fit(x_train, y_train)

# Making predictions
x_test = np.array([2, 4])
pred = model.predict(x_test)

# Printing output
print("Prediction:", pred)
