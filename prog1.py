import numpy as np
import matplotlib.pyplot as plt

# Loading data (random inputs)
input_size = 5
num_samples = 100
data = np.random.rand(num_samples, input_size)                                  

class Neuron:
    def __init__(self, n_features):
        self.w = np.random.randn(n_features)
        self.b = np.random.randn()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        z = np.dot(self.w, x) + self.b
        return self.sigmoid(z)

model = Neuron(input_size)
outputs = np.array([model.forward(x) for x in data])
print(outputs)

sigmoid = lambda x: 1 / (1 + np.exp(-x))
tanh = np.tanh
relu = lambda x: np.maximum(0, x)
softmax = lambda x: np.exp(x) / np.sum(np.exp(x))
x = np.linspace(-5, 5, 1000)

for fn, name in [(sigmoid, "Sigmoid"),(tanh, "Tanh"),(relu, "ReLU"),(softmax, "Softmax")]:
    plt.figure()
    plt.plot(x, fn(x))
    plt.title(name)
    plt.grid(True)
    plt.show()
