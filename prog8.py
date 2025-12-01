import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Load + pad IMDb data
max_words, maxlen = 10000, 200
(xtr, ytr), (xte, yte) = imdb.load_data(num_words=max_words)
xtr = pad_sequences(xtr, maxlen=maxlen)
xte = pad_sequences(xte, maxlen=maxlen)

# Model
m = Sequential([
    Embedding(max_words, 50, input_length=maxlen),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])

m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

h = m.fit(
    xtr, ytr,
    epochs=5,
    batch_size=128,
    validation_split=0.2,
    verbose=2
)

print("Test Accuracy:", m.evaluate(xte, yte, verbose=0)[1])

# Sentiment prediction
wi = imdb.get_word_index()

def predict(text):
    seq = [wi.get(w, 0) for w in text.lower().split() if wi.get(w, 0) < max_words]
    return m.predict(pad_sequences([seq], maxlen=maxlen))[0][0]

print("Positive:", predict("This movie was fantastic and beautifully made"))
print("Negative:", predict("This movie was boring, slow and a waste of time"))

# Plots
for p, t in [('accuracy', "Accuracy"), ('loss', "Loss")]:
    plt.plot(h.history[p])
    plt.plot(h.history['val_' + p])
    plt.title(t)
    plt.legend(['Train', 'Val'])
    plt.show()
