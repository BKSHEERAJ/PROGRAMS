import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
import matplotlib.pyplot as plt

# Load data
data = yf.download("AAPL", start="2010-01-01", end="2023-01-01")[["Open"]]
dataset = data.values

# Train-test split
train_len = int(len(dataset) * 0.8)

# Scale
scaler = MinMaxScaler()
scaled = scaler.fit_transform(dataset)

# Create sequences function
def create_sequences(data, steps=60):
    X, y = [], []
    for i in range(steps, len(data)):
        X.append(data[i-steps:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Create train/test data
x_train, y_train = create_sequences(scaled[:train_len])
x_test, y_test = create_sequences(scaled[train_len-60:])

# Reshape
x_train = x_train.reshape((-1, 60, 1))
x_test  = x_test.reshape((-1, 60, 1))

# Build GRU model
model = Sequential([
    GRU(50, return_sequences=True, input_shape=(60,1)),
    Dropout(0.2),
    GRU(50),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Predict
pred = model.predict(x_test)
pred = scaler.inverse_transform(pred)

# Actual test values
y_test_actual = dataset[train_len:]

# Metrics
rmse = np.sqrt(mean_squared_error(y_test_actual, pred))
mae = mean_absolute_error(y_test_actual, pred)
print("RMSE:", rmse)
print("MAE:", mae)

# Plot
valid = data.iloc[train_len:].copy()
valid["Predictions"] = pred

plt.figure(figsize=(12,6))
plt.plot(data["Open"], label="Train")
plt.plot(valid[["Open","Predictions"]])
plt.legend()
plt.show()

print(valid)
