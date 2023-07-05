import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping
import time

# Function to preprocess data
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

# Function to create dataset for LSTM model
def create_dataset(dataset, lookback):
    data_X, data_y = [], []
    for i in range(len(dataset) - lookback):
        data_X.append(dataset[i:i+lookback])
        data_y.append(dataset[i+lookback])
    return np.array(data_X), np.array(data_y)

# Function to build and train the LSTM model
def build_model(train_X, train_y, num_units, dropout_rate):
    model = Sequential()
    model.add(LSTM(num_units, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
    model.add(LSTM(num_units))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    early_stopping = EarlyStopping(patience=10)  # Early stopping to prevent overfitting
    model.fit(train_X, train_y, epochs=100, batch_size=32, callbacks=[early_stopping])
    return model

# Function to make predictions using the trained model
def make_predictions(model, data):
    predictions = model.predict(data)
    return predictions

# Function to execute trades based on trading signals
def execute_trades(signal):
    # Add your trade execution logic here
    if signal == 1:
        print("Executing buy order")
        # Execute buy order
    elif signal == -1:
        print("Executing sell order")
        # Execute sell order
    else:
        print("No trade signal")

# Load and preprocess the historical price data
data = pd.read_csv('C:\\ANN\\PriceData.csv')  # Replace 'C:\\ANN\\PriceData.csv' with your data file path
prices = data['Close'].values.reshape(-1, 1)  # Assuming the 'Close' column contains price data

scaled_prices, scaler = preprocess_data(prices)

# Split the data into training and testing sets
train_size = int(len(scaled_prices) * 0.8)
train_data = scaled_prices[:train_size]
test_data = scaled_prices[train_size:]

# Prepare the data for LSTM model
lookback = 30  # Number of previous prices to consider
train_X, train_y = create_dataset(train_data, lookback)
test_X, test_y = create_dataset(test_data, lookback)

# Define the model architecture and train the LSTM model
num_units = 50  # Number of LSTM units
dropout_rate = 0.2  # Dropout rate for regularization
model = build_model(train_X, train_y, num_units, dropout_rate)

# Save the model as a frozen model
model.save("frozen_model.h5")

# Real-time trading loop
while True:
    # Retrieve the latest price data
    latest_prices = np.array([data['Close'].iloc[-lookback:]])
    scaled_latest_prices = scaler.transform(latest_prices)

    # Prepare the latest data for prediction
    latest_X = np.reshape(scaled_latest_prices, (1, lookback, 1))

    # Make predictions using the trained model
    prediction = make_predictions(model, latest_X)

    # Inverse transform the prediction to get the actual price
   
