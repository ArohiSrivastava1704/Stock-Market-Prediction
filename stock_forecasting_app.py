import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# Title
st.title("Stock Market Forecasting")

# Upload File
uploaded_file = st.file_uploader("Upload your Stock Data CSV", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:", data.head())

    # Preprocess the data
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    time_series = data['Close']

    # Plot original data
    st.subheader("Stock Closing Price Over Time")
    plt.figure(figsize=(10, 5))
    plt.plot(time_series, label='Close Price')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    st.pyplot(plt)

    # LSTM Model Training
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(time_series.values.reshape(-1, 1))

    seq_length = 60
    X, y = [], []
    for i in range(len(scaled_data) - seq_length):
        X.append(scaled_data[i:i+seq_length])
        y.append(scaled_data[i+seq_length])
    X, y = np.array(X), np.array(y)

    train_size = int(0.8 * len(X))
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=5, batch_size=32)

    # Forecasting
    forecast = model.predict(X_test)
    forecast = scaler.inverse_transform(forecast)
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Plot Forecast vs Actual
    st.subheader("LSTM Forecast vs Actual")
    plt.figure(figsize=(10, 5))
    plt.plot(actual, label='Actual')
    plt.plot(forecast, label='Forecast', alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('Closing Price')
    plt.legend()
    st.pyplot(plt)
