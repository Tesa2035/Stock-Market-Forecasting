import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Streamlit App Title
st.title("📈 Stock Market Prediction using LSTM")

# File Upload
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    # Load and preprocess data
    df = pd.read_csv(uploaded_file)
    df.rename(columns={"Unnamed: 0": "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by="date")

    # Display dataset
    st.subheader("Dataset Overview")
    st.write(df.head())

    # Select numeric columns for scaling
    numeric_cols = ["open", "high", "low", "close", "adjclose", "volume"]
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Create a separate scaler for the "close" column
    close_scaler = MinMaxScaler(feature_range=(0, 1))
    df["close"] = close_scaler.fit_transform(df[["close"]])  # Fit only on close price

    # Plot Closing Price Over Time
    st.subheader("Closing Price Over Time")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["date"], df["close"], label="Normalized Closing Price", color="blue")
    ax.set_xlabel("Date")
    ax.set_ylabel("Closing Price")
    ax.legend()
    st.pyplot(fig)

    # Prepare data for LSTM
    data = df[["close"]].values

    def create_sequences(data, time_step=10):
        X, y = [], []
        for i in range(len(data) - time_step):
            X.append(data[i : i + time_step, 0])
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)

    time_step = 10
    X, y = create_sequences(data, time_step)

    train_size = int(len(X) * 0.9)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    # Reshape input for LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Build LSTM model
    st.subheader("LSTM Model Training")
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X_train, y_train, epochs=5, batch_size=16, verbose=1)

    # Predictions
    y_pred = model.predict(X_test)

    # Inverse transform using only the "close" feature
    y_pred_inv = close_scaler.inverse_transform(y_pred)
    y_test_inv = close_scaler.inverse_transform(y_test.reshape(-1, 1))

    # Model Evaluation
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))

    st.subheader("Model Performance Metrics")
    st.write(f"✅ **Mean Absolute Error (MAE):** {mae:.4f}")
    st.write(f"✅ **Root Mean Squared Error (RMSE):** {rmse:.4f}")

    # Actual vs Predicted Plot
    st.subheader("Actual vs Predicted Stock Prices")
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(df["date"].iloc[-len(y_test):], y_test_inv, label="Actual", color="blue")
    ax2.plot(df["date"].iloc[-len(y_test):], y_pred_inv, label="LSTM Prediction", color="red", linestyle="dashed")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Stock Price")
    ax2.legend()
    st.pyplot(fig2)

    # Future Prediction Input
    st.subheader("Predict Future Stock Prices")
    prev_close = st.number_input("Previous Closing Price", min_value=0.0)
    open_price = st.number_input("Open Price", min_value=0.0)
    high_price = st.number_input("High Price", min_value=0.0)
    low_price = st.number_input("Low Price", min_value=0.0)
    volume = st.number_input("Trading Volume", min_value=0.0)

    if st.button("Predict"):
        user_data = [prev_close]
        input_scaled = close_scaler.transform([user_data])  # Scale only the close price
        sequence = np.array([input_scaled])
        prediction_scaled = model.predict(sequence)
        predicted_price = close_scaler.inverse_transform(prediction_scaled)[0, 0]
        st.success(f"📈 **Predicted Stock Price:** {predicted_price:.2f}")
