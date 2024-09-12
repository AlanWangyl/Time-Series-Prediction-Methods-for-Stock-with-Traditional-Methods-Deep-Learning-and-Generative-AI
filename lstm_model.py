#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def lstm_forecast(time_series_data, time_step, epochs, batch_size):
    """
    Fit an LSTM model and return predictions.

    Parameters:
    - time_series_data: Pandas DataFrame with 'SP500' column.
    - time_step: Number of timesteps for the LSTM input.
    - epochs: Number of training epochs for the LSTM.
    - batch_size: Batch size for training the LSTM.

    Returns:
    - Test predictions from the LSTM model (inverse scaled).
    - Corresponding dates for the test set.
    """

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(time_series_data['SP500'].values.reshape(-1, 1))

    # Split the data into train and test sets
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    # Prepare the data for LSTM
    def create_dataset(dataset, time_step=1):
        X, Y = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset[i:(i + time_step), 0]
            X.append(a)
            Y.append(dataset[i + time_step, 0])
        return np.array(X), np.array(Y)

    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    # Reshape input to be [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile and train the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)

    # Make predictions
    test_predict = model.predict(X_test)

    # Inverse transform predictions back to original scale
    test_predict = scaler.inverse_transform(test_predict)

    # Return the predictions and the corresponding test dates for plotting
    test_dates = time_series_data.index[-len(test_predict):]

    return test_predict.flatten(), test_dates

