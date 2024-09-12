#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def deepar_forecast(time_series_data, time_step, epochs, batch_size):
    """
    Fit a DeepAR-like model and return predictions.
    
    Parameters:
    - time_series_data: Pandas DataFrame with 'SP500' column.
    - time_step: Number of timesteps for the DeepAR-like input.
    - epochs: Number of training epochs for the model.
    - batch_size: Batch size for training the model.

    Returns:
    - Predictions from the DeepAR-like model (inverse scaled).
    - Corresponding dates for the test set.
    """

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(time_series_data['SP500'].values.reshape(-1, 1))

    # Prepare data for the model
    def create_dataset(dataset, time_step=1):
        X, Y = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset[i:(i + time_step), 0]
            X.append(a)
            Y.append(dataset[i + time_step, 0])
        return np.array(X), np.array(Y)

    X, y = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Define DeepAR-like model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    # Make predictions
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)

    # Get the corresponding test dates
    test_dates = time_series_data.index[-len(predictions):]

    return predictions.flatten(), test_dates

