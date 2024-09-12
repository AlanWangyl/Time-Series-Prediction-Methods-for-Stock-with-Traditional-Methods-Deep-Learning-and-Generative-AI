#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten

def tcn_forecast(time_series_data, time_step, epochs, batch_size):
    """
    Fit a TCN model and return predictions.
    
    Parameters:
    - time_series_data: Pandas DataFrame with 'SP500' column.
    - time_step: Number of timesteps for the TCN input.
    - epochs: Number of training epochs for the TCN model.
    - batch_size: Batch size for training the TCN model.

    Returns:
    - Predictions from the TCN model (inverse scaled).
    - Corresponding dates for the test set.
    """

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(time_series_data['SP500'].values.reshape(-1, 1))

    # Prepare data for the TCN model
    def create_dataset(dataset, time_step=1):
        X, Y = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset[i:(i + time_step), 0]
            X.append(a)
            Y.append(dataset[i + time_step, 0])
        return np.array(X), np.array(Y)

    X, y = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Define TCN model
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, dilation_rate=1, activation='relu', input_shape=(time_step, 1)))
    model.add(Conv1D(filters=64, kernel_size=2, dilation_rate=2, activation='relu'))
    model.add(Conv1D(filters=64, kernel_size=2, dilation_rate=4, activation='relu'))
    model.add(Flatten())
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

