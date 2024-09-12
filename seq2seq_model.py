#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, RepeatVector, TimeDistributed
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt

def seq2seq_forecast(time_series_data, time_step=10, epochs=10, batch_size=64):
    """
    Fit the new model and return predictions.

    Parameters:
    - time_series_data: Pandas DataFrame with 'SP500' column.
    - time_step: Number of timesteps for the model input.
    - epochs: Number of training epochs for the new model.
    - batch_size: Batch size for training the model.

    Returns:
    - Test predictions (inverse scaled).
    - Corresponding dates for the test set.
    """

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(time_series_data['SP500'].values.reshape(-1, 1))

    # Prepare data for the model
    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset[i:(i + time_step), 0]
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)

    # Split the data into training and testing sets
    training_size = int(len(data_scaled) * 0.67)
    train_data, test_data = data_scaled[0:training_size, :], data_scaled[training_size:len(data_scaled), :]

    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    # Reshape input for the model
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Seq2Seq Model Definition
    inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))

    # Encoder
    encoder_lstm = LSTM(100, activation='relu', return_sequences=False)(inputs)
    encoder_output = RepeatVector(X_train.shape[1])(encoder_lstm)

    # Decoder
    decoder_lstm = LSTM(100, activation='relu', return_sequences=True)(encoder_output)
    decoder_output = TimeDistributed(Dense(1))(decoder_lstm)

    model = Model(inputs, decoder_output)
    model.compile(optimizer="adam", loss="mean_squared_error")

   # Train the model
    model.fit(X_train, X_train, validation_data=(X_test, X_test), epochs=10, batch_size=64, verbose=0)

    # Make predictions
    test_predict = model.predict(X_test)
    test_predict = test_predict[:, -1, :]
    test_predict = scaler.inverse_transform(test_predict)

    # Get the corresponding test dates
    test_dates = time_series_data.index[-len(test_predict):]

    return test_predict.flatten(), test_dates

