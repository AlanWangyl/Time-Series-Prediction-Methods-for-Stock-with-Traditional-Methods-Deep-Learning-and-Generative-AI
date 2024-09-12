#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, LeakyReLU, Reshape, Input
from tensorflow.keras.optimizers import Adam

def gan_forecast(time_series_data, time_step, epochs, batch_size):
    """
    Fit a GAN model and return predictions.

    Parameters:
    - time_series_data: Pandas DataFrame with 'SP500' column.
    - time_step: Number of timesteps for the GAN input.
    - epochs: Number of training epochs for the GAN.
    - batch_size: Batch size for training.

    Returns:
    - Generated prediction from the GAN model.
    """

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(time_series_data['SP500'].values.reshape(-1, 1))

    # Prepare data for GAN
    def create_dataset(dataset, time_step=1):
        X, Y = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset[i:(i + time_step), 0]
            X.append(a)
            Y.append(dataset[i + time_step, 0])
        return np.array(X), np.array(Y)

    X_train, y_train = create_dataset(scaled_data, time_step)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    # Build Generator
    def build_generator():
        model = Sequential()
        model.add(Dense(100, input_dim=time_step))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(time_step, activation='tanh'))
        model.add(Reshape((time_step, 1)))  # Generator output is shaped as (time_step, 1)
        return model

    # Build Discriminator
    def build_discriminator():
        model = Sequential()
        model.add(LSTM(50, input_shape=(time_step, 1)))
        model.add(Dense(1, activation='sigmoid'))  # Output is a single prediction (real/fake)
        return model

    # Build and compile the discriminator
    discriminator = build_discriminator()
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

    # Build the generator
    generator = build_generator()

    # The generator takes noise as input and generates data
    z = Input(shape=(time_step,))
    generated_data = generator(z)

    # For the combined model, we will only train the generator
    discriminator.trainable = False

    # The discriminator takes generated data as input and determines validity
    validity = discriminator(generated_data)

    # The combined model (stacked generator and discriminator)
    combined = Model(z, validity)
    combined.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

    # Training the GAN
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        # Train Discriminator
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_data = X_train[idx]
        noise = np.random.normal(0, 1, (batch_size, time_step))
        gen_data = generator.predict(noise)
        d_loss_real = discriminator.train_on_batch(real_data, valid)
        d_loss_fake = discriminator.train_on_batch(gen_data, fake)

        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, time_step))
        g_loss = combined.train_on_batch(noise, valid)

        if epoch % 100 == 0:
            print(f"{epoch} [D loss: {d_loss_real[0]} | G loss: {g_loss}]")

    # Make predictions with the trained generator
    noise = np.random.normal(0, 1, (1, time_step))
    generated_prediction = generator.predict(noise)
    generated_prediction = scaler.inverse_transform(generated_prediction.reshape(-1, 1))  # Reshape back and inverse scale

    return generated_prediction.flatten()

