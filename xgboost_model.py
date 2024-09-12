#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# In[ ]:


def xgboost_forecast(time_series_data, lag, test_size):
    """
    Fit an XGBoost model and return predictions and MSE.

    Parameters:
    - time_series_data: Pandas DataFrame with 'DATE' and 'SP500' columns.
    - lag: Number of lag features to use for supervised learning.
    - test_size: Proportion of data to use as the test set.

    Returns:
    - Predictions for the test set.
    - Mean Squared Error for the model.
    - Test set dates for plotting.
    """

    # Prepare data with lag features
    def create_lag_features(data, lag=1):
        df = data.copy()
        for i in range(1, lag + 1):
            df[f'lag_{i}'] = df['SP500'].shift(i)
        return df.dropna()

    # Create lag features
    data_with_lags = create_lag_features(time_series_data, lag=lag)
    x = data_with_lags.drop('SP500', axis=1)
    y = data_with_lags['SP500']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, shuffle=False)

    # Fit XGBoost model
    model = XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.1)
    model.fit(X_train, y_train)

    # Predict and calculate MSE
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # Return predictions, MSE, and the test set index (for plotting)
    return y_pred, mse, X_test.index

