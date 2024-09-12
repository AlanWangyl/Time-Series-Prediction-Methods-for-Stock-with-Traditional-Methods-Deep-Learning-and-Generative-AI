#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def arima_forecast(time_series_data, order, steps):
    """
    Fit ARIMA model and return forecast.
    
    Parameters:
    - time_series_data: Pandas DataFrame with time series data.
    - order: Tuple, order of ARIMA (p, d, q).
    - steps: Number of future steps to forecast.
    
    Returns:
    - Pandas Series of predicted values.
    """
    # Ensure 'Value' column is numeric
#     time_series_data['SP500'] = pd.to_numeric(time_series_data['SP500'], errors='coerce')
#     time_series_data.dropna(inplace=True)
    
    # Fit ARIMA model
    model = ARIMA(time_series_data, order=order)
    model_fit = model.fit()

    # Forecast future values
    predictions = model_fit.forecast(steps=steps)
    
    return predictions

