#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def sarima_forecast(time_series_data, order, seasonal_order, steps):
    """
    Fit SARIMA model and return forecast.
    
    Parameters:
    - time_series_data: Pandas DataFrame with time series data.
    - order: Tuple, order of SARIMA (p, d, q).
    - seasonal_order: Tuple, seasonal order of SARIMA (P, D, Q, s).
    - steps: Number of future steps to forecast.
    
    Returns:
    - Pandas Series of predicted values.
    """
#     # Ensure 'Value' column is numeric
#     time_series_data['Value'] = pd.to_numeric(time_series_data['Value'], errors='coerce')
#     time_series_data.dropna(inplace=True)
    
    # Fit SARIMA model
    model = SARIMAX(time_series_data, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)

    # Forecast future values
    predictions = model_fit.forecast(steps=steps)
    
    return predictions

