#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
from prophet import Prophet

def prophet_forecast(time_series_data, steps):
    """
    Fit a Prophet model and return forecast.
    
    Parameters:
    - time_series_data: Pandas DataFrame with 'Date' and 'SP500' columns.
    - steps: Number of future steps to forecast.
    
    Returns:
    - DataFrame with predicted values.
    """
    # Rename columns for Prophet
    time_series_data = time_series_data.reset_index()
    time_series_data.rename(columns={'DATE': 'ds', 'SP500': 'y'}, inplace=True)

    # Fit Prophet model
    model = Prophet()
    model.fit(time_series_data)

    # Make future predictions
    future = model.make_future_dataframe(periods=steps)
    forecast = model.predict(future)
    
    return forecast[['ds', 'yhat']]


# In[ ]:




