import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

@st.cache_data
def recommend_stock_price(ticker):
    # Load stock data from Yahoo Finance
    stock_data = yf.Ticker(ticker).history(period="max")
    stock_data = stock_data['Close']
    stock_data.index = pd.to_datetime(stock_data.index)
    # Train a model to predict the closing stock price using SARIMAX
    model = SARIMAX(stock_data, order=(1,1,1), seasonal_order=(1,1,1,12))
    model_fit = model.fit(disp=0)

    # Make predictions for the next 30 days
    forecast = model_fit.forecast(steps=31)

    # Create a date range for the next 30 days
    last_date = stock_data.index[-1]
    date_range = pd.date_range(last_date, periods=31, freq='D')[1:]

    # Create a DataFrame with the predicted values
    predicted_data = pd.DataFrame({
        'Date': date_range,
        'Predicted Price': forecast[1:]
    })
    predicted_data.set_index('Date', inplace=True)

    return stock_data, predicted_data



