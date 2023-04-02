import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.graph_objs as go

@st.cache_data
def predict_stock_price(ticker, opening_price,bank_name):
    # Load HDFC stock data from Yahoo Finance
    stock_data = yf.Ticker(ticker).history(period="max")
    stock_data = stock_data[['Close']]
    stock_data.reset_index(inplace=True)
    stock_data.columns = ['Date', 'Close']
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])

    # Train a model to predict the closing stock price of HDFC bank using SARIMAX
    model = SARIMAX(stock_data['Close'], order=(1,1,1), seasonal_order=(1,1,1,12))
    model_fit = model.fit(disp=0)

    # Make predictions for the next 30 days
    forecast = model_fit.forecast(steps=30, exog=np.array([opening_price]*30).reshape(-1, 1))

    # Create a date range for the next 30 days
    last_date = stock_data['Date'].iloc[-1]
    date_range = pd.date_range(start=last_date, periods=31, freq='D')[1:]

    # Create a DataFrame with the predicted values
    predicted_data = pd.DataFrame({
        'Date': date_range,
        'Predicted Price': forecast
    })
    predicted_data.set_index('Date', inplace=True)

    # Plot the actual and predicted stock prices
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Close'], name='Actual'))
    fig.add_trace(go.Scatter(x=predicted_data.index, y=predicted_data['Predicted Price'], name='Predicted'))
    fig.update_layout(title=f'{bank_name} Stock Price Prediction', xaxis_title='Date', yaxis_title='Stock Price')

    return stock_data, fig



