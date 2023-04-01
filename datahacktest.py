import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
import io
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy

df = pd.read_csv("C:/Users/shrey/Desktop/college/datahack 2023/NSE_BANKING_SECTOR.csv")
def graph2(bank):
  j=0;i=0
  for r in range(41231):
    if(df['SYMBOL'][r]==bank):
      if(j==0):
        i=r;
      j=r;
  x=df.iloc[i:j,4:8]
  y=df.iloc[i:j,8:9]
  #print(x)
  minmax_x= MinMaxScaler()
  minmax_x = minmax_x.fit_transform(x)
  bank_data = df[df['SYMBOL'] == bank]
  #plt.figure(figsize=(15, 10))
  #plt.plot(bank_data["DATE"], bank_data["CLOSE"])
  from statsmodels.tsa.seasonal import seasonal_decompose
  result = seasonal_decompose(bank_data["CLOSE"], 
                              model='multiplicative', period= 25)
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
  
  #fig = plt.figure()  
  #fig = result.plot()  
  #fig.set_size_inches(15, 10)
  #pd.plotting.autocorrelation_plot(bank_data["CLOSE"])
  from statsmodels.graphics.tsaplots import plot_pacf
  #plot_pacf(bank_data["CLOSE"], lags = 100)
  p, d, q = 14, 1, 2
  from statsmodels.tsa.arima.model import ARIMA
  model = ARIMA(bank_data["CLOSE"], order=(p,d,q))  
  fitted = model.fit()  
  print(fitted.summary())
  pred = fitted.predict(start=len(bank_data), end=len(bank_data)+30)
  pred
  bank_data["CLOSE"].plot(legend=True, label="Training Data", figsize=(15, 10))
  pred.plot(legend=True, label="Predictions")
  plt.grid(True)
  plt.show()
graph2('HDFC')

import streamlit as st
