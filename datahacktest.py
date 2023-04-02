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
import streamlit as st
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

df = pd.read_csv("C:/Users/shrey/Desktop/college/datahack 2023/NSE_BANKING_SECTOR.csv")
def graph2(bank):
  j=0;i=0
  p = []
  for r in range(41231):
    if(df['SYMBOL'][r]==bank):
      if(j==0):
        i=r
      j=r
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
#   print(fitted.summary())
  for l in range(len(bank_data)):
      pred = 0
  pred = fitted.predict(start=len(bank_data), end=len(bank_data)+30)
  pred
  p=pd.Series(np.array(pred))
  p=p.tolist()
  #bank_data["CLOSE"].plot(legend=True, label="Training Data", figsize=(15, 10))
  # pred.plot(legend=True, label="Predictions")
#   plt.grid(True)
#   plt.show()
  # df2=pd.DataFrame()
  # df2['close2']=bank_data["CLOSE"]
  # df2['pred2']=pd.Series(np.array(pred))
  # df2["date"] = pd.Series(np.array(bank_data["DATE"]))
  # st.line_chart(df2,x='date',y=['close2','pred2'])
  return (p[1], p[-1])
#   print((bank_data['DATE'][0]))
  
#   df3 = pd.DataFrame()
#   df3["date"] = pd.Series(np.array(bank_data["DATE"]))
#   df3["preds"] = pd.Series(np.array(pred))
#   fig, ax = plt.subplots(figsize=(15, 10))
#   bank_data["CLOSE"].plot(ax=ax, legend=True, label="Training Data")
#   plt.plot(df3["date"], df3["preds"], color="red", label="Predictions")
#   plt.xlabel("Date")
#   plt.ylabel("Closing Price")
#   plt.title("Bank Stock Price Prediction")
#   plt.legend()
#   st.write(fig)
# graph2('HDFC')
names=[]
max1=0;max2=0;max3=0;j=0
m=[]
n=[]
names=[df['SYMBOL'].unique()]
for i in range(36):
  [j,k] = graph2(names[0][i])
  if((k-j)>max1 and len(m)<3):
      max1 = (k-j)
      m.append(names[0][i])

  elif((k-j)>max2 and len(m)<3):
      max2 = (k-j)  
      m.append(names[0][i])

  elif((k-j)>max3 and len(m)<3):
      max3 = (k-j)  
      m.append(names[0][i])
print(max3)
print(max2)
print(max1)
#   print(names[0][i])
#   [j,k]=graph2(names[0][i])
#   m.append(k-j)
#   n.append(names[0][i])
# z=zip(m,n)
# print(list(z))







