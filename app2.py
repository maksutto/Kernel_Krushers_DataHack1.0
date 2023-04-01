import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
import datahack_finbert
import requests
import streamlit as st
from streamlit_lottie import st_lottie

data=pd.read_csv("C:\Users\shrey\Desktop\college\datahack 2023\NSE_BANKING_SECTOR.csv")


st.set_page_config(layout="wide")
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

rad = st.sidebar.radio("**Navigation**",["Sentiment","Prediction","Recommendation"])

if rad == "Sentiment":
    st.markdown("<h1 style='text-align: center; color: white;'>Sentiment Analysis</h1>", unsafe_allow_html=True)
    with st.container():
        bank_name = st.selectbox("Enter name of the bank",[1,2,3])
        if st.button("Find Sentiment"):
            sentiment = datahack_finbert.getSentiment(bank_name)
            st.write(sentiment)
    with st.container():
        st.write("---")
        one, two, three = st.columns(3)
        with two:
            lottie_url = "https://assets7.lottiefiles.com/packages/lf20_zNb81tkLPk.json"
            lottie_json = load_lottieurl(lottie_url)
            st_lottie(lottie_json, height = 400)
    with st.container():
        st.write(" ")
        st.write("Written shit here")
    
if rad == "Prediction":
    st.markdown("<h1 style='text-align: center; color: white;'>Prediction</h1>", unsafe_allow_html=True)
    with st.container():
        bank_name = st.selectbox("Enter name of the bank",[1,2,3])
    with st.container():
        st.write("---")
        left, right = st.columns(2)
        with left:
            st.write("Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.")
        with right:
            lottie_url = "https://assets6.lottiefiles.com/packages/lf20_hxXmZsjAZj.json"
            lottie_json = load_lottieurl(lottie_url)
            st_lottie(lottie_json)
            
            
if rad == "Recommendation":
    st.markdown("<h1 style='text-align: center; color: white;'>Recommendation</h1>", unsafe_allow_html=True)
    # with st.container():
    #     graph_bank=st.selectbox("Enter name of bank",[1,2,3])
    with st.container():
        # st.write("---")
        left2, right2 = st.columns(2)
        with left2:
            st.write(" ")
            st.write("**Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.**")
        with right2:
            st.write(" ")
            lottie_url = "https://assets6.lottiefiles.com/packages/lf20_pvso4otf.json"
            lottie_json = load_lottieurl(lottie_url)
            st_lottie(lottie_json,height = 300)
