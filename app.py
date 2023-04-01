import datahack_finbert
import streamlit as st
st.title("Prediction!!!!!!!")
bank_name = st.text_input("Enter name of the bank")
if st.button("Find Sentiment"):
    sentiment = datahack_finbert.getSentiment(bank_name)
    st.write(sentiment)
