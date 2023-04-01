
import datahack_finbert
import requests
import streamlit as st
from streamlit_lottie import st_lottie

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
import pandas as pd
st.title("Stock Sentiment Analysis")

bank_name = st.text_input("Enter name of the bank")
if st.button("Determine Sentiment"):
    #@st.cache_data(ttl=3600,show_spinner=False)
    def writeSentiment():
        return datahack_finbert.getSentiment(bank_name)
        
    with st.spinner(text="Analyzing news articles..."):
        search_res = writeSentiment()
    i=0
    indSentiment = []
    for res in search_res[0]['Sentiment']:
        indSentiment.append(res['label'])

    def finalSentiment():
        neutral = 0
        negative = 0
        positive = 0

        for result in indSentiment:
            if result == 'Neutral':
                neutral+=1
            elif result == 'Negative':
                negative+=1
            elif result == 'Positive':
                positive+=1

        overallLabel = ""
        if max(neutral,negative,positive) == positive:
            overallLabel = "Positive"
        elif max(neutral,negative,positive) == negative:
            overallLabel = "Negative"
        elif max(neutral,negative,positive) == neutral:
            if max(negative,positive) == positive:
                overallLabel = "Positive"
            else: overallLabel = "Negative"
        
        return overallLabel
    sen = finalSentiment()
    if sen=="Positive":
        st.success("Recent news articles indicate that the stock is likely to go up.")
    elif sen=="Negative":
        st.error("Recent news articles indicate that the stock is likely to go down.")
    elif sen=="Neutral":
        st.warning("Recent news articles indicate that the stock trend is likely to remain stable.")
    tableData = list(zip(search_res[0]['Title'],indSentiment))
    df = pd.DataFrame(tableData,columns=['Title','Sentiment'])

    # Define a function to set the font color based on the cell value
    def colorize(val):
        if val == 'Positive':
            return 'color: green'
        elif val == 'Negative':
            return 'color: red'
        else:
            return 'color: gray'

    # Apply the function to the second column of the DataFrame
    styled_df = df.style.applymap(colorize, subset=pd.IndexSlice[:, 'Sentiment'])
    st.write(styled_df)
    # Display the styled DataFrame in Streamlit

