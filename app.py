
import datahack_finbert
import requests
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie
import yfinance as yf
from pandas_datareader import data as pdr
import plotly.figure_factory as ff
import plotly.express as px
from datetime import datetime
from dateutil.relativedelta import relativedelta
import prediction
import experimental
import recommendation

st.set_page_config(layout="wide")
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

rad = st.sidebar.radio("**Navigation**",["Sentiment","Prediction","Recommendation","Custom Fitting"])

if rad == "Sentiment":
    st.markdown("<h1 style='text-align: center; color: white;'>Sentiment Analysis</h1>", unsafe_allow_html=True)
    import pandas as pd

    with st.container():
        st.write("---")
        one, two, three = st.columns(3)
        with two:
            lottie_url = "https://assets7.lottiefiles.com/packages/lf20_zNb81tkLPk.json"
            lottie_json = load_lottieurl(lottie_url)
            st_lottie(lottie_json, height = 200)

    with st.container():
        bank_name = st.selectbox("Enter name of the bank",["HDFC","Yes Bank","ICICI"])
        if st.button("Determine Sentiment"):
            #@st.cache_data(ttl=3600,show_spinner=False)
            def writeSentiment():
                return datahack_finbert.getSentiment(bank_name)
                
            with st.spinner(text="Analyzing news articles..."):
                    search_res = writeSentiment()

            def get_start_date():
                date = datetime.now() - relativedelta(years=5)
                return date.strftime('%Y-%m-%d')

            def is_available(info, key) -> bool:
                if key in info and info[key] != None:
                    return True

                return False

            yf.pdr_override()
            if bank_name == "HDFC":
                ticker = "HDFCBANK.NS"
            elif bank_name == "Yes Bank":
                ticker = "YESBANK.NS"
            elif bank_name == "ICICI":
                ticker = "ICICIBANK.NS"
            df = pdr.get_data_yahoo(ticker, start=get_start_date())
            df["100 day rolling"] = df["High"].rolling(window=100).mean()
            df["200 day rolling"] = df["High"].rolling(window=200).mean()

            company_info = yf.Ticker(ticker).info

            info_container = st.container()
            col1, col2, col3 = info_container.columns(3)

            with col1:
                st.metric(
                    label=company_info["shortName"],
                    value="%.2f" % company_info["regularMarketPrice"],
                    delta="%.2f" % (company_info["regularMarketPrice"] -
                                    company_info["regularMarketPreviousClose"]),
                )
            with col2:
                st.metric(label="Today's High", value="%.2f" % company_info["dayHigh"] if (is_available(company_info, "dayHigh")) else "N/A")
            with col3:
                st.metric(label="Today's Low", value="%.2f" % company_info["dayLow"] if (is_available(company_info, "dayLow")) else "N/A")

            col4, col5, col6 = info_container.columns(3)

            with col4:
                st.metric(
                    label="Revenue Growth (YoY)",
                    value=("%.2f" % (company_info["revenueGrowth"]*100)+"%" if (is_available(company_info, "revenueGrowth")) else "N/A")
                )
            with col5:
                st.metric(label="PE Ratio", value="%.2f" % company_info["trailingPE"] if (is_available(company_info, "trailingPE")) else "N/A")
            with col6:
                st.metric(label="PB Ratio", value="%.2f" % company_info["priceToBook"] if (is_available(company_info, "priceToBook")) else "N/A")

            option = info_container.selectbox('Choose Chart Type',
                                            ('Candlestick chart', 'Line chart'))

            if option == "Candlestick chart":
                fig = ff.create_candlestick(
                    dates=df.index, open=df['Open'], close=df['Close'], high=df['High'], low=df['Low'])
            else:
                fig = px.line(df, x=df.index, y=[
                            "High", "100 day rolling", "200 day rolling"])

            fig.update_layout(xaxis_title="Time", yaxis_title="Stock Value")

            info_container.plotly_chart(
                fig, use_container_width=True, theme="streamlit")

            info_container.markdown("### Company Info")
            info_container.write(company_info["longBusinessSummary"] if is_available(company_info, "longBusinessSummary") else "N/A")

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
    
    
if rad == "Prediction":
    st.markdown("<h1 style='text-align: center; color: white;'>Prediction</h1>", unsafe_allow_html=True)
    with st.container():
        bank_name = st.selectbox("Enter name of the bank",["HDFC","Yes Bank","ICICI"])
    with st.container():
        st.write("---")
        left, right = st.columns(2)
        with left:
            if bank_name == "HDFC":
                ticker = "HDFCBANK.NS"
            elif bank_name == "Yes Bank":
                ticker = "YESBANK.NS"
            elif bank_name == "ICICI":
                ticker = "ICICIBANK.NS"
            prediction.predict_stock_price(ticker)
        with right:
            lottie_url = "https://assets6.lottiefiles.com/packages/lf20_hxXmZsjAZj.json"
            lottie_json = load_lottieurl(lottie_url)
            st_lottie(lottie_json)
            
            
if rad == "Recommendation":
    st.markdown("<h1 style='text-align: center; color: white;'>Recommendation</h1>", unsafe_allow_html=True)
    with st.container():
        # st.write("---")
        left2, right2 = st.columns(2)
        with left2:
            
            # Define the tickers of the 3 best performing banks according to yfinance
            tickers = ['HDFCBANK.NS', 'ICICIBANK.NS','AXISBANK.NS']

            # Create empty lists for the actual and predicted dataframes
            actual_data = []
            predicted_data = []
            with st.spinner('Loading data...'):
                # Loop through the tickers and predict the stock prices
                for ticker in tickers:
                    actual, predicted = recommendation.recommend_stock_price(ticker)
                    actual_data.append(actual)
                    predicted_data.append(predicted)

            # Concatenate the actual and predicted dataframes for each ticker
            actual_data = pd.concat(actual_data, axis=1)
            predicted_data = pd.concat(predicted_data, axis=1)

            # Plot the actual and predicted stock prices
            chart_data = pd.concat([actual_data, predicted_data], axis=1)

            # Set chart title and axes labels
            plt.title('Bank Stock Price Prediction')
            plt.xlabel('Date')
            plt.ylabel('Stock Price')

            # Plot the combined actual and predicted stock prices for all tickers
            st.line_chart(chart_data)

        with right2:
            st.write(" ")
            lottie_url = "https://assets6.lottiefiles.com/packages/lf20_pvso4otf.json"
            lottie_json = load_lottieurl(lottie_url)
            st_lottie(lottie_json,height = 300)
        
        st.success("After analyzing the trends of various banks in the country, it has been found that stocks of HDFC Bank are trending upwards the most")

if rad == "Custom Fitting":
    st.markdown("<h1 style='text-align: center; color: white;'>Customization</h1>", unsafe_allow_html=True)

    opening_price = st.experimental_get_query_params().get("opening_price", None)
    opening_price = float(opening_price[0]) if opening_price else None
    with st.container():
        bank_name = st.selectbox("Enter name of the bank",["HDFC","Yes Bank","ICICI"])
    if bank_name == "HDFC":
        ticker = "HDFCBANK.NS"
    elif bank_name == "Yes Bank":
        ticker = "YESBANK.NS"
    elif bank_name == "ICICI":
        ticker = "ICICIBANK.NS"
    if opening_price:
        stock_data, fig = experimental.predict_stock_price(ticker, opening_price)
        st.plotly_chart(fig)

        # Convert the DatetimeIndex to a regular column
        stock_data['Date'] = stock_data['Date'].dt.date
        st.experimental_data_editor(stock_data)
    else:
        opening_price = st.text_input(f'Enter the opening price of {bank_name} stock', value='2400')
        if st.button('Predict'):
            stock_data, fig = experimental.predict_stock_price(ticker, float(opening_price),bank_name)
            st.plotly_chart(fig)

            # Convert the DatetimeIndex to a regular column
            stock_data['Date'] = stock_data['Date'].dt.date
            st.experimental_data_editor(stock_data)



