import datahack_finbert
import streamlit as st
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

