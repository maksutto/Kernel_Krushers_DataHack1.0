
import time

def getSentiment(bankName):
  from transformers import BertTokenizer , BertForSequenceClassification
  from transformers import pipeline
  from bs4 import BeautifulSoup
  import requests
  bank_name = bankName

  search_url = f"https://www.google.com/search?q={bank_name}&sxsrf=APwXEdfVzpfAF54BQQ6e0mr5fI2tPeW97A:1680341954287&source=lnms&tbm=nws&sa=X&ved=2ahUKEwie8KTKsYj-AhWkU2wGHatNBAkQ_AUoAnoECAEQBA&biw=1536&bih=754&dpr=1.25"
  headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36 Edge/16.16299"
  }
  response = requests.get(search_url, headers=headers)
  soup = BeautifulSoup(response.content, "lxml")
  article_links = []
  links = soup.find_all("a")
  for tag in links:
      href = tag.get("href")
      if href.startswith('/url?esrc=s&q=&rct=j&sa=U&url=')==True:
        index = href.find("&ved=")
        if index != -1:
            href = href[:index] 
        article_links.append(href[30::])


  article_data = []

  for link in article_links:
    response_ = requests.get(link, headers=headers)
    soup_ = BeautifulSoup(response_.content, "lxml")
    meta_tag = soup_.find('meta', attrs={'name': 'description'})
    if meta_tag:
      description = meta_tag.get('content')
      article_data.append(description)
      #print(description)


  # if __name__ == '__main__':
  #     while True:
  #         find_data()
  #         time.sleep(600)
  finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
  tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

  nlp = pipeline("sentiment-analysis",model=finbert,tokenizer=tokenizer)

  sentence = article_data

  results = nlp(sentence)
  #print(results)

  neutral = 0
  negative = 0
  positive = 0

  for result in results:
    if result['label'] == 'Neutral':
      neutral+=1
    elif result['label'] == 'Negative':
      negative+=1
    elif result['label'] == 'Positive':
      positive+=1

  overallLabel = ""
  if max(neutral,negative,positive) == positive:
    overallLabel = "Postive"
  elif max(neutral,negative,positive) == negative:
    overallLabel = "Negative"
  elif max(neutral,negative,positive) == neutral:
    if max(negative,positive) == positive:
      overallLabel = "Postive"
    else: overallLabel = "Negative"

  return overallLabel

  #import yfinance as yf
  #msft = yf.Ticker("MSFT")
  #hist = msft.history(period="1mo")
  #print(msft.major_holders)