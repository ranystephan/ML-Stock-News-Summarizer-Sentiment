#!/usr/bin/env python
# coding: utf-8

# ### Three Steps:
#     1- Scraping news from Google News using Beautiful Soup using Stock and Crypto tickers
#     2- Summarizing each article using HuggingFace Transformers
#     3- Calculating Sentiment and exporting to CSV

# 1. Install and Import Baseline Dependencies
# 2. Setup Summarization Model
# 3. Summarize a Single Article
# 4. Building a News and Sentiment Pipeline
#     4.1. Search for Stock News using Google and Yahoo Finance
#     4.2. Strip out unwanted URLs
#     4.3. Search and Scrape Cleaned URLs
#     4.4 Symmarise all Articles
# 5. Adding Sentiment Analysis
# 6. Exporting Results to CSV


#get_ipython().system('pip install transformers #NLP Library')
#get_ipython().system('pip install sentencepiece')

from transformers import PegasusTokenizer, PegasusForConditionalGeneration, TFPegasusForConditionalGeneration
from bs4 import BeautifulSoup
import requests

# Let's load the model and the tokenizer (pyTorch)
model_name = "human-centered-summarization/financial-summarization-pegasus"
tokenizer = PegasusTokenizer.from_pretrained(model_name) #what incodes and decodes our text
model = PegasusForConditionalGeneration.from_pretrained(model_name, resume_download=True) #model itsef


monitored_tickers = ['AAPL', 'BLK', 'SPY', 'BTC']


#4.1. Search for Stock News using Google and Yahoo Finance

def search_for_stock_news_urls(ticker):
    search_url = "https://www.google.com/search?q=yahoo+finance+{}&tbm=nws".format(ticker)
    r = requests.get(search_url)
    soup = BeautifulSoup(r.text, 'html.parser')
    atags = soup.find_all('a') #atags are the links in the google page (from yahoo in this case)
    hrefs = [link['href'] for link in atags]
    return hrefs



raw_urls = {ticker:search_for_stock_news_urls(ticker) for ticker in monitored_tickers}

#4.2. Strip unwanted URLs
import re #refular expressions

excluded_list = ['maps', 'policies', 'preferences', 'accounts', 'support']


def strip_unwanted_urls(urls, exclude_list):
    val = []
    for url in urls:
        if 'https://' in url and not any(exclude_word in url for exclude_word in exclude_list):
            res = re.findall(r'(https?://\S+)', url)[0].split('&')[0]
            val.append(res)
    return list(set(val))


cleaned_urls = {ticker:strip_unwanted_urls(raw_urls[ticker], excluded_list) for ticker in monitored_tickers}


def scrape_and_process(URLs):
    ARTICLES = []
    for url in URLs:
        r = requests.get(url)
        soup  = BeautifulSoup(r.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = [paragraph.text for paragraph in paragraphs]
        words = ' '.join(text).split(' ')[:350]
        ARTICLE = ' '.join(words)
        ARTICLES.append(ARTICLE)
    return ARTICLES


articles = {ticker:scrape_and_process(cleaned_urls[ticker]) for ticker in monitored_tickers} 

#Summarize all Articles
def summarize(articles):
    summaries = []
    for article in articles:
        input_ids = tokenizer.encode(article, return_tensors = 'pt')
        output = model.generate(input_ids, max_length = 55, num_beams = 5, early_stopping = True)
        summary = tokenizer.decode(output[0], skip_special_tokens = True)
        summaries.append(summary)
    return summaries


summaries = {ticker:summarize(articles[ticker]) for ticker in monitored_tickers}


# # Adding Sentiment Analsis Pipeline

from transformers import pipeline
sentiment = pipeline('sentiment-analysis')

scores = {ticker: sentiment(summaries[ticker]) for ticker in monitored_tickers}

def create_output_array(summaries, scores, urls):
    output = []
    for ticker in monitored_tickers:
        for counter in range(len(summaries[ticker])):
            output_this = [
                ticker,
                summaries[ticker][counter],
                scores[ticker][counter]['label'],
                scores[ticker][counter]['score'],
                urls[ticker][counter]
            ]
            output.append(output_this)
    return output



final_output = create_output_array(summaries, scores, cleaned_urls)

final_output.insert(0, ['Ticker', 'Summary', 'Label', 'Confidence', 'URL' ])

import csv
with open('assetsummaries.csv', mode = 'w', newline = '') as f:
    csv_writer = csv.writer(f, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
    csv_writer.writerows(final_output)




