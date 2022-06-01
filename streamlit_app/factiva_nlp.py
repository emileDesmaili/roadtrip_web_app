
import datetime
import pandas as pd

import spacy
from spacy import displacy
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, AutoModelForSequenceClassification
import transformers
from sentence_transformers import SentenceTransformer, util

def load_transformers_data():    

    tokenizer_sentiment = AutoTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
    model_sentiment = AutoModelForSequenceClassification.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
    return tokenizer_sentiment, model_sentiment

tokenizer_sentiment, model_sentiment= load_transformers_data()



def get_sentiment2(text,lang,tokenizer_sentiment,model_sentiment):
    classifier = pipeline("sentiment-analysis", model=model_sentiment, tokenizer=tokenizer_sentiment)
    sentiment = classifier(text)

    if lang =='en' and sentiment[0]['label'] =='positive':
        sentiment = '😀 Positive sentiment'
    elif lang =='en' and sentiment[0]['label'] =='negative':
        sentiment = '☹️ Negative sentiment'
    else:
        sentiment = ""
    
    return sentiment


def process_news(search_bar, news_dict, tokenizer_sentiment, model_sentiment):
    # first add tags like sentiment and subject codes
    news_dict_firm = news_dict[search_bar]
    sorted_dates = sorted(list(news_dict_firm.keys()), reverse=True)
    

    for date in sorted_dates:
        lang = news_dict_firm[date]['language_code']
        timestamp = news_dict_firm[date]['publication_date'][:10]
        news_dict_firm[date]['publication_datetime'] = datetime.datetime.fromtimestamp(int(timestamp))
        try:
            snippet = news_dict_firm[date]['snippet']
        except:
            snippet=None
        text = news_dict_firm[date]['body']
        article_text = text


        news_dict_firm[date]['sentiment'] = get_sentiment2(snippet,lang,tokenizer_sentiment=tokenizer_sentiment,model_sentiment=model_sentiment)

            
        news_dict_firm[date]['tags'] = []
        if 'c17' in news_dict_firm[date]['subject_codes']:
            news_dict_firm[date]['tags'].append('💰 Funding')
        if 'c18' in news_dict_firm[date]['subject_codes']:
            news_dict_firm[date]['tags'].append('🏛 Ownership Changes')
        if 'cactio' in news_dict_firm[date]['subject_codes']:
            news_dict_firm[date]['tags'].append('🏦 Corporate Actions')
        if 'c411' in news_dict_firm[date]['subject_codes']:
            news_dict_firm[date]['tags'].append('👨‍💼 Management Changes')
        news_dict_firm[date]['tags'] = ','.join(news_dict_firm[date]['tags'])
    df_news = pd.DataFrame(news_dict_firm).transpose()
        
    return df_news