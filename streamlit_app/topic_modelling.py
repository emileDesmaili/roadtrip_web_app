from newsapi import NewsApiClient

from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pyLDAvis
import pyLDAvis.sklearn
from sklearn.decomposition import LatentDirichletAllocation
from datetime import datetime, timedelta

def LDA_viz(query):
    newsapi = NewsApiClient(api_key='57b09c8dcd91403d98299a5b3fc6607a')

    # /v2/top-headlines
    all_articles = newsapi.get_everything(q=query,
                                        from_param=datetime.now()-timedelta(days = 2),
                                        to=datetime.now(),
                                        language='en',
                                        sort_by='relevancy',
                                        page=1)
    desc = []
    date=[]
    for article in all_articles['articles']:
        desc.append(article['description'])
        date.append(article['publishedAt'])

    news_df = pd.DataFrame(desc,index=date,columns=['description']).sort_index(ascending=False)

    vectorizer = TfidfVectorizer(min_df=2, analyzer='word', ngram_range=(1, 2), stop_words='english')

    tfidf = vectorizer.fit_transform(news_df['description'])

    #nmf = NMF(n_components=5, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)



    lda_tfidf = LatentDirichletAllocation(n_components=5, random_state=0)
    lda_tfidf.fit(tfidf)
    viz_data = pyLDAvis.sklearn.prepare(lda_tfidf, tfidf, vectorizer)
    return pyLDAvis.prepared_data_to_html(viz_data)