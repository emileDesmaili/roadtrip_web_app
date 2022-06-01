#library imports





import streamlit as st

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import openrouteservice
from openrouteservice import convert
import json
import folium
import pandas as pd
import requests
from fredapi import Fred
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode,JsCode
from st_aggrid.grid_options_builder import GridOptionsBuilder
import datetime









class City:

    def __init__(self,name,duration=0):
        self.name = name
        self.duration = duration

    
    def geocode(self):
        """geocoding method that turns name and state into a lat & long attributes
        """
        geolocator = Nominatim(user_agent="my_app")
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
        loc_string = self.name
        location = geolocator.geocode(loc_string)
        lat = location.latitude
        lon = location.longitude
        self.lat = lat
        self.lon = lon
    


class Route:

    def __init__(self, start, finish):
        self.start = start
        self.finish = finish
    
    def compute_(self):
        client = openrouteservice.Client(key='5b3ce3597851110001cf624857ac41a9e1574cd5aecd7f089e686084')
        start = City(self.start)
        finish = City(self.finish)
        start.geocode()
        finish.geocode()
        coords = ((start.lon, start.lat),(finish.lon, finish.lat))
        res = client.directions(coords)
        geometry = client.directions(coords)['routes'][0]['geometry']
        self.decoded = convert.decode_polyline(geometry)
        self.distance = res['routes'][0]['summary']['distance']
        self.duration = res['routes'][0]['summary']['duration']
        self.distance_txt = "<h4> <b>Distance :&nbsp" + "<strong>"+str(round(res['routes'][0]['summary']['distance']/1000,1))+" Km </strong>" +"</h4></b>"
        self.duration_txt = "<h4> <b>Duration :&nbsp" + "<strong>"+str(round(res['routes'][0]['summary']['duration']/(60*60*24),1))+" Days. </strong>" +"</h4></b>"
    
    def get_price(self):
        fred_api_key = '7294815d2a10429894fa3423865fea22'
        fred = Fred(api_key= fred_api_key)
        date = '01-06-2022'
        oil = fred.get_series('DCOILWTICO',observation_start=date)[-1]
        eurusd = fred.get_series('DEXUSEU',observation_start=date)[-1]
        # assumptions to be made here
        liters_per_km = 20/100
        liters_per_barrel = 159
        usd_per_barrel = oil  
        usd_per_km = liters_per_km * oil * (1/liters_per_barrel)
        self.price = round(self.distance/1000 * usd_per_km * (1/eurusd),1)


    
def get_prices():
    #Alpha vantage API_KEY = 'CJGXXM6MW5QTI080'
    fred_api_key = '7294815d2a10429894fa3423865fea22'
    fred = Fred(api_key= fred_api_key)
    prices_df = pd.DataFrame()
    prices_df['SPX'] = fred.get_series('SP500')
    prices_df['oil'] = fred.get_series('DCOILWTICO')
    prices_df['USD'] = fred.get_series('DTWEXBGS')
    prices_df['infla'] = fred.get_series('T5YIFR')
    return prices_df.dropna(axis=0)



    





def aggrid_display(df):
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_grid_options(rowHeight=40)
    gb.configure_pagination()
    gb.configure_side_bar()
    gb.configure_default_column(
                                min_column_width =3, groupable=True, value=True, enableRowGroup=True,
                                aggFunc="sum", editable=True, enableRangeSelection=True,
                                )
    gb.configure_selection(selection_mode = 'multiple', use_checkbox=False, groupSelectsChildren=True, groupSelectsFiltered=True)
    gridOptions = gb.build()
    # display main dataframe
    if not df.empty:
        grid_response = AgGrid(df, gridOptions=gridOptions, enable_enterprise_modules=True,
        fit_columns_on_grid_load=False, allow_unsafe_jscode= True, data_return_mode = DataReturnMode.FILTERED_AND_SORTED, update_mode=GridUpdateMode.SELECTION_CHANGED,
        height=300, theme="material"
        )
        #df = grid_response['data']
        # selected = grid_response['selected_rows']
        # selected_df = pd.DataFrame(selected)
    else:
        st.info("No itinerary")
    




@st.cache()
def get_news(client, n):
    title = []
    url = []
    date = []
    import xmltodict
    google_actu_url=f"https://news.google.com/rss/search?q=%7B{client}%7D&hl=fr&gl=FR&ceid=FR:fr"
    google_actu_url = google_actu_url.replace(" ", "+")
    response = requests.get(google_actu_url)
    decoded_response = response.content.decode('utf-8')
    response_json = json.loads(json.dumps(xmltodict.parse(decoded_response)))
    for i in range(0,n):
        try:
            url.append(response_json['rss']['channel']['item'][i]['link'])
        except (IndexError,ValueError, KeyError):
            return 'Oops, something went wrong... try again!'
        title.append(response_json['rss']['channel']['item'][i]['title'])
        date.append(response_json['rss']['channel']['item'][i]['pubDate'])
    df = pd.DataFrame([date,title, url]).transpose()
    df[0] = df[0].apply(parser.parse)
    return df.set_index(df[0])

def st_write_news(client,n=5):
    news_df = get_news(client,n=n)
    st.markdown("""
    <style>
    .big-font {
    font-size:14px;
        }
    </style>
    """, unsafe_allow_html=True)
    if isinstance(news_df,str):
        st.write('Oopps something went wrong... try again!')
    else:
        for i in range(0,n):
            url = news_df[2][i]+' target="_blank"'
            text = news_df[1][i]
            string = f"<p class='big-font'><a href={url}>{text}</a> </p>"
            st.markdown(string, unsafe_allow_html=True)

def plot_news_sentiment(client,news_state_full):
    # analyzer = SentimentIntensityAnalyzer()
    # news_df = get_news(client, n)
    # news_df['sentimentBlob'] = news_df[1].apply(lambda news: TextBlob(news).sentiment.polarity)
    # news_df['sentimentVader'] = news_df[1].apply(lambda news: analyzer.polarity_scores(news)['compound'])
    # news_df['sentiment'] = 0.5*news_df['sentimentBlob']+0.5*news_df['sentimentVader']
    # # translator = Translator()
    # # news_df['EN'] = news_df[1].apply(lambda news: translator.translate(news).text)
    # # news_df['sentimentBlob_EN'] = news_df['EN'].apply(lambda news: TextBlob(news).sentiment.polarity)
    # # news_df['sentimentVader_EN'] = news_df['EN'].apply(lambda news: analyzer.polarity_scores(news)['compound'])
    # # news_df['sentiment_EN'] = 0.5*news_df['sentimentBlob_EN']+0.5*news_df['sentimentVader_EN']
    # senti_fig = px.line(news_df.sort_index(), x=0, y=news_df['sentiment'])
    # senti_fig.update_layout(yaxis_title=None, xaxis_title=None) 
    # st.plotly_chart(senti_fig, use_container_width=False)

    #random
    # brownian = np.cumsum(np.random.randn(200))
    # fig = px.line(pd.Series(brownian),y=0)

    split = min(1,len(client)-2)
    client=' '.join(client.split()[:split])
    pytrends = TrendReq(hl='en-US', tz=360) 
    kw_list = [client] # list of keywords to get data 
    pytrends.build_payload(kw_list, cat=0, timeframe='today 12-m')

    data = pytrends.interest_over_time() 
    data = data.reset_index() 


    fig = px.line(data, x="date", y=[client])

    fig.update_traces(line=dict(width=3))



    
    fig.update_layout(yaxis_title=None, xaxis_title=None,
    width=100, height=150) 
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)','paper_bgcolor': 'rgba(0, 0, 0, 0)',})
    # fig.add_annotation(text="FAKE DATA", align="center", opacity=0.20, font=dict(
    #         family="Helvetica",
    #         size=65,
    #         color="Black"
    #         ),
    #         xref="paper", yref="paper",
    #         x=0.5, y=0.8, showarrow=False)
    st.write(f'##### Search Interest Over Time for {client} from Google Trends')
    try:
        st.metric('Google Trend Score',"{:,.1f}".format(data[client].iloc[-1]),delta="{:.0%}".format((data[client].iloc[-1]/data[client].iloc[-2])-1))
    except:
        pass

    st.plotly_chart(fig, use_container_width=True)



def plot_main_page_financials(data_filtered):
    fin_chart1, fin_chart2, fin_chart3 = st.columns(3)
    turnovers = ['Turnover']
    ebitdas = ['EBITDA']
    net_incomes = ['Net Income']
    color = ['rgb(120, 186, 240)']

    fig1 = px.bar(data_filtered, x='Name', y='Turnover', barmode='group',color_discrete_sequence=color
    )
    fig1.update_layout(title ='<b>Turnover</b>',title_x=0.4,xaxis_title=None,yaxis_title=None) 
    fig1.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)','paper_bgcolor': 'rgba(0, 0, 0, 0)',})
    fig1.update_layout(showlegend=False, xaxis_tickangle=-90)
    fig1.update_layout(
    autosize=False,
    width=500,
    height=450,
    margin=dict(
        l=40,
        r=40,
        b=40,
        t=40,
        pad=4
    )
    )
    fig2 = px.bar(data_filtered, x='Name', y='EBITDA', barmode='group',
    color_discrete_sequence=color)
    fig2.update_layout(title ='<b>EBITDA</b>',title_x=0.4,xaxis_title=None,yaxis_title=None) 
    fig2.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)','paper_bgcolor': 'rgba(0, 0, 0, 0)',})
    fig2.update_layout(showlegend=False, xaxis_tickangle=-90)
    fig2.update_layout(
    autosize=False,
    width=500,
    height=450,
    margin=dict(
        l=40,
        r=40,
        b=40,
        t=40,
        pad=4
    )
    )

    fig3 = px.bar(data_filtered, x='Name', y='Net Income', barmode='group',
    color_discrete_sequence=color)
    fig3.update_layout(title ='<b>Net Income</b>',title_x=0.4,xaxis_title=None,yaxis_title=None)
    fig3.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)','paper_bgcolor': 'rgba(0, 0, 0, 0)',})
    fig3.update_layout(showlegend=False, xaxis_tickangle=-90)
    fig3.update_layout(
    autosize=False,
    width=500,
    height=450,
    margin=dict(
        l=40,
        r=40,
        b=40,
        t=40,
        pad=4
    )
    )

    col1,col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(fig1,use_container_width=True)
    with col2:
        st.plotly_chart(fig2,use_container_width=True)
    with col3:
        st.plotly_chart(fig3,use_container_width=True)
















    




def create_tagbox(tag_state,data):
    """creates a streamlit box from tags in a dataframe

    Args:
        tag_state (string): SessionState variable that is firm specific

    Returns:
        streamlit object: 
    """
    field_data = np.append(data['Tags'].unique(),[['Great Place to Work','Tech','AI','ESG','Healthcare']])
    field_data = set(np.append(field_data,[st.session_state[tag_state]]))
    container = st.container()
    all = st.checkbox(f"Select all tags", value = True)
    if all:
        selected_options = container.multiselect(f"Select one or more Tag:",
         field_data,field_data)
    else:
        selected_options =  container.multiselect(f"Select one or more Tag:",
        field_data)
    return selected_options




@st.cache(show_spinner=False)
def plot_wordcloud(topic,data):
    USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:65.0) Gecko/20100101 Firefox/66.0")
    my_stops = ['Accueil', 'Ligne','contacter','Linkedin','Recherche','wikipedia','avis','annonce','marque','annonces','hiver','automne','officiel','site',
    'acheter','comme','com','description','rejoindre','Wikipédia','quels','bienvenue','chez','Paris','facebook','twitter','Recherches','abandonne','images',
    'Société','bilan','infogreffe','kbis','siret','siren','chiffre',"d'affaires",'tva','societe','ca','bilans','carrieres','adresse','établissements','entreprise','résultat',
    'toute','fr','contact','infos','ouverture','fermeture','homme','femme','groupe','abonnement','bon','code promo','promo','code','recrutement','général','Téléphone',
    'horaires','promotion',"c'bon",'corporation','company','corp','ltd','limited','email']
    data_filtered = data[data["Name"]==topic].reset_index(drop=True)
    address = str(data_filtered['City'][0])
    final_stopwords_list = stopwords.words('english') + stopwords.words('french') + topic.split() + my_stops + address.replace('-', ' ').split()
    topic=topic.replace(' ','+')
    headers = {"user-agent": USER_AGENT}
    url =f"https://google.com/search?q={topic}&num=50"  
    response = requests.get(url, headers= headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    results = soup.find_all('h3')
    descriptions = []
    for result in results:
        try:
            description = result.get_text()
            if description != '': 
                descriptions.append(description)
        except:
            continue
    text = ' '.join(descriptions)
    if len(text)!=0:
        wordcloud = WordCloud(stopwords=set(final_stopwords_list),background_color='white',height=350, font_path='arial', colormap = 'Set2').generate(text)

        return wordcloud.to_array()
  
    

def color_import(val):
    color = '#e6ffe6' if val else '#ffe6e6'
    return f'background-color: {color}'

def tabs(default_tabs = [] ,default_active_tab=0):
        if not default_tabs:
            return None
        active_tab = st.radio("", default_tabs, index=default_active_tab, key='tabs')
        child = default_tabs.index(active_tab)+1
        st.markdown("""  
            <style type="text/css">
            div[role=radiogroup] > label > div:first-of-type, .stRadio > label {
               display: none;               
            }
            div[role=radiogroup] {
                flex-direction: unset
            }
            div[role=radiogroup] label {             
                border: 1px solid #999;
                background: #F8F8FF;
                padding: 4px 12px;
                border-radius: 2px 4px 0 0;
                position: relative;
                top: 1px;
                }
            div[role=radiogroup] label:nth-child(""" + str(child) + """) {    
                background: #FFF !important;
                border-bottom: 1px solid transparent;
            }            
            </style>
        """,unsafe_allow_html=True)        
        return active_tab


def get_sentiment(text,lang):
    analyzer = SentimentIntensityAnalyzer()
    textblob_score = TextBlob(text).sentiment.polarity
    vader_score = analyzer.polarity_scores(text)['compound']
    if textblob_score >0.2 and vader_score >0.2 and lang =='en':
        sentiment = "😀 Positive sentiment"
    elif textblob_score <0 or vader_score <0 and lang =='en':
        sentiment = "☹️ Negative sentiment"
    else:
        sentiment = ""
    return sentiment


def get_sentiment2(text,lang,tokenizer_sentiment,model_sentiment):
    classifier = pipeline("sentiment-analysis", model=model_sentiment, tokenizer=tokenizer_sentiment)
    try:
        sentiment = classifier(text)
        if lang =='en' and sentiment[0]['label'] =='positive':
            sentiment = '😀 Positive sentiment'
        elif lang =='en' and sentiment[0]['label'] =='negative':
            sentiment = '☹️ Negative sentiment'
        else:
            sentiment = ""
    except:
        sentiment = ""
    return sentiment

# @st.cache(hash_funcs={AutoModelForSeq2SeqLM: hash, transformers.models.bart.tokenization_bart_fast.BartTokenizerFast: hash,
# transformers.models.bart.modeling_bart.BartLearnedPositionalEmbedding : hash, AutoModelForSequenceClassification: hash, 
# transformers.models.bert.tokenization_bert_fast.BertTokenizerFast: hash, transformers.models.roberta.tokenization_roberta_fast.RobertaTokenizerFast: hash},
# suppress_st_warning=True)
def process_news(search_bar, news_dict, tokenizer_sentiment, model_sentiment):
    # first add tags like sentiment and subject codes
    news_dict_firm = news_dict[search_bar]
    sorted_dates = sorted(list(news_dict_firm.keys()), reverse=True)[:50]
    

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
        #news_dict_firm[date]['sentiment'] = get_sentiment(snippet,lang)

            
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

def filter_news(df_news,filter_lang,filter_date,filter_sentiment,filter_tag):
    
    df = df_news
    try:
        filter_lang = list(pd.Series(filter_lang).str.replace("French","fr").str.replace("English","en"))
    except:
        pass

    
    df_filtered =  df.loc[(df['language_code']).isin(filter_lang if filter_lang else df['language_code'].unique())
                        & (df['sentiment']).isin(filter_sentiment if filter_sentiment else df['sentiment'].unique())
                        & (df['tags'].str.contains('|'.join(filter_tag)).any(level=0))
                        ]
    return df_filtered


def display_news(df_news,tokenizer, model,nlp):
    WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))
    i=0
    dates = sorted(list(df_news.index),reverse=True)
    dates = dates[:10]
    for date in dates:
        
        st.container()
        title = df_news.loc[date]['title']
        st.write(f'#### {title}')
        #generic variable like language, source, date, body, snippet
        lang = df_news.loc[date]['language_code']
        source = df_news.loc[date]['source_name']
        try:
            snippet = df_news.loc[date]['snippet']
        except:
            snippet=None
        text = df_news.loc[date]['body']
        article_text = text
        time = df_news.loc[date]['publication_datetime']
        

        #list of tags like "Funding, M&A" and sentiment
        news_tags=[]
        for tag in df_news.loc[date]['tags'].split(','):
            news_tags.append(tag)
        news_tags.append(df_news.loc[date]['sentiment'])
        
        news_tags = [str(tag) for tag in news_tags] 
        news_tags_clean = []
        for tag in news_tags:
            if tag != "nan" and tag != "":
                news_tags_clean.append(tag)
        
        
        st.write(f"**{time}  -   {source}**")
        #create columns for tags, the "+3" is for design only

        cols = st.columns(len(news_tags)+3)
        j=0
        for a, x in enumerate(cols):
            with x:
                if j<len(news_tags_clean):
                    if news_tags_clean[j]=='😀 Positive sentiment':
                        st.success(news_tags_clean[j])
                    elif news_tags[j]=='☹️ Negative sentiment':
                        st.error(news_tags_clean[j])
                    else:
                        st.info(news_tags_clean[j])
            j+=1
        
        if snippet:
            if st.checkbox("AutoTag Snippet",key=i):
                #nlp = spacy.load("en_core_web_sm")
                doc1 = nlp(snippet)
                html = displacy.render(doc1, style="ent", page=False)
                st.write(html,unsafe_allow_html=True)
            else:
                st.markdown(snippet.replace('*',' '))
            
    
        with st.expander("See Full Text"):
            st.markdown(article_text)

        if st.button('🧬 Summarize Full Text', key=i):
            with st.spinner('Reading Article...'):     
                # if 'tokenizer' not in st.session_state:
                #     st.session_state['tokenizer'] = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
                # if 'model' not in st.session_state:
                #     st.session_state['model'] = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
                input_ids = tokenizer(
                    [article_text],
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=800
                )["input_ids"]

                output_ids = model.generate(
                    input_ids=input_ids,
                    max_length=120,
                    no_repeat_ngram_size=2,
                    num_beams=4
                )[0]

                summary = tokenizer.decode(
                    output_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )
            st.write("Summary:")
            st.info(summary)
        i+=1

# @st.cache(show_spinner=False,allow_output_mutation=True,
# hash_funcs={spacy.vocab.Vocab: hash, spacy.pipeline.tok2vec.Tok2Vec: hash, spacy.pipeline.tagger.Tagger:hash, spacy.pipeline.dep_parser.DependencyParser: hash,
# spacy.pipeline.senter.SentenceRecognizer: hash, spacy.pipeline.ner.EntityRecognizer: hash})
def NER_news(firm,news_df,nlp):
    firm = firm.title().split()[0]
    names = []
    labels = []
    #news_df = news_df[news_df['language_code']=='en']
    for doc in nlp.pipe(news_df['snippet']):
        for ent in doc.ents:
            names.append(ent.text)
            labels.append(ent.label_)
    ents_df = pd.DataFrame([names,labels]).transpose()
    ents_df.columns=['Name','Label']
    
    ents_df_persons = ents_df[ents_df['Label']=='PERSON'].groupby(['Name']).count().reset_index()
    ents_df_persons.columns=['Name','Count']
    
    ents_df_orgs = ents_df[ents_df['Label']=='ORG'].groupby(['Name']).count().reset_index()
    ents_df_orgs.columns=['Name','Count']

    ents_df_persons = ents_df_persons.loc[(ents_df_persons['Name']==ents_df_persons['Name'].str.title()) &
                                         (~ents_df_persons['Name'].str.contains(firm)) & 
                                         (~ents_df_persons['Name'].str.contains ("[^ a-zA-Z0-9]"))]
    ents_df_orgs = ents_df_orgs.loc[(ents_df_orgs['Name']==ents_df_orgs['Name'].str.title()) &  
                                    (~ents_df_orgs['Name'].str.contains(firm)) &
                                     (~ents_df_orgs['Name'].str.contains ("[^ a-zA-Z0-9]"))]


    return  ents_df_persons, ents_df_orgs

