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
import xmltodict
from dateutil import parser
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from jmd_imagescraper.core import *
import glob
from PIL import Image
import os, stat
import cv2
from pytrends.request import TrendReq
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from newsapi import NewsApiClient





class City:
    """City object 
    """

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
    
    def get_news(self, source='API'):
        """method to retrieve news using RSS feed or NewsAPI

        Args:
            source (str, optional): source (API or RSS). Defaults to 'API'
        """

        if source == 'RSS':
            title = []
            url = []
            date = []
            city = self.name
            n_news = 100
            google_actu_url=f"https://news.google.com/rss/search?q={city}"
            google_actu_url = google_actu_url.replace(" ", "+")
            response = requests.get(google_actu_url)
            decoded_response = response.content.decode('utf-8')
            response_json = json.loads(json.dumps(xmltodict.parse(decoded_response)))
            for i in range(0,n):
                try:
                    url.append(response_json['rss']['channel']['item'][i]['link'])
                except (IndexError,ValueError, KeyError):
                    self.news = 'Oops, something went wrong... try again!'
                title.append(response_json['rss']['channel']['item'][i]['title'])
                date.append(response_json['rss']['channel']['item'][i]['pubDate'])
            df = pd.DataFrame([date,title, url]).transpose()
            df[0] = df[0].apply(parser.parse)
            self.news = df.set_index(df[0])

        if source == 'API':
            newsapi = NewsApiClient(api_key='57b09c8dcd91403d98299a5b3fc6607a')
            all_articles = newsapi.get_everything(q=self.name,
                                                from_param=datetime.datetime.now()-datetime.timedelta(days = 10),
                                                to=datetime.datetime.now(),
                                                language='en',
                                                sort_by='relevancy',
                                                page=1)
            desc = []
            date = []
            for article in all_articles['articles']:
                desc.append(article['description'])
                date.append(article['publishedAt'])

            news_df = pd.DataFrame([date,desc],columns=['date','content']).sort_index(ascending=False)
            self.news = news_df

    def write_news(self,n=10):
        
        self.get_news(n=n)
        news_df = self.news
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

    def plot_trends(self):
        # analyzer = SentimentIntensityAnalyzer()
        # news_df = get_news(client, n)
        # news_df['sentimentBlob'] = news_df[1].apply(lambda news: TextBlob(news).sentiment.polarity)
        # news_df['sentimentVader'] = news_df[1].apply(lambda news: analyzer.polarity_scores(news)['compound'])
        # news_df['sentiment'] = 0.5*news_df['sentimentBlob']+0.5*news_df['sentimentVader']

        # senti_fig = px.line(news_df.sort_index(), x=0, y=news_df['sentiment'])
        # senti_fig.update_layout(yaxis_title=None, xaxis_title=None) 
        # st.plotly_chart(senti_fig, use_container_width=False)
    
        pytrends = TrendReq(hl='en-US', tz=360) 
        kw_list = [self.name] # list of keywords to get data 
        pytrends.build_payload(kw_list, cat=0, timeframe='today 12-m')

        data = pytrends.interest_over_time() 
        data = data.reset_index() 
        fig = px.line(data, x="date", y=[self.name])
        fig.update_traces(line=dict(width=3))

        fig.update_layout(yaxis_title=None, xaxis_title=None,
        width=100, height=150) 
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)','paper_bgcolor': 'rgba(0, 0, 0, 0)',})
        st.write(f'##### Search Interest Over Time for {self.name} from Google Trends')
        try:
            st.metric('Google Trend Score',"{:,.1f}".format(data[self.name].iloc[-1]),delta="{:.0%}".format((data[self.name].iloc[-1]/data[self.name].iloc[-2])-1))
        except:
            pass

        st.plotly_chart(fig, use_container_width=True)

    def plot_news_sentiment(self):
        analyzer = SentimentIntensityAnalyzer()
        self.get_news()
        news_df = self.news
        news_df['sentimentBlob'] = news_df[1].apply(lambda news: TextBlob(news).sentiment.polarity)
        news_df['sentimentVader'] = news_df[1].apply(lambda news: analyzer.polarity_scores(news)['compound'])
        news_df['sentiment'] = 0.5*news_df['sentimentBlob']+0.5*news_df['sentimentVader']

        senti_fig = px.line(news_df.sort_index(), x=news_df.index, y=news_df['sentiment'])
        senti_fig.update_layout(yaxis_title=None, xaxis_title=None) 
        st.plotly_chart(senti_fig, use_container_width=False)



    @st.cache()
    def plot_wordcloud(self):
        """plots wordcloud from html h3 headers of google search

        Returns:
            array: image array of the wordlcoud
        """
        USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:65.0) Gecko/20100101 Firefox/66.0")
        my_stops = ['City', 'News','US','USA','ville','Accueil','Home','Famous','Landmark','Attractions','Tourist','Top','Historic','Monument',
        'places','best','things','visiter','maps','landmarks','com','historical']
        topic = self.name +' monuments to see'
        final_stopwords_list = stopwords.words('english') + stopwords.words('french') + topic.split() + my_stops
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

            self.wordcloud = wordcloud.to_array()
        try:
            st.image(self.wordcloud)
        except:
            st.info('No Wordcloud')
    
    
    def scrape_images(self, max_results=5):
        """method to scrape images from duckduckgo and save them

        Returns:
            
        """
        duckduckgo_search('data/raw/images', self.name,self.name, max_results=max_results)
    
    def display_image(self, max_results=5):
        """methods to display image in streamlit page

        Args:
            max_results (int, optional): number of scraped images. Defaults to 10.
        """
        
        path = 'data/raw/images/' + self.name
        globpath = path + '/*.jpg'
        images = glob.glob(path)
        image_list = []

        #scrape only if directory doesn't exist
        if not os.path.isdir(path):
            self.scrape_images()
        for img in glob.glob(globpath):
            image_list.append(cv2.imread(img))
        cols = st.columns(len(image_list))
        for a, x in enumerate(cols):
            with x:
                st.image(image_list[a], use_column_width=False, width = 250)




class Route:
    """Ruute eobject that is used for itineraries
    """

    def __init__(self, start, finish):
        self.start = start
        self.finish = finish
    
    def compute_(self):
        """compute_ method is used to calculate driving time, route, distance & gas price for a route
        """
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
    """AgGrid styler for streamlit to dipslay a pandas dataframe in a JS AgGrid-like way

    Args:
        df (_type_): pandas dataframe
    """
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
    









