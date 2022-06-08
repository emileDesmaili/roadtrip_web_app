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
import nltk
nltk.download('stopwords')
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
import python_weather
import asyncio
import numpy as np
import plotly.graph_objects as go
from folium import IFrame 
import base64





class City:
    """City object 
    """

    def __init__(self,name,duration=0, bender=0, comfort=1):
        self.name = name
        self.duration = duration
        self.bender = bender
        self.comfort = comfort

    
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
    
    def get_news(self, source):
        """method to retrieve news using RSS feed or NewsAPI

        Args:
            source (str, optional): source (API or RSS). Defaults to 'API'
        """

        if source == 'RSS':
            title = []
            url = []
            date = []
            city = self.name + 'city'
            n_news = 100
            google_actu_url=f"https://news.google.com/rss/search?q={city}"
            google_actu_url = google_actu_url.replace(" ", "+")
            response = requests.get(google_actu_url)
            decoded_response = response.content.decode('utf-8')
            response_json = json.loads(json.dumps(xmltodict.parse(decoded_response)))
            n_articles = 50
            for i in range(0,n_articles):
                try:
                    url.append(response_json['rss']['channel']['item'][i]['link'])
                except (IndexError,ValueError, KeyError):
                    self.news = 'Oops, something went wrong... try again!'
                title.append(response_json['rss']['channel']['item'][i]['title'])
                date.append(response_json['rss']['channel']['item'][i]['pubDate'])
            df = pd.DataFrame([date,title, url]).transpose()
            df[0] = df[0].apply(parser.parse)
            df.columns = ['date','content','url']
            
            self.news = df

        if source == 'API':
            newsapi = NewsApiClient(api_key='57b09c8dcd91403d98299a5b3fc6607a')
            all_articles = newsapi.get_everything(q=self.name,
                                                from_param=datetime.datetime.now()-datetime.timedelta(days = 30),
                                                to=datetime.datetime.now(),
                                                language='en',
                                                sort_by='relevancy',
                                                page_size=100)
            desc = []
            date = []
            for article in all_articles['articles']:
                desc.append(article['description'])
                date.append(article['publishedAt'])

            news_df = pd.DataFrame(list(zip(date,desc)),columns=['date','content']).sort_index(ascending=False)
            self.news = news_df

    def plot_news_sentiment(self, source):
        """computes and plots news sentiment using Textblob and VADER
        """
        analyzer = SentimentIntensityAnalyzer()
        self.get_news(source)
       
        self.news['sentimentBlob'] = self.news['content'].apply(lambda news: TextBlob(news).sentiment.polarity)
        self.news['sentimentVader'] = self.news['content'].apply(lambda news: analyzer.polarity_scores(news)['compound'])
        self.news['sentiment'] = 0.5*self.news['sentimentBlob']+0.5*self.news['sentimentVader']
        self.news['date']= pd.to_datetime(self.news['date'])
        self.news = self.news.sort_values(by='date', ascending=True)
        self.news = self.news.groupby(pd.Grouper(key='date',freq='D')).mean().dropna()

        st.metric('News Sentiment',"{:.0%}".format(self.news['sentiment'].iloc[-1]),delta="{:,.1f}".format(100*(self.news['sentiment'].iloc[-1]-self.news['sentiment'].iloc[-2])))
        fig = px.line(self.news, x=self.news.index, y=['sentiment', 'sentimentBlob','sentimentVader'])
        fig.update_layout(yaxis_title='Sentiment', xaxis_title=None,width=200, height=300) 
        fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)','paper_bgcolor': 'rgba(0, 0, 0, 0)',})
        st.plotly_chart(fig, use_container_width=True)
        

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
        """google trends plot
        """

        pytrends = TrendReq(hl='en-US', tz=360) 
        kw_list = [self.name] # list of keywords to get data 
        pytrends.build_payload(kw_list, cat=0, timeframe='today 12-m')

        data = pytrends.interest_over_time() 
        data = data.reset_index() 
        fig = px.line(data, x="date", y=[self.name])
        fig.update_traces(line=dict(width=3))

        fig.update_layout(yaxis_title=None, xaxis_title=None,
        width=100, height=100) 
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)','paper_bgcolor': 'rgba(0, 0, 0, 0)',})
        st.write(f'##### Search Interest Over Time for {self.name} from Google Trends')
        try:
            st.metric('Google Trend Score',"{:,.1f}".format(data[self.name].iloc[-1]),delta="{:.0%}".format((data[self.name].iloc[-1]/data[self.name].iloc[-2])-1))
        except:
            pass

        st.plotly_chart(fig, use_container_width=True)

    
    def plot_wordcloud(self, location_type):
        """plots wordcloud from html h3 headers of google search

        Returns:
            array: image array of the wordlcoud
        """
        USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:65.0) Gecko/20100101 Firefox/66.0")
        my_stops = ['City', 'News','US','USA','ville','Accueil','Home','Famous','Landmark','Attractions','Tourist','Top','Historic','Monument',
        'places','best','things','visiter','maps','landmarks','com','historical','clubs','tripadvisor','clubbing','night','nightclub','nightclubs',
        'nightlife','night','guide','like','hotel','yelp','meilleurs','cache','www','traduire','https','france','spain','germany','united states',
        'cette','cachetraduire','page','cocktail','','travel','eat','food','dining']
        topic = f'best {location_type} in' + self.name
        final_stopwords_list = stopwords.words('english') + stopwords.words('french') + topic.split() + self.name.split() + my_stops + [self.name,self.name+'https']
        topic=topic.replace(' ','+')
        headers = {"user-agent": USER_AGENT}
        url =f"https://google.com/search?q={topic}&num=50"  
        response = requests.get(url, headers= headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        results = soup.find_all('h3') #tried div 
        descriptions = []
        for result in results:
            try:
                description = result.get_text()
                if (description != '')  : 
                    descriptions.append(description)
            except:
                continue
        text = ' '.join(descriptions)
        #if len(text)!=0:
        wordcloud = WordCloud(stopwords=set(final_stopwords_list),background_color='white',height=350, colormap = 'Set2').generate(text)
        self.wordcloud = wordcloud.to_array()
        
        st.image(self.wordcloud)
  
    
    
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
                st.image(image_list[a], use_column_width=False, width=250, channels='BGR')

    def compute_expenses(self):
        """computes expenses of the trip based on the user's form and the city's price index
        """
        bender = 100* self.bender
        price_per_meal = 15
        n_meals = self.duration * 3
        self.total_meal = self.comfort/2 * price_per_meal * n_meals
        price_per_night = 50
        n_nights = max(0,self.duration-1)
        self.total_nights = n_nights * price_per_night
        total = self.total_nights + self.total_meal + bender
        self.expenses = total
    
    def create_popup(self):
        """ creates a formatted popup for Folium map using expenses, duration
        """
        self.compute_expenses()
        path_ = 'data/raw/images/' + self.name
        if os.path.isdir(path_):
            pass
        else:
            self.scrape_images()

        popup_string = '<strong>' + self.name + '</strong>' + '<br>' + "Duration: " + str(self.duration) + ' days' + '<br>'  + 'Benders: ' + str(self.bender) + '<br>'+"Expenses: " + str(self.expenses) + "EUR"
        path = 'data/raw/images/'+self.name+'/'
        path = path+os.listdir(path)[0]
        encoded = base64.b64encode(open(path, 'rb').read())
        html = popup_string+'<img src="data:image/jpg; base64,{}" style="width:150px; height:100px"><br>'
        html = html.format
        iframe = IFrame(html(encoded.decode('UTF-8')), width=200, height=200)
        popup = folium.Popup(iframe, max_width=400)
        self.popup=popup

    
    async def get_weather(self):
        """asynchronous method to generate weather & forecasts
        """
        # declare the client. format defaults to metric system (celcius, km/h, etc.)
        client = python_weather.Client(format=python_weather.METRIC)
        # fetch a weather forecast from a city
        weather = await client.find(self.name)
        # returns the current day's forecast temperature (int)
        emoji_dict = dict(zip(['Sunny','Cloudy','Light Rain','Partly Sunny','Mostly Cloudy'],['☀️','☁️','🌧️','🌤️','🌥️']))
        text = weather.current.sky_text
        try:
            emoji = emoji_dict[text]
        except KeyError:
            emoji = '🌦️'
        self.current_weather = str(weather.current.temperature) + '°C' + str(emoji)

        st.metric('Current Weather', self.current_weather)
        self.weather_forecasts = weather.forecasts
        # get the weather forecast for a few days
        for forecast in weather.forecasts[2:]:
            temperature = str(forecast.temperature) + '°C'
            date = forecast.date.strftime("%d/%m/%Y")
            text = forecast.sky_text
            try:
                emoji = str(emoji_dict[text])
            except KeyError:
                emoji = '🌦️'
            st.write('Forecast for ',date,' - ', temperature, emoji )

        # close the wrapper once done
        await client.close()
    





class Route:
    """Route object that is used for itineraries
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

    
    
######## regular functions
def make_expense_ts(days_ts, tot_expenses):
    """this function is a messy way to turn lists of trip durations & stays & expenses into a time series dataframe

    Args:
        days_ts (list): session state of days
        tot_expenses (list): session state of expenses

    Returns:
        df: dataframe of days & expenses
    """
    # first pass
    df = pd.DataFrame(list(zip(days_ts,tot_expenses)),columns=['days','expenses'])
    days = []
    expenses = []
    for i in range(len(df)):
        day = df['days'].iloc[i]
        expense = df['expenses'].iloc[i]
        if day>1:
            day = int(day)
            days.extend(np.ones(day))
            expenses.extend(np.ones(day)*expense/day)
        elif day==1:
            day = int(day)
            days.append(1)
            expenses.append(expense)
        else:
            days.append(day)
            expenses.append(expense)

    df = pd.DataFrame(list(zip(days,expenses)),columns=['days','expenses'])
    # second pass
    days = []
    expenses = []
    for i in range(len(df)-1):
        day = df['days'].iloc[i]
        expense = df['expenses'].iloc[i]
        next_day = df['days'].iloc[i+1]
        next_expense = df['expenses'].iloc[i+1]
        if day <1:
            df['days'].iloc[i] = 1
            df['days'].iloc[i+1] = 1-day
            df['expenses'].iloc[i] = expense + (1-day)*next_expense
            df['expenses'].iloc[i+1] = day * next_expense
            
        else:
            df['days'].iloc[i] = 1
            df['expenses'].iloc[i] = expense
        
    df['days'] = df['days'].cumsum().apply(round).apply(int)
    df['cumulative expenses'] = df['expenses'].cumsum()

    return df

def plot_expenses(df, budget, gas_expenses, city_expenses):
    df = df.apply(round)
    df['over_under'] = np.where(df['cumulative expenses']<budget,'Under Budget','Over Budget')
    pie_df = pd.DataFrame(dict(type=['Gas','City'], amount= [sum(gas_expenses),sum(city_expenses)]))
    
    index = 'days'
    columns=['expenses','cumulative expenses']
    tot_exp = df['expenses'].sum()
    mean_exp = df['expenses'].mean()
    balance = budget-tot_exp
    
    
    fig = px.line(df,x=index,y=columns[1],title='Expenses over time')
    fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)','paper_bgcolor': 'rgba(0, 0, 0, 0)',})
    fig.update_traces(line=dict(color="firebrick", width=4), marker_color='firebrick')
    fig2 = px.bar(df, x=index, y=columns[0], text_auto=True, color='over_under')
    if len(fig2.data)==1:
        fig.add_trace(fig2.data[0])
    if len(fig2.data)==2:
        fig.add_trace(fig2.data[0])
        fig.add_trace(fig2.data[1])

    
    col1, col2, col3 = st.columns(3)
    
    #metrics
    with col1:   
        st.metric("Total Expenses",str(round(tot_exp)) + '€')
    with col2:
        st.metric("Average Spend per day",str(round(mean_exp)) + '€')
    with col3:
        if tot_exp> budget:
            st.metric('You are over budget 🥵',value=str(round(balance)) + '€ down',delta=balance)
        else:
            st.metric('You are in line 😃',value=str(round(balance))+"€ left",delta=balance)
    
    #charts
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig)
    with col2:
        fig = px.pie(pie_df, values='amount',names='type',title='Expenses breakdown', hole=.4, color_discrete_sequence=px.colors.sequential.RdBu)
        fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)','paper_bgcolor': 'rgba(0, 0, 0, 0)',})
        st.plotly_chart(fig)


     
   
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
        fit_columns_on_grid_load=True, allow_unsafe_jscode= True, data_return_mode = DataReturnMode.FILTERED_AND_SORTED, update_mode=GridUpdateMode.SELECTION_CHANGED,
        height=400, theme="material"
        )
        #df = grid_response['data']
        # selected = grid_response['selected_rows']
        # selected_df = pd.DataFrame(selected)
    else:
        st.info("No itinerary")
    









