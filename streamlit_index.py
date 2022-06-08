

#library imports

import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit import components
from streamlit_folium import folium_static
import folium
from streamlit_app.components import City, Route, aggrid_display, get_prices, make_expense_ts, plot_expenses
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode,JsCode
from st_aggrid.grid_options_builder import GridOptionsBuilder
import plotly.express as px
import asyncio
import numpy as np








# PAGE SETUP
st.set_page_config(
    page_title="Roadtrippers",
    layout="wide",
    page_icon="streamlit_app/assets/road66.png",

)

# with open("streamlit_app/navbar-bootstrap.html","r") as navbar:
#     st.markdown(navbar.read(),unsafe_allow_html=True)


# From https://discuss.streamlit.io/t/how-to-center-images-latex-header-title-etc/1946/4
with open("streamlit_app/style.css") as f:
    st.markdown("""<link href='http://fonts.googleapis.com/css?family=Roboto:400,100,100italic,300,300italic,400italic,500,500italic,700,700italic,900italic,900' rel='stylesheet' type='text/css'>""", unsafe_allow_html=True)
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)


left_column, center_column,right_column = st.columns([1,3,1])

with left_column:
    st.info("**Demo** Project using streamlit")
with right_column:
    st.write("##### Authors\nThis tool has been developed by [Emile D. Esmaili](https://github.com/emileDesmaili)")
with center_column:
    st.image("streamlit_app/assets/app_logo.PNG")


side1, side2 = st.sidebar.columns([1,3])
with side1:
    st.image("streamlit_app/assets/midoryia.png", width=70)
with side2:
    st.write(f'# Welcome SuperUser')

page_container = st.sidebar.container()
with page_container:
    page = option_menu("Menu", ["Main Page", 'City Explorer','Budget'], 
    icons=['house','cash','dpad'], menu_icon="cast", default_index=0)



if page == 'Main Page':
    if "durations" not in st.session_state:
        st.session_state["durations"] = []
    if "stays" not in st.session_state:
        st.session_state["stays"] = []
    if "cities" not in st.session_state:
        st.session_state["cities"] = []
    if "last_city" not in st.session_state:
        st.session_state["last_city"] = ''
    if "distances" not in st.session_state:
        st.session_state["distances"] = []
    if "routes" not in st.session_state:
        st.session_state["routes"] = []
    if "expenses" not in st.session_state:
        st.session_state["expenses"] = []
    if "starts" not in st.session_state:
        st.session_state["starts"] = []
    if "finishes" not in st.session_state:
        st.session_state["finishes"] = []
    if "gas_expenses" not in st.session_state:
        st.session_state["gas_expenses"] = []
    if "days_ts" not in st.session_state:
        st.session_state["days_ts"] = []
    if "tot_expenses" not in st.session_state:
        st.session_state["tot_expenses"] = []
    
    
    
    st.subheader("Add a city to the road trip")
    with st.form("add_city"):
        col1, col2  = st.columns(2)
        with col1:
            city = st.text_input("City").title()
            bender = st.slider('How many nights will you go on a bender?',0,10)
        with col2:
            stay = st.number_input("Length of stay",1,10,step=1)
            comfort = st.slider('What do you want in terms of comfort (food/hotels) during the stay?',1,3)

        submitted = st.form_submit_button("Add to the Roadtrip")
        if submitted:
            #session state variables for cities
            
            
            #creation of city object
            if city not in st.session_state:
                st.session_state["cities"].append(city.title())
                idx_start = min(len(st.session_state["cities"]),2)*-1
                st.session_state["last_city"] = st.session_state["cities"][idx_start] #penultimate value is the last city aka the starting point for routes
                st.session_state["stays"].append(stay)
                st.session_state["days_ts"].append((stay))
                st.session_state[city] = City(city, stay, bender, comfort)
                st.session_state[city].geocode()
                st.session_state[city].compute_expenses()
                st.session_state["expenses"].append(st.session_state[city].expenses)
                st.session_state["tot_expenses"].append(st.session_state[city].expenses)
            else:
                st.session_state[city] = City(city, stay, bender, comfort)
                st.session_state[city].geocode()
                st.session_state[city].compute_expenses()
                st.session_state["expenses"].append(st.session_state[city].expenses)
            #create map     
            #starting city is the first
            if  len(st.session_state["cities"]) !=0:
                start = st.session_state["last_city"]
                start = st.session_state[start]
                start.create_popup()
                #creation of the session state map
                if "m" not in st.session_state:
                    st.session_state["m"] = folium.Map(location=[start.lat, start.lon], zoom_start=4, control_scale=True,tiles='Stamen Watercolor')
                # add initial marker
                    folium.Marker(
                        [start.lat, start.lon], popup=start.popup, tooltip=start.name
                    ).add_to(st.session_state["m"])

                # if two or more cities, route creation
                if len(st.session_state["cities"]) >=2:
                    
                    start = st.session_state["last_city"]
                    start = st.session_state[start]
                    if start not in st.session_state['starts']:
                        st.session_state['starts'].append(start.name)
                    start.create_popup()
                    
                    finish  = st.session_state["cities"][-1]
                    finish = st.session_state[finish]
                    if finish not in st.session_state['finishes']:
                        st.session_state['finishes'].append(finish.name)
                    finish.create_popup()
                    
                    if start.name == finish.name:
                        #update popup to reflect changes
                        folium.Marker(
                        [finish.lat, finish.lon], popup=finish.popup, tooltip=finish.name
                        ).add_to(st.session_state["m"])
                    else:
                        route_id = "route_id"+ str(len(st.session_state["cities"])-1)
                        st.session_state["routes"].append(route_id)
                        if route_id not in st.session_state:
                            st.session_state[route_id] = Route(start.name, finish.name)
                        
                        st.session_state[route_id].compute_()
                        st.session_state["durations"].append(round(st.session_state[route_id].duration/(60*60*24),1))
                        st.session_state["days_ts"].insert(-1,round(st.session_state[route_id].duration/(60*60*24),1))
                        st.session_state["distances"].append(round(st.session_state[route_id].distance/1000,1))
                        st.session_state["gas_expenses"].append(round(st.session_state[route_id].price,1))
                        st.session_state["tot_expenses"].insert(-1,round(st.session_state[route_id].price,1))

                        
                        #adding markers
                        folium.Marker(
                            [start.lat, start.lon], popup=start.popup, tooltip=start.name
                        ).add_to(st.session_state["m"])

                        folium.Marker(
                        [finish.lat, finish.lon], popup=finish.popup, tooltip=finish.name
                        ).add_to(st.session_state["m"])
                        st.session_state["m"].fit_bounds(st.session_state["m"].get_bounds(), padding=(30, 30))

                        folium.GeoJson(st.session_state[route_id].decoded).add_child(folium.Tooltip(st.session_state[route_id].distance_txt+st.session_state[route_id].duration_txt)).add_to(st.session_state["m"])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Road Trip Duration (Days)",round(sum(st.session_state["durations"])+sum(st.session_state["stays"]),1))
    with col2:
        st.metric("Road Trip Distance (Kms)",round(sum(st.session_state["distances"]),1))
    with col3:
        st.metric("Road Trip Expenses (K€)",round((sum(st.session_state["expenses"])+sum(st.session_state['gas_expenses']))/1000,1))


    # call to render Folium map in Streamlit
    try:
        folium_static(st.session_state["m"],width=1400, height=450)
    except KeyError:
        pass
    # display individual itineraries
    st.subheader ("Itineraries")
        
    

    routes_df = pd.DataFrame(list(zip(st.session_state["starts"],st.session_state["finishes"],st.session_state["durations"],
                                st.session_state["distances"],st.session_state["gas_expenses"])),
                                columns = ['Start','Finish', 'Duration (Days)','Distance (Kms)','Gas Price (EUR)'])

    aggrid_display(routes_df)


if page == 'City Explorer':
    st.info('Please install [this Chrome add-in](https://chrome.google.com/webstore/detail/ignore-x-frame-headers/gleekbfjekiniecknbkamfmkohkpodhe/related?hl=en) ')
    #city page
    my_city = st.sidebar.selectbox('Select City',set(st.session_state["cities"]))
    if my_city == None:
        st.info ('Add Cities to your trip first 😇')
    else:
        #create a city object
        city = st.session_state[my_city]
        #display name and images
        st.title(my_city.title())
        city.display_image(5)
        

        col1,col2 = st.columns(2)
        with col1:
            #plot trends
            city.plot_trends()
            #plot news
            st.write('#### News')
            search = my_city.replace(' ','+').replace('&','and')
            google_news_url = f'https://news.google.com/search?for={search}&hl=en-US&gl=US&ceid=US:en'
            #google news iframe 
            st.markdown(f'<iframe height="400" width="700" src={google_news_url}></iframe>', unsafe_allow_html=True) 
            st.write('#### WordCloud Generator')
            with st.form('Generate Wordcloud'):
                location_type = st.selectbox('What type of location are you interested in?',['bars','restaurants','clubs'])
                submitted = st.form_submit_button('Go!')
            if submitted:
                city.plot_wordcloud(location_type)

        with col2:
            #plot weather
            asyncio.run(city.get_weather())
            
            #plot news sentiment
            source = st.radio('News source',('RSS','API'))
            city.plot_news_sentiment(source)
            st.subheader("Total Expenses")
            city.compute_expenses()
            st.metric(f'Total Expenses in {city.name}', str(round(city.expenses))+ ' €')

if page == 'Budget':
    df = make_expense_ts(days_ts=st.session_state['days_ts'], 
                    tot_expenses= st.session_state['tot_expenses']    
                        )
    budget = st.number_input('What is your budget?')
    if st.session_state['cities'] == []:
        st.info('Add Cities to your trip first 😇')
    else:
        plot_expenses(df, budget,st.session_state['gas_expenses'],st.session_state['expenses'])


    

        


        








        




   
