

#library imports

import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit import components
from streamlit_folium import folium_static
import folium
from streamlit_app.components import City, Route, aggrid_display, get_prices
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode,JsCode
from st_aggrid.grid_options_builder import GridOptionsBuilder
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots







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
    page = option_menu("Menu", ["Main Page", 'Budget','City Explorer'], 
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
    
    
    
    st.subheader("Add a city to the road trip")
    with st.form("add_city"):
        col1, col2  = st.columns(2)
        with col1:
            city = st.text_input("City").title()
            bender = st.checkbox('I will go on a bender in this city (so help me God...)')
        with col2:
            stay = st.number_input("Length of stay",step=1)
            comfort = st.slider('What do you want in terms of comfort (food/hotels) during the stay?',1,3)

        submitted = st.form_submit_button("Add to the Roadtrip")
        if submitted:
            #session state variables for cities
        
            st.session_state["cities"].append(city.title())
            idx_start = min(len(st.session_state["cities"]),2)*-1

            st.session_state["last_city"] = st.session_state["cities"][idx_start] #penultimate value is the last city aka the starting point for routes
            
            st.session_state["stays"].append(stay)
            #creation of city object
            if city not in st.session_state:
                st.session_state[city] = City(city, stay, bender, comfort)
                st.session_state[city].geocode()
                st.session_state[city].compute_expenses()
                st.session_state["expenses"].append(st.session_state[city].expenses)
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
                    st.session_state["m"] = folium.Map(location=[start.lat, start.lon], zoom_start=4, control_scale=True,tiles="cartodbpositron")
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
                        st.session_state["distances"].append(round(st.session_state[route_id].distance/1000,1))
                        st.session_state["gas_expenses"].append(round(st.session_state[route_id].price,1))

                        
                        #adding markers
                        folium.Marker(
                            [start.lat, start.lon], popup=start.popup, tooltip=start.name
                        ).add_to(st.session_state["m"])

                        folium.Marker(
                        [finish.lat, finish.lon], popup=finish.popup, tooltip=finish.name
                        ).add_to(st.session_state["m"])

                        folium.GeoJson(st.session_state[route_id].decoded).add_child(folium.Tooltip(st.session_state[route_id].distance_txt+st.session_state[route_id].duration_txt)).add_to(st.session_state["m"])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Road Trip Duration (Days)",round(sum(st.session_state["durations"])/(60*60*24)+sum(st.session_state["stays"]),1))
    with col2:
        st.metric("Road Trip Distance (Kms)",round(sum(st.session_state["distances"]),1))
    with col3:
        st.metric("Road Trip Expenses (K€)",round((sum(st.session_state["expenses"])+sum(st.session_state['gas_expenses']))/1000,1))


    # call to render Folium map in Streamlit
    try:
        folium_static(st.session_state["m"],width=1200, height=700)
    except KeyError:
        pass
    # display individual itineraries
    st.subheader ("Itineraries")
        
    

    routes_df = pd.DataFrame(list(zip(st.session_state["starts"],st.session_state["finishes"],st.session_state["durations"],
                                st.session_state["distances"],st.session_state["gas_expenses"])),
                                columns = ['Start','Finish', 'Duration (Days)','Distance (Kms)','Gas Price (EUR)'])

    aggrid_display(routes_df)


if page == 'Budget':
    pass


if page == 'City Explorer':
    #city page

    my_city = st.sidebar.selectbox('Select City',set(st.session_state["cities"]))
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
        st.write('#### Weather')
        weather_url = 'https://weather.com/en-US/temps/aujour/l/1a8af5b9d8971c46dd5a52547f9221e22cd895d8d8639267a87df614d0912830'
        st.markdown(f'<iframe height="200" width="600" src={weather_url}></iframe>', unsafe_allow_html=True) 
        
        #plot news sentiment
        city.plot_news_sentiment()
        st.subheader("Total Expenses")
        city.compute_expenses()
        st.metric(f'Total Expenses in {city.name}', city.expenses)


        


        








        




   
