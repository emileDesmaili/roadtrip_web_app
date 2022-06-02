

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
    st.image("streamlit_app/assets/app_logo.png")


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
    if "cities" not in st.session_state:
        st.session_state["cities"] = []
    if "last_city" not in st.session_state:
        st.session_state["last_city"] = ''
    if "distance" not in st.session_state:
        st.session_state["distances"] = []
    if "stays" not in st.session_state:
        st.session_state["stays"] = []
    if "routes" not in st.session_state:
        st.session_state["routes"] = []
    if "expenses" not in st.session_state:
        st.session_state["expenses"] = []
    
    
    
    st.subheader("Add a city to the road trip")
    with st.form("add_city"):
        col1, col2  = st.columns(2)
        with col1:
            city = st.text_input("City")
        with col2:
            length = st.number_input("Length of stay",step=1)

        submitted = st.form_submit_button("Submit")
        if submitted:
            #session state variables for cities
        
            st.session_state["cities"].append(city.title())
            idx_start = min(len(st.session_state["cities"]),2)*-1

            st.session_state["last_city"] = st.session_state["cities"][idx_start] #penultimate value is the last city aka the starting point for routes
            
            st.session_state["stays"].append(length)
            #creation of city object
            if city not in st.session_state:
                st.session_state[city] = City(city, length)
                st.session_state[city].geocode()
            else:
                st.session_state[city] = City(city, length)
                st.session_state[city].geocode()
            #create map     
            #starting city is the first
            if  len(st.session_state["cities"]) !=0:
                start = st.session_state["last_city"]
                start = st.session_state[start]
                tooltip_start = start.name
                #creation of the session state map
                if "m" not in st.session_state:
                    st.session_state["m"] = folium.Map(location=[start.lat, start.lon], zoom_start=4, control_scale=True,tiles="cartodbpositron")
                # add initial marker
                    folium.Marker(
                        [start.lat, start.lon], popup=start.name, tooltip=start.name
                    ).add_to(st.session_state["m"])

                # if two or more cities, route creation
                if len(st.session_state["cities"]) >=2:
                    
                    start = st.session_state["last_city"]
                    start = st.session_state[start]
                    
                    finish  = st.session_state["cities"][-1]
                    finish = st.session_state[finish]
                    

                    route_id = "route_id"+ str(len(st.session_state["cities"])-1)
                    st.session_state["routes"].append(route_id)
                    if route_id not in st.session_state:
                        st.session_state[route_id] = Route(start.name, finish.name)
                    
                    st.session_state[route_id].compute_()
                    st.session_state["durations"].append(st.session_state[route_id].duration/(60*60*24))
                    st.session_state["distances"].append(st.session_state[route_id].distance/1000)

                    
                    #adding markers
                    folium.Marker(
                        [start.lat, start.lon], popup=start.name, tooltip=start.name
                    ).add_to(st.session_state["m"])

                    folium.Marker(
                    [finish.lat, finish.lon], popup=finish.name, tooltip=finish.name
                    ).add_to(st.session_state["m"])

                    folium.GeoJson(st.session_state[route_id].decoded).add_child(folium.Tooltip(st.session_state[route_id].distance_txt+st.session_state[route_id].duration_txt)).add_to(st.session_state["m"])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Road Trip Duration (Days)",round(sum(st.session_state["durations"])/(60*60*24)+sum(st.session_state["stays"]),1))
    with col2:
        st.metric("Road Trip Distance (Kms)",round(sum(st.session_state["distances"]),1))
    with col3:
        st.metric("Road Trip Expenses (K€)",round(sum(st.session_state["expenses"])/1000,1))


    # call to render Folium map in Streamlit
    try:
        folium_static(st.session_state["m"],width=1200, height=700)
    except KeyError:
        pass
    # display individual itineraries
    st.subheader ("Itineraries")
        
    
    def itineraries_df():
        starts = []
        finishes = []
        durations = []
        distances = []
        prices = []
        for route in st.session_state["routes"]:
            route = st.session_state[route]
            route.compute_()
            starts.append(route.start.title())
            finishes.append(route.finish.title())
            durations.append(round(route.duration/(60*60*24),1))
            distances.append(round(route.distance/1000,1))
            prices.append(round(route.price,1))
        df = pd.DataFrame(list(zip(starts,finishes,durations,distances, prices)), columns = ['Start','Finish', 'Duration (Days)','Distance (Kms)','Gas Price (EUR)'])
        return df
          


    df_routes = itineraries_df()
    aggrid_display(df_routes)

if page == 'Budget':
    pass


if page == 'City Explorer':
    #city page

    my_city = st.sidebar.selectbox('Select City',set(st.session_state["cities"]))
    #create a city object
    city = City(my_city)
    #display name and images
    st.title(my_city.title())
    city.display_image(5)
    city.plot_trends()

    col1,col2 = st.columns(2)
    with col1:
        search = my_city.replace(' ','+').replace('&','and')
        google_news_url = f'https://news.google.com/search?for={search}&hl=en-US&gl=US&ceid=US:en'
        #google news iframe 
        st.markdown(f'<iframe height="400" width="700" src={google_news_url}></iframe>', unsafe_allow_html=True) 

    with col2:
        city.plot_news_sentiment()
    city.plot_wordcloud()

        


        








        




   
