

#library imports

import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit import components
from streamlit_folium import folium_static
import folium
from streamlit_app.components import City, get_route







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


left_column, center_column,center_column2,right_column = st.columns([1,3,3,1])

with left_column:
    st.info("**Demo** Project using streamlit")
with right_column:
    st.write("##### Authors\nThis tool has been developed by [Emile D. Esmaili](https://github.com/emileDesmaili)")
with center_column:
    st.image("streamlit_app/assets/roadtrip_wide.jpg")
with center_column2:
    st.title("Road Trip Web App")

side1, side2 = st.sidebar.columns([1,3])
with side1:
    st.image("streamlit_app/assets/midoryia.png", width=70)
with side2:
    st.write(f'# Welcome SuperUser')

page_container = st.sidebar.container()
with page_container:
    page = option_menu("Menu", ["Main Page", 'Budget','Sandbox'], 
    icons=['house','zoom-in','reddit'], menu_icon="cast", default_index=0)



if page == 'Main Page':
    if "duration" not in st.session_state:
        st.session_state["duration"] = 0
    if "cities" not in st.session_state:
        st.session_state["cities"] = []
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

            st.session_state["cities"].append(city)
            
            st.session_state["duration"] += length
            #creation of city object
            if city not in st.session_state:
                st.session_state[city] = City(city, length)
                st.session_state[city].geocode()
            else:
                st.session_state[city] = City(city, length)
                st.session_state[city].geocode()

    
    st.metric("Road Trip Duration",st.session_state["duration"])

    #display map        

    if  len(st.session_state["cities"]) !=0:
        start = st.session_state["cities"][0]
        start = st.session_state[start]

        m = folium.Map(location=[start.lat,start.lon], zoom_start=4, control_scale=True,tiles="cartodbpositron")
        # add markers
        tooltip = start.name
        folium.Marker(
            [start.lat, start.lon], popup=start.name, tooltip=tooltip
        ).add_to(m)

        for city in st.session_state["cities"]:
            city = st.session_state[city]
            tooltip = city.name
            folium.Marker(
            [city.lat, city.lon], popup=city.name, tooltip=tooltip
            ).add_to(m)

        # call to render Folium map in Streamlit
        folium_static(m,width=1200, height=700)
    else:
        pass
    if  len(st.session_state["cities"]) !=0:
        m2 = get_route(st.session_state["cities"][0],st.session_state["cities"][1])
        folium_static(m2,width=1200, height=700)


   
