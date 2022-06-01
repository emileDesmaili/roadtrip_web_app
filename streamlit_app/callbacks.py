import streamlit as st

def get_info_tab(search_bar):
    tab_state = f'tab{search_bar}'
    st.session_state[tab_state] = 'Key infos'

def get_news_tab(search_bar):
    tab_state = f'tab{search_bar}'
    st.session_state[tab_state] = 'News'

def get_financials_tab(search_bar):
    tab_state = f'tab{search_bar}'
    st.session_state[tab_state] = 'Financials'

def get_shareholders_management_tab(search_bar):
    tab_state = f'tab{search_bar}'
    st.session_state[tab_state] = 'Shareholders_Management'