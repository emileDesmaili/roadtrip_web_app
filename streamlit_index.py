

#library imports

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import datetime
from datetime import datetime
import numpy as np
from IPython.display import HTML
from dateutil import parser
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode,JsCode
from st_aggrid.grid_options_builder import GridOptionsBuilder
from streamlit_tags import st_tags, st_tags_sidebar
from streamlit_app.components import st_write_news, generate_shareholders_df, generate_financials_df, generate_management_df, similarity_matrix
from streamlit_app.components import plot_shareholders, plot_news_sentiment,plot_clusters, plot_financials, plot_spider_chart, plot_main_page_financials
from streamlit_app.components import  get_main_df, plot_wordcloud, create_ppt, similarity_matrix2, similarity_matrix3, color_import, tabs, generate_shareholders_treemap
from streamlit_app.components import  network_graph, graph_shareholders,similar_firms_graph, get_sentiment, filter_news, display_news, get_sentiment2, process_news
from streamlit_app.components import NER_news, send_mail, smart_search
from streamlit.state.session_state import SessionState
from streamlit_app.callbacks import get_info_tab, get_news_tab, get_financials_tab, get_shareholders_management_tab
from streamlit_app.topic_modelling import LDA_viz
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, pipeline
import transformers
from streamlit_option_menu import option_menu

from streamlit import components

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pickle

import spacy_streamlit
import time
import spacy
from spacy import displacy

import re
from unidecode import unidecode






# PAGE SETUP
st.set_page_config(
    page_title="Roadtrippers",
    layout="wide",
    page_icon="streamlit_app/assets/favicon-rothschild.jpg",

)

with open("streamlit_app/navbar-bootstrap.html","r") as navbar:
    st.markdown(navbar.read(),unsafe_allow_html=True)


# From https://discuss.streamlit.io/t/how-to-center-images-latex-header-title-etc/1946/4
with open("streamlit_app/style.css") as f:
    st.markdown("""<link href='http://fonts.googleapis.com/css?family=Roboto:400,100,100italic,300,300italic,400italic,500,500italic,700,700italic,900italic,900' rel='stylesheet' type='text/css'>""", unsafe_allow_html=True)
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)


left_column, center_column,center_column2,right_column = st.columns([1,1.7,3,1])

with left_column:
    st.info("**Demo** tool using streamlit to showcase target-screening capabilities")
with right_column:
    st.write("##### Authors\nThis tool has been developed by [Ekimetrics](https://www.ekimetrics.com)")
with center_column:
    st.image("streamlit_app/assets/logo-transparent.png")
with center_column2:
    st.title("Target Screening Tool")

# DATA IMPORT AND CLEANING
@st.cache(allow_output_mutation=True)
def get_data():
    """reads local json files with APIs data and creates the main dataframe
    Returns:
        dict of Orbis, dict of Factiva, main dataframe
    """
    json_dicts = pd.read_pickle('data/raw/json_orbis')
    data = get_main_df(json_dicts)
    #big caveat...
    data['Name'] = data['Name'].replace(['Health Bridge Limited'],'ZAVA')
    json_dicts.index = data['Name']
    with open('streamlit_app/news_list.txt', 'rb') as f:
        factiva_list = pickle.load(f)


    news_dict = dict(zip(sorted(list(data['Name']),reverse=False),factiva_list))
    

    return json_dicts, data, news_dict

json_dicts = get_data()[0]
data = get_data()[1]
news_dict = get_data()[2]
if 'main_df' not in st.session_state:
    st.session_state['main_df'] = data

@st.cache(hash_funcs={AutoModelForSeq2SeqLM: hash, transformers.models.bart.tokenization_bart_fast.BartTokenizerFast: hash,
transformers.models.bart.modeling_bart.BartLearnedPositionalEmbedding : hash, AutoModelForSequenceClassification: hash, 
transformers.models.bert.tokenization_bert_fast.BertTokenizerFast: hash, transformers.models.roberta.tokenization_roberta_fast.RobertaTokenizerFast: hash},
suppress_st_warning=True)
def load_transformers_data():    
    tokenizer_summary = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
    model_summary = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
    tokenizer_sentiment = AutoTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
    model_sentiment = AutoModelForSequenceClassification.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
    return tokenizer_summary, model_summary, tokenizer_sentiment, model_sentiment

tokenizer_summary, model_summary, tokenizer_sentiment, model_sentiment= load_transformers_data()

@st.cache(allow_output_mutation=True)
def load_spacy():
    nlp = spacy.load("en_core_web_lg")
    return nlp
nlp = load_spacy()




# MAIN MENU SETUP

#defining basket variable in session sttae
if "basket" not in st.session_state:
    st.session_state["basket"] = pd.DataFrame([])

#matrix from Word2Vec model
similarity_matrix = pd.DataFrame(similarity_matrix3(st.session_state['main_df']),columns=st.session_state['main_df']['Name'],
index=st.session_state['main_df']['Name'])

#tag_list
tag_list= ['Healthcare','AI','Great place to work','ESG','No Tag','Early-stage','Finance','Luxury','Food','Clothing']

#JavaScript code to format cells in dataframes
cell_million_jscode = JsCode("""
function(number) {
    number = number.value/1e6;
    if(isNaN(number)){
        return '-';
    } else {
    return Math.floor(number).toString().replace(/(\d)(?=(\d{3})+(?!\d))/g, '$1,') + ' M€';
        }
    }
""")
cell_percentage_jscode = JsCode("""
function(number) {
    number = number.value;
    if(isNaN(number)){
        return '-';
    } else {
    return Math.round(number) + '%';
        }
    }
""")
cell_ratio_jscode = JsCode("""
function(number) {
    number = number.value;
    if(isNaN(number)){
        return '-';
    } else {
    return Math.round(number*100)/100;
        }
    }
""")
bold_font_jscode = JsCode("""
function() {
        return {
            'font-weight': 'bold',
        }
};
""")

link_jscode = JsCode("""
function(params) {
var element = document.createElement("span");
var linkElement = document.createElement("a");
var linkText = document.createTextNode('🌐');
link_url = params.value;
linkElement.appendChild(linkText);
linkText.title = params.value;
linkElement.href = link_url;
linkElement.target = "_blank";
element.appendChild(linkElement);
return element;
};
""")
onCellClicked = JsCode("""
onCellClicked(event){
    if (event.column.getColId() === 'Name') {
    return event.column
    }
}
""")

#main menu
side1, side2 = st.sidebar.columns([1,3])
with side1:
    st.image("streamlit_app/assets/arrows_logo.png", width=50)
with side2:
    st.write(f'# Welcome SuperUser')


#import button
uploaded_file = st.sidebar.file_uploader("Import a list of firms")
if uploaded_file is not None:
    imported_firms = pd.read_excel(uploaded_file)
    if imported_firms['Name'] is not None:
        names = imported_firms['Name']
        if st.sidebar.button('⬆ Upload'):
            st.session_state["basket"] = st.session_state['main_df'][st.session_state['main_df']['Name'].isin(list(names))]
            num_success = len(st.session_state['basket'])
            import_message = f'successfully imported {num_success}/{len(names)} firms!'
            st.sidebar.success(import_message)
            report_df = pd.DataFrame(names.isin(list(st.session_state['basket']['Name'])))
            report_df.index = names
            report_df.columns=['Import success']
            st.sidebar.dataframe(report_df.style.applymap(color_import, subset=['Import success']))

page_container = st.sidebar.container()
with page_container:
    page = option_menu("Menu", ["Main Page", 'My Screening','Search Firm','Smart Search'], 
    icons=['house', 'cart','zoom-in','reddit'], menu_icon="cast", default_index=0)
    #page = st.radio('Where to?',('Main Page','My Screening'),key='main')

# search bar
if page == 'Search Firm':
    st.sidebar.header("Get detailed info on a firm")
    buff1, nav1, nav2, buff2 = st.sidebar.columns([1,2,1,1])
    with nav1:
        search_bar = st.sidebar.multiselect(f"Search firm:",
            sorted(list(st.session_state['main_df']['Name'])))
        search_bar = search_bar[0] if search_bar else ''



    # COMPANY PAGE
    if search_bar !='':
        with st.spinner('Loading page'):
        # sidebar with pseudo nearest neighbors, and add to basket button
            if st.sidebar.button(f'⬇ Add {search_bar} to my screening'):
                firm_to_add = st.session_state['main_df'][st.session_state['main_df']["Name"]==search_bar].reset_index(drop=True)
                new_basket = st.session_state['basket'].append(firm_to_add).drop_duplicates(subset = ['Name'])
                st.session_state['basket'] = new_basket

                st.sidebar.success('Added to your screening!')
            

            st.write('')
            st.sidebar.write('Suggested Firms matching ',f'**{search_bar}**')
            st.sidebar.markdown('***')

            similar_firms = similarity_matrix[search_bar].sort_values(ascending=False).drop(search_bar).head(10)
            sidecol1, sidecol2 = st.sidebar.columns([1,5])
            with sidecol2:
                st.write('**Match**')
                for i in range(0,3):
                    st.write(' '.join(similar_firms.index[i].split()[:3]))
            with sidecol1:
                st.write('**Score**')
                for i in range(0,3):
                    st.write("{:.0%}".format(similar_firms.iloc[i]))



        #company main page
            data_filtered = st.session_state['main_df'][st.session_state['main_df']["Name"]==search_bar].reset_index(drop=True)
            website = str(data_filtered['Website'][0])
            st.markdown(f"#### Company card for **[{search_bar}]({website})**", unsafe_allow_html = True)

            active_tab = tabs(["Key Infos", "News", "Financials", "Shareholders & Management","Interactions Log"])
            
            #news processing
            #create two state variables: one is the original which is cached and not filtered, but used for some viz, the other will be filtered and modified by users
            news_state = 'news_state'+search_bar
            news_state_full = 'news_state_full'+search_bar

 
            if news_state not in st.session_state or news_state_full not in st.session_state:
                df_news = process_news(search_bar, news_dict = news_dict, tokenizer_sentiment = tokenizer_sentiment, model_sentiment = model_sentiment)   
                st.session_state[news_state] = df_news
                st.session_state[news_state_full] = df_news

            #end of news processing

            if active_tab == "Key Infos":

                website_and_tags = st.container()
                with website_and_tags:
                    addon = 'https://chrome.google.com/webstore/detail/ignore-x-frame-headers/gleekbfjekiniecknbkamfmkohkpodhe/related?hl=en'
                    preview = f'[Please download this addon to preview websites]({addon})'
                    st.markdown(f'<iframe height="380" width="1600" src={website}></iframe>',
                    unsafe_allow_html=True) 
                buff1, col1, col2, col3,buff2 = st.columns([0.2,1.5,2,2,0.1])
                with col1:
                    st.write('WordCloud')
                    st.markdown('***')
                    firm_cloud = plot_wordcloud(search_bar, data=st.session_state['main_df'])
                    st.image(firm_cloud)

                with col2:
                    with st.form('Description & Tags'):
                        # firm description
                        st.write('Firm Description')
                       
                        description_state = f'description{search_bar}'
                        if description_state not in st.session_state:
                            st.session_state[description_state]  = str(data_filtered['Activity'][0])
                        new_text = st.text_area("",st.session_state[description_state])
                        # tags
                        tag_state = f'last_tag{search_bar}'
                        if tag_state not in st.session_state:
                            st.session_state[tag_state] = [data_filtered['Tags'][0]]
                        keywords=st.multiselect('Select Tags',options=tag_list,default=st.session_state[tag_state])
                        submit = st.form_submit_button('Edit Description or Tags')
                        if submit:
                            st.session_state[description_state] = new_text

                            st.session_state[tag_state] = keywords
                            my_tags = ', '.join(st.session_state[tag_state])
                            idx = st.session_state['main_df'][st.session_state['main_df']['Name'] ==search_bar].index
                            st.session_state['main_df'].loc[idx,'Tags']=my_tags
                            st.success(f'Edited by SuperUser on {datetime.now():%Y-%m-%d} at {datetime.now():%H:%M}')


                with col3:
                    st.write('Additional Info')
                    st.markdown('***')
                    st.write('**National ID Number**: ', str(data_filtered['National ID'][0]))
                    st.write('**Legal Entity Number**: ', str(data_filtered['LEI'][0]))
                    st.write('**Sector Classification**: ', str(data_filtered['Sector Code'][0]), ' - ',str(data_filtered['Sector description'][0]))
                    st.write('**Founded in**: ', str(data_filtered['Founded in'][0]))
                    st.write('**Employee Range**: ', str(data_filtered['Employees'][0]))
                    st.write('**Address**: ', str(data_filtered['Address'][0]).title()+', '+str(data_filtered['City'][0]).title())
                    map_data = pd.DataFrame({'lat': [data_filtered['Lat'][0]], 'lon': [data_filtered['Lon'][0]]})
                    fig = px.scatter_mapbox(map_data, lat="lat", lon="lon",
                            color_discrete_sequence=["red"], zoom=12, height=150)
                    fig.update_layout(mapbox_style="open-street-map")
                    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
                    st.plotly_chart(fig,use_container_width=True)
                

                st.subheader("**Related People & Firms**") 

                top_persons = NER_news(search_bar, st.session_state[news_state_full], nlp)[0].sort_values(by='Count',ascending='False').tail(15)
                top_orgs = NER_news(search_bar, st.session_state[news_state_full], nlp)[1].sort_values(by='Count',ascending='False').tail(15)
                df2string = ' '.join(top_orgs['Name'])

                col1, col2 = st.columns(2)
                with col1:
                    fig1 = px.bar(top_orgs,x='Count',y='Name',orientation='h',title='Top Firms')
                    fig1.update_layout(yaxis_title=None)
                    st.plotly_chart(fig1)

                with col2:
                    fig2 = px.bar(top_persons,x='Count',y='Name',orientation='h',title='Top People')
                    fig2.update_layout(yaxis_title=None)
                    st.plotly_chart(fig2)
                
                st.write("### Network View")
                #similar_firms_graph(search_bar, similar_firms)
                network_graph(search_bar, json_dicts[0].loc[search_bar], similar_firms)

            if active_tab == "News":
                #google trends api
                
                try:
                    plot_news_sentiment(search_bar, st.session_state[news_state_full])
                except:
                    pass

            
                # search = search_bar.replace(' ','+').replace('&','and')
                # google_actu_url=f"https://news.google.com/search?for={search}&hl=fr&gl=FR&ceid=FR%3Afr"
                #google news iframe 
                #st.markdown(f'<iframe height="500" width="1000" src={google_actu_url}></iframe>', unsafe_allow_html=True)  
            
                #Factiva news display
                st.subheader("**Latest News**")
            
                #filters for form
                with st.form('news_search'):
                    filter1,filter2, filter3, filter4 = st.columns([2,2,2,2])
                    with filter1:
                        filter_lang = st.multiselect('Language',['English','French'])
                    with filter2:
                        filter_tag = st.multiselect('Topic',['💰 Funding','🏛 Ownership Changes','🏦 Corporate Actions','👨‍💼 Management Changes'])
                    with filter3:
                        filter_sentiment = st.multiselect('Sentiment',["😀 Positive sentiment","☹️ Negative sentiment"])
                    with filter4:
                        filter_date = st.multiselect('Date',['Last week','Last month','Last quarter','Last 6 months','All'])
                    submitted = st.form_submit_button("Search News")
                    if submitted:
                        st.session_state[news_state] = filter_news(st.session_state[news_state_full],filter_lang,filter_date,filter_sentiment,filter_tag)
                #display filtered news
                        
                i=0
                dates = sorted(list(st.session_state[news_state].index),reverse=True)
                dates = dates[:10]
                for date in dates:
                    
                    st.container()
                    title = st.session_state[news_state].loc[date]['title'].replace('$','USD ')
                    st.write(f'#### {title}')
                    #generic variable like language, source, date, body, snippet
                    lang = st.session_state[news_state].loc[date]['language_code']
                    source = st.session_state[news_state].loc[date]['source_name']
                    try:
                        snippet = st.session_state[news_state].loc[date]['snippet'].replace('$','USD ')
                    except:
                        snippet=None
                    text = st.session_state[news_state].loc[date]['body']
                    article_text = text.replace('$','USD ')
                    time = st.session_state[news_state].loc[date]['publication_datetime']
                    

                    #list of tags like "Funding, M&A" and sentiment
                    news_tags=[]
                    for tag in st.session_state[news_state].loc[date]['tags'].split(','):
                        news_tags.append(tag)
                    news_tags.append(st.session_state[news_state].loc[date]['sentiment'])
                    
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
                            st.write(snippet.replace('--','-'))
                        
                
                    with st.expander("See Full Text"):
                        st.write(article_text.replace('--','-'))

                    if st.button('🧬 Summarize Full Text', key=i):
                        with st.spinner('Reading Article... 🤓 '):     
                            # if 'tokenizer' not in st.session_state:
                            #     st.session_state['tokenizer'] = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
                            # if 'model' not in st.session_state:
                            #     st.session_state['model'] = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
                            input_ids = tokenizer_summary(
                                [article_text],
                                return_tensors="pt",
                                padding="max_length",
                                truncation=True,
                                max_length=800
                            )["input_ids"]

                            output_ids = model_summary.generate(
                                input_ids=input_ids,
                                max_length=120,
                                no_repeat_ngram_size=2,
                                num_beams=4
                            )[0]

                            summary = tokenizer_summary.decode(
                                output_ids,
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=False
                            )
                        st.write("**Summary:**")
                        st.info(summary)
                    i+=1                
                #display_news(st.session_state['news_state'],tokenizer=tokenizer_summary,model=model_summary,nlp=nlp)
                


            if active_tab =='Financials':
                if not generate_financials_df(json_dicts[0].loc[search_bar]).empty:
                    
                    last_financials = generate_financials_df(json_dict=json_dicts[0][search_bar]).copy()
                    last_financials = last_financials.iloc[0].apply(pd.to_numeric)
                    with st.form('financials'):
                    # using session state to hack the collab part
                        st.write('**Internal Input on Financials**')
                        col1, col2, col3, col4,col5 = st.columns(5)
                        with col1:
                            metric = st.selectbox('Select Metric', last_financials.index)
                        with col2:
                            metric_value = st.number_input('Value')
                        with col3:
                            source = st.text_input('Source')
                        with col4:
                            st.selectbox('Confidentiality:',['Internal','Public','Confidential'])
                        with col5:
                            st.write('#')
                            submitted = st.form_submit_button('Submit')
                    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                    timestamp=f"{metric} added by SuperUser on {now},  Data Source: {source}"
                    date_col = f"{datetime.now():%Y-%m-%d %H:%M:%S}"
                    last_financials = pd.DataFrame(last_financials,index=last_financials.index)

                    # persist state of dataframe
                    firm_state = f'last_fin{search_bar}'
                    if firm_state not in st.session_state:
                        st.session_state[firm_state] = last_financials

                    # new value to append
                    new_df = pd.DataFrame(index=last_financials.index,columns=[date_col])
                    new_df.loc[metric] = metric_value

                    if submitted:
                        st.info(timestamp)
                    # update dataframe state
                        st.session_state[firm_state] = new_df.merge(st.session_state[firm_state],on=st.session_state[firm_state].index).set_index('key_0')
                        st.session_state[firm_state]['Metric'] = st.session_state[firm_state].index
                        st.write('#### KPIs')
                        plot_financials(search_bar,json_dicts=json_dicts)
                    else:
                        st.write('#### Evolution of Key Financials')
                        plot_financials(search_bar,json_dicts=json_dicts)

                    st.write('#### Detailed Statements')
                    financials_raw = generate_financials_df(json_dict=json_dicts[0][search_bar])
                    financials_raw['Date'] = financials_raw.index.copy()
                    gb2 = GridOptionsBuilder.from_dataframe(financials_raw.fillna('-'))
                    gb2.configure_column("Date",pinned=True,sort='desc')
                    gb2.configure_columns(["Turnover", "EBITDA", "Net Income", "EBIT", "Taxes","Financial Debt",'Cash','Cash Flow','Working Capital' ],
                        type=["numberColumn", "numberColumnFilter"],valueFormatter =cell_million_jscode)

                    gb2.configure_columns(['Leverage', 'ROE', 'ROA', 'Gross Margin','EBITDA Margin','Operating Margin'],
                        type=["numberColumn", "numberColumnFilter"],valueFormatter =cell_percentage_jscode)

                    gb2.configure_columns(['Gearing','Leverage Ratio','Liquidity Ratio'],
                        type=["numberColumn", "numberColumnFilter"],valueFormatter =cell_ratio_jscode)
                    
                    gb2.configure_default_column(
                                                min_column_width =2, groupable=True, value=True, enableRowGroup=True,
                                                aggFunc="sum", editable=True, enableRangeSelection=True
                                                )
                    gridOptions = gb2.build()
                    AgGrid(
                            financials_raw.fillna('-'), enable_enterprise_modules=True, gridOptions = gridOptions,
                                fit_columns_on_grid_load=False, allow_unsafe_jscode= True,theme="material")

                else:
                    st.info('**No Financials to Display**')

            if active_tab == "Shareholders & Management":
                
                contacts = st.container()
                with contacts:
                    col1, col2, = st.columns(2)
                    with col1:
                        st.write('**Management**: ')
                        client_dict = json_dicts[0].loc[search_bar]
                        board_df = generate_management_df(client_dict)
                        if not board_df.empty:
                            for i in range(len(board_df.index)):
                                st.write(
                                    f'<img src="https://img.icons8.com/nolan/32/linkedin.png"/>',
                                    board_df['Title'].iloc[i],'-',f"[{board_df['Name'].iloc[i]}]({board_df['linkedin'].iloc[i]})",
                                    unsafe_allow_html=True
                                        )
                        else:
                            st.info('No Management Data available')   
                    with col2:
                        shareholders_cont = st.container()
                        with shareholders_cont:
                            st.write('**Shareholders**: ')
                            # client_dict = json_dicts[0].loc[search_bar]
                            # shareholders_df = generate_shareholders_df(client_dict)
                            # if not shareholders_df.empty:
                            #     for i in range(len(shareholders_df.columns)):
                            #         st.write(str(shareholders_df.columns[i]).title()+' ('+ shareholders_df.iloc[1,i] +')')  
                            #     st.write('')
                            #     sh_data = plot_shareholders(search_bar,json_dicts)
                            # else:
                            #     st.info('No Shareholders data available')
                            try:
                                graph_shareholders(search_bar,json_dicts[0].loc[search_bar])
                            except:
                                st.info('No Shareholders data available')

                
            if active_tab == "Interactions Log":
                user = 'SuperUser'
                interaction_state = search_bar+'interaction'
                if interaction_state not in st.session_state:
                    st.session_state[interaction_state] = []



                with st.expander('Add Interaction'):
                    with st.form(f'interaction'):
                        date = st.date_input('Date')
                        contact = st.text_input('Point of contact')
                        desc = st.text_area('Details')
                        confidentiality = st.selectbox('Confidentiality Level',['Internal','Confidential','Public'])
                        submitted = st.form_submit_button('Add Interaction')
                    if submitted:
                        st.success(f'Added by {user} on {datetime.now():%Y-%m-%d} at {datetime.now():%H:%M}')
                        st.session_state[interaction_state].append([date,contact,desc,confidentiality,user])
                        
                interactions_df = pd.DataFrame(st.session_state[interaction_state],columns=['Date','Contact','Details','Confidentiality','Added by'])
                gb = GridOptionsBuilder.from_dataframe(interactions_df)
                gridOptions = gb.build()
                df = AgGrid(interactions_df, enable_enterprise_modules=True, gridOptions = gridOptions,
                    fit_columns_on_grid_load=True, allow_unsafe_jscode= True, 
                    theme="material"
                    )

                    

if page =='Smart Search':
    text = st.text_input('Type keywords')
    if st.button('Smart Search 💡'):
        sim_vec = smart_search(text,st.session_state['main_df'])
        targets = sim_vec[sim_vec['score']>0.39].sort_values(by='score',ascending=False)
        if len(targets)==0:
            st.info('No Matches :(')
        else:
            st.success(f'{len(targets)} matches found, displaying the top results')
            
            cols = st.columns(min(len(targets),4))
            for a, x in enumerate(cols):
                with x:
                    firm = targets['name'][a]
                    firm_df = st.session_state['main_df'][st.session_state['main_df']["Name"]==firm].reset_index(drop=True)
                    st.subheader(firm)
                    score = targets['score'][a]
                    scores = [score,1-score] 
                    labels=['A','B'] 
                    marker_colors = ['aquamarine','snow']
                    fig = go.Figure(data=[go.Pie(values=scores, labels=labels, direction='clockwise', hole=0.5,marker_colors = marker_colors, sort=False)])
                    fig.update_traces(textinfo='none')
                    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, width=100,height=100, showlegend=False)
                    st.plotly_chart(fig,use_container_width=True)
                    
                    st.metric('Score',"{:.0%}".format(score))
                    st.write(firm_df['Activity'][0])
                    turnover = "{:.0f} M€".format(firm_df['Turnover'][0] /1e6)
                    location = firm_df['Country'][0]
                    
                    st.metric('Turnover',turnover)
                    st.write(f'**Location**: {location}')
            #second row
            st.write('***')
            if len(targets)>4:
                cols2 = st.columns(min(len(targets)-4,4))
                for a2, x2 in enumerate(cols2):
                    with x2:
                        firm = targets['name'][a2+4]
                        st.subheader(firm)
                        score = targets['score'][a2+4]
                        scores = [score,1-score] 
                        labels=['A','B']  
                        marker_colors = ['aquamarine','snow']
                        fig = go.Figure(data=[go.Pie(values=scores, labels=labels, direction='clockwise', hole=0.5,marker_colors = marker_colors, sort=False)])
                        fig.update_traces(textinfo='none')
                        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, width=100,height=100, showlegend=False)
                        st.plotly_chart(fig,use_container_width=True)
                        
                        firm_df = st.session_state['main_df'][st.session_state['main_df']["Name"]==firm].reset_index(drop=True)
                        
                        st.metric('Score',"{:.0%}".format(score))
                        st.write(firm_df['Activity'][0])
                        turnover = "{:.0f} M€".format(firm_df['Turnover'][0] /1e6)
                        location = firm_df['Country'][0]
                        
                        st.metric('Turnover',turnover)
                        st.write(f'**Location**: {location}')

        
                


       




#  MAIN DASHBOARD AND BASKET ARE ONLY DISPLAYED WHEN NO FIRM IS SELECTED       
else:
    # MAIN PAGE
    if page == 'Main Page':


        #st.write('##### **Filter Database**')
        buff1, filter1,filter2, filter3, buff2 = st.columns([1,4,2,3,1])
        buffer,filter4, microbuff, filter5, filter6, filter7, filter8, filter9, buffer2 = st.columns([1,2,0.5,2,2,3,3,3,1])

        with filter1:
            filter_naf = st.multiselect('Sector',data['Sector description'].unique())
        with filter2:
            filter_text = st.text_input('Keywords')
        with filter3:
            filter_name = st.multiselect('Name',data['Name'].unique())
        with filter4:
            age = st.slider('Creation',1900, 2021,(1900,2021),1)
        with filter5:
            CA_m = st.text_input('Min Turnover (€M)',key='1')
            CA_min = -1e25 if CA_m =='' else float(CA_m)
        with filter6:
            CA_M = st.text_input('Max Turnover (€M)',key='1')
            CA_max = 1e25 if CA_M =='' else float(CA_M)
        with filter7:
            filter_employee = st.multiselect('Employee Range',data['Employees'].unique())
        with filter8:
            filter_city = st.multiselect('Country',data['Country'].dropna().unique())
        with filter9:
            filter_tag = st.multiselect('Tags',tag_list)
        
        # dataframe filtering
        
        data_filtered = st.session_state['main_df'].loc[(pd.to_numeric(st.session_state['main_df']['Founded in']).between(age[0],age[1])) 
                                & (st.session_state['main_df']['Sector description'].isin(filter_naf if filter_naf else st.session_state['main_df']['Sector description'].unique()))
                                & ~(pd.to_numeric(st.session_state['main_df']['Turnover']).ge(CA_max*1e6))
                                & ~(pd.to_numeric(st.session_state['main_df']['Turnover']).le(CA_min*1e6))
                                & (st.session_state['main_df']['Country'].isin(filter_city if filter_city else st.session_state['main_df']['Country'].unique()))
                                & (st.session_state['main_df']['Employees'].isin(filter_employee if filter_employee else st.session_state['main_df']['Employees'].unique()))
                                &  (st.session_state['main_df']['Activity'].apply(unidecode).str.contains(filter_text,flags=re.IGNORECASE, regex=True))  
                                & (st.session_state['main_df']['Name'].isin(filter_name if filter_name else st.session_state['main_df']['Name'].unique()))
                                & (st.session_state['main_df']['Tags'].str.contains('|'.join(filter_tag)).any(level=0))
                                ]

        # AGGrid DataFrame for streamlit
    
        gb = GridOptionsBuilder.from_dataframe(st.session_state['main_df'])


        gb.configure_grid_options(rowHeight=40)
        gb.configure_pagination()
        gb.configure_side_bar()

        gb.configure_column('Name', cellStyle = bold_font_jscode, cellClicked = onCellClicked)
        gb.configure_column("Website", cellRenderer=link_jscode)
        gb.configure_column("Name",pinned=True,sort='asc')
        gb.configure_column('Sector description',autoHeight=False, wrapText=True)
        gb.configure_columns(['Activity','SIREN','Tags','Lat','Lon','LEI'],hide =True)

        
        gb.configure_columns(["EBITDA", 'Turnover', 'Net Income','Cash Flow', 'Debt'], type=["numberColumn", "numberColumnFilter"],valueFormatter =cell_million_jscode)
        gb.configure_default_column(
                                    min_column_width =2, groupable=True, value=True, enableRowGroup=True,
                                    aggFunc="sum", editable=True, enableRangeSelection=True,
                                    )

        gb.configure_selection(selection_mode = 'multiple', use_checkbox=True, groupSelectsChildren=True, groupSelectsFiltered=True)
        gridOptions = gb.build()
        # display main dataframe
        if not data_filtered.empty:
            grid_response = AgGrid(data_filtered, gridOptions=gridOptions, enable_enterprise_modules=True,
            fit_columns_on_grid_load=False, allow_unsafe_jscode= True, data_return_mode = DataReturnMode.FILTERED_AND_SORTED, update_mode=GridUpdateMode.SELECTION_CHANGED,
            height=600, theme="material"
            )
            df = grid_response['data']
            selected = grid_response['selected_rows']
            selected_df = pd.DataFrame(selected)
        else:
            st.error("No firm match your filters!")

        if st.button('🔁 Update my screening'):
            basket_df = selected_df.replace('nan',np.nan) # nan come back as strings from the grid response I suspect..
            st.session_state["basket"] = st.session_state['basket'].append(basket_df, ignore_index=True)
            st.session_state["basket"] = st.session_state['basket'].drop_duplicates(subset = ['Name'])
            st.success('Successfully updated!')


        if not data_filtered.empty:
            #financials
            fin_icon, fin_head, buff = st.columns([1,3,17])
            with fin_icon:
                st.markdown(f'<img src="https://img.icons8.com/external-flatart-icons-flat-flatarticons/54/000000/external-chart-data-science-and-cyber-security-flatart-icons-flat-flatarticons.png"/>',
         unsafe_allow_html=True)
            with fin_head:
                st.subheader('Financials')
            plot_main_page_financials(data_filtered)

        #spider chart
        spider_icon, spider_head, buff = st.columns([1,6,17])
        with spider_icon:
            st.markdown(f'<img src="https://img.icons8.com/color/56/000000/radar-plot.png"/>',
         unsafe_allow_html=True)
        with spider_head:
            st.write('## Compare Metrics')
        plot_spider_chart(data=st.session_state['main_df'])

        #cluster chart
        cluster_icon, cluster_head, buff = st.columns([1,6,17])
        with cluster_icon:
            st.markdown('<img src="https://img.icons8.com/external-icongeek26-flat-icongeek26/64/000000/external-chart-data-analytics-icongeek26-flat-icongeek26.png"/>',
            unsafe_allow_html=True)
        with cluster_head:
            st.write('## Cluster View of the Database')
        dim_box = st.radio('Select Graph Dimension', ('3D','2D'))
        plot_clusters(st.session_state['main_df'],dim=int(dim_box.replace('D','')))

        #newsfeed
        news_icon, news_head, buff = st.columns([1,3,17])
        with news_icon:            
            st.markdown(f'<img src="https://img.icons8.com/fluency/58/000000/news.png"/>',
         unsafe_allow_html=True)  
        with news_head:
            st.header('NewsRoom')

        col1, buff1, col2, buff2, col3, buff4 = st.columns([4,1,4,1,4,0.4])
        fintimes = 'https://www.ft.com/mergers-acquisitions'
        bloomberg = 'https://www.bloomberg.com/europe'
        echos = 'https://www.lesechos.fr/finance-marches/ma'
        with col1:
            st.markdown(f'<iframe height="600" width="530" src={bloomberg}></iframe>',
            unsafe_allow_html=True) 
        with col2:
            st.markdown(f'<iframe height="600" width="530" src={fintimes}></iframe>',
            unsafe_allow_html=True) 
        with col3:
            st.markdown(f'<iframe height="600" width="500" src={echos}></iframe>',
            unsafe_allow_html=True) 
        

        





    # BASKET PAGE
    if page == 'My Screening':
        st.write('### My Screening')
        #AgGRid building for dataframe formatting
        gb = GridOptionsBuilder.from_dataframe(st.session_state['basket'])

        gb.configure_grid_options(rowHeight=40)
        gb.configure_pagination()
        gb.configure_side_bar()
        gb.configure_column('Name', cellStyle = bold_font_jscode)
        gb.configure_column("Website", cellRenderer=link_jscode)
        gb.configure_column("Name",pinned=True,sort='asc')
        gb.configure_column('Sector description',autoHeight=False, wrapText=True)
        gb.configure_column('Activity',hide =True)
        gb.configure_column('SIREN',hide =True)
        gb.configure_columns(["EBITDA", 'Turnover', 'Net Income','Cash Flow', 'Debt'], type=["numberColumn", "numberColumnFilter"],valueFormatter =cell_million_jscode)
        gb.configure_columns(["Leverage", 'Liquidity'], type=["numberColumn", "numberColumnFilter"],valueFormatter =cell_percentage_jscode)
        gb.configure_default_column(
                                    min_column_width =2, groupable=True, value=True, enableRowGroup=True,
                                    aggFunc="sum", editable=True, enableRangeSelection=True,
                                    )

        gb.configure_selection(selection_mode = 'multiple', use_checkbox=False, groupSelectsChildren=True, groupSelectsFiltered=True)
        gridOptions = gb.build()
        if st.button(' ❌ Clear basket'):
            st.session_state['basket']=pd.DataFrame([])
        if not st.session_state["basket"].empty:
            df = AgGrid(st.session_state['basket'], enable_enterprise_modules=True, gridOptions = gridOptions,
                    fit_columns_on_grid_load=False, allow_unsafe_jscode= True, 
                    theme="material"
                    )
            file_name=f'extract_{datetime.today()}.pptx'.replace(':','_')
            col1, col2, col3 = st.columns([2,1,20])
            with col1:
                #ppt creation
                clients = list(pd.DataFrame(df['data'])['Name'])
                ppt = create_ppt(clients,json_dicts)
                out = BytesIO()
                ppt.save(out)
                out.seek(0)
                st.download_button('Create report',out,file_name = file_name)
            with col2:
                st.markdown('<img src="https://img.icons8.com/color/40/000000/microsoft-powerpoint-2019--v1.png"/>', unsafe_allow_html=True)

            plot_main_page_financials(st.session_state['basket'])
            plot_spider_chart(data=st.session_state['basket'])
        else:
                st.info('Your screening list is empty! :(') 










    

    

 

