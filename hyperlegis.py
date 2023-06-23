import streamlit as st
ss = st.session_state
import pandas as pd
import numpy as np
import hyperdemocracy as hd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from super_dataframe import super_dataframe
import os
from dotenv import load_dotenv
load_dotenv()

st.title('Hyperlegis - Ask Questions About Legislation')

@st.cache_data
def _load_assembleco_records():
    assembleco_records = hd.load_assembleco_records(
        process=True, 
        strip_html=True, 
        remove_empty_body=True, 
        col_order=["key", "name", "summary", "congress_gov_url", "sponsors"]
    )
    assembleco_records['select'] = False
    cols = ["select"] + [col for col in assembleco_records.columns if col != 'select']
    return assembleco_records[cols]

# borrowed from https://github.com/mobarski/ask-my-pdf/blob/main/src/gui.py
def on_api_key_change():
	api_key = ss.get('api_key') or os.getenv('OPENAI_KEY')
	# model.use_key(api_key) # TODO: empty api_key
	#
	if 'data_dict' not in ss:ss['data_dict'] = {} # used only with DictStorage
	# ss['storage'] = storage.get_storage(api_key, data_dict=ss['data_dict'])
	# ss['cache'] = cache.get_cache()
	ss['user'] = ss['storage'].folder # TODO: refactor user 'calculation' from get_storage
	# model.set_user(ss['user'])
	# ss['feedback'] = feedback.get_feedback_adapter(ss['user'])
	ss['feedback_score'] = ss['feedback'].get_score()
	#
	ss['debug']['storage.folder'] = ss['storage'].folder
	ss['debug']['storage.class'] = ss['storage'].__class__.__name__

with st.expander("Add your OpenAI API key"):
    st.text_input('OpenAI API key', type='password', key='api_key', on_change=on_api_key_change, label_visibility="collapsed")

docsearch = hd.get_pinecone_index()

docquery = st.text_input("Enter your question here","")

if docquery != "":

    found_docs = docsearch.max_marginal_relevance_search(docquery, k=2, fetch_k=10)

    for doc in found_docs:
        st.subheader(doc.metadata['key'])
        st.write(doc.page_content)


def search_filter(df, search_term):
    df[df.apply(lambda row: search_term.lower() in row.astype(str).str.lower().any(), axis=1)]

data = _load_assembleco_records()

st.subheader('Assembleco Records 📜')
editable_df = st.data_editor(super_dataframe(data))
st.caption('use ⌘ Cmd + F or Ctrl + F to search the table')

