import streamlit as st
import pandas as pd
import numpy as np
import hyperdemocracy as hd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from super_dataframe import super_dataframe
import os
from dotenv import load_dotenv
load_dotenv()
from langchain.schema import Document 

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

oa_key =  st.text_input("Enter your OpenAI API key here","")

def search_filter(df, search_term):
    df[df.apply(lambda row: search_term.lower() in row.astype(str).str.lower().any(), axis=1)]

data = _load_assembleco_records()

st.subheader('Assembleco Records 📜')
editable_df = st.data_editor(super_dataframe(data))
st.caption('use ⌘ Cmd + F or Ctrl + F to search the table')

docsearch = hd.get_pinecone_index()

docquery = st.text_input("Enter your question here","")

if docquery != "":

    found_docs = docsearch.max_marginal_relevance_search(docquery, k=2, fetch_k=10)

    for doc in found_docs:
        st.subheader(doc.metadata['key'])
        st.write(doc.page_content)