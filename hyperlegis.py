import streamlit as st
import pandas as pd
import numpy as np
import hyperdemocracy as hd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from super_dataframe import super_dataframe


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
editable_df = st.data_editor(super_dataframe(data))

# def get_editable_df(df):
#     return st.data_editor(df,
#     column_config={
#         "congress_gov_url": st.column_config.LinkColumn("Congress.gov URL"),
#     },
#     hide_index=True
# )

# search_query = st.text_input("Search for legislation by keyword")
# if st.button('Search') and search_query != '':
#     editable_df = hd.filter_aco_df(data, search_query)

# st.subheader('Assembleco Records 📜')
# editable_df = get_editable_df(data)
# st.caption('use ⌘ Cmd + F or Ctrl + F to search the table')

def langchain_setup(n_docs=100): 
    docs = hd.get_legislative_documents_from_df(data)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
    split_docs = text_splitter.split_documents(docs)