import streamlit as st
ss = st.session_state
import pandas as pd
import numpy as np
import hyperdemocracy as hd
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os
from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import ChatOpenAI


st.title('📜✨ Hyperlegis - Ask Questions About Legislation')

with st.expander("Add your OpenAI API key"):
    st.text_input(
        'OpenAI API key', 
        type='password', 
        key='api_key', 
        label_visibility="collapsed"
    )

llm = ChatOpenAI(model_name="gpt-4", request_timeout=120, temperature = 0.9, openai_api_key=ss.get('api_key'))
qaws = hd.get_qa_with_sources_chain(llm)

query = st.text_input("Enter your question here","")
kdocs = st.slider("Number of source documents to return", 1, 10, 2)

if query != "":

    out = qaws(query)

    st.subheader('Answer')
    st.write(out['answer'])
    st.subheader(out['sources'])

