import streamlit as st
import pandas as pd
import numpy as np
import hyperdemocracy as hd

st.title('Hyperlegis - Ask Questions About Legislation')

assembleco_records = hd.load_assembleco_records(
    process=True, 
    strip_html=True, 
    remove_empty_body=True, 
    col_order=["key", "name", "summary", "congress_gov_url", "sponsors"]
)

st.subheader('Assembleco Records 📜')
st.dataframe(assembleco_records)
st.caption('use ⌘ Cmd + F or Ctrl + F to search the table')