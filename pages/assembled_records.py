from super_dataframe import super_dataframe
import streamlit as st
ss = st.session_state

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

data = _load_assembleco_records()

st.title('Assembleco Records 📜')
_df = super_dataframe(data)
editable_df = st.data_editor(_df)
st.caption('use ⌘ Cmd + F or Ctrl + F to search the table')

