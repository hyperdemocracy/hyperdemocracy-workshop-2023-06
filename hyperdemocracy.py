from datasets import load_dataset
import pandas as pd
from bs4 import BeautifulSoup
import re
import openai

def load_assembleco_records(
    ds_name="assembleco/hyperdemocracy",
    process=False, 
    strip_html=False, 
    remove_empty_body=False,
) -> pd.DataFrame: 
    ds = load_dataset(ds_name, split="train")
    df = ds.to_pandas()
    if process: 
        df['congress_num'] = None
        df['legis_class'] = None
        df['legis_num'] = None
        for irow, row in df.iterrows():
            congress_num, legis_class, legis_num = split_key(row['key'])
            df.loc[irow, 'congress_num'] = congress_num
            df.loc[irow, 'legis_class'] = legis_class
            df.loc[irow, 'legis_num'] = legis_num
    if strip_html: 
        df['body'] = df['body'].apply(lambda x: BeautifulSoup(x, 'html.parser').get_text())
        df['summary'] = df['summary'].apply(lambda x: BeautifulSoup(x, 'html.parser').get_text())
        df['congress_gov_url'] = df['key'].apply(address_on_congress)
        df['assembled_url'] = df['key'].apply(address_on_assembled)

    if remove_empty_body: 
        df = df[df['body']!='']


    return df

def address_on_congress(key): 
    """Return congress.gov url from key."""
    # TODO add assembled url builder option here as well
    url_map = {
        "HR": "house-bill",
        "HCONRES": "house-concurrent-resolution",
        "HRES": "house-resolution",
        "HJRES": "house-joint-resolution",
        "S": "senate-bill",
        "SCONRES": "senate-concurrent-resolution",
        "SRES": "senate-resolution",
        "SJRES": "senate-joint-resolution",
    }
    congress_num, legis_class, legis_num = split_key(key)
    url_legis_class = url_map[legis_class]
    url = f"https://www.congress.gov/bill/{congress_num}th-congress/{url_legis_class}/{legis_num}"
    return url

def address_on_assembled(key):
    return f"https://assembled.app/measures/{key}"

def split_key(key):
    """
    TODO: add a link explaining this notation and variable names
    """
    congress_num, legis_class, legis_num = re.match("(\d+)(\D+)(\d+)", key).groups()
    return congress_num, legis_class, legis_num

def get_openai_embedding(word): 
    openai_embd = openai.Embedding.create(input=word, model='text-embedding-ada-002')['data'][0]['embedding']
    return openai_embd