#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

# !ls records/session/118/bill/hr1.*
base = Path("~/.cache.channel/us-congress/session/118/bill")
bills = sorted(list(base.glob('*')))

len(bills)
path = bills[0]
path.parts[-1].split('.')

frame = []
for path in bills:
    frame.append([path, *path.parts[-1].split('.')])

paths = (
    pd
    .DataFrame(frame, columns=['path', 'key', 'date'])
    .sort_values(['key', 'date'])
)
measures = paths.groupby('key').last()
# gabe says this is guaranteed.
# assert len(measures) == len(measures.index.unique())

rows = []
for path in tqdm(measures['path']): #[0:120]
    path = Path(path)
    row = {}
    row['path'] = path

    # goal: look up ordering of `ih`, `eh`, `rh`, and so on.
    copies = list(path.glob('copies/*.htm'))
    if(copies):
        with open(copies[-1], 'r') as copy:
            row['body'] = copy.read()
    else:
        row['body'] = ''

    for file_name in path.glob('*.json'):
        column = file_name.name.split('.')[0]
        with file_name.open('r') as fp:
            dd = json.load(fp)
        row[column] = dd
    rows.append(row)

records = pd.DataFrame(rows)

def summarize(r):
    grades = pd.DataFrame(r['summaries']['summaries'], columns=['updateDate', 'text'])
    return grades.sort_values('updateDate').iloc[-1].text if len(grades) else ''
def sponsors(r):
    return (list([m['bioguideId'], m['fullName'], 'sponsor'] for m in r['index']['bill']['sponsors'])
            + list([m['bioguideId'], m['fullName'], 'cosponsor'] for m in r['cosponsors']['cosponsors']))
def thema(r):
    base = [r.subjects['subjects']['policyArea']['name']] if 'policyArea' in r.subjects['subjects'] else []
    return base + list(s['name'] for s in r.subjects['subjects']['legislativeSubjects'])

records['key'] = records.apply(
    lambda r:'%s%s%s' % (r['index']['bill']['congress'], r['index']['bill']['type'], r['index']['bill']['number']),
    axis=1
)
records['name'] = records.apply(
    lambda r: r['index']['bill']['title'],
    axis=1
)
records['summary'] = records.apply(summarize, axis=1)
records['sponsors'] = records.apply(sponsors, axis=1)
records['themes'] = records.apply(thema, axis=1)

cols = [
  'key', 'name', 'sponsors', 'summary', 'body', 'themes',
  'index', 'actions', 'amendments', 'committees', 'relatedbills',
  'cosponsors', 'subjects', 'text', 'titles',
]

records = records[cols]
# records
records.to_parquet('records.parquet')
dataset = load_dataset('parquet', data_files={ 'train': 'records.parquet' })
# dataset
# get_ipython().run_line_magic('pinfo', 'dataset.push_to_hub')

with open('.call', 'r') as blob:
    calls = blob.readlines()
    call = dict(line.strip().split("=") for line in calls)

dataset.push_to_hub('assembleco/hyperdemocracy', token = call['huggingface.key'])
