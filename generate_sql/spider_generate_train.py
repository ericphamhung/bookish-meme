import pandas as pd
import pprint as pprint
import json

path_to_file = ''

df1 = pd.read_json(path_to_file + "train_spider.json")

df1 = pd.read_json("train_spider.json")
df2 = df1[(~df1['query'].str.contains('JOIN',regex=False))]
df2 = df2[(~df2['query'].str.contains('UNION',regex=False))]
df2 = df2[(~df2['query'].str.contains('INTERSECT',regex=False))]
df2 = df2[(~df2['query'].str.contains('EXCEPT',regex=False))]
df2 = df2[(~df2['query'].str.contains('NOT IN',regex=False))]

train = {}
for index, row in df2.iterrows():
    train[index]={}
    train[index]['table'] = row['db_id']
    train[index]['sql'] = row['query']
    train[index]['eng'] = row['question']
    
with open(path_to_file + 'spider_train_subset.json', 'w') as fp:
    json.dump(train, fp)
