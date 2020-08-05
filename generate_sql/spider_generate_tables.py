import pandas as pd
import pprint as pprint
import json

path_to_data='/home/eric/Triplet/data/spider/'

tables = pd.read_json(path_to_data + 'tables.json')
tables

tables_dict = {}
for index, row in tables.iterrows():
    tables_dict[row['db_id']] = row['table_names']
    
with open(path_to_file + 'spider_tables.json', 'w') as fp:
    json.dump(tables_dict, fp)
