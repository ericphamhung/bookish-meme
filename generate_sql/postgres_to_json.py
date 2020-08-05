import json
import pandas as pd
import psycopg2
from tqdm import tqdm_notebook as tqdm
sqluser = 'postgres'
dbname = 'alcohol'
pw = 'postgres'
port = 5432
# host = "/var/run/postgresql"
host = "localhost"
con=psycopg2.connect(user=sqluser,
                    password=pw,
                     host=host,
                     port = port,
                    dbname=dbname)

cursor = con.cursor()



outpath = '/home/eric/Cannabis/json_result.txt'
con.commit()

string = """
SELECT json_agg(t)FROM ({}) t;
""".format(query)

cursor.execute(string)

fetch = cursor.fetchall()


#TODO MULT ROWS, SINGLE COLUMN

#checks if more than 2 columns, tag to produce data table
if len([k for k in fetch[0][0][0].items()]) > 2:
    data = {}
    data['table'] = []
    for f in fetch[0][0]:
        data['table'].append(f)

#checks if location variables are included, tag as geo graph
elif 'store_coord' in [k for k,v in fetch[0][0][0].items()]:
    data = {}
    data['geo'] = []
    for f in fetch[0][0]:
        data['geo'].append(f)

#checks if more than one row, tag as bar graph
elif len(fetch[0][0]) > 1 and len([k for k in fetch[0][0][0].items()]) == 2:
    data = {}
    data['bar'] = []
    for f in fetch[0][0]:
        data['bar'].append(f)

#checks for multple row, tag as a table
elif len(fetch[0][0]) > 1 and len([k for k in fetch[0][0][0].items()]) == 1:
    data = {}
    data['table'] = []
    for f in fetch[0][0]:
        data['table'].append(f)

else:
    data = {}
    data['single'] = []
    for f in fetch[0][0]:
        data['single'].append(f)

        
#data
        
with open(outpath, 'w') as outfile:
    json.dump(data, outfile)
