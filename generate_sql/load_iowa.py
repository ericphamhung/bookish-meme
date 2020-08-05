import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from tqdm import tqdm_notebook as tqdm

path_to_file = 'data/iowa/'

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



df = pd.read_csv(path_to_file + 'last_quarter_2018.csv')

df['store_coord']=df['store location'].str.split("\n").str[2]
df = df.drop(['store location'], axis=1)
df = df.drop(['Unnamed: 0'], axis=1)

df.to_csv(path_to_file + 'last_quarter_2018_v2.csv', index=False)

#CREATING TABLE IN POSTGRES
string = '''
DROP TABLE IF EXISTS iowa;
CREATE TABLE iowa (
"invoice/item number"       varchar(50),
"date"                      timestamp,
"store number"               integer,
"store name"                varchar(50),
"address"                   varchar(50),
"city"                      varchar(50),
"zip code"                   varchar(50),
"county number"              real,
"county"                    varchar(50),
"category"                   real,
"category name"             varchar(50),
"vendor number"              real,
"vendor name"               varchar(50),
"item number"                real,
"item description"          varchar(500),
"pack"                       real,
"bottle volume (ml)"         real,
"state bottle cost"        real,
"state bottle retail"      real,
"bottles sold"               real,
"sale (dollars)"           real,
"volume sold (liters)"     real,
"volume sold (gallons)"    real,
"store_coord"              varchar(50), 
PRIMARY KEY("invoice/item number")
);
'''
cursor.execute(string)
con.commit()
print('created table')

string = '''
COPY iowa from '{}' CSV HEADER;
'''.format(path_to_file + 'last_quarter_2018_v2.csv')
cursor.execute(string)
con.commit()
print('copied table')


#RENAMING COLUMNS
rename_columns = {
 'invoice/item number':'invoice_id',
 'store number':'del2',
 'store name':'store',
 'zip code':'zipcode',
 'county number':'del4',
 'category':'del5',
 'category name':'category',
 'vendor number':'del6',
 'vendor name':'del7',
 'item number':'del8',
 'item description':'product',
 'bottle volume (ml)':'del9',
 'state bottle cost':'cost',
 'state bottle retail':'del10',
 'bottles sold':'del11',
 'sale (dollars)':'sale',
 'volume sold (liters)':'volume',
 'volume sold (gallons)':'del12'}

for k,v in tqdm(rename_columns.items()):
    string1 = '''
        ALTER TABLE iowa
        RENAME COLUMN "{0}" TO "{1}";
        '''
    cursor.execute(string1.format(k,v))
    con.commit()

    
#REMOVING IRRELEVANT COLUMNS
drop_string = """
ALTER TABLE iowa
DROP COLUMN IF EXISTS {};
"""    
del_list= []
[del_list.append('del{}'.format(i)) for i in range(13)]
for del_ in del_list:
    cursor.execute(drop_string.format(del_))
    con.commit()
    print("dropped {}".format(del_))
    
cursor.execute('''
DELETE
FROM iowa
WHERE store_coord IS NULL
''')
con.commit()



print('done')

