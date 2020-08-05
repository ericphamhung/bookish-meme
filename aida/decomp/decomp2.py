import pandas as pd
import numpy as np
import psycopg2
from copy import copy


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

class DecomposeCompare:

    def __init__(self, table, lookat = ['category', 'sku', 'customer', 'rep'], 
                    reps = ['JG', 'AO', 'FM', 'Other']):

        df = pd.read_sql('''select sum(sale), category, sku, customer, rep, 
                            extract(month from date) as month,
                            extract(year from date) as year from {}
                            group by rep, category, sku, customer, month, year'''.format(table), con)
        self.reps = reps
        if 'Other' in reps:
            indx = [a in reps for a in df['rep']]
            df.loc[indx, 'rep'] = 'Other'
        
        df['date'] = df[['month', 'year']].apply(lambda x: pd.to_datetime('28-{}-{}'.format(int(x[0]), int(x[1]))), axis=1)
        df.set_index('date', inplace=True)
        self.df = df
                                                 
            
    def get_season_rest(self, subselect = None):
        if subselect is not None:
            assert False
            #subselection = 
        else:
            subselection = copy(self.df)
        
        series = subselection.sum(axis=1)
        season = series.groupby(series.index.month).mean()
        year = series.groupby(series.index.year).mean()
        
        season_std = series.groupby(series.index.month).std()
        rest = series - season

        return season, rest