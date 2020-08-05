import pandas as pd
import psycopg2
from time import time
import matplotlib.pyplot as plt
from pprint import pprint
from matplotlib.dates import (DateFormatter)
from datetime import datetime
import numpy as np
import itertools
from collections.abc import Iterable

#below temporary

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

table = 'ttcc'


allowed = ['rep', 'product', 'customer']

#um, with is pointless here
start_sql = '''
with first as (
    select {1}, min(date) first_date from {0} group by {1}
)
select * from first
'''

end_sql = '''
with last as (
    select {1}, max(date) last_date from {0} group by {1}
)
select * from last
'''

def single_call(string, start = True):
    if start:
        string = start_sql.format(table, string)
    else:
        string = end_sql.format(table, string)
    return pd.read_sql(start_sql.format(table, string), con=con)

def get_starts(which=1):
    if isinstance(which, int):
        assert which in [1, 2, 3]
        if which == 1:
            which = allowed
        elif which == 2:
            which = [a for a in itertools.product(allowed, allowed) if a[0]!=a[1]]
        else:
            which = [a for a in itertools.product(allowed, allowed, allowed) if a[0]!=aa[1] and a[0]!=a[2] and a[1]!=a[2]]
    elif isinstance(which, string):
        assert which in allowed
        which = [which]
    elif isinstance(which, Iterable):
        assert isinstance(which, str)
        assert all([a in allowed for a in which])
    
    dfs = []
    if isinstance(which[0], str):
        for w in which:
            dfs.append(single_call(w))
    elif isinstance(which[0], Iterable):
        for w in which:
            string = ','.join([w0 for w0 in w])
            dfs.append(single_call(string))

    dfs = pd.concat(dfs)
    return dfs


def get_ends(which=1):
    if isinstance(which, int):
        assert which in [1, 2, 3]
        if which == 1:
            which = allowed
        elif which == 2:
            which = [a for a in itertools.product(allowed, allowed) if a[0]!=a[1]]
        else:
            which = [a for a in itertools.product(allowed, allowed, allowed) if a[0]!=aa[1] and a[0]!=a[2] and a[1]!=a[2]]
    elif isinstance(which, string):
        assert which in allowed
        which = [which]
    elif isinstance(which, Iterable):
        assert isinstance(which, str)
        assert all([a in allowed for a in which])
    
    dfs = []
    if isinstance(which[0], str):
        for w in which:
            dfs.append(single_call(w, False))
    elif isinstance(which[0], Iterable):
        for w in which:
            string = ','.join([w0 for w0 in w])
            dfs.append(single_call(string, False))

    dfs = pd.concat(dfs)
    return dfs

effective_range_sql = '''
select {1}, min(date) first_date, max(date) last_date from {0} group by {1}
'''

def get_effective_range():
    pass
#don't need to normalize for lifetime
def get_lifetime_value():
    pass

all_entries_effect_sql = '''
with reference as (
    select sum(sale) sales from {0} where extract(year from date)={1}
),
test as (
    select sum(sale) as sales from {0} where {2}
)
select sales from reference
union
select sales from test
'''

def one_call(ref, dd):
    string = all_entries_effect_sql.format(table, ref, dd)
    return pd.read_sql(string, con=con)


all_entries_effect_sql = '''
with reference as (
    select sum(sale) sales from {0} where extract(year from date)={1}
),
test as (
    select sum(sale) as sales from {0} where {2}
)
select sales from reference
union
select sales from test
'''

def one_call(ref, dd):
    string = all_entries_effect_sql.format(table, ref, dd)
    return pd.read_sql(string, con=con)


def dformat(date):
    return date.strftime("%Y-%m-%d")

def all_entries_effect_oneY(reference_year = 2015):
    base_str = "date >= '{}'::date and date <= '{}'::date"
    datesdf = get_starts(1)
    datesdf = datesdf[datesdf.first_date.dt.year>reference_year]
    yoy_diff_normalized = []
    last_date = pd.read_sql('select max(date) m from {0}'.format(table), con=con).loc[0, 'm']
    for i, row in datesdf.iterrows():
        date = row['first_date']
        val = (last_date - date)
        if val < pd.Timedelta(1, 'y'):
            factor = val.days/365.0
            
            string = base_str.format(dformat(date), dformat(last_date))
        else:
            factor = 1.0
            next_date = date+pd.Timedelta(1, 'y')
            string = base_str.format(dformat(date), dformat(next_date))
        vals = one_call(reference_year, string)['sales'].tolist()
        yoy_diff_normalized.append((vals[1]-vals[0])*factor)
    
    datesdf['yoy_diff_normalized'] = yoy_diff_normalized
    return datesdf