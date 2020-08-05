import json
import numpy as np
from dateutil.parser import parse

#metadata kind of on bucket, in uniques

with open('/home/jq/software/triplet-all/spider_train_subset.json') as f:
    dat = json.load(f)

def float_all(lst):
    try:
        [float(l) for l in lst if l is not None]
        return True
    except:
        return False


def int_all(lst):
    try:
        a = [int(l) for l in lst if l is not None]
        b = [float(l) for l in lst if l is not None]
        return all([aa==bb for aa, bb in zip(a, b)])
    except:
        return False

def dates_all(lst):
    try:
        [parse(l) for l in lst if l is not None]
        return True
    except:
        return False

metadata = {}
ints = []
floats = []
dates = []
for db in dat.keys():
    db_ = db.lower()
    for tab in dat[db].keys():
        tab_ = tab.lower()
        metadata[db_][tab_] = {}
        print(dat[db][tab])
        for col in dat[db][tab].keys():
            col_ = col.lower()
            if dates_all(dat[db][tab][col]):
                metadata[db_][tab_][col_] = 'dates'
                dates.append((db_, tab_, col_))
            elif int_all(dat[db][tab][col]):
                metadata[db_][tab_][col_] = 'integers'
                ints.append((db_, tab_, col_))
            elif float_all(dat[db][tab][col]):
                metadata[db_][tab_][col_] = 'floats'
                floats.append((db_, tab_, col_))
            else:
                metadata[db_][tab_][col_] = 'others'
print('dates are {}'.format(dates))
print('ints are {}'.format(ints))
print('floats are {}'.format(floats))
with open('metadata.json', 'w') as f:
    json.dump(f, metadata)
    

