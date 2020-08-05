import json
import pandas as pd
import glob
from grammar import Grammar

def rem_json_add(f, add=''):
    if 'wiki' in f:
        f = f.replace('.bz2', '')
    return f.replace('.json', add)







if __name__ == '__main__':
    directory = '/home/jq/software/triplet-all/text2sql-data/data/'
    qq_all = [f for f in glob.glob(directory+'*.json')]
    #qq_all.extend([f for f in glob.glob(directory+'*.json.bz2')])
    fields = [rem_json_add(f, '-fields.txt') for f in qq_all]
    schema = [rem_json_add(f, '-schema.csv') for f in qq_all]
    # print(len(schema))
    q = [q for q in qq_all if 'wiki' in q][0]
    print(q)
    with open(q, 'r') as file:
        ff = json.load(file)
    g = Grammar('sql_simple_transition.bnf')
    vvlst = ['select *', 'select * from tab INNER JOIN tab2']
    for v in vvlst:
        assert not g.match(v)
    for f in ff:
        #print(len(f['sql-original']))
        mat = g.match(f['sql-original'][0])
        if not mat:
            print(f['sql-original'])

    # for f in fields:
    #     with open(f, 'r') as file:
    #         file.read()
    #
    # for f in schema:
    #     if 'wiki' not in f:
    #         pd.read_csv(f)
    #     else:
    #         with open(f, 'r') as file:
    #             print(f)
    #             first = file.readline()
    #             first_len = len(first.split(','))
    #             print(first)
    #             for line in file.readlines():
    #                 toks = line.split(',')
    #                 if len(toks) > first_len:
    #                     print(line)
