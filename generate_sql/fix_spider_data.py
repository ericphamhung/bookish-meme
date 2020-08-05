import json

with open('spider_tables.json') as f:
    ff = json.load(f)

fixed = {}
for db in ff.keys():
    db_ = db.lower()
    fixed[db_] = {}
    for tab in ff[db].keys():
        tab_ = tab.lower()
        fixed[db_][tab_] = []
        for col in ff[db][tab]:
            fixed[db_][tab_].append(col.lower())

with open('spider_tables_lowercase.json', 'w') as f:
    json.dump(fixed, f)