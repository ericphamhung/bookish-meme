import re
import json
grammar_tok = re.compile("(?<=\<)(.+?)(?=\>)")
begin_tok = re.compile("\<(.+)\>\s*::=")
beginning_split = re.compile("::=")
resolve_tok = re.compile("(?<=\[\[)(.+?)(?=\]\])")
or_tok = re.compile('\|')
comment = re.compile('#.*')
fn_tok = re.compile("[^,]*\(.*\)")

# list_of_loops = [{'from':'loopcond', 'to':['loopcondand', 'loopcondor'], 'max':4}]
resolve_order = ['[[value_pattern]]', '[[int_pattern]]', '[[like_pattern]]']
resolve_loop_in = {'<colname_list3>':"([^ ,]+)(?:\s*,\s*([^ ,]+))+\s+",
                "<sellist2>":"([^ ,]+)(?:\s*,\s*([^ ,]+))+\s+",
                "<sellist4>":"([^ ,]+)(?:\s*,\s*([^ ,]+))+\s+"}

resolve_loop = re.compile("([^ ,]+)(?:\s*,\s*([^ ,]+))+\s+")
loop_in = ['<colname_list3>', "<sellist2>", "<sellist4>"]


resolve_dict = {'[[table_name]]':"([^ ,]+)\s*",
                '[[colname_pattern]]':"([^ ,]+)\s+",
                '[[value_pattern]]':"\s+[-+]?[0-9]*\.?[0-9]+\s*",
                '[[int_pattern]]':"\s+[0-9]+\s*",
                '[[like_pattern]]':"([^ ,]+)(?:\s*,\s*([^ ,]+))*\s+"}

resolve_sampling = {'[[value_pattern]]':"real",
                '[[int_pattern]]':"int"}

with open('spider_tables_lowercase.json', 'r') as f:
    spider_db = json.loads(f.read())

max_tables = 0
max_columns = 0
for k in spider_db.keys():
    max_tables = max(max_tables, len(spider_db[k]))
    for t in spider_db[k].keys():
        max_columns = max(max_columns, len(spider_db[k][t]))

# def add_table_name(k, dict):
#     k = k + '_added'
#     if k in dict:
#         k = add_table_name(k, dict)
#     return k


# resolve_sampling['[[db_name]]'] = list(spider_db.keys())
# resolve_sampling['[[db_name]]'].sort()

# resolve_sampling['[[table_name]]'] = {}
# resolve_sampling['[[colname_pattern]]'] = {}

# for t in spider_db:
#     resolve_sampling['[[table_name]]'][t] = spider_db[t]
#     resolve_sampling['[[colname_pattern]]'][t] = {}
#     for s in spider_db[t]:
#         resolve_sampling['[[colname_pattern]]'][t][s] = spider_db[t][s]
