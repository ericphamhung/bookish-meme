import re
from copy import copy


with open('sql_92_complete.bnf', 'r') as f:
    grammar = f.read()

lst = re.findall('\<(.*?)\>', grammar)

new_lst =[]

for m in lst:
    n = copy(m)
    n = n.replace(' ', '_')
    grammar = re.sub(m, n, grammar)

grammar, cnt = re.subn('::=', ':', grammar)

lst = re.findall('^(\<(.*?)\>).*', grammar)

print(lst)
# print(grammar)
