from lark import Lark

with open('sql_92_complete.bnf', 'r') as f:
    grammar = f.read()

parser = Lark(grammar, parser='lalr')

print(parser)
