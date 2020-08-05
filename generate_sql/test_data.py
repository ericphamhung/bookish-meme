from simpler_grammar import SimpleGrammar
from copy import copy
import re
import json
from definitions import *

mathtoks = ['=', '<', '>', '<>', '!=', '>=', '<=', '(',')']

def rem_terminals_tokenize_string(grammar, string):

    for tok in grammar.terminal_toks:
        if '[[' in tok:
            if tok == '[[value_pattern]]' or tok == '[[int_pattern]]':
                string = re.sub(resolve_dict[tok], ' ', string)
        else:
            if tok in mathtoks:
                string = string.replace(tok, ' ')
            elif tok != 'SELECT' :
                string = string.replace(' ' + tok.lower()+ ' ', ' ')
                string = string.replace(' ' + tok.upper()+ ' ', ' ')
            else:
                string = string.replace(tok.lower()+ ' ', ' ')
                string = string.replace(tok.upper()+ ' ', ' ')
    toks = string.split(' ')
    ret = []
    #not needed, I think
    for tok in toks:
        if tok != "" and tok != " " and tok.upper() not in grammar.terminal_toks:
            ret.append(tok)
    return ret

def all_toks_not_matched(tokens, tables):
    toks_not_in = []
    for tok in tokens:
        matched = False
        for key in tables:
            assert isinstance(tables[key], list)
            if tok == key:
                matched = True
            elif tok in tables[key]:
                matched = True
            if matched:
                break
        if not matched:
            toks_not_in.append(tok)
    return set(toks_not_in)



def iterate_through_data_tables(grammar, data, tables):
    for k in data.keys():
        toks = rem_terminals_tokenize_string(grammar, data[k]['sql'])
        toks = all_toks_not_matched(toks, tables)
        print('{} unmatched tokens in {}'.format(len(toks), data[k]['sql']))
        print(toks)



if __name__ == '__main__':
    jsonfile = '/home/jq/software/triplet-all/spider_train_subset.json'

    with open(jsonfile, 'r') as f:
        data = json.loads(f.read())

    with open('spider_tables.json', 'r') as f:
        spider_tables = json.loads(f.read())

    g = SimpleGrammar('sql_simple_transition.bnf')
    iterate_through_data_tables(g, data, spider_tables)
