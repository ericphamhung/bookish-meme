import torch
import re
import numpy as np

## BETTER WAY -
  ## UNKNOWN START STRINGS:
  ## http://inform7.com/sources/src/inweb/Woven/3-bnf.pdf
  ## use known
  ##
  ## KNOWN -
  ##  Just use transition matrix from each state, M
  ##  Cycles detected by M_i. (col) and nM_i. (col for nth degree) being the same
  ## do known only for now

  ## Only known for now


grammar_tok = re.compile("\<(.+)\>")
begin_tok = re.compile("\<(.+)\>\s*::=")
beginning_split = re.compile("::=")
resolve_tok = re.compile("\[\[(.+)\]\]")
or_tok = re.compile('\|')
comment = re.compile('#.*')


def rem_ws_from_list(lst):
    ret = []
    for w in lst:
        w = w.replace('  ', ' ')
        w = w.replace('  ', ' ')
        w = w.strip()
        if w == "":
            continue
        ret.append(w)
    return ret

def xor(a, b):
    return bool(a) ^ bool(b)

def split_token(token):
    token = token.replace('><', '> <')
    token = token.replace('>[', '> [')
    token = token.replace(']<', '] <')
    tok_lst = grammar_tok.split(token.strip())
    if type(tok_lst) is str:
        tok_lst = [tok_lst]
    return rem_ws_from_list(tok_lst)

def get_grammar_dict(fname):
    with open(fname, 'r') as f:
        text_lst = f.readlines()

    #if this works, don't need to have ;;; in anymore
    #  ... it works
    # new_lst = []
    # for l in text_lst:
    #     l = l.replace(';;;', '')
    #     new_lst.append(l)
    # text_lst = new_lst

    text_lst = rem_ws_from_list(text_lst)

    gram_dct = dict()
    last_key = None

    # makes_sense_to_cont = False

    for l in text_lst:
        l = l.strip()
        m = comment.match(l)

        if m:
            val = m.start()
        else:
            val = -1

        if val == 0:
            continue
        elif val > 0:
            l = l[0:(val-1)]

        if l == "": continue

        in_parse = (begin_tok.search(l) is None)
        if in_parse:
            assert last_key in gram_dct
            gram_dct[last_key] += " "
            gram_dct[last_key] += l
        else:
            after_split = beginning_split.split(l)
            assert len(after_split) == 2
            last_key = after_split[0].strip()
            gram_dct[last_key] = after_split[1].strip()

    for k in gram_dct.keys():
        string = gram_dct[k]
        lst = or_tok.split(string)
        if isinstance(lst, str):
            lst = [lst]
        lst = rem_ws_from_list(lst)

        lst_larger = []

        for l in lst:
            ll = l.split()
            if isinstance(ll, str):
                ll = [ll]
            ll = rem_ws_from_list(ll)
            lst_larger.append(ll)
        gram_dct[k] = lst_larger

    return gram_dct

def get_ind_of(string, lst):
    ind = np.where(string == np.array(lst))[0][0]
    return ind


#Not useful at the moment...
st = '<start>'
def create_grammar_matrix(gram_dct):
    assert st in gram_dct, "Only dealing with defined start grammars ATM"
    m = len(gram_dct)-1
    assert len(gram_dct[st]) == 1, "Only dealing with one start string ATM"

    gram_array = np.zeros((m, m))
    gram_start = np.zeros(m)
    key_lst = [k for k in gram_dct.keys() if k != st]

    _start = get_ind_of(key_lst, gram_dct[st][0])
    gram_start[_start] = 1





if __name__ == '__main__':
    g = get_grammar_dict('sql_simple.gr')
    print(g)
    #create_grammar_matrix(g)
