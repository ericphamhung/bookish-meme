from definitions import *

def file_num(num):
    return '{:05d}'.format(num)

def tok_is_definition(string):
    return beginning_tok.search(string) is not None

def line_is_definition(string):
    return grammar_tok.search(string) is not None

def tok_is_resolve(string):
    return resolve_tok.search(string) is not None

def test_resolve(string):
    val = None
    if resolve_loop.search(string) is not None:
        val = loop_in
    else:
        for k in resolve_dict.keys():
            if re.search(resolve_dict[k], string) is not None:
                val = k
    return val

def match_string(comm, string):
    string = string.strip()
    return comm in string

def rem_ws_from_list(lst):
    ret = []
    for w in lst:
        w = w.strip()
        if w == '':
            continue
        ret.append(w)
    return ret

def notin(string, lst):
    lst11 = [l.upper() for l in lst]
    val = string.upper() not in lst11
    return val

def isin(string, lst):
    return not notin(string, lst)

def is_terminal_2(string, depth = 0):
    if isinstance(string, str):
        val = (grammar_tok.search(string) is None)
    elif isinstance(string, list) and depth < 100:
        val = True
        for s in string:
            if not val:
                break
            val = val and is_terminal_2(s, depth+1)
    else:
        raise ValueError('Depth greater than 100!')
    return val

def is_terminal(string):
    assert isinstance(string, str)
    val = (grammar_tok.search(string) is None)
    return val

def add_connection(mat, fr, to, and_ = False):
    if and_:
        # mat[to, fr] += 1
        mat[to, fr] = 1
    else:
        mat[to, fr] = 1
    return mat


def xor(a, b):
    return bool(a) ^ bool(b)

def split_string_or(token):
    tok_lst = token.strip().split('|')
    return rem_ws_from_list(tok_lst), len(tok_lst)>1

def split_string_and(token):
    token = token.replace('  ', ' ')
    token = token.replace('  ', ' ')
    token = token.replace('><', '> <')
    token = token.replace('> <', '>+<')
    tok_lst = token.strip().split('+')
    return rem_ws_from_list(tok_lst), len(tok_lst)>1
