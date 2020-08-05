import sys
import itertools
from definitions import *
from utils import *
import numpy as np
from copy import copy
from time import time
from functools import reduce



## BETTER WAY -
  ## UNKNOWN START STRINGS:
  ## http://inform7.com/sources/src/inweb/Woven/3-bnf.pdf
  ## use known
  ##
  ## KNOWN -
  ##  Just use transition matrix from each state, M
  ##  Cycles detected by M_i. (col) and nM_i. (col for nth degree) being the same
  ## do known only for now


#don't see the lists in the dict, so cant use this way
# def _finditem(obj, key):
#     if key in obj: return obj[key]
#     for k, v in obj.items():
#         if isinstance(v,dict):
#             item = _finditem(v, key)
#             if item is not None:
#                 return item
# def _item_in(obj, key):
#     if _finditem(obj, key) is not None:
#         return 1
#     else:
#         return 0
#
# def follow_grammar_path_in(gr, tok):
#     if not is_terminal(tok):
#

class SimpleGrammar:

    def __init__(self, file_name, maxdepth = -1, verbose = False):
        super(SimpleGrammar, self).__init__()


        self.gr = self.parse(file_name)
        self.gram_keys = [k for k in self.gr.keys()]
        m = len(self.gram_keys)
        self.or_mat = np.zeros((m, m))
        self.gr2 = {}
        for k in self.gram_keys:
            # print(self.gr[k])
            self.gr2[k] = self.gr[k]['items']
        #self.or_mat, self.and_mat = self.get_or_and_mat()

        self.learn_ = self.learn_these()
        self.reverse_gr = {}

        for k in self.gram_keys:
            lst = self.gr[k]['items']
            assert isinstance(lst, list)
            # self.reverse_gr[string] = []
            for string in lst:
                assert isinstance(string, str)
                if string in self.reverse_gr:
                    self.reverse_gr[string].append(k)
                else:
                    self.reverse_gr[string] = [k]
        self.terminals, self.terminal_toks = [], []


    def parse(self, fn):
        with open(fn, 'r') as f:
            text_lst = f.readlines()

        text_lst = rem_ws_from_list(text_lst)

        gram_dct = dict()
        numor = 0

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

        gram_keys = [k for k in gram_dct.keys()]

        m = len(gram_keys)

        # or_mat = np.zeros((m, m))
        # and_mat = np.zeros((m, m))
        fin_dict = dict()

        for j, k in enumerate(gram_keys):
            fin_dict[k] = dict()

            string = gram_dct[k]
            lst, cond = split_string_or(string)

            if not cond and is_terminal(lst[0]):
                fin_dict[k]['type'] = 'term'
            elif not cond:
                fin_dict[k]['type'] = 'and'
                lst, _ = split_string_and(lst[0])
            else:
                fin_dict[k]['type'] = 'or'
                # self.or_mat = add_connection(self.or_mat, j)


            if not isinstance(lst, list):
                print('{}, {}'.format(k, lst))
                assert False
            fin_dict[k]['num'] = len(lst)
            fin_dict[k]['items'] =  lst

        return fin_dict


if __name__ == '__main__':
    g = SimpleGrammar('sql_simple_transition_3.bnf')

    # t = time()
    # print(g.is_on_path('<query>', 'SELECT'))
    # print(time()-t)
    # for i in range(10):
    #     print(g.get_string(i))


    #gr = g.get_grammar()
    #revg = g.reverse_gr
    #print(revg['[[like_pattern]]'])
    #print(gr['<condition>'])
    #print(g.match('SELECT * FROM [[table_name]]'))

    #print(g.get_grammar_tree())
