import sys
import itertools
from definitions import *
from utils import *
import numpy as np
from copy import copy
from time import time
from functools import reduce
from networkx import Graph
import networkx as nx

def add_vals(val1, val2):
    if isinstance(val1, int) and isinstance(val2, int):
        val = val1+val2
    elif isinstance(val1, list) and isinstance(val2, int):
        val = add_vals(val2, val1)
    elif isinstance(val1, int) and isinstance(val2, list):
        val = [add_vals(val1, v) for v in val2]
    elif isinstance(val1, list) and isinstance(val2, list):
        val = []
        for v1 in val1:
            val.append([add_vals(v1, v2) for v2 in val2])
    else:
        raise ValueError("add vals cant deal with {} and {}".format(type(val1), type(val2)))
    return val


class SimpleGrammar:

    def __init__(self, file_name, maxdepth = -1, verbose = False, nograph = True):
        super(SimpleGrammar, self).__init__()


        self.gr = self.parse(file_name)
        self.gram_keys = [k for k in self.gr.keys()]
        m = len(self.gram_keys)
        self.or_mat = np.zeros((m, m))
        
        self.learn_ = self.learn_these()
        self.inverse_gr = {}

        for k in self.gram_keys:
            lst = self.gr[k]['items']
            assert isinstance(lst, list)
            # self.inverse_gr[string] = []
            for string in lst:
                assert isinstance(string, str)
                if string in self.inverse_gr:
                    self.inverse_gr[string].append(k)
                else:
                    self.inverse_gr[string] = [k]
        self.terminals, self.terminal_toks = [], []
        self.ands, self.ors = [], []
        for key in self.gr.keys():
            m = ''.join(itertools.chain(*self.gr[key]['items']))
            if (grammar_tok.search(m) is None):
                self.terminals.append(key)
                mm = self.gr[key]['items']
                assert isinstance(mm, list)
                assert len(mm) == 1
                self.terminal_toks.append(mm[0])
            elif self.gr[key]['type'] == 'and':
                self.ands.append(key)
            else:
                self.ors.append(key)

        self.common_sentence = self._common_sentence(verbose=False)
        self.nograph = nograph
        if not self.nograph:
            self.graph = Graph()
            for g in self.gram_keys:
                self.graph.add_node(g)
                self.graph.nodes[g]['type'] = self.gr[g]['type']

            nodes = list(self.graph.nodes)
            for g in nodes:
                for c in self.gr[g]['items']:
                    d = self.pathone(c)
                    if d != c:
                        self.graph.remove_node(c)
                    self.graph.add_edge(g, d, weight = 1.0)
        
        # nodes = list(self.graph.nodes)
        # for n in nodes:
        #     if n != self.pathone(n):
        #         print(n)
        #         print(self.pathone(n))
        #         print(self.graph.edges(n))
    
    def num_or(self, tok):
        if tok in self.terminals:
            return [0]
        if self.gr[tok]['type'] != 'or':
            val = 0
        else:
            val = 1
        items=self.gr[tok]['items']
        return [val+i for it in items for i in self.num_or(it)]


    def get_subgrammar(self, start):
        reachable_tokens = self.get_reachable_tokens(start)
        ret = copy(self)
        rem = []
        tokrem = []
        for r in self.gram_keys:
            if r not in reachable_tokens:
                if r in self.terminals:
                    for t in ret.gr[r]['items']:
                        tokrem.append(t)
                rem.append(r)
        for r in rem:
            del ret.gr[r]
        ret.terminal_toks = [r for r in ret.terminal_toks if r not in tokrem]
        ret.gram_keys = [r for r in ret.gr.keys()]
        wrong = [r for r in rem if r in ret.gram_keys]
        assert len(wrong) == 0
        ret.terminals = [r for r in ret.gr.keys() if r in self.terminals]
        ret.ands = [a for a in self.ands if a in ret.gram_keys]
        ret.ors = [a for a in self.ors if a in ret.gram_keys]
        ret.inverse_gr = {}
        for k in ret.gram_keys:
            lst = ret.gr[k]['items']
            assert isinstance(lst, list)
            for string in lst:
                assert isinstance(string, str)
                if string in ret.inverse_gr:
                    ret.inverse_gr[string].append(k)
                else:
                    ret.inverse_gr[string] = [k]
        if not self.nograph:
            ret.graph = Graph()
            for g in ret.gram_keys:
                ret.graph.add_node(g)
                ret.graph.nodes[g]['type'] = ret.gr[g]['type']

            nodes = list(ret.graph.nodes)
            for g in nodes:
                for c in ret.gr[g]['items']:
                    d = ret.pathone(c)
                    if d != c:
                        ret.graph.remove_node(c)
                    ret.graph.add_edge(g, d, weight = 1.0)
        return ret
                
                
            
    def get_reachable_tokens(self, start, plus_len = 0):
        reachable_tokens = [start]
        if start not in self.terminals:
            its = self.gr[start]['items']
            reachable_tokens.extend(its)
            for it in its:
                reachable_tokens.extend(self.get_reachable_tokens(it))
        return reachable_tokens
    
    def get_len_to_term(self, start):
        if start not in self.gram_keys:
            ret = 0
        elif start in self.terminals:
            ret = 1
        else:
            if start not in self.terminals:
                its = self.gr[start]['items']
                ret = 0
                for it in its:
                    m = 1+self.get_len_to_term(it)
                    ret = max(ret, m)
        return ret
        

    def len_to_terminal(self, tok):
    
        pmin = 1000
        pmax = -1
        for t in self.terminals:
            print('for {} from {}'.format(t, tok))
            for path in nx.all_simple_paths(self.graph, tok, t):
                p = len(path)
                print(p)
                pmin = min(pmin, p)
                pmax = max(pmax, p)
        return pmin, pmax

    def shortest_to_term(self, tok):
        pmin = 1000
        for t in self.terminals:
            for path in nx.all_shortest_paths(self.graph, tok, t):
                pmin = min(pmin, len(path))
        return pmin
    
    def longest_to_term(self, tok):
        assert False #because not right
        return max(self.num_or(tok))

    def num_tokens(self, tok):
        # print(tok)
        if tok in self.terminals or tok in self.terminal_toks:
            return 1
        elif tok in self.ands:
            val = 0
            for v in self.gr[tok]['items']:
                n_ = self.num_tokens(v)
                val = add_vals(val, n_) 
            return val
        else:
            val = []
            for v in self.gr[tok]['items']:
                val.append(self.num_tokens(v))
            return val



    def nterm(self, tok, depth = 0):
        if tok in self.terminals:
            val = 1
        elif tok in self.ands:
            items = self.gr[tok]['items']
            val = self.nterm(items[0])
            for it in items[1:]:
                itt = self.nterm(it, depth = depth+1)
                val = add_vals(val, itt)
        else:
            items = self.gr[tok]['items']
            val = [0]
            for it in items:
                val = add_vals(val, self.nterm(it, depth=depth+1))
        return val
        

    def recurse_common_sentence(self, tok, depth = 0, bef_term = False, verbose = False):
        tlist = []

        tok = self.pathone(tok)
        if verbose:
            print('{} at depth {}'.format(tok, depth))
        if bef_term:
            if tok not in self.gr:
                return None
            elif self.gr[tok]['type'] == 'term':
                return [tok]
        elif is_terminal(tok):
            return [tok]
        if self.gr[tok]['type'] == 'and':
            if verbose:
                print('{} is and'.format(tok))
            for t in self.gr[tok]['items']:
                vv = self.recurse_common_sentence(t, depth = depth + 1, bef_term=bef_term, verbose = verbose)
                if vv is not None:
                    tlist.extend(vv)
                else:
                    tlist.append(t)
        elif self.gr[tok]['type'] == 'term':
            if verbose:
                print('{} is term'.format(tok))
            tlist = self.gr[tok]['items']
        else:
            if verbose:
                print('{} is or'.format(tok))
            tlist.append(tok)

        return tlist

    def _sub_sentence(self, tok, bef_term = False, verbose = False):
        return ' '.join(self.recurse_common_sentence(tok, bef_term = bef_term, verbose=verbose))

    def _common_sentence(self, bef_term = False, verbose = False):
        return self._sub_sentence('<query>', bef_term = bef_term, verbose = verbose)


    def get_common_sentence(self):
        return self.common_sentence




    def tokenized_form(self):
        lst0 = copy(self.gr['<query>']['items'])
        return lst0, copy(self.gr), copy(self.inverse_gr)


    def _match_string(self, string):
        comm = self.common_sentence.split('<')[0]
        if not match_string(comm, string):
            return False
        rest = string.replace(comm, '')



    def _tokenize(self, string):
        fn_ = fn_tok.match(string)
        string_tokens = string.split(' ')
        if fn_ is not None:
            all_tokens = []
            for tok in string_tokens:
                if fn_tok.match(tok) is not None:
                    ttok = tok.replace('(', " ")
                    ttok = ttok.replace(')', " ")
                    ttok = ttok.split(' ')
                    ttok.insert(0, '(')
                    ttok.append(')')
                    all_tokens.extend(ttok)
                else:
                    all_tokens.append(tok)

            string_tokens = rem_ws_from_list(all_tokens)
        return string_tokens




    def in_terminal_toks(self, key):
        return key in self.terminal_toks

    def is_resolve(self, key):
        if self.in_terminal_toks(key):
            val = False
        else:
            val = test_resolve(key)
        return val



    def parse(self, fn):
        with open(fn, 'r') as f:
            text_lst = f.readlines()

        text_lst = rem_ws_from_list(text_lst)

        gram_dct = dict()
        numor = 0

        for l in text_lst:
            l = l.strip().lower()
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


    def get_num_learn(self):
        return len(self.learn_)
    def learn_these(self):
        dct = {}
        for k in self.gram_keys:
            if self.gr[k]['type'] == 'or':
                dct[k] = self.gr[k]['items']
        return dct

    def get_grammar_keys(self):
        return self.gram_keys

    def get_grammar(self):
        return self.gr

    def get_inverse_grammar(self):
        return self.inverse_gr

    def get_terminals(self):
        return self.terminals

    def get_terminal_toks(self):
        return self.terminal_toks


    def pathone(self, tok):
        v = tok
        while v in self.gr and self.gr[v]['num'] == 1:
            if v in self.terminals:
                break
            v = self.gr[v]['items'][0]
        return v

    def is_on_path(self, start_, terminal):
        #print(self.gr2[start_])
        return _item_in(self.gr2[start_], terminal)



if __name__ == '__main__':
    gram = SimpleGrammar('sql_simple_transition_2.bnf')
    subgram = gram.get_subgrammar('<sellist>')
    for g in subgram.gram_keys:
        print(g)
        print(subgram.get_reachable_tokens(g))
    # sentence = gram.gr['<query>']['items']
    # for g in sentence:
    #     if g in gram.ors:
    #         print(g)
    #         print(gram.num_tokens(g))
    #         print(gram.shortest_to_term(g))
    #         print(gram.longest_to_term(g))
    
 