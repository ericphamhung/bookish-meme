# def is_terminal(string):
#     if isinstance(string, list) and isinstance(string[0], str):
#         val = True
#         for s in string:
#             if not val:
#                 break
#             val =  val and (grammar_tok.search(s) is None)
#     elif isinstance(string, str):
#         val = (grammar_tok.search(string) is None)
#     elif isinstance(string, list) and isinstance(string[0], list):
#         val = False
#     else:
#         raise ValueError('Error in is_terminal, got {}'.format(string))
#     return val
#def add_connection(mat, i, j, type = 'a')

    #below kept as a reminder.
    #not currently necessaty
    # if type == 'a':
    #     divby = 1.0
    # else:
    #     divby = 1.0*len(j)
    # mat[i, j] += 1.0/divby


## grammar

from torch import nn

from pampy import match, HEAD, TAIL, _
import sys
from itertools import chain
from utils import *
from node import Node
import numpy as np

#at least colname list and table name should be given in the actual
# this is currently just for chekcing
#no longer needed. Commenting
# regex_strings = {'<<value_pattern>>':re.compile("[-+]?[0-9]+\.?[0-9]*"),
#                 '<<like_pattern>>':re.compile('\"[^\"]\"'),
#                 '<<colname_list>>':re.compile('[a-z]+'),
#                 '<<table_name>>':re.compile('[a-z]+')}









## BETTER WAY -
  ## UNKNOWN START STRINGS:
  ## http://inform7.com/sources/src/inweb/Woven/3-bnf.pdf
  ## use known
  ##
  ## KNOWN -
  ##  Just use transition matrix from each state, M
  ##  Cycles detected by M_i. (col) and nM_i. (col for nth degree) being the same
  ## do known only for now

class Grammar(nn.Module):

    def __init__(self, file_name, maxdepth = -1, verbose = False):
        super(Grammar, self).__init__()
        self.loop_in = resolve_loop_in.keys()
        self.gr = {}
        self.maxdepth = maxdepth
        self.verbose = verbose
        self.klist = []
        self.parse(file_name)
        self.reverse_gr = {}
        for k in self.gr:
            lst = self.gr[k]
            assert isinstance(lst, list)
            for string in lst:
                assert isinstance(string, str)
                self.reverse_gr[string] = k
        self.terminals = []
        for key in self.gr.keys():
            m = ''.join(chain(*self.gr[key]))
            if (grammar_tok.search(m) is None):
                self.terminals.append(key)

        # self.loops = {}
        # for l in list_of_loops:
        #     lf = l['from']
        #     self.loops[lf] = l
        #     self.loops[lf]['num_left'] = l['max']

        start_ = '<start>'
        self.all_strings = self.new_follow(start_)

        num_paths = np.sum(self.or_mat)
        print('{} strings, {} predicted paths'.format(len(self.all_strings), num_paths))
        #assert len(self.all_strings) == num_paths


    def new_follow(self, tok, in_loop_lev = 0):

        if isinstance(tok, list):
            assert len(tok) == 1
            tok = tok[0]

        if tok in self.loop_in:
            new_val = "\s" + resolve_loop_in[tok]

        elif tok not in self.gram_keys:
            if is_terminal(tok):
                if tok in resolve_dict.keys():
                    new_val = resolve_dict[tok]
                else:
                    new_val = tok + "\s"
            else:
                raise ValueError("{} is not a terminal nor in grammar keys".format(tok))
        else:
            val = ""
            # if tok in self.loops:
            #     if self.loops[tok]['num_left'] > 0:
            #         self.loops[tok]['num_left'] -= 1
            #     else:
            #         breaker
            new_val = []
            if self.trans_dict[tok] == 'or':
                ind = self.gram_keys.index(tok)
                num_ = int(np.sum(self.or_mat[:, ind]))
                assert num_ > 1 and num_ == len(self.gr[tok])
                val = [val] * num_
                for ii, va in enumerate(val):
                    vhat = self.new_follow(self.gr[tok][ii])
                    if isinstance(vhat, str):
                        vhat = [vhat]
                    vv =  [va]*len(vhat)
                    for iii, v in enumerate(vv):
                        vr = v+vhat[iii]
                        new_val.append(vr)

            elif self.trans_dict[tok] == 'and':
                ind = self.gram_keys.index(tok)
                num_ = int(np.sum(self.and_mat[:, ind]))
                assert num_ == len(self.gr[tok])
                for ii in range(num_):
                    vhat = self.new_follow(self.gr[tok][ii])
                    if isinstance(vhat, str):
                        vhat = [vhat]
                    vv =  [val]*len(vhat)
                    for iii, v in enumerate(vv):
                        vr = v+vhat[iii]
                        new_val.append(vr)

            elif self.trans_dict[tok] == 'term':
                new_val = self.new_follow(self.gr[tok])
            else:
                raise ValueError("I've gotten to a place that shouldn't be possible")
        return new_val

    # self.all_strings = self.new_follow(start_)
    # for string in self.all_strings:
    #     print(string)


#     self.get_loops(verbose=True)
#
#
#
#
# def get_loops(self, max_len_follow = 10, verbose = False):
#     for k in self.gr.keys():
#         if verbose: print(k)
#         lst = [k]
#         for i in range(max_len_follow):
#             if verbose: print(i)
#             if i == 0:
#                 kk = self.gr[k]
#             else:
#                 kk = self.gr[kk]
#
#
#             if is_terminal(kk):
#                 break
#             if kk in lst:
#                 self.num_loops += 1
#                 if verbose: print('loop in {}'.format(k))
#                 break
#



def insert_in(self, lst1, lst2):
    if isinstance(lst1, list):
        m1 = len(lst1)
        if isinstance(lst2, list):
            m2 = len(lst2)
            new_lst = []
            l2lst = isinstance(lst2[0], list)
            l1lst = isinstance(lst1[0], list)
            for l2 in lst2:
                if l1lst:
                    for l1 in l1lst:
                        if l2lst:
                            new_lst.append(l1.extend(l2))
                        else:
                            new_lst.append(l1.append(l2))
                else:
                    for l1 in l1lst:
                        if l2lst:
                            vvv = l1 + " {}"
                            new_lst.append()

#def better_follow(self, tok, in_loop_lev = 0):
def stringify(self, tok, addto):
    if tok in self.terminals:
        return addto.format(self.gr[tok])

    ii = self.gram_keys.index(tok)
    if self.or_mat[:, ii].sum() > 0:

def new_follow(self, tok, in_loop_lev = 0):

    if isinstance(tok, list):
        assert len(tok) == 1
        tok = tok[0]

    if tok in self.loop_in:
        new_val = "\s" + resolve_loop_in[tok]

    elif tok not in self.gram_keys:
        if is_terminal(tok):
            if tok in resolve_dict.keys():
                new_val = resolve_dict[tok]
            else:
                new_val = tok + "\s"
        else:
            raise ValueError("{} is not a terminal nor in grammar keys".format(tok))
    else:
        val = ""
        # if tok in self.loops:
        #     if self.loops[tok]['num_left'] > 0:
        #         self.loops[tok]['num_left'] -= 1
        #     else:
        #         breaker
        new_val = []
        if self.trans_dict[tok] == 'or':
            ind = self.gram_keys.index(tok)
            num_ = int(np.sum(self.or_mat[:, ind]))
            assert num_ > 1 and num_ == len(self.gr[tok])
            val = [val] * num_
            for ii, va in enumerate(val):
                vhat = self.new_follow(self.gr[tok][ii])
                if isinstance(vhat, str):
                    vhat = [vhat]
                vv =  [va]*len(vhat)
                for iii, v in enumerate(vv):
                    vr = v+vhat[iii]
                    new_val.append(vr)

        elif self.trans_dict[tok] == 'and':
            ind = self.gram_keys.index(tok)
            num_ = int(np.sum(self.and_mat[:, ind]))
            assert num_ == len(self.gr[tok])
            for ii in range(num_):
                vhat = self.new_follow(self.gr[tok][ii])
                if isinstance(vhat, str):
                    vhat = [vhat]
                vv =  [val]*len(vhat)
                for iii, v in enumerate(vv):
                    vr = v+vhat[iii]
                    new_val.append(vr)

        elif self.trans_dict[tok] == 'term':
            new_val = self.new_follow(self.gr[tok])
        else:
            raise ValueError("I've gotten to a place that shouldn't be possible")
    return new_val
    def follow(self, string):
        if string not in self.gr.keys() and not is_terminal(string):
            val = False
        elif string not in self.gr.keys():
            val = False
        elif not is_terminal(string):
            val = self.follow(self.gr[string])
        else:
            val = True

        return val

    def reverse_follow(self, string):
        if string == "":
            val = True
        elif notin(string, self.reverse_gr.keys()) and is_terminal(string):
            val = False
            print(self.reverse_gr.keys())
            print(1)
            print(string)
        elif notin(string, self.reverse_gr.keys()):
            val = False
            print(string)
        elif is_terminal(string):
            val = self.reverse_follow(self.reverse_gr[string])
        else:
            val = True

        return val

    def match_old(self, string):
        string_tokens = string.split(' ')
        for t in self.terminals:
            p = match(string_tokens, [self.gr[t], TAIL], lambda h, t: (h, t))
            print(p)

    def match(self, string):
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

        val = True
        for t in string_tokens:
            if not val:
                break
            val = val and self.reverse_follow(t)
        return val

    #not quite correct for
    def num_paths_this_branch(self, loc):
        if is_terminal(self.gr[loc]) and isinstance(self.gr[loc], list):
            val = len(self.gr[loc][0])
        else:
            val = len(self.gr[loc])

        return val

    # def all_strings(self):
    #     for l



    def get_subtrees(self, loc):

        dct = self.gr[loc]
        n_subtree = len(dct)
        length_ = 1 # starting with the current location
        terminal_flag = False
        stree_ = []
        for i, branch in enumerate(dct):
            if terminal_flag:
                assert is_terminal(branch)
            if isinstance(branch, list) and isinstance(branch[0], list):
                assert len(branch) == 1 #shouldn't need to check, but...
                length, stree = self.get_subtrees(branch[0])
            elif is_terminal(branch):
                if i > 0:
                    assert terminal_flag
                else:
                    terminal_flag = True
                    length = 1
                stree = [branch]
            else: #will this even happen?  unsure atm
                length, stree = get_subtrees(branch)

            stree_.append(stree)
            length_ += length

        return stree_, length_




    def parse(self, fn):
        with open(fn, 'r') as f:
            text_lst = f.readlines()

        text_lst = rem_ws_from_list(text_lst)

        gram_dct = dict()
        trans_dict = dict()
        last_key = None


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

        or_mat = np.zeros((m, m))
        and_mat = np.zeros((m, m))

        for j, k in enumerate(gram_keys):
            string = gram_dct[k]
            lst, cond = split_string_or(string)

            if not cond and is_terminal(lst[0]):
                trans_dict[k] = 'term'
            elif not cond:
                trans_dict[k] = 'and'
                lst, _ = split_string_and(lst[0])
                for l in lst:
                    and_mat = add_connection(and_mat, j,  \
                        gram_keys.index(l), True)
            else:
                trans_dict[k] = 'or'
                for l in lst:
                    or_mat = add_connection(or_mat, j,  \
                        gram_keys.index(l))


            if not isinstance(lst, list):
                print('{}, {}'.format(k, lst))
                assert False
            gram_dct[k] = lst

        self.gr = gram_dct
        self.trans_dict = trans_dict
        self.or_mat = or_mat
        self.and_mat = and_mat
        self.gram_keys = gram_keys


    def get_grammar_keys(self):
        return self.gram_keys

    def get_grammar_tree(self):
        return self.tree

    def get_grammar(self):
        return self.gr

    def get_reverse_grammar(self):
        return self.reverse_gr

    def get_terminals(self):
        return self.terminals
    # def get_grammar_tree(self, starting_pos, maxdepth):
    #     return self.tree
        # for

    def get_string(self, i):
        return self.all_strings[i]

    def get_num_strings(self):
        return len(self.all_strings)

    def evaluate_string(self, string):
        pass #root =

if __name__ == '__main__':
    g = Grammar('sql_simple_transition.bnf')
    print(g.match('SELECT COUNT(*) FROM [[table_name]]'))
    # for i in range(10):
    #     print(g.get_string(i))


    #gr = g.get_grammar()
    #revg = g.reverse_gr
    #print(revg['[[like_pattern]]'])
    #print(gr['<condition>'])
    #print(g.match('SELECT * FROM [[table_name]]'))

    #print(g.get_grammar_tree())
