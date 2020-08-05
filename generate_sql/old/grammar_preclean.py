import torch
import re
from pampy import match, HEAD, TAIL, _
import sys
from itertools import chain

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





grammar_tok = re.compile("\<(.+)\>")
begin_tok = re.compile("\<(.+)\>\s*::=")
beginning_split = re.compile("::=")
resolve_tok = re.compile("\[\[(.+)\]\]")
or_tok = re.compile('\|')
comment = re.compile('#.*')

sys.setrecursionlimit(4500)

def rem_ws_from_list(lst):
    ret = []
    for w in lst:
        w = w.strip()
        if w == '':
            continue
        ret.append(w)
    return ret

# def get_basis(lst):
#     print(lst)
#     if not isinstance(lst, list):
#         val = lst
#     elif len(lst)==1:
#         val = get_basis(lst[0])
#     else:
#         val = lst
#     return val
#
# def is_terminal(string):
#     string = get_basis(string)
#     if isinstance(string, list) and len(string)>1:
#         ret = True
#         for s in string:
#             ret  = ret and is_terminal(s)
#             if not ret:
#                 return False
#     else:
#         if(isinstance(str, list)):
#             string = string[0]
#         assert isinstance(string, str)
#         ret = (grammar_tok.search(string) is None)
#     return ret

# def use_dct_recognizer



def is_terminal(string):
    if isinstance(string, list) and isinstance(string[0], list) and len(string)==1:
        string = string[0]
    if isinstance(string, list):
        if len(string)>1:
            val = False
        else:
            print(string[0])
            val = (grammar_tok.search(string[0]) is None)
    else:
        val = (grammar_tok.search(string) is None)
    return val

def xor(a, b):
    return bool(a) ^ bool(b)

def split_token(token):
    token = token.replace('><', '> <')
    token = token.replace('>[', '> [')
    token = token.replace(']<', '] <')
    tok_lst = or_tok.split(token.strip())
    if type(tok_lst) is str:
        tok_lst = [tok_lst]
    return rem_ws_from_list(tok_lst)

class Node:

    def __init__(self, name, parent, dct, choice_num, depth=0):
        # if isinstance(name, list):
        #     assert len(name) == 1
        #     name = name[0]

        if parent is None:
            assert depth == 0
        if depth == 0:
            assert parent is None

        self.name = name
        self.choice_num = choice_num
        self.root = (parent is None)
        self.parent = parent
        print(dct[name])
        self.leaf = is_terminal(dct[self.name])
        #this would never happen, put in grammar
        # actually, eventually we want loops in select...
        # but maybe we should evaluate those separately...
        # would require a data change, though, likely...
        lst_ancestors = list(filter(bool, self.anscestor_names()))

        self.looped = False

        if self.name in lst_ancestors:
            print('Note that there is a loop in {}'.format(self.name))
            self.looped = True
            raise ValueError('loop with name ', self.name)
        if self.leaf:
            self.content = dct[self.name]
            self.children = None
            self.num_choices = len(self.content)
        else:
            self.content = None

            self.children = []
            self.num_choices = len(self.children)

            for i, c in enumerate(dct[self.name]):

                if (isinstance(c, list) and len(c) == 1):
                    if self.looped and self.name == c[0]:
                        continue
                    self.add_child(Node(name = c[0], parent = self, dct = dct, \
                        choice_num = i, depth = depth+1))
                elif isinstance(c, str):
                    if self.looped and self.name == c:
                        continue
                    self.add_child(Node(name = c, parent = self, dct = dct, \
                        choice_num = i, depth = depth+1))
                else:
                    for j, d in enumerate(c):
                        assert len(d) == 1 or isinstance(d, str)
                        if isinstance(d, str):
                            if self.looped and self.name == d:
                                continue
                            self.add_child(Node(name = d, parent = self,  \
                                dct = dct, choice_num = j, depth = depth+1))
                        else:
                            if self.looped and self.name == d[0]:
                                continue
                            self.add_child(Node(name = d[0], parent = self,  \
                                dct = dct, choice_num = j, depth = depth+1))





    def __eq__(self, other):
        assert type(other) is Node
        children_ = len(self.children) == len(other.children)
        if children_:
            try:
                for u, v in zip(self.children, other.children):
                    children_ = children_ and (u == v)
                    if not children_:
                        break
            except:
                children_ = False
        val = children_ and (self.name == other.name) and \
                (self.parent == self.parent)
        return val

    def get_num(self):
        return self.choice_num

    #need to add choices, etc as well, not right
    def strict_eq(self, other):
        assert type(other) is Node
        num_ = self.choice_num == other.get_num()
        children_ = len(self.children) == len(other.children)
        if children_:
            try:
                for u, v in zip(self.children, other.children):
                    children_ = children_ and (u == v) and \
                        (u.get_num()==v.get_num())
                    if not children_:
                        break
            except:
                children_ = False
        val = children_ and (self.name == other.name) and \
                (self.parent == self.parent)
        return val

    def add_child(self, child):
        if self.leaf:
            self.leaf = False
        self.children.append(child)

    def add_children(self, children_):
        for child in children_:
            self.add_child(child)

    def num_children(self):
        return len(self.children)

    def len_to_leaf(self):
        if self.leaf:
            return 0
        else:
            max_len = 0
            for child in children:
                l = child.len_to_leaf()
                if l > max_len:
                    max_len = l
            return max_len + 1

    def get_name(self):
        return self.name

    def anscestor_names(self):
        if self.root:
            lst = ''
        else:
            lst = [self.parent.get_name()]
            if self.name in lst:
                #Node has no attribute looped... wtf?
                #assert self.looped
                return []
            lst.extend(self.parent.anscestor_names())
        return lst

    def check_content(self):
        return xor(self.leaf, len(self.children)>1)



class Grammar:

    def __init__(self, file_name, maxdepth = -1, verbose = False):
        self.gr = {}
        self.maxdepth = maxdepth
        self.verbose = verbose
        self.klist = []
        self.parse(file_name)
        self.reverse_gr = {}
        for k in self.gr:
            for lst in self.gr[k]:
                assert isinstance(lst, list)
                for string in lst:
                    assert isinstance(string, str)
                    self.reverse_gr[string] = k
        self.terminals = []
        for key in self.gr.keys():
            m = ''.join(chain(*self.gr[key]))
            if (grammar_tok.search(m) is None):
                self.terminals.append(key)

        # self.top_level = self.gr['<start>']
        # assert len(self.top_level) == 1, "Only dealing with one top level atm"
        # self.tree = []
        # #need to add
        # self.loops = []
        # for i, vv in enumerate(self.top_level[0]):
        #     root_node = Node(name = vv, parent = None, dct = self.gr, \
        #             choice_num = i, depth = 0)
        #     self.tree.append(root_node)

    # def get_top_level(self):
    #     unique_keys = set(self.get_grammar_keys())
    #     keys = self.get_grammar_keys()
    #     for k in keys:
    #         for r in self.gr[k]:
    #             unique_keys = unique_keys - set(r)
    #
    #     return list(unique_keys)
    #
    # def get_tree(self):
    #     return self.tree
    #
    # def get_num_roots(self):
    #     return len(self.tree)
    #
    # def get_root(self, i):
    #     return self.tree[i]
    #
    # #not currently used
    # def depth_exceeded(self, depth):
    #     if self.maxdepth > 0 and depth > self.maxdepth:
    #         if self.verbose:
    #             print('Recursion limit exceeded')
    #         return True
    #     return False

    # def recurse(self, k, depth):
    #
    #     if k in self.klist:
    #         raise ValueError('Loop in grammar!')
    #
    #     if(self.depth_exceeded(depth)):
    #         return ''
    #
    #     choices = []
    #     for i, l in enumerate(dct[k]):
    #         if not is_terminal(l):
    #             val = recurse(dct, l, depth+1, maxdepth)
    #         else:
    #             val = copy(l)
    #
    #         if len(val)>0:
    #             if type(val) is str:
    #                 pass

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
        if string not in self.reverse_gr.keys() and is_terminal(string):
            val = False
        elif string not in self.reverse_gr.keys():
            val = False
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
        string_tokens = string.split(' ')
        val = True
        for t in string_tokens:
            if not val:
                break
            val = val and self.reverse_follow(t)
        return val



    def parse(self, fn):
        with open(fn, 'r') as f:
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
            lst = split_token(string)
            if isinstance(lst, str):
                lst = [lst]
            lst = rem_ws_from_list(lst)

            lst_larger = []
            for l in lst:
                ll = l.split(" ")
                if isinstance(ll, str):
                    ll = [ll]
                ll = rem_ws_from_list(ll)
                lst_larger.append(ll)
            gram_dct[k] = lst_larger

        self.gr = gram_dct
    # def parse(self, fn):
    #     with open(fn, 'r') as f:
    #         text = f.read()
    #
    #     text = text.strip()
    #
    #     lst = text.split(';;;')
    #
    #     cnt = 0
    #     first = True
    #
    #     klist = []
    #
    #     for l in lst:
    #         l = l.replace('\n', ' ')
    #         l = l.replace('\t', ' ')
    #         l = l.replace('  ', ' ')
    #         l = l.replace('  ', ' ')
    #         l = l.strip()
    #
    #         if not len(l):
    #             continue
    #
    #         if l[0]=='#':
    #             continue
    #
    #
    #         valid_ = re.findall('::=', l)
    #         continued = (len(valid_) == 0)
    #
    #         if len(valid_) > 1:
    #             raise ValueError(l + ' is ill-formed ', cnt)
    #
    #         if not continued:
    #             if first:
    #                 first = False
    #
    #             f = l.strip().split('::=')
    #
    #             key = f[0].strip()
    #
    #             self.gr[key] = []
    #
    #             klist.append(key)
    #
    #             cnt += 1
    #
    #             if len(f) > 1:
    #                 if len(f) > 2:
    #                     raise ValueError(l + ' is ill-formed ', cnt)
    #
    #                 toks = f[1].strip().split('|')
    #
    #
    #                 if type(toks) is str:
    #                     toks = [toks]
    #
    #                 toks = rem_ws_from_list(toks)
    #                 toks = list(filter(bool, toks))
    #
    #                 for tok in toks:
    #                     self.gr[key].append(split_token(tok))
    #         else:
    #             if first:
    #                 raise ValueError('Ill-formed grammar!')
    #
    #             toks = l.strip().split('|')
    #
    #             if type(toks) is str:
    #                 toks = [toks]
    #
    #             toks = rem_ws_from_list(toks)
    #
    #             toks = list(filter(bool, toks))
    #
    #             for tok in toks:
    #                 self.gr[key].append(split_token(tok))

    def get_grammar_keys(self):
        lst = [k for k in self.gr.keys()]
        return lst

    def get_grammar(self):
        return self.gr

    def get_terminals(self):
        return self.terminals
    # def get_grammar_tree(self, starting_pos, maxdepth):
    #     return self.tree
        # for

    def evaluate_string(self, string):
        pass #root =

if __name__ == '__main__':
    g = Grammar('sql_simple.bnf')

    #print(g.get_grammar())
    # for k in gg.keys():
    #     print(k)
    #     print(len(gg[k]))

    gr = g.get_grammar()
    revg = g.reverse_gr
    print(revg['[[like_pattern]]'])
    print(gr['<condition>'])
    print(g.match('SELECT * FROM [[table_name]]'))
    #
    # gkeys = g.get_grammar_keys()
    # select_key_idxs = []
    #
    # for i, gkey in enumerate(gkeys):
    #     if 'select' in gkey.lower():
    #         select_key_idxs.append(i)
    # print([gkeys[i] for i in select_key_idxs])
