import torch
import re

regex_strings = {'<<value_pattern>>':re.compile("[-+]?[0-9]+\.?[0-9]*"),
                '<<like_pattern>>':re.compile('\"[^\"]\"')}

#
# def check_valid(string, possibly_continuation):
#     if len(string) == 0:
#         ret = '001'
#     else:
#         num = len(re.findall('::==', string))
#         if num == 1:
#             continued = 0
#             valid = 1
#             blank = 0
#         elif num == 0:
#             continued = 1
#

splitter = re.compile("\s+(?=\<^\\>)}\>*(\<\\<({\>|$))")

def rem_ws_from_list(lst):
    ret = []
    for w in lst:
        w = w.strip()
        ret.append(w)
    return ret

def is_terminal(string, dct):
    lst = []
    print(string)
    print(dct[string])
    for val in dct[string]:
        for v in val:
            lst.extend(re.findall('\<(.*?)\>', v))
    if(len(lst)):
        return False
    return True


def xor(a, b):
    return bool(a) ^ bool(b)

def split_token(token):
    token = token.replace('><', '> <')
    tok_lst = splitter.split(token.strip())
    if type(tok_lst) is str:
        tok_lst = [tok_lst]
    return rem_ws_from_list(tok_lst)

class Node:

    def __init__(self, name, parent, dct, choice_num, depth=0):
        if parent is None:
            assert depth == 0
        if depth == 0:
            assert parent is None
        self.name = name
        self.choice_num = choice_num
        self.root = (parent is None)
        self.parent = parent

        self.leaf = is_terminal(self.name, dct)
        #this would never happen, put in grammar
        lst_ancestors = list(filter(bool, self.anscestor_names()))
        if self.name in lst_ancestors:
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
                self.add_child(Node(name = c, parent = self, dct = dct, \
                    choice_num = i, depth = depth+1))





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
            lst.extend(self.parent.anscestor_names())
        return lst

    def check_content(self):
        return xor(self.leaf, len(self.children)>1)

# class Tree:
#     def __init__(self, grammar_dct, maxdepth = -1, verbose = False):
#         self.dct = grammar_dct
#         self.maxdepth = maxdepth
#         self.verbose = verbose
#         self.klist = []
#         self.nodes = []
#         for k in self.dct.keys():
#
#
#     def __eq__(self, other):
#         assert type(other) is Tree
#         # more...
#
#     def depth_exceeded(self, depth):
#         if self.maxdepth > 0 and depth > self.maxdepth:
#             if self.verbose:
#                 print('Recursion limit exceeded')
#             return True
#         return False
#
#     def recurse(self, k, depth):
#
#         if k in self.klist:
#             raise ValueError('Loop in grammar!')
#
#         if(self.depth_exceeded(depth)):
#             return ''
#
#         choices = []
#         for i, l in enumerate(dct[k]):
#             if not is_terminal(l):
#                 val = recurse(dct, l, depth+1, maxdepth)
#             else:
#                 val = copy(l)
#
#             if len(val)>0:
#                 if type(val) is str:
#
#

class Grammar:

    def __init__(self, file_name, maxdepth = -1, verbose = False):
        self.gr = {}
        self.maxdepth = maxdepth
        self.verbose = verbose
        self.klist = []
        self.parse(file_name)
        self.top_level = self.get_top_level()
        self.tree = []
        for i, vv in enumerate(self.top_level):
            root_node = Node(name = vv, parent = None, dct = self.gr, \
                    choice_num = i, depth = 0)
            self.tree.append(root_node)

    def get_top_level(self):
        unique_keys = set(self.get_grammar_keys())
        keys = self.get_grammar_keys()
        for k in keys:
            for r in self.gr[k]:
                unique_keys = unique_keys - set(r)

        return list(unique_keys)

    def get_tree(self):
        return self.tree

    def get_num_roots(self):
        return len(self.tree)

    def get_root(self, i):
        return self.tree[i]

    #not currently used
    def depth_exceeded(self, depth):
        if self.maxdepth > 0 and depth > self.maxdepth:
            if self.verbose:
                print('Recursion limit exceeded')
            return True
        return False

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

    def parse(self, fn):
        with open(fn, 'r') as f:
            lst = f.readlines()

        cnt = 0
        first = True

        klist = []

        for l in lst:
            l = l.strip()

            if not len(l):
                continue

            if l[0]=='#':
                continue


            valid_ = re.findall('::=', l)
            continued = (len(valid_) == 0)

            if len(valid_) > 1:
                raise ValueError(l + ' is ill-formed ', cnt)

            if not continued:
                if first:
                    first = False

                f = l.strip().split('::=')

                key = f[0].strip()

                self.gr[key] = []

                klist.append(key)

                cnt += 1

                if len(f) > 1:
                    if len(f) > 2:
                        raise ValueError(l + ' is ill-formed ', cnt)

                    toks = f[1].strip().split('|')


                    if type(toks) is str:
                        toks = [toks]

                    toks = rem_ws_from_list(toks)
                    toks = list(filter(bool, toks))

                    for tok in toks:
                        self.gr[key].append(split_token(tok))
            else:
                if first:
                    raise ValueError('Ill-formed grammar!')

                toks = l.strip().split('|')

                if type(toks) is str:
                    toks = [toks]

                toks = rem_ws_from_list(toks)

                toks = list(filter(bool, toks))

                for tok in toks:
                    self.gr[key].append(split_token(tok))

    def get_grammar_keys(self):
        lst = [k for k in self.gr.keys()]
        return lst

    def get_grammar(self):
        return self.gr

    def get_grammar_tree(self, starting_pos, maxdepth):
        pass
        # for

if __name__ == '__main__':

    g = Grammar('sql_simple.gr')

    print(g.get_tree())
    #
    # gkeys = g.get_grammar_keys()
    # select_key_idxs = []
    #
    # for i, gkey in enumerate(gkeys):
    #     if 'select' in gkey.lower():
    #         select_key_idxs.append(i)
    # print([gkeys[i] for i in select_key_idxs])
