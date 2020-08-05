from utils import *
import numpy as np
from itertools import chain

'''
SimpleTree!
Either is terminal or is not, make sure of terminal
If >>not<<, no contents, but children
If yes, contents
That's it, data structure is dict
'''
def is_or(lst):
    if isinstance(lst, list) and isinstance(lst[0], list):
        val = True
    else:
        val = False
    return val


class SimpleNode:

    def __init__(self, name, parent, dct, choice_num, depth=0):

        if parent is None:
            assert depth == 0
        if depth == 0:
            assert parent is None

        self.depth = depth
        self.choice_num = choice_num
        self.root = (parent is None)
        self.parent = parent
        self.leaf = is_terminal(dct[self.name])
        self.val = dct[self.name]


class SimpleTree:
    def __init__(self, dct, terminals):
        assert isinstance(dct, dict)
        self.dct = dct
        terms_ = [dct[k] for k in terminals]
        #self.terminals = chain(*terms_)
        #self.nonterminals = [k for k in dct.keys()]
        self.all_items = [k for k in dct.keys()]
        self.all_items.extend(list(chain(*terms_)))
        # self.m = len(self.nonterminals)+len(self.terminals)
        self.m  = len(self.all_items)
        self.matrix = np.zeros((self.m, self.m))


class Node:

    def __init__(self, name, parent, dct, choice_num, depth=0):

        if parent is None:
            assert depth == 0
        if depth == 0:
            assert parent is None


        self.name = name

        assert (self.name in dct.keys()) #or (self.name in dct.values())
        self.choice_num = choice_num
        self.root = (parent is None)
        self.parent = parent
        self.leaf = is_terminal(dct[self.name]) #(self.name not in dct.keys()) or
        #this would never happen, put in grammar
        # actually, eventually we want loops in select...
        # but maybe we should evaluate those separately...
        # would require a data change, though, likely...
        lst_ancestors = list(filter(bool, self.anscestor_names()))

        self.looped = False

        if self.name in lst_ancestors:
            print('Note that there is a loop in {}'.format(self.name))
            self.looped = True
            raise ValueError('loop with name {}, not should not happen at the moment'.format(self.name))
        if self.leaf:
            assert len(dct[self.name]) == 1
            self.content = dct[self.name][0]
            self.children = None
            self.num_choices = len(self.content)
        else:

            self.content = None

            self.children = []
            self.num_choices = len(self.children)
            if len(dct[self.name]) == 1:
                lst = dct[self.name][0]
            else:
                lst = dct[self.name]
                print('000')
            print(lst)
            for i, c in enumerate(lst):

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
