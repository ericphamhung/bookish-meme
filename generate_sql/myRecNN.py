import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import OrderedDict
import os
from simpler_grammar import SimpleGrammar as Grammar
from torch.nn.modules.rnn import RNN, LSTM
from copy import copy
from definitions import *
from utils import *

flatten = lambda l: [item for sublist in l for item in sublist]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HIDDEN_SIZE = 50 #30
ROOT_ONLY = False
BATCH_SIZE = 20
EPOCH = 5
LR = 0.001

def get_vocabulary_word2index(grammar, db_dict):
    vocab = grammar.terminal_toks
    for db in db_dict:
        for tab in db_dict[db]:
            vocab.append(tab)
            for col in db_dict[db][tab]:
                vocab.append(col)
    vocab = list(set(vocab))
    word2index = {'[[UNKOWN]]':0}
    for vo in vocab:
        if word2index.get(vo) is None:
            word2index[vo] = len(word2index)
    return vocab, word2index

class Node:  # a node in the tree
    def __init__(self, label, word=None):
        self.label = label
        self.word = word
        self.parent = None  # reference to parent
        self.left = None  # reference to left child
        self.right = None  # reference to right child
        # true if I am a leaf (could have probably derived this from if I have
        # a word)
        self.isLeaf = False
        # true if we have finished performing fowardprop on this node (note,
        # there are many ways to implement the recursion.. some might not
        # require this flag)

    def __str__(self):
        if self.isLeaf:
            return '[{0}:{1}]'.format(self.word, self.label)
        return '({0} <- [{1}:{2}] -> {3})'.format(self.left, self.word, self.label, self.right)

def get_possibles(word2index, grammar, start):
    if start in grammar.terminals:
        return word2index[grammar.gr[start]['items'][0]]
    elif start in grammar.terminal_toks:
        return word2index.gr[start]
    elif start in grammar.ands:
        items = grammar.gr[start]['items']
        return tuple(get_possibles(word2index, grammar, it) for it in items)
    elif start in grammar.ors:
        items = grammar.gr[start]['items']
        return list(get_possibles(word2index, grammar, it) for it in items)
    else:
        raise ValueError('unknown in possibles')

def len1(lst):
    if lst_or_tup(lst):
        if len(lst) == 1:
            return lst[0]
        else:
            return lst
    else:
        return lst

def minimize_lens(lst):
    lst = len1(lst)
    if isinstance(lst, list):
        fun = list
    elif isinstance(lst, tuple):
        fun = tuple
    else:
        return lst
    
    return fun(len1(l) for l in lst)


        

def get_massaged_possibles(word2index, grammar, start):
    possibles = get_possibles(word2index, grammar, start)
    possibles = minimize_lens(possibles)
    return possibles

def lst_or_tup(p):
    return isinstance(p, list) or isinstance(p, tuple)

def pad_possibles(possibles, length):
    if not lst_or_tup(possibles):
        return possibles
    tup = isinstance(possibles, tuple)
    possibles = list(possibles)
    if tup:
        _num = sum(longest_num(p) for p in possibles)
        to_add = length - _num
        ext = [-1]*to_add
        possibles.extend(ext)
    else:
        _nums = [longest_num(p) for p in possibles]
        possibles = [pad_possibles(p, l) for p, l in zip(possibles, _nums)]
    if tup:
        possibles = tuple(possibles)
    return possibles

# def pad_possibles(possibles, length):
#     if not lst_or_tup(possibles):
#         return possibles
#     tup = isinstance(possibles, tuple)
#     possibles = list(possibles)
#     if tup:
#         to_add = length - len(possibles)
#         if to_add > 0:
#             possibles.append(-1)
#     else:
#         possibles = [pad_possibles(p, length) for p in possibles]
#     if tup:
#         possibles = tuple(possibles)
#     return possibles

def longest_num(possibles):
    if isinstance(possibles, tuple):
        return sum(longest_num(it) for it in possibles)
    elif isinstance(possibles, list):
        return max(longest_num(it) for it in possibles)
    else:
        return 1

def lowest_level(possibles):
    if lst_or_tup(possibles):
        return 1+max(lowest_level(it) for it in possibles)
    else:
        return 1


class RecursiveDecoder(nn.Module):

    # outputsize - predict the token, and if we're done
    def __init__(self, grammar, db_dict, hidden_size, word_dim, rnn_hid, start):#, max_sentence_len):

        x_dim = 2

        super(RecursiveDecoder, self).__init__()
        assert x_dim == 2
        self.word_dim = word_dim

        _, self.word2index = get_vocabulary_word2index(grammar, db_dict)
        self.index2word =dict([[str(i),k] for k,i in self.word2index.items()])
        self.term_toks = grammar.terminal_toks
        # a dictionary
        self.db_dict = db_dict
        self.x_dim = x_dim
        self.start = start
        self.possibles = get_massaged_possibles(self.word2index, grammar, self.start)
        self.longest_path = longest_num(self.possibles)
        self.rnn_eng = RNN(word_dim, rnn_hid, 1)
        self.rnn_hist = RNN(hidden_size, rnn_hid, 1, batch_first = True)
        self.rnn_add = LSTM(rnn_hid, rnn_hid, 1)
        self.hidden_size = hidden_size
        self.rnn_hid = rnn_hid
        self.embed_size = len(self.word2index)
        self.embed = nn.Embedding(self.embed_size, self.hidden_size)
        # self.V = nn.ModuleList([nn.Linear(hidden_size*2,hidden_size*2) for _ in range(hidden_size)])
        # self.W = nn.Linear(hidden_size*2,hidden_size)
        # self.V = nn.ParameterList(
        #     [nn.Parameter(torch.randn(hidden_size * 2, hidden_size * 2)) for _ in range(hidden_size)])  # Tensor
        self.in_node = nn.Linear(self.rnn_hid*2, self.rnn_hid)#nn.Parameter(torch.randn(hidden_size * 2, hidden_size))
        self.in_or = nn.Linear(self.rnn_hid*2, self.rnn_hid)#nn.Parameter(torch.randn(hidden_size * 2, 1))
        self.for_or_choice = nn.Linear(self.rnn_hid, 1)
        self.sigmoid = nn.Sigmoid()

    def get_word2index(self):
        return self.word2index

    def _get_sql(self, index):
        return self.index2word[str(index)]


    def get_sql(self, indxes):
        return [self._get_sql(i) for i in indxes]

        

    def init_weight(self):
        nn.init.xavier_uniform_(self.embed.state_dict()['weight'])
        nn.init.xavier_uniform_(self.W_out.state_dict()['weight'])
        for param in self.V.parameters():
            nn.init.xavier_uniform_(param)
        nn.init.xavier_uniform_(self.W)
        

    def get_value(self, node, x, heng, hsql):
        if isinstance(node, int):
            sql_emb = self.embed(torch.tensor(node)).view(-1, 1, self.hidden_size)
            for x_ in x.split(1, dim=2):
                out, heng = self.rnn_eng(x_.view(-1, 1, self.word_dim), heng)
            sql_emb, hsql = self.rnn_hist(sql_emb, hsql)
            hist = hsql
            current = self.in_node(torch.cat([heng, hsql], dim = 2))
            current = F.relu(current)
        elif isinstance(node, list):
            l = len(node)
            max_, ind_ = -1.0, -1
            for i, t in enumerate(node):
                v, _, _ = self.get_value(t, x, heng, hsql)
                val = self.sigmoid(self.for_or_choice(v))
                
                if val > max_:
                    max_ = val
                    if isinstance(t, int):
                        ind_ = t
                    best = v
            
            current = best
        elif isinstance(node, tuple):
            hadd = torch.zeros(1,1,self.rnn_hid)
            cadd = torch.zeros(1,1,self.rnn_hid)
            lst = []
            for i, t in enumerate(node):
                v, heng, hsql = self.get_value(t, x, heng, hsql)
                lst.append(v)
                
            
            hist = hsql
            for v in lst:
                output, (hadd, cadd) = self.rnn_add(v, (hadd, cadd))
            
            current = hadd
        else:
            assert False
        
        return current, heng, hsql

    def grammar_propagation(self, node, x, heng, hsql, inds = [], cnt = 0, ccount = 0):
        hist = None
        #recursive_tensor = OrderedDict()
        choice_tensor = OrderedDict()
        if isinstance(node, int):
            sql_emb = self.embed(torch.tensor(node)).view(-1, 1, self.hidden_size)
            for x_ in x.split(1, dim=2):
                out, heng = self.rnn_eng(x_.view(-1, 1, self.word_dim), heng)
            sql_emb, hsql = self.rnn_hist(sql_emb, hsql)
            hist = hsql
            current = self.in_node(torch.cat([heng, hsql], dim = 2))
            current = F.relu(current)
            inds.append(node)
        elif isinstance(node, list):
            l = len(node)
            ind_ = None
            max_, ind_ = -1.0, -1
            for i, t in enumerate(node):
                cnt += 1
                v, heng, hsql = self.get_value(t, x, heng, hsql)
                val = self.sigmoid(self.for_or_choice(v))
                
                if val > max_:
                    max_ = val
                    ind_ = t
                    best = v
            
            
            _, v, _, inds, _ = self.grammar_propagation(ind_, x, heng, hsql, inds, cnt, ccount)
            choice_tensor[ccount] = best
            ccount += 1
            current = best
        elif isinstance(node, tuple):
            hadd = torch.zeros(1,1,self.rnn_hid)
            cadd = torch.zeros(1,1,self.rnn_hid)
            # hadd = torch.zeros(self.rnn_hid, x.dim()-1, self.rnn_hid)
            # cadd = torch.zeros(self.rnn_hid, x.dim()-1, self.rnn_hid)
            cc = copy(ccount)
            lst = []
            for i, t in enumerate(node):
                ccc = copy(ccount)
                v, d, hsql, inds, ccount = self.grammar_propagation(t, x, heng, hsql, inds, cnt, ccount)
                lst.append(d)
                if ccc > ccount:
                    choice_tensor.update(v)
                    

                # recursive_tensor[cnt] = v
            
            hist = hsql
            for v in lst:
                # cnt += 1
                output, (hadd, cadd) = self.rnn_add(v, (hadd, cadd))
            
            current = hadd

            # recursive_tensor[cnt] = current
        else:
            assert False
        return choice_tensor, current, hist, inds, ccount

    def forward(self, x):
        assert x.size()[0] == self.word_dim
    
        #heng = torch.zeros(self.rnn_hid, 1, self.word_dim)
        #hsql = torch.zeros(self.rnn_hid, 1, self.embed_size)
        heng = torch.zeros(1,1,self.rnn_hid)
        hsql = torch.zeros(1,1,self.rnn_hid)


        #rets, _, inds, ccount = self.grammar_propagation(self.possibles, x, heng, hsql)
        
        return self.grammar_propagation(self.possibles, x, heng, hsql)

    # def get_tokens(self, probs):





if __name__ == '__main__':

    gram = Grammar('sql_simple_transition_2.bnf')
    st = '<modifyop>'
    rsl = RecursiveDecoder(gram, spider_db, hidden_size=256, word_dim=1024, rnn_hid =256, start = st)
    x = torch.randn(1024, 1, 11)
    _, _, _, toks, ccount = rsl.forward(x)
    print(ccount)
    print(toks)
    print(rsl.get_sql(toks))

    

    # if os.path.exists('model/RecNN.pkl'):
    #     model = torch.load('model/RecNN.pkl').to(device)
    #     print('Model loaded')
    # else:
    #     model = RNTN(word2index, HIDDEN_SIZE, 5).to(device)
    #     model.init_weight()
    # # Train a model
    # train(model, LR, EPOCH)
    # # test a model
    # test(model)



