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
# from torch.nn.modules.rnn import RNN, LSTM
from copy import copy
from definitions import *
from utils import *
import pickle
import warnings
from topologies import *
from run_defs import *
import json

device = torch.device('cuda')

def get_grammar2index(grammar, db_dict, dontuse):
    if isinstance(dontuse, str):
        dontuse = [dontuse]
    elif dontuse is None:
        dontuse = []
    vocab = [v for v in grammar.gram_keys if v not in dontuse]
    vocab = [v for v in vocab if v not in grammar.terminal_toks]
    
    for db in db_dict:
        for tab in db_dict[db]:
            vocab.append(tab)
            for col in db_dict[db][tab]:
                vocab.append(col)
    vocab = list(set(vocab))
    word2index = OrderedDict()
    for vo in vocab:
        if word2index.get(vo) is None:
            word2index[vo] = len(word2index)
    return vocab, word2index

def get_word2index(grammar, db_dict, dontuse):
    if isinstance(dontuse, str):
        dontuse = [dontuse]
    elif dontuse is None:
        dontuse = []
    vocab = [v for v in grammar.terminal_toks if v not in dontuse]
    
    for db in db_dict:
        for tab in db_dict[db]:
            vocab.append(tab)
            for col in db_dict[db][tab]:
                vocab.append(col)
    vocab = list(set(vocab))
    word2index = OrderedDict()
    word2index['[[UNKOWN]]'] = 0
    word2index['[[CONTINUATION]]'] = 1
    for vo in vocab:
        if word2index.get(vo) is None:
            word2index[vo] = len(word2index)
    return vocab, word2index

def get_possibles(word2index, grammar, start):
    if start in grammar.terminal_toks:
        return word2index.gr[start]
    elif start in grammar.terminals:
        return word2index[grammar.gr[start]['items'][0]]
    elif start in grammar.ands:
        items = grammar.gr[start]['items']
        return tuple(get_possibles(word2index, grammar, it) for it in items)
    elif start in grammar.ors:
        items = grammar.gr[start]['items']
        return list(get_possibles(word2index, grammar, it) for it in items)
    else:
        raise ValueError('unknown in possibles')

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

def max_num_tokens(possibles):
    if isinstance(possibles, list):
        ret = 0
        for p in possibles:
            ret = max(ret, max_num_tokens(p))
    elif isinstance(possibles, tuple):
        ret = sum([max_num_tokens(p) for p in possibles])
    elif isinstance(possibles, int):
        ret = 1
    return ret

class TrainableRepresentation(nn.Module):

    def __init__(self, grammar, x_dim = 1024, start = '<sellist>', dim = 1024, db_dict = None, dontuse = ['<start>'], \
            hist_dim = 0, cuda = True, words = False, return_log = False):
        super().__init__()

        if db_dict is None:
            db_dict = {}
        if start is None:
            start = '<query>'
        
        #if start not in grammar.gr:
        if 'sell' in start:
            start = '<sellist>'
            with open('grammar2index_grammar_2_sellist.json') as f:
                self.gram2index = json.load(f)
        elif 'modif' in start:
            start = '<modifyop>'
            with open('grammar2index_grammar_2_modifyop.json') as f:
                self.gram2index = json.load(f)
        else:
            raise ValueError("Don't yet know how to deal with {} in TrainableRepresentation".format(start))
        #print(start)
        grammar = grammar.get_subgrammar(start)
        self.words = None
        if words:
            self.words = SentenceRepresenation(1024, x_dim)
        self.dim = dim
        self.grammar = grammar
        self.return_log = return_log
        self.eps = 1e-7
        with open('terminals2index_grammar_2.json') as f:
            self.word2index = json.load(f)
        
            
        # print(self.grammar.terminal_toks)
        # assert 'ORDER' in self.grammar.terminal_toks
        # assert 'ORDER' in self.word2index
        
        #self.softmax = nn.Softmax(dim=0)
        self.index2word =dict([[str(i),k] for k,i in self.word2index.items()])
        self.term_toks = grammar.terminal_toks
        self.history_allowed = (hist_dim > 0)
        if self.history_allowed:
            assert False
        # a dictionary
        self.db_dict = db_dict
        self.x_dim = x_dim
        self.start = start
        self.possibles = get_massaged_possibles(self.word2index, grammar, self.start)
        
        self.max_length = max_num_tokens(self.possibles)

        self.continuation_index = self.word2index['[[CONTINUATION]]']
        
        self.embed_size = len(self.gram2index)
        
        self.token_size = len(self.word2index)
        
        self.embedding = nn.Embedding(self.embed_size, self.dim)
        representations = OrderedDict()
        use_keys = []
        num_below = []
        for tok in self.gram2index.keys():
            if tok in self.grammar.gram_keys:
                use_keys.append(tok)
                l = self.grammar.get_len_to_term(tok)
                num_below.append(l)
            # elif tok in self.grammar.
            #elif tok in self.db_dict.keys():
            else:
                print(tok)
                # not using this for table or columns any more
                assert False
                use_keys.append(tok)
                num_below.append(0)
        mm = [i for i, k in enumerate(use_keys)]
        arr = np.array([num_below, mm]).T
        arr = arr[arr[:, 0].argsort()]
        for c, a in arr:
            tok = use_keys[a]
            
            a = GrammarTokenRepresentation(grammar, tok, dim, x_dim, hist_dim, representations)
            if cuda:
                a = a.cuda()
            representations[tok] = a 
        self.representations = representations
        self.arr = None
        self.max_index = -1
    
    def get_all_token_length(self):
        return self.max_length
    
    def get_continuation_index(self):
        return self.continuation_index
    
    def calculate_probs(self, token, mult):
        pass

    # -1s because of unknown.  Remove?
    def iterative_calculate(self, repres):
        probability = torch.zeros(self.token_size-1, self.max_length-1, dtype = torch.float, device = device)
        for i in range(self.max_length-1):
            if i == 0:
                toks = [self.start]
            else:
                toks = extended_toks

            for tok in toks:
                if self.grammar.gr[tok] == 'or':
                    _toks = self.grammar.gr[tok]['items']
                    mask = [(self.word2index[t]-1) for t in _toks]
                    last = 1.0#torch.ones([1], dtype=torch.float, device=device, requires_grad = True)
                    if i > 1:
                        last *= probability[self.word2index[t]-1, i-1]
                    probability[mask, i] = self.representations[tok].chooser(vs)*last
                elif self.grammar.gr[tok] == 'and':
                    _toks = self.grammar.gr[tok]['items']
                    mask = [(self.word2index[t]-1)for t in _toks]
                    last = torch.ones([1], dtype=torch.float, device=device, requires_grad = True)
                    if i > 1:
                        last *= probability[self.word2index[t], i-1]
                    probability[mask, i] = last
                elif self.grammar.gr[tok] == 'or':
                    _toks = self.grammar.gr[tok]['items']
                    mask = [(self.word2index[t]-1) for t in _toks]
                    last = torch.ones([1], dtype=torch.float, device=device, requires_grad = True)
                    if i > 1:
                        last *= probability[self.word2index[t], i-1]
                    probability[mask, i] = last



            extended_toks = {tok:self.grammar.gr[tok]['items'] for tok in toks}
            tt = {tok:self.grammar.gr[tok]['type'] for tok in toks}
    
    def recursive_calculate(self, repres, token, mult, index):
        tt = self.grammar.gr[token]['type']
        toks = self.grammar.gr[token]['items']

        dct = {}
        if tt == 'term':
            for tok in toks:
                dct[tok] = {}
                dct[tok]["ind"] = (self.word2index[tok], index)
                dct[tok]['prob'] = mult
                
        elif tt ==  'and':
            for i, tok in enumerate(toks):
                dct[tok] = {}
                #print(toks)
                if self.grammar.gr[tok]['type'] == 'term':
                    index += 1
                    self.max_index = max(self.max_index, index)
                    assert index <= self.max_length
                
                dct[tok]['path'] = self.recursive_calculate(repres, token = tok, mult = mult, index = index)
        else:
            assert tt == 'or'
            probs = self.representations[token].chooser(repres[token])
            for i, (prob, tok) in enumerate(zip(probs[0], toks)):
                dct[tok] = {}
                if self.grammar.gr[tok]['type'] == 'term':
                    index += 1
                    self.max_index = max(self.max_index, index)
                    assert index <= self.max_length
                
                dct[tok]['path'] = self.recursive_calculate(repres, token = tok, mult = mult*prob, index = index) 
        return dct       

    def recursive_calculate_old(self, repres, token, mult, index, last = False, level = 0):
        tt = self.grammar.gr[token]['type']
        toks = self.grammar.gr[token]['items']
        if tt == 'term':
            for tok in toks:
                self.arr[self.word2index[tok], index] += mult*torch.ones([4], device=device)
                self.arr[self.continuation_index, index] -= mult*torch.ones(self.arr[self.continuation_index, index].size())
                # if last:
                #     print(tok)
                #     self.arr[self.continuation_index, (index+1):] += mult
            # print(index)
            # print(self.max_length)
            # if index < self.max_length:
            #     print(mult)
            #     for i in range(index, self.max_length):
            #         self.arr[self.continuation_index, i] = mult
        elif tt ==  'and':
            for i, tok in enumerate(toks):
                #print(toks)
                if self.grammar.gr[tok]['type'] == 'term':
                    index += 1
                    self.max_index = max(self.max_index, index)
                    assert index <= self.max_length
                
                self.recursive_calculate(repres, token = tok, mult = mult, index = index, last = (i==len(toks)-1), level = level+1)
        else:
            assert tt == 'or'
            vs = repres[token]
            probs = self.representations[token].chooser(vs)
            for i, (prob, tok) in enumerate(zip(probs, toks)):
                if self.grammar.gr[tok]['type'] == 'term':
                    self.max_index = max(self.max_index, index)
                    assert index <= self.max_length
                    last = (level==0)
                else:
                    last = False
                self.recursive_calculate(repres, token = tok, mult = mult*prob, index = index, last =last, level = level+1)
    # -1s because of unknown.  Remove?            
    def forward(self, sentence_embedding, history = None):
        if self.words is not None:
            sentence_embedding = self.words.forward(sentence_embedding)
        self.max_index = -1
        if not self.history_allowed and history is not None:
            warnings.warn("In TrainableRepresentation, hist_dim was 0, passing a history object won't do anything")
        sz0 = sentence_embedding.size()[0]
        arr = torch.zeros(sz0, self.token_size-1, self.max_length, dtype = torch.float, device = device)
        repres = self.get_representations(sentence_embedding, history)
        dct = self.recursive_calculate(repres, token = self.start, mult = 1.0, index = -1)
        return prob_dict_recurse(arr, dct)
        


                
            


    def forward_old(self, sentence_embedding, history = None):
        if self.words is not None:
            sentence_embedding = self.words.forward(sentence_embedding)
        self.max_index = -1
        if not self.history_allowed and history is not None:
            warnings.warn("In TrainableRepresentation, hist_dim was 0, passing a history object won't do anything")
        
        self.arr = torch.zeros(self.token_size, self.max_length, dtype = torch.float, device = device)
        self.arr[self.continuation_index, :] = 1.0
        repres = self.get_representations(sentence_embedding, history)
        self.recursive_calculate(repres, token = self.start, mult = 1.0, index = -1)
        
        #self.arr = torch.t(self.arr)
        # for i in range
        # self.arr[:, self.continuation_index]

        
        assert (self.arr.sum(dim=0)).allclose(torch.ones(self.max_length, dtype=torch.float, device=device))
        if self.return_log:
            self.arr = torch.log(self.arr)
        return torch.t(self.arr), repres
    


    def get_representations(self, sentence_embedding, history):
        if not self.history_allowed and history is not None:
            warnings.warn("In TrainableRepresentation, hist_dim was 0, passing a history object won't do anything")
        
        dct = OrderedDict()
        for tok in self.representations.keys():
            lookup_tensor = torch.tensor([self.gram2index[tok]], dtype=torch.long, device=device)
            embedded = self.embedding(lookup_tensor)
            v = self.representations[tok].forward(sentence_embedding, embedded, dct)
            dct[tok] = v
        return dct#torch.cat(lst, dim = 1)
    
    def get_scores(self, sentence_embedding):
        scores = {}
        for tok in self.use_keys:
            v = self.representations[tok].score(sentence_embedding)
            scores[tok] = v
        return scores

    def get_use_keys(self):
        return self.use_keys
    
    def generate_toks(self, dct, tok = None):
        if tok is None:
            tok = self.start
        tok = self.grammar.pathone(tok)
        string = ' '
        if tok in self.grammar.terminals:
            string += self.grammar.gr[tok]['items'][0]
        elif tok in self.grammar.ands:
            for t in self.grammar.gr[tok]['items']:
                string += self.generate_toks(dct, t)
        else:
            val = dct[tok]
            choice = self.representations[tok].choose(val)
            probs = choice.detach().cpu().numpy()
            ind = probs.argmax()
            string += self.generate_toks(dct, self.grammar.gr[tok]['items'][ind])
        return string
    
    #actually a vector
    def matrix_of_tokens_old(self, tokens):
        if not isinstance(tokens[0], str):
            ret = torch.zeros(self.max_length, len(tokens), dtype = torch.long, device = device)
            for i in range(len(tokens)):
                ret[i, :] = self.matrix_of_tokens(tokens[i])
            
        else:
            print(len(tokens))
            print(tokens)
            ret = torch.zeros(1, self.max_length, dtype = torch.long, device = device)
            
            for i, tok in enumerate(tokens):
                
                if tok == 'ORDER' or tok == 'GROUP':
                    tok += ' BY'
                elif tok == 'BY':
                    continue
                if isinstance(tok, tuple):
                    assert len(tok) == 1
                    tok = tok[0]

                tok = tok.lower()
                ret[:, i] = self.word2index[tok]
            
            ret[:, len(tokens):] = self.continuation_index
        return ret
    #-1s because of unknown... really may want to get rid of
    def matrix_of_tokens(self, tokens):
#         print(tokens)
        if not isinstance(tokens[0], str):
            assert False
            
        else:
            
            ret = torch.zeros(self.token_size-1, self.max_length, dtype = torch.float, device = device)
            
            for i, tok in enumerate(tokens):
                
                if tok == 'ORDER' or tok == 'GROUP':
                    tok += ' BY'
                elif tok == 'BY':
                    continue
                if isinstance(tok, tuple):
                    assert len(tok) == 1
                    tok = tok[0]

                tok = tok.lower()
                ret[self.word2index[tok]-1, i] = 1
            
            ret[self.continuation_index-1, len(tokens):] = 1
        return ret
    
    def save(self, fname):
        for key in self.representations:
            self.representations[key].save(fname+'_'+key+'.pyt')
        torch.save(self.state_dict(), fname+'.pyt')
    
    def load(self, fname):
        for key in self.representations:
            self.representations[key].load_state_dict(torch.load(fname+'_'+key+'.pyt'))
        self.load_state_dict(torch.load(fname+'.pyt'))

        

class GrammarTokenRepresentation(nn.Module):

    def __init__(self, grammar, tok, dim, x_dim, hist_dim, reps):
        super().__init__()
        self.type = grammar.gr[tok]['type']

        self.dim = dim
        self.x_dim = x_dim
        self.hist_dim = hist_dim
        self.faux_and = False
        self.subs = OrderedDict()
        self.tok = tok

        proj = base_representations(x_dim, dim, BASE_REP_TOP)
        self.proj = proj
        if self.type == 'or':
            ret_dim = 2
        else:
            ret_dim = 1
        self.summarizer = None
        self.sigmoid = nn.Sigmoid()


        if self.type in ['and', 'or']:
            if len(grammar.gr[tok]['items']) == 1:
                assert self.type == 'and'
                self.faux_and = True
                its = grammar.gr[tok]['items']
                assert its[0] in reps
                self.subs = its
            else:
                its = grammar.gr[tok]["items"]
                self.subs = its
                for it in its:
                    assert it in reps
                ddim = dim*(len(its)+1)
                seqlist = and_or_representations(ddim, dim, AND_OR_TOP)
                self.summarizer = seqlist

                if self.type == 'or':
                    self.chooser = MakeChoices(dim, len(its))
            

    def forward(self, sentence_embedding, embedded, embs):
#         print(sentence_embedding.size())
#         print(embedded.size())
        if len(sentence_embedding.size())>len(embedded.size()):
            sentence_embedding = sentence_embedding.squeeze(1)
        if sentence_embedding.size()[0]>1:
            assert embedded.size()[0]== 1
            lst = [embedded]
            for i in range(1, sentence_embedding.size()[0]):
                lst.append(embedded)
            embedded = torch.cat(lst, dim = 0)
        if self.type == 'or':
            assert len(embs) > 0
            repres = self.proj(torch.cat([sentence_embedding, embedded], dim = 1))
            its = torch.cat([torch.cat([embs[a] for a in self.subs], dim = 1), repres], dim = 1)
            ret = self.summarizer(its)
        elif self.type == 'and':
            assert len(embs) > 0
            if self.faux_and:
                return embs[self.subs[0]]
            else:
                repres = self.proj(torch.cat([sentence_embedding, embedded], dim = 1))
                its = torch.cat([torch.cat([embs[a] for a in self.subs], dim = 1), repres], dim = 1)
                ret = self.summarizer(its)
        elif self.type in ['term', 'resolve']:
            ret = self.proj(torch.cat([sentence_embedding, embedded], dim = 1))
        else:
            ValueError("I shouldn't be able to get here!")

        return ret
    
    def choose(self, x):
        return self.chooser(x)
    
    def save(self, fname):
        torch.save(self.state_dict(), fname)
    
    def load(self,fname):
        self.load_state_dict(torch.load(fname))

class MakeChoices(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.choose = nn.Linear(dim, out_dim)
        self.sm = nn.Softmax(dim = 1)

    def forward(self, x):
        return self.sm(self.choose(x))

class DBRepresentation:
    pass

class TableRepresentation:
    pass

class ColumnRepresentation:
    pass


def train_representation_prep(grammar, others = []):
    vocab = [v for v in grammar.terminal_toks]

    vocab = list(set(vocab))
    word2index = OrderedDict()
    word2index['[[UNKOWN]]'] = 0
    word2index['[[ERRONEOUSLY_CONTINUED]]'] = 1
    for vo in vocab:
        if word2index.get(vo) is None:
            word2index[vo] = len(word2index)

    def represent_toks(toks):
        arr = np.zeros(len(toks), len(word2index))
        for j, t in enumerate(toks):
            arr[word2index[tok], j] = 1
        return arr
    def represent_probs(probs_dict):
        for j, t in enumerate(probs_dict.keys()):
            arr[:, j] = probs_dict[t]
        return arr
    def get_probs_dict(cls):
        pass



    return vocab, word2index

def save_2index():
    grammar = Grammar('sql_simple_transition_2.bnf')
    #this shouldn't be necessary but it is
    grammar2 = Grammar('sql_simple_transition_2.bnf')
    
    sellist_gram = grammar.get_subgrammar('<sellist>')
    assert '<modifyop>' in grammar.gram_keys
    modifyopgram = grammar2.get_subgrammar('<modifyop>')
    dontuse = ['<start>']
    db_dict = []
    _, gram2ind = get_grammar2index(sellist_gram, db_dict, dontuse = dontuse)
    with open('grammar2index_grammar_2_sellist.json', 'w') as f:
        json.dump(gram2ind, f)
    _, gram2ind = get_grammar2index(modifyopgram, db_dict, dontuse = dontuse)
    with open('grammar2index_grammar_2_modifyop.json', 'w') as f:
        json.dump(gram2ind, f)
    _, w2ind = get_word2index(grammar, db_dict, dontuse = dontuse)
    with open('terminals2index_grammar_2.json', 'w') as f:
        json.dump(w2ind, f)
    with open('spider_tables_lowercase.json', 'r') as f:
        spider_db = json.loads(f.read())
    tab2ind = {}
    col2ind = {}
    allcols_db2ind = {}
    for db in spider_db.keys():
        dbp = db.lower()
        tab2ind[dbp] = {tab.lower():i for i, tab in enumerate(spider_db[db].keys())}
        all_cols = list(set([c.lower() for t in spider_db[db].keys() for c in spider_db[db][t]]))
        allcols_db2ind[dbp] = {c.lower():i for i, c in enumerate(all_cols)}
        col2ind[dbp] = {}
        for tab in spider_db[db].keys():
            tabp = tab.lower()
            col2ind[dbp][tabp] = {col.lower():i for i, col in enumerate(spider_db[db][tab])}
    with open('spider_tab2index.json', 'w') as f:
        json.dump(tab2ind, f)
    with open('spider_col2index.json', 'w') as f:
        json.dump(col2ind, f)
    

    with open('spider_db_cols2ind.json', 'w') as f:
        json.dump(allcols_db2ind, f)
            

#-1 here!!!        
def prob_dict_recurse(arr, dct, level = 0):
    
    while len(dct)==1:
        keys = [k for k in dct.keys()]
        dct = dct[keys[0]]
    
    if 'prob' in dct:
        idx1, idx2 = dct['ind']
        arr[:, idx1-1, idx2] = dct['prob']
    else:
        for k in dct.keys():
            arr = prob_dict_recurse(arr, dct[k], level = level+1)
    return arr

if __name__ == '__main__':
#     save_2index()
#     assert False

    #gram = Grammar('sql_simple_transition_2.bnf')
    # print(gram.get_subgrammar('<sellist>').terminal_toks)
    # assert False
    rep = TrainableRepresentation(gram, x_dim = 1024, start = 'sellist', cuda = True).cuda()
    x = torch.randn(1, 1024).cuda()
    xx = rep.forward(x)
    lst =rep.generate_toks(xx[1]).split()
    lst = [l for l in lst if not (l == '' or l == " " or l == "  ")]
    print(lst)
    loss = nn.CrossEntropyLoss()
    mat = rep.matrix_of_tokens(lst)
    print(loss(torch.t(xx[0]), mat))


    # rnn = nn.RNN(12, 15, 1)
    # t = torch.randn(1, 1, 12)
    # h = torch.zeros(1, 1, 15)
    # for cnt in range(0, 4):
    #     print(cnt)
    #     print(t)
    #     print(h)
    #     h, _ = rnn(t, h)