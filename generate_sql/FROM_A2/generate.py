from fastai.text.models.awd_lstm import AWD_LSTM
from fastai.text.models.transformer import Transformer, TransformerXL
import torch
from grammar import SimpleGrammar
from torch import nn, data
from copy import copy
from definitions import *
from predictors import Regressor, Classifier

def language_model_learner(data:DataBunch, arch, config:dict=None, drop_mult:float=1., pretrained:bool=True,
                           pretrained_fnames:OptStrTuple=None, **learn_kwargs) -> 'LanguageLearner':
    "Create a `Learner` with a language model from `data` and `arch`."
    model = get_language_model(arch, len(data.vocab.itos), config=config, drop_mult=drop_mult)
    meta = _model_meta[arch]
    learn = LanguageLearner(data, model, split_func=meta['split_lm'], **learn_kwargs)
    if pretrained:
        if 'url' not in meta: 
            warn("There are no pretrained weights for that architecture yet!")
            return learn
        model_path = untar_data(meta['url'], data=False)
        fnames = [list(model_path.glob(f'*.{ext}'))[0] for ext in ['pth', 'pkl']]
        learn.load_pretrained(*fnames)
        learn.freeze()
    if pretrained_fnames is not None:
        fnames = [learn.path/learn.model_dir/f'{fn}.{ext}' for fn,ext in zip(pretrained_fnames, ['pth', 'pkl'])]
        learn.load_pretrained(*fnames)
        learn.freeze()
    return learn

n_output_eng_emb = 200

class Generate_not_simple:

    def __init__(self,eng_emb:nn.Module,  emb_dim:int, gram:SimpleGrammar, clsfier:Classifier, regressor:Regressor, 
    intregress:Regressor, dataloader:data.DataLoader, model_dct:dict):

        self.tokenized_grammar, self.terms, self.gr, self.inv_gr = gram.tokenized_form()
        self.grammar = gram
        self.setup()   
        self.table_predictor = OrderedDict()
        self.column_predictor = OrderedDict()
        for i, db in enumerate(resolve_sampling['db_name']):
            n_tabs = len(resolve_sampling['table_name'][tab]))
            self.table_predictor[db] = clsfier(n_tabs) #will need to initialize predictor with table names
            self.column_predictor[db] = OrderedDict()
            for j, tab in enumerate(resolve_sampling['table_name'][tab]):
                self.column_predictor[db][tab] = copy(predictor) #will need to initialize predictor with column names
        self.cts_resolve = regressor(model_dct['regress'])
        self.int_resolve = regressor(model_dct['int_regress'])

        self.eng_emb = eng_emb #need to get output dim
        self.emb = nn.Embedding(len(self.terms), embedding_dim = 4, max_norm = 5)    
        self.dataloader = dataloader



    def setup(self):
        terminals = self.grammar.get_terminals()
        for t in self.tokenized_grammar:
            cnt = 0
            t = self.grammar.pathone(t)
            if self.gr[t]['type'] == 'or'


    def train_one_sentence(self, i):
        db, sql, eng = dataloader(i) #NOT CORRECT!
        
class Generate_simple:

    def __init__(self,eng_emb:nn.Module,  emb_dim:int, gram:SimpleGrammar, clsfier:Classifier, regressor:Regressor, 
    intregress:Regressor, dataloader:data.DataLoader, model_dct:dict):

        self.tokenized_grammar, self.terms, self.gr, self.inv_gr = gram.tokenized_form()
    
        self.predictor = clsfier(len(self.terms)) 
        self.cts_resolve = regressor
        self.int_resolve = intregress
        self.eng_emb = eng_emb #need to get output dim
        self.emb = nn.Embedding(len(self.terms), embedding_dim = 4, max_norm = 5)    
        self.dataloader = dataloader

    def train_one_sentence(self, i):
        db, sql, eng = dataloader(i) #NOT CORRECT!