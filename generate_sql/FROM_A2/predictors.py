import torch
from torch import nn
import numpy as np
import math
from copy import copy

class Regressor(nn.Module):

    def __init__(layer_type, n_hidden, n_in, integer_out = False):
        super().__init__()
        
        layers = []
        if isinstance(n_hidden, int):
            num_divide = round(math.log(n_in, (n_hidden+1)))
            to_div = n_in//num_divide
            layers.append(nn.Linear(n_in, to_div))
            for i in range(n_hidden):
                layers.append(layer_type(to_div, to_div//num_divide))
                to_div =  to_div//num_divide
            assert to_div == 1
        else:
            assert False, "I can't do this yet"
        
        self.layers = nn.Sequential(*layers)
        self.integer_out = integer_out
    
    def forward(x):
        y = self.layers(x)
        if self.integer_out:
            y = y.to(dtype = torch.int32)
        return y


# class TablePredictor_givenDB:
#     pass

# class ColumnPredictor_givenTable:

class RecursivePredictor(nn.Module):
    def __init__(grammar, n_in, token, level = 0, extra_ = None):
        if token == 'table':
            pass #table given db here, available from definitions
        elif token == 'column':
            pass #column given table/db here, as above
        else:
            self.list_of_preds = grammar.gr[token]['items']
            self.passthrough, self.adder, self.resolver = False, False, False
            self.to_resolve = None
            if len(self.list_of_preds) == 1:
                if self.list_of_preds[0] in resolve_dict:
                    self.resolver = True
                    self.to_resolve = self.list_of_preds[0]
                else:
                    self.passthrough = True
            elif grammar.gr[token]['type'] == 'and':
                    self.adder = True
            
            if not self.passthrough and not self.adder and not self.resolver:
                self.classes = grammar.gr[token]['items']
                self.n_classes = len(self.classes)
                self.forward = self.or_pred
                self.sub_pred = nn.ModuleList()
                for tok self.classes:
                    self.sub_pred.append(RecursivePredictor(grammar, n_in, tok, level+1))
            elif self.resolver:
                if self.to_resolve == '[[value_pattern]]':
                    self.forward = self.value_pred
                elif self.to_resolve == '[[int_pattern]]'
                    self.forward = self.int_pred
                elif 'table' in self.to_resolve:
                    self.sub_pred = OrderedDict()
                    for db in dbs:
                        self.sub_pred[db] = RecursivePredictor(grammar, n_in, token = 'table', extra_ = db)
    
                elif 'column' in self.to_resolve:
                    self.sub_pred = OrderedDict()
                    for db in dbs:
                        self.sub_pred = OrderedDict()
                        for tab in tables:
                            self.sub_pred[db][tab] = RecursivePredictor(grammar, n_in, token = 'column', extra_ = [db, tab])
                elif 'like' in self.to_resolve:
                    self.sub_pred = OrderedDict()
                    for db in dbs:
                        self.sub_pred = OrderedDict()
                        for tab in tables:
                            self.sub_pred[tab] = OrderedDict()
                            for c in like_columns:
                                self.sub_pred[db][tab][c] = RecursivePredictor(grammar, n_in, token = 'column', extra_ = [db, tab])

class Classifier(nn.Module):
    def __init__(n_classes, layer_type, n_in):
        super().__init__()
        layers = []
        if isinstance(n_hidden, int):
            num_divide = round(math.log(n_in-n_classes, (n_hidden+1)))
            to_div = (n_in-n_classes)//num_divide
            layers.append(nn.Linear(n_in, to_div))
            for i in range(n_hidden):
                layers.append(layer_type(to_div, to_div//num_divide))
                to_div =  to_div//num_divide
            assert to_div == n_classes
        else:
            assert False, "I can't do this yet"
        layers.append(nn.Softmax(dim = 0))
        self.layers = nn.Sequential(*layers)
    
    def forward(x):
        y = self.layers(x)
        return y
        