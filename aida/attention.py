import torch
import torch.nn as nn
import torch.nn.functional as F
from extended import ExtendedModule
import numpy as np

#Note - these are simplified attentive models!
# No query embedding!!

class ConvolutionDotProdNormAttention(ExtendedModule):

    def __init__(self, hidden_sz, max_len, filen = None, bidirection = True, num_tasks = 1, seq_dim = 0):
        super().__init__(filen=filen, tanh=True)
        assert seq_dim != 2
        self.seq_dim = seq_dim
        
        self.hidden_sz = hidden_sz*2 if bidirection else hidden_sz
        
        if seq_dim == 0:
            self.viewer = (1, -1, self.hidden_sz)
        else:
            self.viewer = (-1, 1, self.hidden_sz)
        self.bidirection = bidirection
        self.max_len = max_len
        self.num_tasks = num_tasks
        pars = []
        for _ in range(num_tasks):
            p = torch.randn(1, 1, self.hidden_sz)*np.sqrt(2/self.hidden_sz)
            pars.append(p)    
            
        self.params = nn.ParameterList([nn.Parameter(data=p) for p in pars])
        
        self.softmax = nn.Softmax(dim=2)
        self.initialize()

    def forward(self, hidden_x):
        assert (len(hidden_x.size()) == 3) and (hidden_x.size()[2] == self.hidden_sz)
        lst = []
        
        for i in range(self.num_tasks):
            raw_score = self.params[i]*hidden_x/np.sqrt(1.0*self.hidden_sz)
            raw_score = raw_score.sum(dim=self.seq_dim).view(*self.viewer)
            score = self.softmax(raw_score)
            to_return = score*hidden_x
            # to_return = to_return.sum(dim=self.seq_dim)
            lst.append(to_return)
        if len(lst) > 1:
            to_return = torch.cat(lst, dim = -1)
        else:
            to_return = lst[0]
        return to_return, self.seq_dim


class ConcatLinearTanhAttention(ExtendedModule):

    def __init__(self, hidden_sz, max_len, filen = None, bidirection = True, num_tasks = 1, seq_dim = 0):
        super().__init__(filen=filen)
        assert seq_dim != 2
        assert num_tasks ==1 #for now
        self.seq_dim = seq_dim
        
        self.hidden_sz = hidden_sz*2 if bidirection else hidden_sz
        self.max_len = max_len
        if seq_dim == 0:
            self.viewer1 = (self.max_len, -1, self.hidden_sz)
            self.viewer2 = (1, -1, self.hidden_sz)
        else:
            self.viewer1 = (-1, self.max_len, self.hidden_sz)
            self.viewer2 = (1, -1, self.hidden_sz)
        self.bidirection = bidirection
        
        self.num_tasks = num_tasks
        mods = []
        mods.append(nn.Linear(self.hidden_sz, self.hidden_sz))
        mods.append(nn.Tanh())
        mods.append(nn.Linear(self.hidden_sz, self.hidden_sz))
        self.modules_ = nn.Sequential(*mods)
        self.softmax = nn.Softmax(dim=2)
        self.initialize()

    def forward(self, hidden_x):
        hxs = hidden_x.size()
        assert (len(hxs)==3) and (hxs[2] == self.hidden_sz)
        lst = []
        hidden_x = hidden_x.view(-1, self.hidden_sz)
        for i in range(self.num_tasks):
            raw_score = self.modules_(hidden_x).view(*self.viewer1)
            raw_score = raw_score.sum(dim=self.seq_dim).view(*self.viewer2)
            score = self.softmax(raw_score)
            to_return = score*hidden_x.view(hxs)
            lst.append(to_return)
        if len(lst) > 1:
            to_return = torch.cat(lst, dim = -1)
        else:
            to_return = lst[0]
        return to_return, self.seq_dim
