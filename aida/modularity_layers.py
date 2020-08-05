import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import BCELoss as BCE
from pathlib import Path
from copy import copy
import time
from extended import ExtendedModule, ExtendedGRU
from torch.utils.data import Dataset, TensorDataset, DataLoader
# from allennlp.commands.elmo import ElmoEmbedder
from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.commands.elmo import ElmoEmbedder

class ModularityLayerGRUAttn(ExtendedModule):
    def __init__(self, attentionmod):
        super().__init__()
        self.hidden_sz = attentionmod.hidden_sz//2
        mods = []
        mods.append(nn.GRU(self.hidden_sz, self.hidden_sz, bidirectional=True))
        mods.append(attentionmod)
        self.modules_ = nn.ModuleList(modules = mods)
        self.initialize()

    #x - of shape (seq_len, batch_sz, hidden_sz)
    #mask - (seq_len, batch_sz, hidden_sz*2)
    # - 1 if still embedding, 0 o/w
    def forward(self, x, mask):
        h, _ = self.modules_[0](x)
        h, sum_dim = self.modules_[1](h)
        h *= mask
        h = h.sum(sum_dim)
        return h

        # h, _ = self.process(x)
        # #this creates the alphas
        # sc = self.score(x.view(-1, self.hidden_sz*MAX_LEN*2)).view(-1, MAX_LEN, 1)
        # #this makes sure we don't go past the end of the sentence
        # print(mask.sum())
        # print(h.sum())
        # # print(h.size())
        # # print(mask.size())

        # # h = h.view(MAX_LEN, -1, self.hidden_sz*2)
        # h = h * mask
        # print(h.sum())
        # #these three take the sum of alpha_t*h_t, although the compactification is non-standard
        # s = h.view(-1, MAX_LEN, self.hidden_sz*2)*sc
        # s = self.compact(s.view(-1, 2*self.hidden_sz)).view(-1, MAX_LEN, self.hidden_sz)
        # s = s.sum(dim=1)
        # return s