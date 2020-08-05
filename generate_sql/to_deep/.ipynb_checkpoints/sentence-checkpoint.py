import torch.nn as nn
import torch.nn.functional as F
import torch

import json
from copy import copy
from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.commands.elmo import ElmoEmbedder
from allennlp.modules.scalar_mix import ScalarMix
device = torch.device('cuda')

def get_elmo_sentence_mixture(sentence):
    character_ids = batch_to_ids(sentence).to(device)
    embedding = elmo(character_ids)['elmo_representations'][0]
    sm = ScalarMix(embedding.size()[1]).cuda()
    return  (sm, embedding)


options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

elmo = Elmo(options_file, weight_file, 1, dropout=0).cuda()
class SentenceRepresentation(nn.Module):

    def __init__(self, sentence):#sentence_dim, dropout_p = 0  max_length, 
        super().__init__()
        character_ids = batch_to_ids(sentence).to(device)
        _embedding = elmo(character_ids)['elmo_representations'][0]
        self.embedding = []
        for i in range(_embedding.size()[0]):
            self.embedding.append(_embedding[i, :, :])
        self.mixture = ScalarMix(len(self.embedding))
        

    def forward(self):
        return self.mixture(self.embedding)
