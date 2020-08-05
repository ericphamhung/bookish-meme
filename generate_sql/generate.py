from definitions import *
import torch.nn as nn
from simpler_grammar import SimpleGrammar as Grammar 
import torch.nn.functional as F
from collections import OrderedDict
import torch

emb_sz = 1024

# Create predic

class Classifier(nn.Module):

    def __init__(self, n_classes):
        super(Classifier, self).__init__()
        self.n_classes = n_classes
        self.fc1 = nn.Linear(emb_sz, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_classes)
       
    
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim = -1)
        return x



class Generator(nn.Module):

    def __init__(self, fname, cuda = True):

        super().__init__()

        gr = Grammar(fname)
        self.gr = gr
        self.grammar = gr.graph
        self.terminals = gr.terminals
        self.terminal_toks = gr.terminal_toks
        self.ors = gr.ors
        self.or_loc = {l:i for i, l in enumerate(self.ors)}
        self.ands = gr.ands
        self.learners = self.create_learners(cuda = cuda)

    
    def create_learners(self, cuda):
        mdict = []
        for o in self.ors:
            n_ = len(self.gr.gr[o]['items'])
            cls_ = Classifier(n_)
            if cuda:
                cls_ = cls_.cuda()
            mdict.append(cls_)
        return mdict
    
    # def forward_or(self, x, or_tok):
    #     assert or_tok in self.ors
    #     l = self.learners[or_tok]
    #     xx = l.forward(x)
    #     return xx
    
    def generate(self, x, tok):
        tok = self.gr.pathone(tok)
        string = ' '
        if tok in self.terminals:
            string += self.gr.gr[tok]['items'][0]
        elif tok in self.ands:
            for t in self.gr.gr[tok]['items']:
                string += self.generate(x, t)
        else:
            i = self.or_loc[tok]
            l = self.learners[i]
            probs = l.forward(x)
            probs = probs.detach().cpu().numpy()
            ind = probs.argmax()
            string += self.generate(x, self.gr.gr[tok]['items'][ind])
        return string

    def train_sentence(self, sentence, sql):
        sql_toks = self.gr._tokenize(sql)
        #just take the "right" sql tokens here




            


if __name__ == '__main__':
    g = Generator('sql_simple_transition_4_no_like.bnf')
    mm = torch.rand(1024).cuda()
    print(g.generate(mm, '<query>'))