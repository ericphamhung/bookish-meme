import torch.nn as nn
import torch
from torch.distributions import Normal
import numpy as np

def initializer(m, activation = None):
    if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    elif type(m) in [nn.Linear]:
        
    
        if activation == 'tanh':
            nn.init.xavier_uniform_(m.weight)
        else:
            nn.init.kaiming_uniform_(m.weight)
    
        try:
            m.bias.data.fill_(0.0)
        except:
            pass
    else:
        pass



def parameter_initializer(shape, seed = None):
    if seed is not None:
        torch.manual_seed(seed)
    return Normal(torch.zeros(shape), 0.1*torch.ones(shape))



class ExtendedModule(nn.Module):

    def __init__(self, cuda = True, 
                initializer = initializer,
                parameter_initializer = parameter_initializer, 
                modules = None, 
                parameters = None, 
                device = None, 
                last_layer = None, 
                filen = None, 
                tanh = False):
        super().__init__()
        if filen is not None:
            self.initializer = filen
        else:
            if tanh:
                def initial(x):
                    initializer(x, activation='tanh')
                self.tanh = True
            else:
                def initial(x):
                    initializer(x)
                self.tanh = False
            self.initializer = initial
        self.last_layer = last_layer
        self.use_cuda = cuda
        if modules is not None:
            try:
                _ = iter(modules)
            except TypeError:
                modules = [modules]
            
            self.modules_ = nn.Sequential(*modules)
        else:
            self.modules_ = modules
        self.params = parameters
        if self.cuda and device is None:
            self.device = torch.device('cuda')
    

    def initialize(self, check = False):
        #if self.params is not None:
        if type(self.initializer) == str:
            self.load(self.initializer)
        else:
            if self.modules_ is not None:
                for i, m in enumerate(self.modules()):
                    m.apply(self.initializer)
            

            if check:
                for i, m in enumerate(self.modules()):
                    print(m.parameters())

        if self.use_cuda:
            self = self.cuda()
            if self.modules_ is not None:
                for i, m in enumerate(self.modules_):
                    self.modules_[i] = m.cuda()
            if self.params is not None:
                for i, p in enumerate(self.params):
                    self.params[i] = p.to(self.device)
    
    def save(self, fname):
        if self.modules_ is not None:
            for i, m in enumerate(self.modules_):
                try:
                    m.save('{}_{}'.format(fname, i))
                except:
                    torch.save(m.state_dict(), '{}_{}'.format(fname, i))
        if self.params is not None:
            for i, p in enumerate(self.params):
                torch.save(p.detach(), '{}_parameter_{}'.format(fname, i))
        torch.save(self.state_dict(), fname)
    
    def load(self, fname):
        if self.modules_ is not None:
            for i, m in enumerate(self.modules_):
                try:
                    m.load('{}_{}'.format(fname, i))
                except:
                    m.load_state_dict(torch.load('{}_{}'.format(fname, i)))
        if self.params is not None:
            for i, p in enumerate(self.params):
                p.data = torch.load('{}_parameter_{}'.format(fname, i))
        self.load_state_dict(torch.load(fname))
    

    def forward(self, x):
        pass

    def get_last_layer(self):
        return self.last_layer
    
    def get_modules(self):
        return self.modules_

class ExtendedGRU(ExtendedModule):
    def __init__(self, in_sz, out_sz, bidirectional = False):
        super().__init__()
        self.modules_ = nn.Sequential(nn.GRU(in_sz, out_sz, bidirectional=bidirectional))
    
    def forward(self, x):
        return self.modules_(x)
