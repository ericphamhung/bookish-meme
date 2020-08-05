import numpy as np
import pandas as pd
import pyro
import torch
import torch.nn as nn
import pyro.distributions as dist
from pyro import poutine
from pyro.contrib.autoguide import AutoDelta
from pyro.optim import Adam
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, infer_discrete


# class DecompositionLocalSeasonal(nn.Module):

#     def __init__(self, )

# def create_model_guide(seas, local_=True, global_=True):
    
#     if len(seas) 
#     def model(y, seas_ind):

'''
DECOMPOSITION!

use from custom objectives to add extra terms, see http://pyro.ai/examples/custom_objectives.html

Per SKU/For all

With MONTH (FEB-DEC, 11) as regressor, M

PER REP & CUST, or PER REP, or PER CUST, or ALL

DECOMPOSE SALES, S, into
(St = current, Stm1 = S t minus 1, etc)
St = beta1*M + beta2*(Stm1-Stm2)*(1-a) + a*beta3 + 
'''



'''
Might sort of work, for switch...

But need sequential

Below is for GMM


@config_enumerate
def model(data):
    # Global variables.
    weights = pyro.sample('weights', dist.Dirichlet(0.5 * torch.ones(K)))
    scale = pyro.sample('scale', dist.LogNormal(0., 2.))
    with pyro.plate('components', K):
        locs = pyro.sample('locs', dist.Normal(0., 10.))

    with pyro.plate('data', len(data)):
        # Local variables.
        assignment = pyro.sample('assignment', dist.Categorical(weights))
        pyro.sample('obs', dist.Normal(locs[assignment], scale), obs=data)

global_guide = AutoDelta(poutine.block(model, expose=['weights', 'locs', 'scale']))
'''

'''
Idea - 

Decompose y_t into

y_t = T_t + S_t + E_t

s.t.

T_t = a_t*(T_{t-1}+u) + (1-a_t)*w

where a_t ~ Ber(p_t), u ~ N(0, 0.1), w ~ N(0, 10)

S_t = S_i + v

where v ~ N(0, 1), and i is the seasonal index (i=1, ..., s, where s might = 7 or 12)

E_t ~ N(0, sig)

want to add sum(a_t) (or sum(p_t), probably, a) to the loss, we can, just need to look at pyro.optim
 -- or in poutine docs, or similar, was looking on the phone

will want to tweak standard deviations (0.1, 1, 10) somewhat

The changepoints, where a_t are one, are likely what we're looking out for

'''


@config_enumerate
def model(y, seas = 12):
    trends = []
    seases = []
    resids = []
    tr = pyro.sample('trend_0', dist.Normal(0, 10))
    for i in range(1, len(y)):
        
