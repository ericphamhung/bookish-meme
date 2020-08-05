import torch.nn as nn
import torch.nn.functional as F
import torch

def and_or_representations(ddim, dim, index):
    if index == 0:
        ret = nn.Sequential(nn.Linear(ddim, dim), nn.ReLU(inplace = True), nn.Linear(dim, dim), nn.ReLU(inplace = True))
    elif index == 1:
        ret = nn.Sequential(nn.Linear(ddim, dim), nn.ReLU(inplace = True))
    else:
        raise ValueError('Unrecognized index {} in and_or_representations'.format(i))
    return ret

def base_representations(x_dim, dim, index):
    if index == 0:
        ret = nn.Sequential(nn.Linear(dim+x_dim, dim), nn.ReLU(inplace = True), nn.Linear(dim, dim), nn.ReLU(inplace = True))
    elif index == 1:
        ret = nn.Sequential(nn.Linear(dim_x+dim, dim), nn.ReLU(inplace = True))
    else:
        raise ValueError('Unrecognized index {} in base_representations'.format(i))
    return ret

