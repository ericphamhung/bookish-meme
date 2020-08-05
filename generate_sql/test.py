import torch
import pandas as pd
from scipy.stats import multivariate_normal as mvn
from glob import glob
import numpy as np

df = pd.read_pickle("/home/jq/software/triplet-all/spider_pickles/spider_df_train.pkl")
for c in df:
    print(c)
    print(df[c].head())
assert False

data_loc = '/home/jq/software/triplet-all/spider_pickles/'

train_ = glob(data_loc+'train_emb*')

all_train = []
means = []

for t in train_:
    p = torch.load(t).squeeze()
    means.append(p.detach().cpu().numpy().mean(axis = 1))
    #print(p.size())
    #assert False
    all_train.append(torch.load(t))

all_ = torch.cat(all_train, dim = 0).squeeze().detach().cpu().numpy()

all_mean = all_.mean(axis = 1)
#all_var = np.cov(all_)
#print(all_var.shape)
assert False
sument = 0.0

def get_ent(x):
    return mvn(x, mean =all_mean, cov=all_var)

for j in a_:
    pass
    
print(mvn)