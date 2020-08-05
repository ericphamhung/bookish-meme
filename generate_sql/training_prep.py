import torch
import pandas as pd
from scipy.stats import multivariate_normal as mvn
from glob import glob
import numpy as np
from utils import *

data_loc = '/home/jq/software/triplet-all/spider_pickles/'
emb_files = glob(data_loc+'emb*')



train = pd.read_pickle('/home/jq/software/triplet-all/original/spider_df_train.pkl')
train['file_number'] = train['file_number'].apply(file_num)
train.reset_index(inplace = True)
print(train.index[:5])
train.to_pickle('/home/jq/software/triplet-all/spider_df_train.pkl')

test = pd.read_pickle('/home/jq/software/triplet-all/original/spider_df_test.pkl')
test['file_number'] = test['file_number'].apply(file_num)
test.reset_index(inplace = True)
print(test.index[:5])
test.to_pickle('/home/jq/software/triplet-all/spider_df_test.pkl')

# assert False
train_idx = train.file_number.tolist()
test_idx = test.file_number.tolist()
# print(test_idx)
# assert False

#emb_nums = [int(t.split('spider_')[-1].split('.')[0]) for t in emb_files]
emb_nums = [t.split('spider_')[-1].split('.')[0] for t in emb_files]

trainembs = np.intersect1d(train_idx, emb_nums)
testembs = np.intersect1d(test_idx, emb_nums)

# print('test/train df overlap {}'.format(len(np.intersect1d(train_idx, test_idx))))
# print('{} in dfs, {} in embeddings'.format(train.shape[0]+test.shape[0], len(emb_nums)))
# print('{} in dfs not in embeddings'.format(len(np.setdiff1d(np.union1d(train_idx, test_idx),emb_nums))))
# print('{} in embeddings not in dfs'.format(len(np.setdiff1d(emb_nums, np.union1d(train_idx, test_idx)))))

def num_words(arr, axis = 0):
    return np.array(arr.shape[axis]).reshape((1, ))
use_funcs = [num_words, np.mean, np.median, np.std, np.min, np.max]   

def get_description(arr):
    return np.concatenate([f(arr, axis = 0) for f in use_funcs])

tr_arrs, te_arrs, err_arrs = {}, {}, {}
for t, num in zip(emb_files, emb_nums):
    arr = torch.load(t).squeeze().detach().cpu().numpy()
    if num in trainembs:
        newnum = int(train[train['file_number']==num].index[0])
        assert isinstance(newnum, np.int)
        tr_arrs[str(newnum)] = get_description(arr)
    elif num in test_idx:
        newnum = int(test[test['file_number']==num].index[0])
        assert isinstance(newnum, int)
        te_arrs[str(newnum)] = get_description(arr)
    # else:
    #     err_arrs[num] = get_description(arr)


np.save('/home/jq/software/triplet-all/train_sentence_summaries.npy', tr_arrs)

np.save('/home/jq/software/triplet-all/test_sentence_summaries.npy', te_arrs)

# if len(err_arrs) >0:
#     print('{} errors'.format(len(err_arrs)))
#     np.save('/home/jq/software/triplet-all/mismatch_sentence_summaries.npy', err_arrs)