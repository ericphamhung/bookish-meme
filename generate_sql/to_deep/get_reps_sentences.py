import spacy
import pandas as pd
import json
import numpy as np 
from dates import *

train = pd.read_pickle('/home/jq/software/triplet-all/spider_df_train.pkl')
test = pd.read_pickle('/home/jq/software/triplet-all/spider_df_test.pkl')

lst = []
nlp = spacy.load('en_core_web_sm')
all_ = train['eng'].tolist()
all_.extend(test['eng'].tolist())
for sent in all_:
    
    

    
#     sent = nlp(tr)
#     lss = np.unique([t.pos_ for t in sent]).tolist()
#     if 'X' in lss:
#         print('X')
#         print(tr)
#         print([t for t in sent if t.pos_ == 'X'])
#     if 'SYM' in lss:
#         print('SYM')
#         print(tr)
#         print([t for t in sent if t.pos_ == 'SYM'])
    
#     lst.extend(lss)

# print(np.unique(lst))