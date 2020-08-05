import torch.nn as nn
import torch.nn.functional as F
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from run_defs import *
from definitions import *
from torch.optim.lr_scheduler import MultiStepLR
from representations import *
from predictors import MultiLabelClassifier, Regressor, MultiLabelColumnGivenDBAndTableClassifier
import json
from copy import copy
from utils import *


device = torch.device('cuda')
with open('terminals2index_grammar_2.json') as f:
    terminals2index = json.load(f)
with open('spider_tab2index.json') as f:
    tables2index = json.load(f)
with open('spider_col2index.json') as f:
    cols2index = json.load(f)

with open('spider_db_cols2ind.json') as f:
    dbcols2index = json.load(f)

def float_poss(x):
    try: 
        if isinstance(x, list):
            if len(x) == 0:
                return False
            else:
                x = x[0]
        int(x)
        val = True
    except ValueError:
        val = False
    return val

def get_first(x):
    if isinstance(x, list):
        if len(x) == 0:
            raise Error("Shouldn't be albe to get there")
        x = x[0]
    return x



class Dset_v1(Dataset):

    def __init__(self, filen, embedfile, data_t, label = False):
        self.df = pd.read_pickle(filen)
        self.embed = np.load(embedfile).item()
        self.data_t = data_t
        self.label = label
        dropkeys = []
        if 'value' in self.data_t:
            for key in self.embed.keys():
                if not float_poss(self.df.loc[int(key), self.data_t]):
                    dropkeys.append(key)
            print('In value conversion for {}, lost {}.  NOTE ONLY FIRST USED'.format(self.data_t, len(dropkeys)))
        elif 'int' in self.data_t:
            for key in self.embed.keys():
                if self.df.loc[int(key), self.data_t] is None:
                    dropkeys.append(key)
            print('In integer/limit conversion for {}, lost {}'.format(self.data_t, len(dropkeys)))
        elif 'all_cols' == self.data_t[:2]: #all columns
            cnt = 0
            
            for key in self.embed.keys():
                if len(self.df.loc[int(key), self.data_t]) == 0:
                    dropkeys.append(key)
                for kk in gram.terminal_toks:
                    if kk in self.df.loc[int(key), self.data_t]:
                        cnt += 1
                        dropkeys.append(key)
                        break
            print('In all column conversion for {}, {} dropped, {} BECAUSE OF ENCODING ERRORS, rest empty'.format(self.data_t, len(dropkeys), cnt))
        elif '_col' in self.data_t:
            col = copy(self.df[self.data_t])
            for lst in col:
                for kk in gram.terminal_toks:
                    if kk in lst:
                        lst.remove(kk)
                for kk in SQL_TOKS_REM:
                    if kk in lst:
                        lst.remove(kk)
            self.df[self.data_t] = col
                
            cnt = 0
            for key in self.embed.keys():
                if len(self.df.loc[int(key), self.data_t]) == 0:
                    dropkeys.append(key)
                
                    
            print('In all column conversion for {}, {} dropped'.format(self.data_t, len(dropkeys)))#, {} BECAUSE OF ENCODING ERRORS, rest empty cnt
            
            
        elif 'modify_ta' in self.data_t:
            cnt = 0
            for key in self.embed.keys():
                if self.df.loc[int(key), self.data_t] is None:
                    dropkeys.append(key)
            print('In table conversion for {}, {} are empty'.format(self.data_t, len(dropkeys)))
        elif 'tab' in self.data_t and self.data_t != 'table_name':
            cnt = 0
            for key in self.embed.keys():
                if len(self.df.loc[int(key), self.data_t]) == 0:
                    dropkeys.append(key)
            print('In table conversion for {}, {} are empty'.format(self.data_t, len(dropkeys)))
        elif self.data_t == 'table_name':
            cnt = 0
            for key in self.embed.keys():
                if self.df.loc[int(key), self.data_t] is None:
                    dropkeys.append(key)
            print('In table conversion for {}, {} are empty'.format(self.data_t, len(dropkeys)))
        if len(dropkeys)>0:
            for k in dropkeys:
                del self.embed[k]
                self.df = self.df.drop(int(k))
        if 'int' in self.data_t or 'valu' in self.data_t:
            # print(self.df[self.data_t].unique())
            firstones = self.df[self.data_t].apply(get_first)
            self.df[self.data_t] = firstones.astype(np.float32)

        # assert self.df.shape[0] == len(self.embed)
        assert (all([str(a) in self.embed for a in self.df.index.tolist()]))
        self.keymap = {str(i):ind for i, ind in enumerate(self.embed.keys())}
        self.table_ints = None
        self.column_ints = None
        self.db_columns = None
        i= int(self.keymap[str(0)])
        self.x_dim = self.embed[str(i)].shape[0]
        assert self.data_t in self.df
        if 'col' in self.data_t or 'table' in self.data_t:
            if 'table' in self.data_t:
                self.table_ints = copy(tables2index)
            elif self.data_t == 'all_cols':
                self.db_columns = copy(dbcols2index)
            else:
                self.column_ints = copy(cols2index)
                mm = 0
                for db in spider_db:
                    for tab in spider_db[db]:
                        mm = max(mm, len(spider_db[db][tab]))
                self.max_columns = mm
                
    
    def get_x_dim(self):
            
            return self.x_dim

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, i):
        i = int(self.keymap[str(i)])
        db = self.df.loc[i, 'db_name'].lower()
        if 'modify' in self.data_t:
            table = self.df.loc[i, 'modify_table_name'].lower()
            if table is None:
                table = self.df.loc[i, 'table_name'].lower()
        else:
            table = self.df.loc[i, 'table_name'].lower()

        # print(self.df.loc[i])
        x = self.embed[str(i)].astype(np.float32)
        y = self.df.loc[i, self.data_t]
        
        if self.data_t == 'modifyop':
            for i, tok in enumerate(y):
                if tok == '[[string_pattern]]':
                    y[i] = '[[value_pattern]]'


        # for t in SQL_TOKS_REM:
        #     y = y.replace(t, '')

        if self.table_ints is not None:
            
            if self.label:
                ret = torch.zeros(len(self.table_ints[db]), dtype = torch.long, device=device)
                if isinstance(y, str):
                    y = self.table_ints[db][y.lower()]
                    ret[y] = 1
                else:
                    for yy in y:
                        yy = self.table_ints[db][yy.lower()]
                        ret[yy] = 1
                y = ret
            else:
                y = torch.tensor(y,dtype = torch.long, device=device)
        elif self.column_ints is not None:
            x = x.reshape(-1, self.x_dim)
            if self.label:
                ret = torch.zeros(self.max_columns, dtype = torch.long, device=device)
                if isinstance(y, str):
                    y = self.column_ints[db][table][y.lower()]
                    ret[y] = 1
                else:
                    for yy in y:
                        yy = self.column_ints[db][table][yy.lower()]
                        ret[yy] = 1
                y = ret
            else:
                if isinstance(y, str):
                    y = self.column_ints[db][table][y.lower()]
                else:
                    lst = []
                    for yy in y:
                        lst.append(self.column_ints[db][table][yy.lower()])
                    y = lst
                y = torch.tensor(y,dtype = torch.long, device=device)
            y = y.unsqueeze(0)
           
        elif self.db_columns is not None:
            ret = torch.zeros(len(self.db_columns[db]), dtype = torch.long, device = device)
            ret[self.db_columns[db][y.lower()]] = 1
            y = ret
        elif 'value' in self.data_t or 'int' in self.data_t:
            y = torch.tensor(y, dtype = torch.float, device=device)

        sample = {'x':torch.tensor(x, requires_grad=True, device=device), 'y':y, 'db':db, 'table':table}
        return sample


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = False
    return model


def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = True
    return model


def summarize_loss(item, max_, min_, mean_, n, bs):
    max_ = max(max_, item*bs)
    min_ = min(min_, item*bs)
    mean_ = (mean_*(n-bs)+item*bs)/n
    return max_, min_, mean_

def train(model, dl, optimizer, criterion, printevery, prepy):
    bs1 = dl.batch_size
    running_loss = 0.0
    max_, min_, mean_, n = -1.0, 10000.0, 0.0, 0
    ll = len(dl)
    for i, batch in enumerate(dl):
        if bs1 > 1:
            if i == ll-1:
                bs = y.size()[0]
            else:
                bs = bs1
        else:
            bs = bs1
        model.zero_grad()
        
        x = batch['x'].to(device)
        table = batch['table']
        db = batch['db']
        try:
            ypred = model.forward(x)
        except:
            ypred = model.forward(x, db, table)
        #only happens for the corner case
        # if isinstance(ypred, tuple):
        #     ypred = ypred[0]
        #     y = model.matrix_of_tokens(batch['y'])
        #     y = y.to(device)
        # else:
        if prepy is None:
            y = batch['y'].to(device)
        else:
            # assert bs == 1
            y = prepy(batch['y'][0]).to(device)
            ypred = torch.t(ypred)
        loss = criterion(ypred, y)
        loss.backward()
        optimizer.step()
        lit = loss.item()
        running_loss += lit
        ave_item = lit/bs
        n = n+bs
        max_, min_, mean_ = summarize_loss(ave_item, max_, min_, mean_, n, bs)
        if i % printevery == (printevery-1):
            print('{} of {}, average loss = {}'.format(i, len(dl), running_loss/printevery))
            running_loss = 0.0
    return model, max_, min_, mean_


def test(model, dl, optimizer, criterion, printevery, prepy):
    bs1 = dl.batch_size
    model = freeze(model)
    bs = dl.batch_size
    running_loss = 0.0
    max_, min_, mean_, n = -1.0, 10000.0, 0.0, 0
    ll = len(dl)
    for i, batch in enumerate(dl):
        if bs1 > 1:
            if i == ll-1:
                bs = y.size()[0]
            else:
                bs = bs1
        else:
            bs = bs1
        model.zero_grad()
         
        x = batch['x'].to(device)
        table = batch['table']
        db = batch['db']
        try:
            ypred = model.forward(x)
        except:
            ypred = model.forward(x, db, table)
        #only happens for the corner case
        # if isinstance(ypred, tuple):
        #     ypred = ypred[0]
        #     y = model.matrix_of_tokens(batch['y'])
        #     y = y.to(device)
        # else:
        if prepy is None:
            y = batch['y'].to(device)
        else:
            # assert bs == 1
            y = prepy(batch['y'][0]).to(device)
            ypred = torch.t(ypred)
        loss = criterion(ypred, y)
        lit = loss.item()
        running_loss += lit
        ave_item = lit/bs
        n = n+bs
        max_, min_, mean_ = summarize_loss(ave_item, max_, min_, mean_, n, bs)
        if i % printevery == (printevery-1):
            print('{} of {}, average loss = {}'.format(i, len(dl), running_loss/printevery))
            running_loss = 0.0
    model = unfreeze(model)
    return max_, min_, mean_

def final_test(model, dl, optimizer, criterion, printevery, prepy):
    bs1 = dl.batch_size
    model = freeze(model)
    bs = dl.batch_size
    running_loss = 0.0
    max_, min_, mean_, n = -1.0, 10000.0, 0.0, 0
    ll = len(dl)
    for i, batch in enumerate(dl):
        if bs1 > 1:
            if i == ll-1:
                bs = y.size()[0]
            else:
                bs = bs1
        else:
            bs = bs1
        model.zero_grad()
         
        x = batch['x'].to(device)
        table = batch['table']
        db = batch['db']
        try:
            ypred = model.forward(x)
        except:
            ypred = model.forward(x, db, table)
        #only happens for the corner case
    
        if prepy is None:
            y = batch['y'].to(device)
        else:
            y = prepy(batch['y'][0]).to(device)
        
        print('Predicted {}, got {}'.format(ypred, y))
    return max_, min_, mean_

def tests_done():
    
    for d in dnames:
        dsettr = Dset_v1('/home/jq/software/triplet-all/spider_df_train.pkl', '/home/jq/software/triplet-all/train_sentence_summaries.npy', d)
        failed = False
        for i in range(len(dsettr)):
            try:
                dsettr.__getitem__(i)
                
            except:
                
                print('{} training failed at item {}'.format(d, i))
                assert False
                failed = True
        if not failed:
            print('{} train passed'.format(d))
        else:
            print('{} train has an error'.format(d))
        dsette = Dset_v1('/home/jq/software/triplet-all/spider_df_test.pkl', '/home/jq/software/triplet-all/test_sentence_summaries.npy', d)
        failed = False
        for i in range(len(dsette)):
            try:
                dsette.__getitem__(i)
                
            except:
                failed = True
                print('{} testing failed at item {}'.format(d, i))
        if not failed:
            print('{} test passed'.format(d))
        else:
            print('{} test has an error'.format(d))
    
    print("x dim is {}".format(dsettr.get_x_dim()))

def run_get(num, summaries = True):
    if not summaries:
        raise ValueError('Not done yet')
    to_run = dnames[num]
    if '_col' in to_run:
        label = True
    else:
        label = False
    data_train = Dset_v1(trdf, trfile, to_run, label)
    data_test = Dset_v1(tedf, tefile, to_run, label)
    dimx = data_train.get_x_dim()

    if to_run in ['sellist', 'modifyop']:
        model = TrainableRepresentation(gram, x_dim = dimx, start = to_run, return_log = False)
        model = model.cuda()
    elif to_run in ['sellist_cols', 'modify_cols']:
        model = MultiLabelColumnGivenDBAndTableClassifier(dimx, spider_db, data_train.max_columns)
    elif 'int' in to_run or 'value' in to_run:
        model = Regressor(dimx)
        model = model.cuda()
    elif 'tab' in to_run:
        model = Classifier(dimx, max_tables)
        model = model.cuda()
    elif 'col' in to_run:
        model = Classifier(dimx, max_columns)
        model = model.cuda()


    
    return data_train, data_test, model

### NOTE - we need to get selection list and value pattern better, we're only getting the fist at the moment
### NOTE 2 - Currently not using 'like_pattern', 
##### Really not hard to use, except it is for me now

### NOTE - Batch size needs to be 1 for anything that needs the dict, need to change data set to accomodate somehow 
#### Eric can probably fix if I can coherently describe the problem
#
#  LATER:
#  'like_pattern', -- named entity recognition
#  'string_pattern' -- named entity recognition + whether a value pattern is a string or a value pattern
#  What is standard NER?  Was going to go through sentence to predict ELMO embedding closest to the one we want (somehow)
dnames =['sellist', 'sellist_cols', 'table_name', 'modifyop', 'int_pattern',  'modify_table_name', 'value_pattern', 'modify_cols', 'all_cols']
if __name__ == '__main__':
    # print(pd.read_pickle('/home/jq/software/triplet-all/spider_df_train.pkl').columns.tolist())
    # assert False
    # tests_done()

    # num = 0
    # data_train, data_test, model = run_get(num)
    # print(data_train.__getitem__(0))
    # assert False
    for num in [1]: #range(len(dnames)):
        

            

        num_epochs = 100

        data_train, data_test, model = run_get(num)
        if num == 0 or num == 6:
            prepy = model.matrix_of_tokens
        else:
            prepy = None
        if num == 4 or num == 5:
            criterion = nn.L1Loss()
        elif num == 1 or num == 7:
            criterion = nn.BCEWithLogitsLoss()
        else:   
            criterion = nn.CrossEntropyLoss()
        if num == 0 or num == 6:
            printevery = 100
            batch_size = 10
        else:
            printevery = 10
            batch_size = 100
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        best_since = 0
        best = 1e6*1.0
        best_since_limit = 10
        test_every = 1
        trl = DataLoader(data_train, batch_size=batch_size, shuffle=True)
        tel = DataLoader(data_test, batch_size=batch_size, shuffle=True)

        scheduler = MultiStepLR(optimizer, milestones=[30,num_epochs], gamma=0.1)
        for i in range(num_epochs):
            model, trmax, trmin, trmean = train(model, trl, optimizer, criterion, printevery, prepy)
            print('Epoch {}, training:\nLosses:\n'.format(i))
            print(' min:{},\n max:{}\n mean:{}:'.format(trmin, trmax, trmean))
            if i % test_every == test_every-1:
                temax, temin, temean = test(model, tel, optimizer, criterion, printevery, prepy)
                print('Epoch {}, testing:\nLosses:\n'.format(i))
                print(' min:{},\n max:{}\n mean:{}:'.format(temin, temax, temean))
                if temean < best:
                    best = temean
                    best_since = 0
                    print('Best model so far.  Saving')
                    bname = 'best/{}_best_{:7.4e}_test_loss'.format(dnames[num], temean)
                    model.save(bname)
                    print('Model saved')
                else:
                    best_since += 1
                if best_since >= best_since_limit:
                    print("Haven't improved mean since {}, gave {}.  Stopping".format(i-best_since_limit, best))
                    model.load(bname)
                    print(final_test(model, tel, optimizer, criterion, printevery))
                    break
                    
                
            scheduler.step()





    