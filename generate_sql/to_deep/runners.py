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
from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.commands.elmo import ElmoEmbedder
from allennlp.modules.scalar_mix import ScalarMix
from sentence import SentenceRepresentation
from simpler_grammar import SimpleGrammar as Grammar
dnames =['sellist', 'sellist_cols',  'modifyop', 'int_pattern',  'value_pattern', 'modify_cols', 'all_cols']

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



class Dset_v2(Dataset):

    def __init__(self, filen, data_t, prepy = None, label = False):
        self.df = pd.read_pickle(filen)
        self.prepy = prepy
        self.data_t = data_t
#         for dn in dnames:
#             del self.df[dn]
        self.label = label
        dropkeys = []
        if 'value' in self.data_t:
            for key in self.df.index.tolist():
                if not float_poss(self.df.loc[int(key), self.data_t]):
                    dropkeys.append(key)
            print('In value conversion for {}, lost {}.  NOTE ONLY FIRST USED'.format(self.data_t, len(dropkeys)))
        elif 'int' in self.data_t:
            for key in self.df[self.data_t]:
                if self.df.loc[int(key), self.data_t] is None:
                    dropkeys.append(key)
            print('In integer/limit conversion for {}, lost {}'.format(self.data_t, len(dropkeys)))
        elif 'all_cols' == self.data_t[:2]: #all columns
            cnt = 0
            
            for key in self.df.index.tolist():
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
            for key in self.df.index.tolist():
                if len(self.df.loc[int(key), self.data_t]) == 0:
                    dropkeys.append(key)
                
                    
            print('In all column conversion for {}, {} dropped'.format(self.data_t, len(dropkeys)))#, {} BECAUSE OF ENCODING ERRORS, rest empty cnt
            
            
        elif 'modify_ta' in self.data_t:
            cnt = 0
            for key in self.df.index.tolist():
                if self.df.loc[int(key), self.data_t] is None:
                    dropkeys.append(key)
            print('In table conversion for {}, {} are empty'.format(self.data_t, len(dropkeys)))
        elif 'tab' in self.data_t and self.data_t != 'table_name':
            cnt = 0
            for key in self.df.index.tolist():
                if len(self.df.loc[int(key), self.data_t]) == 0:
                    dropkeys.append(key)
            print('In table conversion for {}, {} are empty'.format(self.data_t, len(dropkeys)))
        elif self.data_t == 'table_name':
            cnt = 0
            for key in self.df.index.tolist():
                if self.df.loc[int(key), self.data_t] is None:
                    dropkeys.append(key)
            print('In table conversion for {}, {} are empty'.format(self.data_t, len(dropkeys)))
        elif self.data_t == 'modifyop':
            cnt = 0
            for key in self.df.index.tolist():
                if self.df.loc[int(key), self.data_t] is None or len(self.df.loc[int(key), self.data_t])==0:
                    dropkeys.append(key)
            print('In conversion for {}, {} are empty'.format(self.data_t, len(dropkeys)))
        if len(dropkeys)>0:
            for k in dropkeys:
                self.df = self.df.drop(int(k))
        if 'int' in self.data_t or 'valu' in self.data_t:
            # print(self.df[self.data_t].unique())
            firstones = self.df[self.data_t].apply(get_first)
            self.df[self.data_t] = firstones.astype(np.float32)


        self.keymap = {str(i):ind for i, ind in enumerate(self.df.index.tolist())}
        self.table_ints = None
        self.column_ints = None
        self.db_columns = None
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
        lst = []
        self.emb_map = {}
        for cnt, ind in enumerate(self.df.index.tolist()):
            lst.append(SentenceRepresentation(self.df.loc[ind, 'eng']).cuda())
            self.emb_map[str(ind)] = cnt
        self.sentence_embedding = nn.ModuleList(lst)
                
    
    def get_x_dim(self):
            
            return 1024

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, i):
        i = int(self.keymap[str(i)])
        row = self.df.loc[i]
#         print(row)
        db = row['db_name'].lower()
        x = self.sentence_embedding[self.emb_map[str(i)]].forward()
        if 'modify' in self.data_t:
            table = row['modify_table_name']
            if table is None:
                table = row['table_name'].lower()
            else:
                table = table.lower()
        else:
            table = row['table_name'].lower()

        y = row[self.data_t]
        
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
        if self.prepy is not None:
            y = self.prepy(y)
        sample = {'x':x, 'y':y, 'db':db, 'table':table}
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

def train(model, dl, optimizer, criterion, printevery, num, shp1, shp2):
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
#         try:
        ypred = model.forward(x)#, db, table)
#         except:
#             ypred = model.forward(x, db, table)
        #only happens for the corner case
        # if isinstance(ypred, tuple):
        #     ypred = ypred[0]
        #     y = model.matrix_of_tokens(batch['y'])
        #     y = y.to(device)
        # else:
        y = batch['y'].to(device)
#         if num == 0 or num == 6:
                
#             y = y.view(-1, shp1)
#             ypred = ypred.view(-1, shp2, shp1)
        
        loss = criterion(ypred, y)
        loss.backward(retain_graph = True)
        optimizer.step()
        lit = loss.item()
        running_loss += lit
        ave_item = lit/bs
        n = n+bs
        max_, min_, mean_ = summarize_loss(ave_item, max_, min_, mean_, n, bs)
        if i % printevery == (printevery-1):
            print('{} of {}, average loss = {}'.format(i, len(dl), running_loss/printevery))
            running_loss = 0.0
    return model, optimizer, criterion, max_, min_, mean_


def test(model, dl, criterion, printevery, num, shp1, shp2):
    bs1 = dl.batch_size
    model = freeze(model)
    bs = dl.batch_size
    running_loss = 0.0
    max_, min_, mean_, n = -1.0, 10000.0, 0.0, 0
    ll = len(dl)
    with torch.no_grad():
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
    #         try:
            ypred = model.forward(x)#, db, table)
    #         except:
    #             ypred = model.forward(x, db, table)
            #only happens for the corner case
            # if isinstance(ypred, tuple):
            #     ypred = ypred[0]
            #     y = model.matrix_of_tokens(batch['y'])
            #     y = y.to(device)
            # else:
            y = batch['y'].to(device)
#             if num == 0 or num == 6:
#                 y = y.view(-1, shp1)
#                 ypred = ypred.view(-1, shp2, shp1)

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

def final_test(model, dl, optimizer, criterion, printevery):
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
#         try:
        ypred = model.forward(x)
#         except:
#             ypred = model.forward(x, db, table)
        #only happens for the corner case
    
        y = batch['y'].to(device)
        
        
        print('Predicted {}, got {}'.format(ypred, y))
    return max_, min_, mean_

def tests_done():
    
    for d in dnames:
        dsettr = Dset_v2('~/spider_df_train.pkl', d)
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
        dsette = Dset_v2('~/spider_df_test.pkl', d)
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
    
    dimx = 1024
    if to_run in ['sellist', 'modifyop']:
        print(to_run)
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
    if num == 0 or num == 2:
        prepy = model.matrix_of_tokens
    else:
        prepy = None

    data_train = Dset_v2(trdf, to_run, prepy, label)
    data_test = Dset_v2(tedf, to_run, prepy, label)
    dimx = data_train.get_x_dim()
    
    return data_train, data_test, model

def check_token_in_list(tok, lst):
    return any([t.upper() == tok.upper() for t in lst])
def check_token_not_in_list(tok, lst):
    return not any([t.upper() == tok.upper() for t in lst])
def check_any_items_not_in_list(its, lst):
    return any([check_token_not_in_list(it, lst) for it in its])

def check_len(its, length):
    return len(its) > length


def check_(its, length, lst):
    return check_len(its, length) or check_any_items_not_in_list(its, lst)
def resave_data(gfile = 'sql_simple_transition_2.bnf'):
    cols = [0, 2]
    grm = Grammar(gfile)
    assert '<modifyop>' in grm.gr
    grm_terms = grm.terminal_toks
#     grm_terms1 = [g.upper() for g in grm_terms if '[' not in g]
#     grm_terms = [g for g in grm_terms if '[' in g]
#     grm_terms.extend(grm_terms1)
    td = pd.read_pickle(trdf)
    te = pd.read_pickle(tedf)
    td.to_pickle(trdf+'.bak')
    te.to_pickle(tedf+'.bak')

    
    for i, c in enumerate(cols):
        if i == 0:
            continue
        grm1 = Grammar(gfile)
        assert '<modifyop>' in grm1.gr
        max_ = TrainableRepresentation(grm1, x_dim = 1, start = dnames[c], return_log = False).max_length
        numtr, numte = 0, 0
        td['drop_{}'.format(i)] = td[dnames[c]].apply(lambda x: check_(x, max_, grm_terms))
        te['drop_{}'.format(i)] = te[dnames[c]].apply(lambda x: check_(x, max_, grm_terms))
    print('Before shapes {}, {}'.format(td.shape, te.shape))
    td = td[~td['drop_0']]
    td = td[~td['drop_1']]
    te = te[~te['drop_0']]
    te = te[~te['drop_1']]
    del td['drop_0']
    del td['drop_1']
    del te['drop_0']
    del te['drop_1']
    print('After shapes {}, {}'.format(td.shape, te.shape))
    
    td.to_pickle(trdf)
    te.to_pickle(tedf)


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
if __name__ == '__main__':
#     resave_data()
#     assert False
#     print(pd.read_pickle(trdf).columns.tolist())
#     assert False
#     tests_done()
#     assert False
    # num = 0
    # data_train, data_test, model = run_get(num)
    # print(data_train.__getitem__(0))
    # assert False
    for num in [2]: #range(len(dnames)):
        

            

        num_epochs = 250

        data_train, data_test, model = run_get(num)
        
        if num == 4 or num == 5:
            criterion = nn.L1Loss()
        elif num == 1 or num == 7 or num == 0 or num == 2:
            criterion = nn.BCEWithLogitsLoss()
            test_crit = nn.BCEWithLogitsLoss()
        else:   
            criterion = nn.CrossEntropyLoss()
        if num == 0 or num == 2:
            printevery = 10
            batch_size = 100
        else:
            printevery = 10
            batch_size = 100
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        best_since = 0
        best = 1e6*1.0
        best_since_limit = 25
        test_every = 1
        trl = DataLoader(data_train, batch_size=batch_size, shuffle=True)
        tel = DataLoader(data_test, batch_size=batch_size, shuffle=True)
        if num == 0 or num == 2:
            #-1 because of the uknown token that's not used!!!
            shp2 = model.token_size-1
            shp1 = model.max_length
        else:
            shp1, shp2 = 0, 0

        scheduler = MultiStepLR(optimizer, milestones=[10,num_epochs], gamma=0.1)
        for i in range(num_epochs):
            model, optimizer, criterion, trmax, trmin, trmean = train(model, trl, optimizer, criterion, printevery, num, shp1, shp2)
            print('Epoch {}, training:\nLosses:\n'.format(i))
            print(' min:{},\n max:{}\n mean:{}:'.format(trmin, trmax, trmean))
            if i % test_every == test_every-1:
                temax, temin, temean = test(model, tel, test_crit, printevery, num, shp1, shp2)
                print('Epoch {}, testing:\nLosses:\n'.format(i))
                print(' min:{},\n max:{}\n mean:{}:'.format(temin, temax, temean))
                if temean < best:
                    best = temean
                    best_since = 0
                    print('Best model so far.  Saving')
                    bname = 'best/{}_best'.format(dnames[num], temean)#_{:7.4e}_test_loss
                    model.save(bname)
                    print('Model saved')
                else:
                    best_since += 1
                if best_since >= best_since_limit:
                    print("Haven't improved mean since {}, gave {}.  Stopping".format(i-best_since_limit, best))
#                     model.load(bname)
#                     print(final_test(model, tel, optimizer, criterion, printevery))
                    break
                    
                
            scheduler.step()





    