from torch.utils.data.dataloader import default_collate
from fastai.torch_core import *
from fastai.basic_data import DataBunch
from torch.utils.data import DataLoader, Dataset
from grammar import SimpleGrammar
from definitions import *
import re
import json
from copy import copy
from functools import reduce
dataf = '/home/jq/software/triplet-all/spider_train_subset.json'

#from A2 notebook
class SeparatedDataset(Dataset):
    # Class to load dataset    
    def __init__(self, dpath, test_size = 0.1, seed = 123432):
        datastr = dpath+'/spider_train_subset.json'
        data = pd.read_json(datastr).T
        self.train, self.test = train_test_split(data, test_size=test_size, random_state = seed)
        self.items = ['like', 'gb', 'ob', 'hv','lm','asc','dsc','whr']
        self.curr_item = self.items[0]
        self.test_ = False
    
    
    def selector(self, index):
        assert index < len(self.items)
        self.curr_item = self.items[index]
    def set_train(self):
        self.test_ = False
    def set_test(self):
        self.test_ = True
    
    def __len__(self):
        if self.test_:
            return self.test.shape[0]
        else:
            return self.train.shape[0]
    
    def __getitem__(self, index):
        if self.test_:
            dset = self.test
        else:
            dset = self.train
        
        d = dset.iloc[index]
        x = d['eng']
        if self.curr_item == 'like':
            y = 1*('LIKE' in d['sql'].upper())
        elif self.curr_item == 'gb':
            y = 1*('GROUP BY' in d['sql'].upper())
        elif self.curr_item == 'ob':
            y = 1*('ORDER BY' in d['sql'].upper())
        elif self.curr_item == 'hv':
            y = 1*('HAVING' in d['sql'].upper())
        elif self.curr_item == 'lm':
            y = 1*('LIMIT' in d['sql'].upper())
        elif self.curr_item == 'asc':
            y = 1*('ASC' in d['sql'].upper())
        elif self.curr_item == 'dsc':
            y = 1*('DESC' in d['sql'].upper())
        elif self.curr_item == 'whr':
            y = 1*('WHERE' in d['sql'].upper())

        return x, y

class AugmentedDataset:#(Dataset):
    def __init__(self, jsonfile, grammar_file):
        self.grammar = SimpleGrammar(grammar_file)
        self.grammar_terminals = self.grammar.get_terminal_toks()
        with open(jsonfile, 'r') as f:
            self.data = json.loads(f.read())


    def augment_data(self, test_string):
        to_learn = self.grammar.learn_

    def get_likes_nonlikes(self):
        l_counter = 0
        nl_counter = 0
        kk = [k for k in self.data.keys()]
        for k in kk:
            string = self.data[k]['sql']
            if 'LIKE' in string:
                l_counter += 1
            else:
                nl_counter +=1
        print('{} likes, {} non likes'.format(l_counter, nl_counter))

    def test(self):
        counter = 0
        not_resolved = 0
        kk = [k for k in self.data.keys()]
        for k in kk:
            string = self.data[k]['sql']
            val, reas = self.grammar.check_string_tokens(string, verbose = True)
            if not val:
                counter += 1
                if reas == 'res':
                    not_resolved += 1


        print('{} out of {} are errors, {} are with resolution'.format(counter, len(kk), not_resolved))


class DataBunchWithAppend(DataBunch):
    "Bind `train_dl`,`valid_dl` and `test_dl` in a data object."

    def __init__(self, train_dl:DataLoader, valid_dl:DataLoader, fix_dl:DataLoader=None, test_dl:Optional[DataLoader]=None,
                 device:torch.device=None, dl_tfms:Optional[Collection[Callable]]=None, path:PathOrStr='.',
                 collate_fn:Callable=data_collate, no_check:bool=False):
        super().__init__(self, train_dl, valid_dl, fix_dl, test_dl, device, dl_tfms, path, collate_fn, no_check)

    def append_to(self, ys):
        self.ys = ys



class AugmentedDataLoader(DataLoader):
    def __init__(self, dataset, filen, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, collate_fn=default_collate, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):
        super().__init__(dataset, batch_size, shuffle, sampler, batch_sampler,
                num_workers, collate_fn, pin_memory, drop_last, timeout, worker_init_fn)

        self.grammar = SimpleGrammar(filen)
        self.database_calls

    def get_choices_key(self, key):
        return self.grammar.gr[key]['items']

    def get_values_terminal(self, terminal):
        if terminal in resolve_dict:
            pass
        else:
            lst = self.grammar.gr[terminal]
            assert len(lst) == 1
            return lst[0]

    def is_terminal_on_path(self, tok, terminal):
        return self.grammar.from_terminal_to_token(tok, terminal)

if __name__=='__main__':
    a = AugmentedDataset(dataf, 'sql_simple_transition_2.bnf')
    a.get_likes_nonlikes()
    # with open(dataf, 'r') as f:
    #     data = json.loads(f.read())
    # for k in data.keys():
    #     print(k)
    #     for kk in data[k].keys():
    #         print(kk)
    #         print(data[k][kk])
    #     break
