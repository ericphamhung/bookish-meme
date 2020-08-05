from torch.utils.data.dataloader import default_collate
from fastai.torch_core import *
from fastai.basic_data import DataBunch
from torch.utils.data import DataLoader, Dataset
from simpler_grammar import SimpleGrammar
from definitions import *
import re
import json
from copy import copy
from functools import reduce
dataf = '/home/jq/software/triplet-all/spider_train_subset.json'

class AugmentedDataset:#(Dataset):
    def __init__(self, jsonfile, grammar_file):
        self.grammar = SimpleGrammar(grammar_file)
        self.grammar_terminals = self.grammar.get_terminal_toks()
        with open(jsonfile, 'r') as f:
            self.data = json.loads(f.read())


    def augment_data(self, test_string):
        to_learn = self.grammar.learn_

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
    assert 'movie' in spider_tables
    assert 'Director' in spider_tables['movie'] or 'Director' in spider_tables['movie_added']
    a = AugmentedDataset(dataf, 'sql_simple_transition.bnf')
    a.test()
    # with open(dataf, 'r') as f:
    #     data = json.loads(f.read())
    # for k in data.keys():
    #     print(k)
    #     for kk in data[k].keys():
    #         print(kk)
    #         print(data[k][kk])
    #     break
