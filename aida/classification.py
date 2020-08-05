import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import BCELoss as BCE
from pathlib import Path
from copy import copy
import time
from extended import ExtendedModule
from attention import ConcatLinearTanhAttention, ConvolutionDotProdNormAttention
from torch.utils.data import Dataset, TensorDataset, DataLoader
# from allennlp.commands.elmo import ElmoEmbedder
from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.commands.elmo import ElmoEmbedder
from modularity_layers import ModularityLayerGRUAttn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

emb_loc = "/home/jq/software/triplet-all/embeddings"
elmo_emb_wt = "elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
elmo_cfg = "elmo_2x4096_512_2048cnn_2xhighway_options.json"



data_loc = '/home/jq/software/triplet-all/aclImdb/'



def get_elmo():
    weight_file = emb_loc+'/'+elmo_emb_wt
    options_file = emb_loc+'/'+elmo_cfg
    dropout = 0.25
    elmo = Elmo(options_file, weight_file, 1, dropout=dropout, do_layer_norm=False)
    return elmo

def get_sset(path_of, max_num = 2):
    label = None
    if path_of[-3:]=='neg':
        label = 0
    elif path_of[-3:]=='pos':
        label = 1
    if label is not None:
        pp = Path(path_of)
        docs = [d for d in pp.iterdir()]
        doc_list = []
        ml = 0
        for d in docs:
            if max_num > 0 and len(doc_list) >= max_num:
                break
            with open(d, 'r') as f:
                a = f.read()
                ml = max(ml, len(a))
                doc_list.append(a)
        return doc_list, [label]*len(doc_list), ml
    elif 'train' in path_of or 'test' in path_of:
        data, labs, ml1 = get_sset(path_of+'/pos')
        dat2, neg, ml2 = get_sset(path_of+'/pos')
        data.extend(dat2)
        labs.extend(neg)
        return data, labs, max(ml1, ml2)
    else:
        train, train_labs, ml1 = get_sset(path_of+'/train')
        test, test_labs, ml2 = get_sset(path_of+'/test')
        return train, train_labs, test, test_labs, max(ml1, ml2)




def get_data():
    return get_sset(data_loc)

#cl = time.clock()
# dd = get_data()
#MAX_LEN = get_data()[-1]
MAX_LEN = 100
# print(MAX_LEN)

# print(time.clock()-cl)

class ElmoDset(Dataset):
    def __init__(self, xs, ys, lens):
        assert xs.size()[0]==len(ys)
        assert len(lens) == len(ys)
        self.xs = xs
        self.ys = torch.tensor(ys, device=device)
        self.lens = lens
    
    def __len__(self):
        return len(self.ys)
    
    def __getitem__(self, i):
        mask = torch.ones(1, MAX_LEN, 2048, device = device)
        if self.lens[i] < MAX_LEN:
            mask[:, :self.lens[i], :] = 0
        
        return {'x':self.xs[i, :, :], 'y':self.ys[i], 'mask':mask}


# def get_longest(set_of_sentences):
#     MAX_LEN_ = 0
#     for s in set_of_sentences:

#         s = s.replace('.', ' ')
#         s = s.replace(',', " ")
#         s = s.replace("  ", " ")
#         s = s.replace("  ", " ")
#         slist = s.split(' ')
#         MAX_LEN_ = max(len(slist), MAX_LEN_)
#     return MAX_LEN_
    
# MAX_LEN = max(get_longest(dd[0]), get_longest(dd[2]))
# print(MAX_LEN)
# def process_sentences(elmo_embedder, set_of_sentences):
#     tlist = []
#     for s in set_of_sentences:
#         print(s)
#         s = s.replace('.', ' ')
#         s = s.replace(',', " ")
#         slist = s.split(' ')
#         print(slist)
#         print(len(slist))
#         append_to = [" "]*(MAX_LEN-len(slist))
#         if len(append_to) > 0:
#             slist.extend(append_to)
#         print(slist)
#         embs = elmo_embedder.embed_sentences(slist)
        
#         embs = list(embs)
#         print(embs[0].shape)
#         assert False
#         embs = emsb[2]
#         print(embs.shape)
#         assert embs.shape[-1]==MAX_LEN
#         tlist.append(embs)
#     return torch.tensor(tlist)
    
    



def create_embed_loaders():
    train, train_labs, test, test_labs, _ = get_data()
    ttr, tte = [], []
    for t in train:
        ttr.append(len(t))
    for t in test:
        tte.append(len(t))
    # train = train[:100]
    # test = test[:50]
    lentrain = len(train)
    data = copy(train)
    data.extend(test)
    elmo = get_elmo()
    # elmo = ElmoEmbedder()
    
    # te = process_sentences(elmo, test)
    # print(tr.size())
    # assert False
    da_ids = batch_to_ids(data)
    da = elmo(da_ids)['elmo_representations'][0]
    da = da.to(device)
    print(da.size())
    tr = da[:lentrain, :, :]
    te = da[lentrain:, :, :]
    trdata = ElmoDset(tr, train_labs, ttr)
    tedata = ElmoDset(te, test_labs, tte)
    trload = DataLoader(trdata, shuffle = True, batch_size = 2)
    teload = DataLoader(tedata, shuffle = True, batch_size = 2)
    return trload, teload
    


"""
Basically, the idea should be to use an attention type 
network to summarize the entire embedding, no matter the type, 
into a 1xhsize network. 

"""



class ClassifierFromElmo(ExtendedModule):
    """Some Information about ClassifierFromElmo"""
    def __init__(self, hidden_sz, attn = 0, filen=None):
        super(ClassifierFromElmo, self).__init__(filen=filen)
        if attn == 0:
            attentionmod = ConvolutionDotProdNormAttention(hidden_sz, MAX_LEN)
        else:
            attentionmod = ConcatLinearTanhAttention(hidden_sz, MAX_LEN)
        mods = [ModularityLayerGRUAttn(attentionmod)]
        mods.append(nn.Linear(hidden_sz*2, 1))
        mods.append(nn.Sigmoid())
        self.modules_ = nn.ModuleList(mods)
        self.initialize()

    def forward(self, x, mask):
        y = self.modules_[0](x, mask)
        
        y = self.modules_[1](y)
        y = self.modules_[2](y)
        return y

if __name__ == '__main__':
    
    classifier = ClassifierFromElmo(1024, attn = 1)
    data = torch.randn(MAX_LEN, 200, 1024).to(device)
    mask = torch.ones(MAX_LEN, 200, 2*1024).to(device)
    yy = classifier(data, mask)
    print(yy.size())
    classifier.save('classifiertry')
    
    newc = ClassifierFromElmo(1024, attn = 1, filen = 'classifiertry')
    zz = classifier(data, mask)
    ff = (yy-zz).sum()
    assert ff.abs() < 0.000001
    print('works')



    
    assert False
    cl = time.clock()
    trload, teload = create_embed_loaders()
    print('done in {}'.format(time.clock()-cl))
    classifier = ClassifierFromElmo(1024).cuda()
    for i in range(100):
        for j, b in enumerate(trload):
            x, y, mask = b['x'], b['y'], b['mask']
            mask = mask.squeeze(1)
            classifier.zero_grad()
            yhat = classifier(x, mask)
            loss = BCE(yhat, y)
            loss.backward()
            print(loss.item())
