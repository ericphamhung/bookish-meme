import torch.nn as nn
import torch.nn.functional as F
import torch
from collections import OrderedDict



class Classifier(nn.Module):

    def __init__(self, emb_sz, n_classes):
        super(Classifier, self).__init__()
        self.n_classes = n_classes
        self.fc1 = nn.Linear(emb_sz, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_classes)
    
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x))#, dim = -1
        return x

    def save(self, fname):
        torch.save(self.state_dict(), fname+'.pyt')
    
    def load(self, fname):
        self.load_state_dict(torch.load(fname+'.pyt'))


class MultiLabelClassifier(nn.Module):

    def __init__(self, emb_sz, n_labels):
        super(Classifier, self).__init__()
        self.n_classes = n_classes
        self.fc1 = nn.Linear(emb_sz, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_classes)
        self.sigmoid = nn.Sigmoid()
    
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))#, dim = -1
        return x

    def save(self, fname):
        torch.save(self.state_dict(), fname+'.pyt')
    
    def load(self, fname):
        self.load_state_dict(torch.load(fname+'.pyt'))


class MultiLabelColumnGivenDBAndTableClassifier(nn.Module):

    def __init__(self, emb_sz, db_dict, n_labels):
        super().__init__()
        self.module_dct = {}
        self.n_labels = n_labels
        lst = []
        cnt = 0
        #self.sigmoid = nn.Sigmoid()
        for db in db_dict.keys():
            self.module_dct[db] = {}
            for table in db_dict[db].keys():
                fc1 = nn.Linear(emb_sz, 512)
                fc2 = nn.Linear(512, 256)
                fc3 = nn.Linear(256, self.n_labels)
                self.module_dct[db][table] = cnt
                lst += [nn.Sequential(fc1, fc2, fc3)]#, self.sigmoid
                cnt += 1
        self.modules = nn.ModuleList(lst)
        
        
    
    
    def forward(self, x, db, table):
        x = self.modules[self.module_dct[db][table]](x)
        return x

    def save(self, fname):
        torch.save(self.state_dict(), fname+'.pyt')
    
    def load(self, fname):
        self.load_state_dict(torch.load(fname+'.pyt'))

def rounder(to_int = True):

    if to_int:
        def rr(x):
            return torch.round(x)*1.0
    else:
        def rr(x):
            return x
    return rr

class Regressor(nn.Module):

    def __init__(self, emb_sz, to_int = True):
        super(Regressor, self).__init__()
        self.fc1 = nn.Linear(emb_sz, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.rr = rounder()
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.rr(x)
        return x

    def save(self, fname):
        torch.save(self.state_dict(), fname+'.pyt')
    
    def load(self, fname):
        self.load_state_dict(torch.load(fname+'.pyt'))