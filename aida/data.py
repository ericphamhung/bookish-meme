import fasttext
from torch.utils.data import Dataset, TensorDataset, DataLoader


__allowed_embeds__ = ['fasttext']
__allowed_tasks__ = ['extract']
__allowed_embeds_forms__ = ['word'] #fasttext
#fasttext train vecs in ../wiki.en.zip

class TaskDataSet(Dataset):
    def __init__(self, embedding, typeembed, task, source):
        assert embedding in __allowed_embeds__
        assert task in __allowed_tasks__
        assert ty in __allowed_embeds_forms__
        # basically get length of sentence etc
    
    def __len__(self):
        return len(self.ys)
    
    def __getitem__(self, i):
        mask = torch.ones(1, MAX_LEN, 2048, device = device)
        if self.lens[i] < MAX_LEN:
            mask[:, :self.lens[i], :] = 0
        
        return {'x':self.xs[i, :, :], 'y':self.ys[i], 'mask':mask}