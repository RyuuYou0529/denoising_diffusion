import numpy as np
from functools import partial

import torch
from torch.utils.data import Dataset, DataLoader

class npz_dataset(Dataset):
    def __init__(self, path, type='train', split=0.9):
        file = np.load(path)
        index = int(np.ceil(len(file['X']) * split))
        if type == 'train':
            self.lr_data = torch.from_numpy(file['X'][0:index])
            self.hr_data = torch.from_numpy(file['Y'][0:index])
        elif type == 'val':
            self.lr_data = torch.from_numpy(file['X'][index:])
            self.hr_data = torch.from_numpy(file['Y'][index:])

    def __len__(self):
        return self.lr_data.size(0)
    
    def __getitem__(self, index):
        return self.lr_data[index], self.hr_data[index]

def get_Dataloader(path, train_batch_size=32, val_batch_size=16, split=0.9):
    train_dataset = npz_dataset(path=path, type='train', split=split)
    train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True)

    val_dataset = npz_dataset(path=path, type='val', split=split)
    val_loader = DataLoader(dataset=val_dataset, batch_size=val_batch_size, shuffle=True)
    return train_loader, val_loader