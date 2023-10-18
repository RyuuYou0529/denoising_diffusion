import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

import numpy as np
from PIL import Image

from functools import partial
from pathlib import Path


# ====================
# npz dataset
# ====================
class npz_dataset(Dataset):
    def __init__(
            self, 
            path, 
            npz_file_name,
            normalize_mode: str='min_max',
            transform = None
    ):
        file = np.load(path)
        self.data = torch.from_numpy(file[npz_file_name])
        self.len = self.data.shape[0]

        self.normalize = lambda t:(t-t.mean())/(t.std())
        self.scale = lambda t:(t - t.min())/(t.max()-t.min())
        self.normalize_mode = normalize_mode

        assert normalize_mode in ['z_score', 'min_max', 'no_normalize', None], 'Invalid args: "normalize_mode"'
        if normalize_mode == 'z_score':
            self.normalize = lambda t:(t-t.mean())/(t.std())
        elif normalize_mode == 'min_max':
            self.normalize = lambda t:(t - t.min())/(t.max()-t.min())
        else:
            self.normalize = lambda t:t

        self.transform = transform

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        item = self.data[index]
        if self.transform is not None:
            item = self.transform(item)
        item = self.normalize(item)
        return item

# ====================
# folder dataset
# ====================
class folder_dataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts = ['jpg', 'jpeg', 'png', 'tiff']
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.transform = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)