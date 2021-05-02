import os
import torch
import numpy as np
import pandas as pd

from .utils import read_clip
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    """Custom Dataset"""
    
    def __init__(self, root, subdir, dstype, ext, transform=None):
        """
        Args:
            root (string): Root directory
            subdir (string): Root sub-directory
            dstype (string): Dataset type ('train' or 'test')
            ext (string): Data files file extension
            transform (callable, optional): Optional transform to be applied on a sample
        
        Directory Structure:
            root/                     # Root directory
            ├── subdir/               # Sub-directory
            │   └── dstype/           # Dataset type
            │       └── 123/          # Class index
            │           └── 456.ext   # Data files
            ├── train_list.csv        # Sample id, class index
            ├── test_list.csv         # Sample id, class index
            └── classes.txt           # List of class names
        """
        assert dstype in ('train', 'test'), f"dstype must be 'train' or 'test', not '{dstype}'"
        
        self.root = root
        self.subdir = subdir
        self.dstype = dstype
        self.ext = ext
        self.transform = transform
        
        self._load_classes()
        self._load_sample_list()
    
    def _load_classes(self):
        with open(os.path.join(self.root, 'classes.txt'), 'r') as f:
            self._classes = [line.strip() for line in f]
            self._num_classes = len(self._classes)
    
    def _load_sample_list(self):
        list_path = os.path.join(self.root, f'{self.dstype}_list.csv')
        self._sample_list = pd.read_csv(list_path)
        self._num_samples = len(self._sample_list)
    
    def get_info(self, idx):
        return tuple(self._sample_list.iloc[idx])
    
    def _get_sample(self, path):
        raise NotImplementedError
    
    def _get_label(self, cls_idx):
        label = torch.zeros(self._num_samples)
        label[cls_idx] = 1.
        return label
    
    def __len__(self):
        return self._num_samples
    
    def __getitem__(self, idx):
        sample_id, cls_idx = self.get_info(idx)
        name = str(sample_id) + os.extsep + self.ext
        path = os.path.join(self.root, self.subdir, self.dstype, str(cls_idx), name)
        
        sample = self._get_sample(path)
        label = self._get_label(cls_idx)
        return sample, label

class SSV2_Dataset(CustomDataset):
    """Something Something V2 Dataset"""

    def __init__(self, root, subdir, dstype, transform=None):
        super().__init__(root, subdir, dstype, 'webm', transform)
    
    def _get_sample(self, path):
        clip = read_clip(path)
        if self.transform is not None:
            clip = self.transform(clip)
        return clip

class SSV2_ProcessedDataset(CustomDataset):
    """Something Something V2 Processed Dataset"""
    
    def __init__(self, root, subdir, dstype):
        super.__init__(root, subdir, dstype, 'npy')
    
    def _get_sample(self, path):
        return torch.from_numpy(np.load(path))