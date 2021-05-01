import os
import pandas as pd

from utils import read_clip
from torch.utils.data import Dataset

class SSV2_Dataset(Dataset):
    """Something Something V2 Dataset"""
    
    def __init__(self, root, dstype, transform=None):
        """
        Args:
            root (string): Root directory (see directory structure)
            dstype (string): Type of dataset ('train' or 'test')
            transform (callable, optional): Optional transform to be applied on a sample.
        
        Directory Structure:
            root/
            ├── raw/
            │   ├── train/
            │   └── test/
            │       ├── 123/          # Class index
            │       │   ├── 456.webm  # Raw clip files
            │       │   └── ..
            │       └── ..
            ├── train_list.csv        # Clip id, label (class index)
            ├── test_list.csv         # Clip id, label (class index)
            └── classes.txt           # List of class names
        """
        assert dstype in ('train', 'test'), f"dstype must be 'train' or 'test', not '{dstype}'"
        
        self.root = root
        self.dstype = dstype
        self.transform = transform
        
        self._load_classes()
        self._load_clip_list()
    
    def _load_classes(self):
        with open(os.path.join(self.root, 'classes.txt'), 'r') as f:
            self.classes = [line.strip() for line in f]
            self.num_classes = len(self.classes)
    
    def _load_clip_list(self):
        self.clip_list = pd.read_csv(f'{self.dstype}_list.csv')
    
    def __len__(self):
        return len(self.clip_paths)
    
    def __getitem__(self, idx):
        clip_id, cls_idx = self.clip_list.iloc[idx]
        
        label = torch.zeros(self.num_classes)
        label[cls_idx] = 1.
        
        clip_path = os.path.join(self.root, self.dstype, cls_idx, clip_id)
        clip = read_clip(clip_path)
        
        if self.transform:
            clip = self.transform(clip)
        
        return clip, label