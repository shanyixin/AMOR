import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common_prep import load_preprocessed, apply_412


class HypergraphDataset412(Dataset):
    def __init__(self, items, raw=False):
        self.items = items
        self.raw = raw

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        vol = apply_412(it['ct'], it.get('seg'), side=it.get('side'), crop_box=it.get('crop_box')) if self.raw else load_preprocessed(it['ct'])
        arr = vol.astype(np.float32)
        for _ in range(2):
            std = arr.std()
            arr = (arr - arr.mean()) / (std if std >= 1e-6 else 1.0)
        return {'input': torch.from_numpy(arr).unsqueeze(0), 'id': it['id']}
