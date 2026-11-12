import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common_prep import load_preprocessed, apply_412


class NNMambaDataset(Dataset):
    def __init__(self, items, raw=False):
        self.items = items
        self.raw = raw

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        vol = apply_412(it['ct'], side=it.get('side'), crop_box=it.get('crop_box')) if self.raw else load_preprocessed(it['ct'])
        out = {
            'input': torch.from_numpy(vol.astype(np.float32, copy=False)).unsqueeze(0),
            'id': it['id'],
        }
        if it.get('points') is not None:
            out['points'] = torch.tensor(it['points'], dtype=torch.float32)
        return out
