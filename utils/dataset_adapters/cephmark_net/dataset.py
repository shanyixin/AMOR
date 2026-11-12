import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common_prep import load_preprocessed, apply_412

SLICE_AXIS = 0
N_SLICES = 256


class CephMarkDataset412(Dataset):


    def __init__(self, items, raw=False):
        self.items = items
        self.raw = raw

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        vol = apply_412(it['ct'], side=it.get('side'), crop_box=it.get('crop_box')) if self.raw else load_preprocessed(it['ct'])
        slices = np.stack([vol[z] for z in range(N_SLICES)], axis=0)
        return {'slices': torch.from_numpy(slices.astype(np.float32)).unsqueeze(1),
                'slice_index': torch.arange(N_SLICES), 'id': it['id']}
