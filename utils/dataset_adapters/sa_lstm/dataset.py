import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common_prep import load_preprocessed, apply_412

COARSE_SIZE = (256, 256, 256)
class SALSTMDataset412(Dataset):
    def __init__(self, items, raw=False):
        self.items = items
        self.raw = raw

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        vol = apply_412(it['ct'], it.get('seg'), side=it.get('side'), crop_box=it.get('crop_box')) if self.raw else load_preprocessed(it['ct'])
        full = torch.from_numpy(vol).unsqueeze(0)
        coarse = torch.nn.functional.interpolate(full[None], size=COARSE_SIZE,
                                                 mode='trilinear', align_corners=False)[0]
        return {'image_full': full, 'image_coarse': coarse,
                'shape_zyx': torch.tensor(vol.shape), 'id': it['id']}
