import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common_prep import load_preprocessed, apply_412


class HourglassDataset412(Dataset):
    def __init__(self, items, raw=False):
        self.items = items
        self.raw = raw

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        vol = apply_412(it['ct'], it.get('seg'), side=it.get('side'), crop_box=it.get('crop_box')) if self.raw else load_preprocessed(it['ct'])
        t = torch.from_numpy(vol).unsqueeze(0)
        t = (t - t.mean()) / (t.std() + 1e-5)
        return {'input': t, 'id': it['id']}
