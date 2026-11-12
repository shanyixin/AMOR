import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common_prep import load_preprocessed, apply_412

INPUT_SIZE = (256, 256, 256)
class DCeLRDataset412(Dataset):
    def __init__(self, items, raw=False):
        self.items = items
        self.raw = raw

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        vol = apply_412(it['ct'], it.get('seg'), side=it.get('side'), crop_box=it.get('crop_box')) if self.raw else load_preprocessed(it['ct'])
        t = torch.from_numpy(vol)[None, None]
        t = torch.nn.functional.interpolate(t, size=INPUT_SIZE, mode='trilinear', align_corners=False)
        x = t[0, 0].numpy().astype(np.float32)
        std = x.std()
        x = (x - x.mean()) / (std + 1e-8) if std > 0 else x - x.mean()
        return {'input': torch.from_numpy(x).unsqueeze(0), 'id': it['id']}
