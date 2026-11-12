import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common_prep import (
    load_preprocessed, load_preprocessed_segmentation, apply_412,
    normalize_01, structural,
)


def amor_input(volume01: np.ndarray, seg: np.ndarray | None):
    volume01 = np.asarray(volume01, dtype=np.float32)
    if seg is not None:
        if volume01.shape != seg.shape:
            raise ValueError('CT and segmentation must share the same input grid')
        volume01 = normalize_01(volume01 + np.float32(0.75) * structural(seg))
    return torch.from_numpy(np.ascontiguousarray(volume01)).unsqueeze(0)


class ShoulderDataset412(Dataset):
    def __init__(self, items, raw=False):
        self.items = items
        self.raw = raw

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        seg_path = it.get('seg')
        if self.raw:
            vol, seg = apply_412(
                it['ct'], seg_path, side=it.get('side'),
                crop_box=it.get('crop_box'), return_seg=True,
            )
        else:
            vol = load_preprocessed(it['ct'])
            seg = load_preprocessed_segmentation(it['ct'], seg_path) if seg_path is not None else None
        out = {'input': amor_input(vol, seg), 'id': it['id']}
        if it.get('points') is not None:
            out['points'] = torch.tensor(it['points'], dtype=torch.float32)
        return out
