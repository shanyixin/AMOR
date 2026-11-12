import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common_prep import load_preprocessed, apply_412

STAGE1_SIZE = (256, 256, 256)
STAGE2_SIZE = (256, 256, 256)
FOCUS_RADIUS = 15
FOCUS_RADIUS_STAGE2 = 5


class MaxillofacialDataset412(Dataset):
    def __init__(self, items, raw=False):
        self.items = items
        self.raw = raw

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        vol01 = apply_412(it['ct'], side=it.get('side'), crop_box=it.get('crop_box')) if self.raw else load_preprocessed(it['ct'])
        shape = np.asarray(vol01.shape, dtype=np.int64)
        predicted_box = it.get('predicted_box')
        if predicted_box is None:
            lower = np.zeros(3, dtype=np.int64)
            upper = shape
        else:
            box = np.asarray(predicted_box, dtype=np.float32)
            if box.shape != (6,) or not np.isfinite(box).all():
                raise ValueError('predicted_box must contain six finite normalized coordinates')
            if np.any(box < 0) or np.any(box > 1):
                raise ValueError('predicted_box coordinates must lie in [0, 1]')
            lower = (box[:3] * shape).astype(np.int64)
            upper = (box[3:] * shape).astype(np.int64)
            if np.any(upper <= lower):
                raise ValueError('predicted_box must define a nonempty region')
        z0, y0, x0 = (int(v) for v in lower)
        z1, y1, x1 = (int(v) for v in upper)
        roi = (z0, z1, y0, y1, x0, x1)
        crop = vol01[z0:z1, y0:y1, x0:x1]
        t = torch.from_numpy(crop)[None, None]
        t = torch.nn.functional.interpolate(
            t, size=STAGE1_SIZE, mode='trilinear', align_corners=False,
        )
        stage1 = t[0, 0].numpy().astype(np.float32)
        return {'stage1': torch.from_numpy(stage1).unsqueeze(0), 'roi': roi, 'id': it['id']}
