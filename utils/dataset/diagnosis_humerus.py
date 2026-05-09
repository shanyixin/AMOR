import os
import json
import torch
import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset
from skimage import measure
import shutil
import random
import scipy.ndimage as ndimage
import re

def natural_key(name):
    # Internal processing step.
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', name)]

def random_flip(image):
    axis = random.choice([0, 1, 2])
    return np.flip(image, axis=axis).copy()

def random_intensity_shift(image, max_shift=0.1):
    shift = random.uniform(-max_shift, max_shift)
    return np.clip(image + shift, 0.0, 1.0)

def random_gaussian_noise(image, std=0.05):
    noise = np.random.normal(0, std, size=image.shape)
    return np.clip(image + noise, 0.0, 1.0)

def random_elastic_deformation(image, alpha=1, sigma=8):
    for z in range(image.shape[0]):
        dx = ndimage.gaussian_filter((np.random.rand(*image[z].shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = ndimage.gaussian_filter((np.random.rand(*image[z].shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        x, y = np.meshgrid(np.arange(image[z].shape[1]), np.arange(image[z].shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        image[z] = ndimage.map_coordinates(image[z], indices, order=1, mode='reflect').reshape(image[z].shape)
    return image

def random_gamma(image, gamma_range=(0.7, 1.5)):
    gamma = random.uniform(*gamma_range)
    image = np.clip(image, 0, 1)
    return image ** gamma

def random_contrast(image, factor_range=(0.8, 1.2)):
    factor = random.uniform(*factor_range)
    mean = np.mean(image)
    return np.clip((image - mean) * factor + mean, 0, 1)

def apply_augments(image, transform_set):
    basic_ops = ["flip", "shift", "contrast"]
    advanced_ops = ["noise", "gamma", "elastic"]
    random.shuffle(basic_ops)
    random.shuffle(advanced_ops)

    used = 0

    # Internal processing step.
    for op in basic_ops:
        if op in transform_set:
            if op == "flip" and random.random() < 0.7:
                image = random_flip(image)
                used += 1
            elif op == "shift" and random.random() < 0.6:
                image = random_intensity_shift(image)
                used += 1
            elif op == "contrast" and random.random() < 0.6:
                image = random_contrast(image)
                used += 1
        if used >= 4:
            return image

    # Internal processing step.
    for op in advanced_ops:
        if op in transform_set:
            if op == "noise" and random.random() < 0.4:
                image = random_gaussian_noise(image)
                used += 1
            elif op == "gamma" and random.random() < 0.4:
                image = random_gamma(image)
                used += 1
            elif op == "elastic" and random.random() < 0.3:
                image = random_elastic_deformation(image)
                used += 1
        if used >= 4:
            break

    # Internal processing step.
    if used == 0:
        fallback_ops = list(transform_set & set(basic_ops + advanced_ops))
        if fallback_ops:
            op = random.choice(fallback_ops)
            if op == "flip":
                image = random_flip(image)
            elif op == "shift":
                image = random_intensity_shift(image)
            elif op == "contrast":
                image = random_contrast(image)
            elif op == "noise":
                image = random_gaussian_noise(image)
            elif op == "gamma":
                image = random_gamma(image)
            elif op == "elastic":
                image = random_elastic_deformation(image)

    return image

class Humerus_diagnosis(Dataset):
    def __init__(self, img_size, train_path, cache_path, dataset_type=1, transform=True, rebuild_cache=False):
        """
                    
        """
        self.type = dataset_type
        data_type = ['train', 'valid', 'test'][dataset_type]
        self.img_size = img_size[0]
        if data_type == 'train':
            if transform:
                self.transform = {"flip", "noise", "shift", "elastic", "gamma", "contrast"}
            elif isinstance(transform, (set, list)):
                self.transform = set(transform)
            else:
                self.transform = None
        else:
            self.transform = None

        # Internal processing step.
        self.ImagePath = os.path.join(train_path, "img")
        self.SegPath = os.path.join(train_path, "seg")
        self.TypePath = os.path.join(train_path, "type")
        self.DiagnosisPath = os.path.join(train_path, "diagnosis_zoom")
        self.cache_dir = os.path.join(cache_path, data_type)

        # Internal processing step.
        self.img_paths = sorted([os.path.join(self.ImagePath, f) for f in os.listdir(self.ImagePath) if f.endswith(".nrrd")])
        self.seg_paths = sorted([os.path.join(self.SegPath, f) for f in os.listdir(self.SegPath) if f.endswith(".nrrd")])
        self.type_paths = sorted([os.path.join(self.TypePath, f) for f in os.listdir(self.TypePath) if f.endswith(".json")])
        self.diagnosis_paths = sorted([os.path.join(self.DiagnosisPath, f) for f in os.listdir(self.DiagnosisPath) if f.endswith(".nrrd")])

        self.sample_origin = {}
        self.idx_to_name = {}  # Internal processing step.

        # Internal processing step.
        if rebuild_cache:
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
            self.preprocess_and_cache()
        elif not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
            self.preprocess_and_cache()
        else:
            # Internal processing step.
            self.sample_origin = {}
            self.idx_to_name = {}  # Internal processing step.
            for fname in os.listdir(self.cache_dir):
                if fname.endswith('_img.npy'):
                    name = fname.replace('_img.npy', '')  # Internal processing step.
                    idx_str = name.replace('sample_', '').split('_')[0]  # e.g., '3'
                    if idx_str.isdigit():
                        self.sample_origin[name] = int(idx_str)
                        self.idx_to_name[name] = idx_str  # Internal processing step.


        self.index_list = sorted(list(self.sample_origin.keys()), key=natural_key)
        
    def preprocess_and_cache(self):
        attention_weight = [0.6, 2]
        for idx in range(len(self.img_paths)):
            img = self.load_nrrd(self.img_paths[idx])
            seg = self.fliter_seg(self.load_nrrd(self.seg_paths[idx]))
            diagnosis = self.load_nrrd(self.diagnosis_paths[idx])
            norm_img = self.normalize_image(img, seg, diagnosis, attention_weight).numpy()
            type = self.load_type(self.type_paths[idx])
            
            basename = os.path.splitext(os.path.basename(self.img_paths[idx]))[0]
            basename = basename.split('.')[0]  # Internal processing step.
            name = f"sample_{idx}"
            np.save(os.path.join(self.cache_dir, f"sample_{idx}_img.npy"), norm_img)
            np.save(os.path.join(self.cache_dir, f"sample_{idx}_type.npy"), type)
            self.sample_origin[name] = idx
            self.idx_to_name[idx] = basename  # Internal processing step.

            if idx == 0:
                norm_img_sitk = sitk.GetImageFromArray(norm_img.astype(np.float32))
                vis_dir = os.path.join(os.path.dirname(self.cache_dir), "vis")
                os.makedirs(vis_dir, exist_ok=True)
                sitk.WriteImage(norm_img_sitk, os.path.join(vis_dir, "norm_img.nrrd"))
            
            if self.transform:
                num_aug = random.randint(5, 8)
                for j in range(num_aug):
                    aug_img = apply_augments(norm_img.copy(), self.transform)
                    aug_name = f"sample_{idx}_aug{j}"
                    np.save(os.path.join(self.cache_dir, f"{aug_name}_img.npy"), aug_img)
                    np.save(os.path.join(self.cache_dir, f"{aug_name}_type.npy"), type)
                    self.sample_origin[aug_name] = idx
                
    def __len__(self):
        return len(self.index_list)
    
    def __getitem__(self, idx):
        name = self.index_list[idx]
        img = np.load(os.path.join(self.cache_dir, f"{name}_img.npy"), mmap_mode='r')
        label = np.load(os.path.join(self.cache_dir, f"{name}_type.npy"), allow_pickle=True)
        img_tensor = torch.from_numpy(img.copy()).unsqueeze(0).float()
        label_tensor = torch.from_numpy(label.copy()).unsqueeze(0).float()
        return img_tensor, label_tensor

    def normalize_image(self, img, seg, diagnosis, attention_weight):
        img_tensor = torch.from_numpy(img).float()
        seg_tensor = torch.from_numpy(seg).float()
        zoom_tensor = torch.from_numpy(diagnosis).float()
        img_tensor = img_tensor + attention_weight[0] * seg_tensor + attention_weight[1] * zoom_tensor
        img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min())
        return img_tensor

    def load_nrrd(self, filepath):
        itk_img = sitk.ReadImage(filepath)
        return sitk.GetArrayFromImage(itk_img)

    def load_type(self, filepath):
        with open(filepath, 'r') as f:
            points_data = json.load(f)
        return points_data['type']

    def fliter_seg(self, seg_array, type=2):
        return (seg_array == type).astype(np.float32)

    def clear_cache(self):
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)

    def get_sample_name(self, idx):
        name = self.index_list[idx]
        return self.idx_to_name.get(idx, name)
    

