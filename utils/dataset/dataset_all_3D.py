import os
import json
import torch
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter
from skimage import measure
import shutil
from scipy.ndimage import binary_erosion, distance_transform_edt, binary_dilation

class Shoulder_Joint(Dataset):
    def __init__(self, img_size, train_path, cache_path, sigma=3, dataset_type=0, transform=True, rebuild_cache=False):
        """
        :param img_size:         (512, 512, 512)
        :param train_path:           img/ seg/   scapula/ 
        :param cache_path:                 
        :param transform:         
        :param sigma:         
        :param dataset_type:       0: train 1: valid 2: test 
        :param rebuild_cache:         
        """
        self.type = dataset_type
        data_type = ['train', 'valid', 'test'][dataset_type]

        self.img_size = img_size[0]
        self.patch_size = 128
        self.stride = 64 if dataset_type == 0 else 128   # Internal processing step.
        self.transform = transform
        self.sigma = sigma
        self.pacth_index = []
        self.seg_thresh = 0.01

        # Internal processing step.
        self.ImagePath = os.path.join(train_path, "img")
        self.SegPath = os.path.join(train_path, "seg")
        self.ScapulaPath = os.path.join(train_path, "scapula")
        self.cache_dir = os.path.join(cache_path, data_type)

        # Internal processing step.
        self.scapula_fixed_labels = ["ss", "sm", "si", "AA", "AP", "C"]
        self.img_paths = sorted([os.path.join(self.ImagePath, f) for f in os.listdir(self.ImagePath) if f.endswith(".nrrd")])
        self.seg_paths = sorted([os.path.join(self.SegPath, f) for f in os.listdir(self.SegPath) if f.endswith(".nrrd")])
        self.scapula_paths = sorted([os.path.join(self.ScapulaPath, f) for f in os.listdir(self.ScapulaPath) if f.endswith(".json")])

        # Internal processing step.
        if rebuild_cache:
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
            self.preprocess_and_cache()
        else:
            if not os.path.exists(self.cache_dir) or not os.listdir(self.cache_dir):
                os.makedirs(self.cache_dir, exist_ok=True)
                self.preprocess_and_cache()
            else:
                self._load_cached_index()

    def preprocess_and_cache(self):
        attention_weight = 0.75
        patch_size = self.patch_size
        stride = self.stride
        for idx in range(len(self.img_paths)):
            img = self.load_nrrd(self.img_paths[idx])
            seg = self.fliter_seg(self.load_nrrd(self.seg_paths[idx]))
            norm_img = self.normalize_image(img, seg, attention_weight).numpy()
            scapula_heatmap, scapula_points = self.create_heatmaps(self.scapula_paths[idx], img.shape)

            D, H, W = img.shape
            patch_id = 0
            for z in range(0, D - patch_size + 1, stride):
                for y in range(0, H - patch_size + 1, stride):
                    for x in range(0, W - patch_size + 1, stride):
                        patch_img = norm_img[z:z+patch_size, y:y+patch_size, x:x+patch_size]
                        patch_seg = seg[z:z+patch_size, y:y+patch_size, x:x+patch_size]
                        patch_heatmap = scapula_heatmap[:, z:z+patch_size, y:y+patch_size, x:x+patch_size]

                        # Internal processing step.
                        patch_id += 1
                        structure_ratio = (patch_seg > 0).sum() / (patch_size ** 3)
                        if structure_ratio < self.seg_thresh:
                            continue
                        
                        patch_name = f"sample_{idx}_patch_{patch_id}"
                        patch_boundary = self.create_boundary_map(patch_seg)
                        np.save(os.path.join(self.cache_dir, f"{patch_name}_img.npy"), patch_img)
                        np.save(os.path.join(self.cache_dir, f"{patch_name}_seg.npy"), patch_seg)
                        np.save(os.path.join(self.cache_dir, f"{patch_name}_scapula_heatmap.npy"), patch_heatmap)
                        np.save(os.path.join(self.cache_dir, f"{patch_name}_scapula_points.npy"), scapula_points)
                        np.save(os.path.join(self.cache_dir, f"{patch_name}_boundary.npy"), patch_boundary)
                        self.pacth_index.append((idx, patch_id))

    def __len__(self):
        return len(self.pacth_index)

    def __getitem__(self, idx):
        sample_idx, patch_id = self.pacth_index[idx]
        base_name = f"sample_{sample_idx}_patch_{patch_id}"
        img = np.load(os.path.join(self.cache_dir, f"{base_name}_img.npy"), mmap_mode='r')
        heatmap = np.load(os.path.join(self.cache_dir, f"{base_name}_scapula_heatmap.npy"), mmap_mode='r')
        points = np.load(os.path.join(self.cache_dir, f"{base_name}_scapula_points.npy"), allow_pickle=True)
        boundary = np.load(os.path.join(self.cache_dir, f"{base_name}_boundary.npy"), mmap_mode='r')

        img_tensor = torch.from_numpy(img.copy()).unsqueeze(0).float()
        heatmap_tensor = torch.from_numpy(heatmap.copy()).float()
        boundary_tensor = torch.from_numpy(boundary.copy()).unsqueeze(0).float()
        return img_tensor, heatmap_tensor, points, boundary_tensor

    def normalize_image(self, img, seg, attention_weight):
        """          """
        img_tensor = torch.from_numpy(img).float()
        seg_tensor = torch.from_numpy(seg).float()
        img_tensor = img_tensor + attention_weight * seg_tensor
        img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min())
        
        # norm_img = img_tensor.numpy()
        # sitk_img = sitk.GetImageFromArray(norm_img)
        # Internal processing step.
        # Internal processing step.
        # output_file = "./normalized_image.nrrd"
        # sitk.WriteImage(sitk_img, output_file)
        
        return img_tensor
    
    def load_nrrd(self, filepath):
        """   NRRD 3D   """
        itk_img = sitk.ReadImage(filepath)
        img_array = sitk.GetArrayFromImage(itk_img)  # (D, H, W)
        return img_array

    def create_heatmaps(self, json_path, img_shape):
        """   6                """
        with open(json_path, 'r') as f:
            points_data = json.load(f)

        # Internal processing step.
        label_dict = {p['label']: p['position'] for p in points_data}

        # Internal processing step.
        heatmap = np.zeros((len(self.scapula_fixed_labels), *img_shape), dtype=np.float32)
        points = []

        for i, label in enumerate(self.scapula_fixed_labels):
            if label in label_dict:
                x, y, z = label_dict[label]
                x, y, z = round(x), round(y), round(z)
                points.append((x, y, z))

                if 0 <= x < img_shape[2] and 0 <= y < img_shape[1] and 0 <= z < img_shape[0]:
                    heatmap[i, z, y, x] = 1  # Internal processing step.

        # Internal processing step.
        for i in range(len(self.scapula_fixed_labels)):
            heatmap[i] = gaussian_filter(heatmap[i], sigma=self.sigma)
            heatmap[i] /= heatmap[i].max()
            # Internal processing step.
            # sitk_image = sitk.GetImageFromArray(heatmap[i])
            # Internal processing step.
            # Internal processing step.
            # output_file = os.path.join(f"./{i}.nrrd")
            # sitk.WriteImage(sitk_image, output_file)
        
        # Internal processing step.
        # enhanced_heatmap = np.copy(heatmap)
        # enhanced_heatmap[heatmap > 0.99] *= 30
        # enhanced_heatmap[(heatmap > 0.9) & (heatmap <= 0.99)] *= 20
        # enhanced_heatmap[(heatmap > 0.5) & (heatmap <= 0.9)] *= 10
        # enhanced_heatmap[(heatmap > 0.1) & (heatmap <= 0.5)] *= 5
        # heatmap = enhanced_heatmap
        
        # sitk_image = sitk.GetImageFromArray(heatmap[0])
        # Internal processing step.
        # Internal processing step.
        # output_file = os.path.join(f"./0.nrrd")
        # sitk.WriteImage(sitk_image, output_file)
        
        return heatmap, points
    
                
    
    def get_sample_name(self, idx):
        """
                    patch ID
        :param idx: patch   
        :return: (    , patch ID)
        """
        sample_idx, patch_id = self.pacth_index[idx]
        input_img_path = self.img_paths[sample_idx]
        sample_name = os.path.splitext(os.path.basename(input_img_path))[0]
        return sample_name, patch_id

    def fliter_seg(self, seg_array, type=1):
        # Internal processing step.
        # scapula - 1, humerus - 2
        flitered_seg = (seg_array == type).astype(np.float32)

        # Internal processing step.
        boundary = np.zeros_like(flitered_seg)
        for z in range(flitered_seg.shape[0]):
            slice_ = flitered_seg[z, :, :]
            contours = measure.find_contours(slice_, level=0.5)

            # Internal processing step.
            for contour in contours:
                contour = np.round(contour).astype(int)
                for point in contour:
                    boundary[z, point[0], point[1]] = 3

        # Internal processing step.
        flitered_seg = np.where(boundary == 3, 3, flitered_seg)
        
        return flitered_seg
    
    def clear_cache(self):
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)


    def create_boundary_map(self, seg_array, iterations=50):
        """
                =3 seg_array            =1   =0 
        """
        # Internal processing step.
        structure_mask = np.logical_or(seg_array == 1, seg_array == 3).astype(np.uint8)

        # Internal processing step.
        labeled = measure.label(structure_mask, connectivity=1)
        props = measure.regionprops(labeled)
        if not props:
            dist_map = np.zeros_like(seg_array, dtype=np.float32)
        else:
            largest_region = max(props, key=lambda x: x.area)
            max_mask = (labeled == largest_region.label).astype(np.uint8)

            # Internal processing step.
            eroded = binary_erosion(max_mask)
            boundary = max_mask ^ eroded

            # Internal processing step.
            dist_map = boundary.astype(np.float32)
            current = boundary.copy()
            for i in range(1, iterations):
                current = binary_dilation(current) & (~max_mask)
                dist_map += current.astype(np.float32)

            # Internal processing step.
            dist_map = (iterations - dist_map) / iterations
            dist_map = np.clip(dist_map, 0, 1)

        # Internal processing step.
        # sitk_img = sitk.GetImageFromArray(dist_map)
        # sitk_img.SetOrigin((0, 0, 0))
        # sitk_img.SetSpacing((1, 1, 1))
        # sitk.WriteImage(sitk_img, "./boundary_map.nrrd")

        return dist_map.astype(np.float32)
    
    def _load_cached_index(self):
        """    cache     pacth_index   """
        self.pacth_index = []
        for fname in os.listdir(self.cache_dir):
            if fname.endswith('_img.npy'):
                name_parts = fname.replace('_img.npy', '').split('_patch_')
                sample_idx = int(name_parts[0].replace('sample_', ''))
                patch_id = int(name_parts[1])
                self.pacth_index.append((sample_idx, patch_id))




