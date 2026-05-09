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

class Shoulder_Joint(Dataset):
    def __init__(self, img_size, train_path, cache_path, sigma=3, dataset_type = 0,  transform=None):
        """
        :param img_size:         (512, 512, 512)
        :param train_path:           img/   scapula/ 
        :cache_path:                 
        :param transform:          
        :param sigma:           
        :param dataset_type:       0:     1:     2:     
        """
        self.ImagePath = os.path.join(train_path, "img")
        self.SegPath = os.path.join(train_path, "seg")
        self.ScapulaPath = os.path.join(train_path, "scapula")
        self.HumerusPath = os.path.join(train_path, "humerus")
        self.TypePath = os.path.join(train_path, "type")
        
        if dataset_type == 0:
            data_type = "train"
        elif dataset_type == 1:
            data_type = "valid"
        elif dataset_type == 2:
            data_type = "test"
        
        self.cache_dir = os.path.join(cache_path, data_type)
        if os.path.exists(self.cache_dir) and os.listdir(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.img_size = img_size[0]  
        self.transform = transform  
        self.sigma = sigma  

        # Internal processing step.
        # ["ss","sm","si","AA","AP","C","gc","bi12","bi2","bi3","bi4","bi5","bi6","bi7","bi8","bi9","SCP6","SCP12"]
        self.scapula_fixed_labels = ["ss","sm","si","AA","AP","C"]
        # slef.humerus_fixed_labels = []

        # Internal processing step.
        self.img_paths = sorted([os.path.join(self.ImagePath, f) for f in os.listdir(self.ImagePath) if f.endswith(".nrrd")])
        self.seg_paths = sorted([os.path.join(self.SegPath, f) for f in os.listdir(self.SegPath) if f.endswith(".nrrd")])
        self.scapula_paths = sorted([os.path.join(self.ScapulaPath, f) for f in os.listdir(self.ScapulaPath) if f.endswith(".json")])
        # self.humerus_paths = sorted([os.path.join(self.HumerusPath, f) for f in os.listdir(self.HumerusPath) if f.endswith(".json")])
        # self.type_paths = sorted([os.path.join(self.TypePath, f) for f in os.listdir(self.TypePath) if f.endswith(".json")])
        
        self.preprocess_and_cache()

    def preprocess_and_cache(self):
        attention_weight = 0.75
        for idx in range(len(self.img_paths)):
            cache_file = os.path.join(self.cache_dir, f"sample_{idx}.npz")
            if os.path.exists(cache_file):
                continue

            img = self.load_nrrd(self.img_paths[idx])
            seg = self.fliter_seg(self.load_nrrd(self.seg_paths[idx]))
            norm_img = self.normalize_image(img, seg, attention_weight).numpy()
            scapula_heatmap, scapula_points = self.create_heatmaps(self.scapula_paths[idx], img.shape)

            np.save(os.path.join(self.cache_dir, f"sample_{idx}_img.npy"), norm_img)
            np.save(os.path.join(self.cache_dir, f"sample_{idx}_seg.npy"), seg)
            np.save(os.path.join(self.cache_dir, f"sample_{idx}_scapula_heatmap.npy"), scapula_heatmap)
            np.save(os.path.join(self.cache_dir, f"sample_{idx}_scapula_points.npy"), scapula_points)

    def normalize_image(self, img, seg, attention_weight):
        """          """
        img_tensor = torch.from_numpy(img).float()
        seg_tensor = torch.from_numpy(seg).float()
        img_tensor = img_tensor + attention_weight * seg_tensor
        img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min())
        
        norm_img = img_tensor.numpy()
        sitk_img = sitk.GetImageFromArray(norm_img)
        sitk_img.SetOrigin((0, 0, 0))  # Internal processing step.
        sitk_img.SetSpacing((1, 1, 1))  # Internal processing step.
        output_file = "./normalized_image.nrrd"
        sitk.WriteImage(sitk_img, output_file)
        
        return img_tensor
    
    def __len__(self):
        return len(self.img_paths) * self.img_size

    def __getitem__(self, idx):
        img_idx = idx // 256
        z_idx = idx % 256

        img = np.load(os.path.join(self.cache_dir, f"sample_{img_idx}_img.npy"), mmap_mode='r')
        seg = np.load(os.path.join(self.cache_dir, f"sample_{img_idx}_seg.npy"), mmap_mode='r')
        scapula_heatmap = np.load(os.path.join(self.cache_dir, f"sample_{img_idx}_scapula_heatmap.npy"), mmap_mode='r')
        scapula_points = np.load(os.path.join(self.cache_dir, f"sample_{img_idx}_scapula_points.npy"), mmap_mode='r')

        img_slice = torch.from_numpy(img[z_idx].copy()).unsqueeze(0)
        seg_slice = torch.from_numpy(seg[z_idx].copy()).unsqueeze(0)
        scapula_heatmap_slice = torch.from_numpy(scapula_heatmap[:, z_idx].copy()).float()
        scapula_points = scapula_points.copy()
        
        return img_slice.float(), seg_slice.float(), scapula_heatmap_slice, scapula_points


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
        
        # sitk_image = sitk.GetImageFromArray(enhanced_heatmap[0])
        # Internal processing step.
        # Internal processing step.
        # output_file = os.path.join(f"./0.nrrd")
        # sitk.WriteImage(sitk_image, output_file)
        
        return heatmap, points
    
    def create_single_heatmap(self, json_path, img_shape):
        """           3D    (D, H, W)"""
        with open(json_path, 'r') as f:
            points_data = json.load(f)
        
        # Internal processing step.
        label_dict = {p['label']: p['position'] for p in points_data}
        heatmap = np.zeros(img_shape, dtype=np.float32)  # Internal processing step.
        
        # Internal processing step.
        for label in self.scapula_fixed_labels:
            if label in label_dict:
                x, y, z = label_dict[label]
                x, y, z = round(x), round(y), round(z)

                # Internal processing step.
                if 0 <= x < img_shape[2] and 0 <= y < img_shape[1] and 0 <= z < img_shape[0]:
                    single_heatmap = np.zeros(img_shape, dtype=np.float32)
                    single_heatmap[z, y, x] = 1  # Internal processing step.

                    # Internal processing step.
                    filtered_heatmap = gaussian_filter(single_heatmap, sigma=self.sigma)

                    # Internal processing step.
                    heatmap += filtered_heatmap
        
        # Internal processing step.
        heatmap /= heatmap.max()  # Internal processing step.
        
        return heatmap
                
    
    def get_sample_name(self, idx):
        """
                                 
            :param idx:     
            :return:     
        """
        idx = idx // 256
        input_img_path = self.img_paths[idx]
        sample_name = os.path.splitext(os.path.basename(input_img_path))[0]  # Internal processing step.
        return sample_name

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


