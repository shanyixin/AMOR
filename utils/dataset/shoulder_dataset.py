"""Dataset implementation for 3D shoulder landmark heatmap regression."""

import json
import os
import shutil
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter
from skimage import measure
from torch.utils.data import Dataset


DATASET_SPLITS = ("train", "valid", "test")
VOLUME_EXTENSIONS = (".nii.gz", ".nrrd")


class ShoulderLandmarkDataset(Dataset):
    """Load volumes, segmentation masks, landmark JSON files, and cached heatmaps."""

    def __init__(
        self,
        points,
        img_size,
        train_path,
        cache_path,
        sigma=3,
        dataset_type=1,
        transform=True,
        rebuild_cache=False,
        vis=True,
        use_labels=True,
        landmark_dir_name="scapula",
        segmentation_label=1,
        cache_prefix="scapula",
        exclude_prefixes=None,
    ):
        self.dataset_type = dataset_type
        self.split = DATASET_SPLITS[dataset_type]
        self.use_labels = self.split in {"train", "valid"} or use_labels
        self.img_size = img_size[0] if isinstance(img_size, (list, tuple)) else img_size
        self.patch_size = 128
        self.transform = transform
        self.sigma = sigma
        self.vis = vis
        self.landmark_labels = list(points)
        self.landmark_dir_name = landmark_dir_name
        self.segmentation_label = segmentation_label
        self.cache_prefix = cache_prefix
        self.exclude_prefixes = tuple(exclude_prefixes or ())

        root = Path(train_path)
        self.image_dir = root / "img"
        self.segmentation_dir = root / "seg"
        self.landmark_dir = root / landmark_dir_name
        self.cache_dir = Path(cache_path) / self.split

        self.img_paths = self._list_files(self.image_dir, VOLUME_EXTENSIONS)
        self.seg_paths = self._list_files(self.segmentation_dir, VOLUME_EXTENSIONS)
        self.landmark_paths = self._list_files(self.landmark_dir, (".json",)) if self.use_labels else None

        if len(self.img_paths) != len(self.seg_paths):
            raise ValueError("The number of images and segmentation files does not match.")
        if self.use_labels and len(self.img_paths) != len(self.landmark_paths):
            raise ValueError("The number of images and landmark files does not match.")

        if rebuild_cache and self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        if rebuild_cache or not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.preprocess_and_cache()

        self.index_list = list(range(len(self.img_paths)))

    def _list_files(self, directory, extensions):
        if not directory.exists():
            raise FileNotFoundError(f"Required dataset directory does not exist: {directory}")
        files = []
        for name in os.listdir(directory):
            if name.startswith(self.exclude_prefixes):
                continue
            if any(name.endswith(ext) for ext in extensions):
                files.append(directory / name)
        return sorted(files)

    def preprocess_and_cache(self):
        attention_weight = 0.75
        for idx, image_path in enumerate(self.img_paths):
            image = self.load_volume(image_path)
            segmentation = self.filter_segmentation(self.load_volume(self.seg_paths[idx]))
            normalized_image = self.normalize_image(image, segmentation, attention_weight).numpy()
            boundary = self.create_boundary_map(segmentation)

            base_name = f"sample_{idx}"
            np.save(self.cache_dir / f"{base_name}_img.npy", normalized_image)
            np.save(self.cache_dir / f"{base_name}_seg.npy", segmentation)
            np.save(self.cache_dir / f"{base_name}_boundary.npy", boundary)

            if self.use_labels:
                heatmap, points = self.create_heatmaps(self.landmark_paths[idx], image.shape)
                np.save(self.cache_dir / f"{base_name}_{self.cache_prefix}_heatmap.npy", heatmap)
                np.save(self.cache_dir / f"{base_name}_{self.cache_prefix}_points.npy", points)
                if self.vis and idx == 0:
                    self.visualize_sample(normalized_image, heatmap[0], boundary, segmentation)

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        base_name = f"sample_{idx}"
        image = np.load(self.cache_dir / f"{base_name}_img.npy", mmap_mode="r")
        boundary = np.load(self.cache_dir / f"{base_name}_boundary.npy", mmap_mode="r")
        segmentation = np.load(self.cache_dir / f"{base_name}_seg.npy", mmap_mode="r")

        image_tensor = torch.from_numpy(image.copy()).unsqueeze(0).float()
        boundary_tensor = torch.from_numpy(boundary.copy()).unsqueeze(0).float()
        segmentation_tensor = torch.from_numpy(segmentation.copy()).unsqueeze(0).float()

        if not self.use_labels:
            return image_tensor, [], [], boundary_tensor, segmentation_tensor

        heatmap = np.load(self.cache_dir / f"{base_name}_{self.cache_prefix}_heatmap.npy", mmap_mode="r")
        points = np.load(self.cache_dir / f"{base_name}_{self.cache_prefix}_points.npy", allow_pickle=True)
        heatmap_tensor = torch.from_numpy(heatmap.copy()).float()
        points_tensor = torch.as_tensor(points, dtype=torch.float32)
        return image_tensor, heatmap_tensor, points_tensor, boundary_tensor, segmentation_tensor

    @staticmethod
    def normalize_image(image, segmentation, attention_weight):
        image_tensor = torch.from_numpy(image).float()
        segmentation_tensor = torch.from_numpy(segmentation).float()
        image_tensor = image_tensor + attention_weight * segmentation_tensor
        value_range = image_tensor.max() - image_tensor.min()
        if value_range <= 0:
            return torch.zeros_like(image_tensor)
        return (image_tensor - image_tensor.min()) / value_range

    @staticmethod
    def load_volume(filepath):
        image = sitk.ReadImage(str(filepath))
        return sitk.GetArrayFromImage(image)

    def create_heatmaps(self, json_path, image_shape):
        label_dict = self._load_landmark_positions(json_path)
        heatmap = np.zeros((len(self.landmark_labels), *image_shape), dtype=np.float32)
        points = []

        for channel, label in enumerate(self.landmark_labels):
            if label not in label_dict:
                raise KeyError(f"Landmark '{label}' was not found in {json_path}.")

            x, y, z = [round(float(value)) for value in label_dict[label]]
            points.append((x, y, z))
            if 0 <= x < image_shape[2] and 0 <= y < image_shape[1] and 0 <= z < image_shape[0]:
                heatmap[channel, z, y, x] = 1.0

        for channel in range(len(self.landmark_labels)):
            heatmap[channel] = gaussian_filter(heatmap[channel], sigma=self.sigma)
            peak = heatmap[channel].max()
            if peak > 0:
                heatmap[channel] /= peak

        return heatmap, np.asarray(points, dtype=np.float32)

    @staticmethod
    def _load_landmark_positions(json_path):
        with open(json_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        if isinstance(data, list):
            points = data
        else:
            markups = data.get("markups", [])
            points = markups[0].get("controlPoints", []) if markups else []

        positions = {}
        for point in points:
            label = point.get("label")
            position = point.get("position") or point.get("center")
            if label and position is not None:
                positions[label] = position
        return positions

    def filter_segmentation(self, segmentation):
        filtered = (segmentation == self.segmentation_label).astype(np.float32)
        boundary = np.zeros_like(filtered)
        for z_index in range(filtered.shape[0]):
            contours = measure.find_contours(filtered[z_index], level=0.5)
            for contour in contours:
                for row, col in np.round(contour).astype(int):
                    if 0 <= row < boundary.shape[1] and 0 <= col < boundary.shape[2]:
                        boundary[z_index, row, col] = 3.0
        return np.where(boundary == 3.0, 3.0, filtered)

    @staticmethod
    def create_boundary_map(segmentation, erosion_iters=1, dilation_iters=5):
        structure_mask = np.logical_or(segmentation == 1, segmentation == 3).astype(np.uint8)
        labeled = measure.label(structure_mask, connectivity=1)
        regions = measure.regionprops(labeled)
        if not regions:
            return np.zeros_like(segmentation, dtype=np.float32)

        largest_region = max(regions, key=lambda region: region.area)
        mask = (labeled == largest_region.label).astype(np.uint8)

        eroded = mask.copy()
        for _ in range(erosion_iters):
            eroded = binary_erosion(eroded)

        boundary = mask ^ eroded
        distance_map = boundary.astype(np.float32)
        current = boundary.copy()

        for _ in range(1, dilation_iters):
            current = binary_dilation(current) & (~mask)
            distance_map += current.astype(np.float32)

        distance_map = (dilation_iters - distance_map) / dilation_iters
        return np.clip(distance_map, 0, 1).astype(np.float32)

    def clear_cache(self):
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)

    def get_sample_name(self, idx):
        return self.img_paths[idx].name.split(".")[0]

    def visualize_sample(self, normalized_image, heatmap, boundary, segmentation):
        vis_dir = self.cache_dir.parent / "vis" / self.split
        vis_dir.mkdir(parents=True, exist_ok=True)

        outputs = {
            "norm_img.nrrd": normalized_image,
            "heatmap.nrrd": heatmap,
            "boundary.nrrd": boundary,
            "seg.nrrd": segmentation,
        }
        for name, array in outputs.items():
            sitk.WriteImage(sitk.GetImageFromArray(array.astype(np.float32)), str(vis_dir / name))

