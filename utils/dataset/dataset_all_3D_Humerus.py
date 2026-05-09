"""Compatibility wrapper for humerus landmark datasets."""

from .shoulder_dataset import ShoulderLandmarkDataset


class Shoulder_Joint_Whole(ShoulderLandmarkDataset):
    def __init__(self, points, img_size, train_path, cache_path, sigma=3, dataset_type=1,
                 transform=True, rebuild_cache=False, vis=True, use_labels=True):
        super().__init__(
            points=points,
            img_size=img_size,
            train_path=train_path,
            cache_path=cache_path,
            sigma=sigma,
            dataset_type=dataset_type,
            transform=transform,
            rebuild_cache=rebuild_cache,
            vis=vis,
            use_labels=use_labels,
            landmark_dir_name="humerus",
            segmentation_label=2,
            cache_prefix="humerus",
            exclude_prefixes=("a68d1c1",),
        )


ShoulderJointWhole = Shoulder_Joint_Whole
