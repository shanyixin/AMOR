"""Run inference and evaluation for shoulder landmark localization."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

import utils.logger as Logger
from utils.loss_function.boundary_aware_loss import bmc_loss
from utils.model_tool import upload_model


POINT_GROUPS = {
    "scapula": [
        "ss", "sm", "si", "AA", "AP", "C", "gc", "SCP6", "SCP12",
        "bi3", "bi9", "bi12", "bi2", "bi4", "bi5", "bi6", "bi7", "bi8",
    ],
    "humerus": ["HC", "HI", "HS", "HP", "HA", "HBGS", "HBGI"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained shoulder landmark model.")
    parser.add_argument("-c", "--config", type=str, default="config/test.json", help="Path to a JSON config.")
    parser.add_argument("-p", "--phase", type=str, choices=["train", "test"], default="test")
    parser.add_argument("-gpu", "--gpu_ids", type=str, default="0")
    parser.add_argument("-debug", "-d", action="store_true")
    return parser.parse_args()


def resolve_landmarks(dataset_opt):
    points_part = dataset_opt["points_part"]
    if points_part == "scapula":
        import utils.dataset.dataset_all_3D_Scapula as DatasetModule

        return POINT_GROUPS["scapula"], DatasetModule
    if points_part == "humerus":
        import utils.dataset.dataset_all_3D_Humerus as DatasetModule

        return POINT_GROUPS["humerus"], DatasetModule
    raise ValueError("datasets.test.points_part must be either 'scapula' or 'humerus'.")


def build_test_loader(opt):
    dataset_opt = opt["datasets"]["test"]
    points, dataset_module = resolve_landmarks(dataset_opt)
    dataset_opt["num_landmarks"] = len(points)

    test_set = dataset_module.Shoulder_Joint_Whole(
        points,
        dataset_opt["image_size"],
        dataset_opt["testroot"],
        dataset_opt["cacheroot"],
        dataset_opt["heat_sigma"],
        dataset_type=2,
        transform=False,
        rebuild_cache=dataset_opt["rebuild_cache"],
        use_labels=dataset_opt["use_labels"],
    )
    loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=dataset_opt["batch_size"],
        shuffle=dataset_opt["use_shuffle"],
        num_workers=int(dataset_opt.get("num_workers") or 0),
        pin_memory=torch.cuda.is_available(),
    )
    return points, test_set, loader


def find_center_argmax(volume, boundary=None, refine=False):
    volume_np = volume.detach().cpu().numpy() if isinstance(volume, torch.Tensor) else volume
    volume_np = np.squeeze(volume_np)
    z_coord, y_coord, x_coord = np.unravel_index(np.argmax(volume_np), volume_np.shape)
    coords = np.array([x_coord, y_coord, z_coord], dtype=np.float32)
    return _project_to_surface(coords, boundary) if refine else tuple(coords)


def find_center_softargmax(volume, boundary, refine=True, eps=1e-6):
    volume_tensor = torch.from_numpy(volume).float().clamp(min=0)
    depth, height, width = volume_tensor.shape
    volume_tensor = volume_tensor.unsqueeze(0).unsqueeze(0)

    grid_z = torch.arange(depth).view(1, 1, depth, 1, 1).float()
    grid_y = torch.arange(height).view(1, 1, 1, height, 1).float()
    grid_x = torch.arange(width).view(1, 1, 1, 1, width).float()

    mass = volume_tensor.sum(dim=[2, 3, 4], keepdim=True) + eps
    center_z = (volume_tensor * grid_z).sum(dim=[2, 3, 4], keepdim=True) / mass
    center_y = (volume_tensor * grid_y).sum(dim=[2, 3, 4], keepdim=True) / mass
    center_x = (volume_tensor * grid_x).sum(dim=[2, 3, 4], keepdim=True) / mass
    coords = torch.cat([center_x, center_y, center_z], dim=-1).view(3).cpu().numpy()

    return _project_to_surface(coords, boundary) if refine else tuple(coords)


def _project_to_surface(coords, boundary):
    if boundary is None:
        return tuple(coords)

    boundary_np = boundary.detach().cpu().numpy() if isinstance(boundary, torch.Tensor) else boundary
    boundary_np = np.squeeze(boundary_np)
    surface_mask = boundary_np == boundary_np.max()
    if not np.any(surface_mask):
        return tuple(coords)

    surface_indices = np.argwhere(surface_mask)
    surface_indices_xyz = np.stack([surface_indices[:, 2], surface_indices[:, 1], surface_indices[:, 0]], axis=1)
    closest_idx = np.argmin(np.linalg.norm(surface_indices_xyz - coords, axis=1))
    return tuple(surface_indices_xyz[closest_idx].astype(np.float32))


def save_slicer(landmarks, output_path, origin=(0, 0, 0), spacing=(1, 1, 1)):
    slicer_format = {
        "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
        "markups": [
            {
                "type": "Fiducial",
                "controlPoints": [],
                "measurements": [],
                "display": {
                    "visibility": True,
                    "opacity": 1.0,
                    "color": [0.4, 1.0, 1.0],
                    "selectedColor": [1.0, 0.5, 0.5],
                    "activeColor": [0.4, 1.0, 0.0],
                    "propertiesLabelVisibility": False,
                    "pointLabelsVisibility": True,
                    "textScale": 2.0,
                    "glyphType": "Sphere3D",
                    "glyphScale": 1.0,
                    "glyphSize": 5.0,
                    "useGlyphScale": True,
                    "sliceProjection": False,
                    "sliceProjectionUseFiducialColor": True,
                    "sliceProjectionOutlinedBehindSlicePlane": False,
                    "sliceProjectionColor": [1.0, 1.0, 1.0],
                    "sliceProjectionOpacity": 0.6,
                    "lineThickness": 0.2,
                    "lineColorFadingStart": 1.0,
                    "lineColorFadingEnd": 10.0,
                    "lineColorFadingSaturation": 1.0,
                    "lineColorFadingHueOffset": 0.0,
                    "handlesInteractive": False,
                    "translationHandleVisibility": True,
                    "rotationHandleVisibility": True,
                    "scaleHandleVisibility": True,
                    "interactionHandleScale": 3.0,
                    "snapMode": "toVisibleSurface",
                },
            }
        ],
    }

    for index, landmark in enumerate(landmarks, start=1):
        position = (np.array(landmark["center"]) - np.array(origin)) / np.array(spacing)
        slicer_format["markups"][0]["controlPoints"].append(
            {
                "id": f"point_{index}",
                "label": landmark["label"],
                "position": position.tolist(),
                "selected": True,
                "locked": False,
                "visibility": True,
            }
        )

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(slicer_format, handle, indent=4)


def evaluate_checkpoint(opt, logger, device, points, test_set, test_loader):
    model = upload_model(opt, logger, device)
    if model is None:
        logger.info("Model loading failed. Exiting.")
        print("Model loading failed. Exiting.")
        sys.exit(1)

    model.eval()
    pointwise_errors = []
    total_heatmap_loss = 0.0
    total_prior_loss = 0.0
    total_samples = 0
    has_labels = bool(opt["datasets"]["test"]["use_labels"])

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            images, labels, label_points, boundary, segmentation = batch
            images = images.to(device)
            boundary = boundary.to(device)

            if has_labels:
                labels = labels.to(device)
                label_points_loss = label_points.to(device)

            outputs = model(images).squeeze(1).float()
            if has_labels:
                labels = labels.squeeze(1).float()
            boundary = boundary.squeeze(1).float()

            if has_labels:
                heatmap_loss, prior_loss, _, _, _, _ = bmc_loss(outputs, labels, label_points_loss, boundary, None, "valid")
                total_heatmap_loss += heatmap_loss.item()
                total_prior_loss += prior_loss.item()

            total_samples += 1
            outputs_np = outputs.squeeze(0).cpu().numpy()
            sample_name = test_set.get_sample_name(idx)
            result_dir = Path(opt["path"]["results"]) / sample_name
            result_dir.mkdir(parents=True, exist_ok=True)

            ground_truth_points = label_points[0] if has_labels else None
            pred_coords, errors = [], []

            for landmark_idx, label in enumerate(points):
                heatmap = outputs_np[landmark_idx]
                point = find_center_softargmax(heatmap, segmentation)
                pred_coords.append({"label": label, "center": point})

                heatmap_image = sitk.GetImageFromArray(heatmap)
                heatmap_image.SetSpacing((1.0, 1.0, 1.0))
                heatmap_image.SetOrigin((0.0, 0.0, 0.0))
                sitk.WriteImage(heatmap_image, str(result_dir / f"{label}.nii.gz"))

                if has_labels:
                    error = np.linalg.norm(np.array(ground_truth_points[landmark_idx]) - np.array(point))
                    errors.append(error)

            save_slicer(pred_coords, result_dir / "landmarks.json")
            if has_labels:
                pointwise_errors.append(errors)
                logger.info("%s | Mean error: %.3f", sample_name, np.mean(errors))
            else:
                logger.info("%s | Saved predicted landmarks.", sample_name)

    if has_labels:
        summarize_errors(pointwise_errors, points, total_heatmap_loss, total_prior_loss, total_samples, logger)
    else:
        logger.info("No ground-truth labels were provided; error and loss summaries were skipped.")


def summarize_errors(pointwise_errors, points, total_heatmap_loss, total_prior_loss, total_samples, logger):
    pointwise_errors = np.asarray(pointwise_errors)
    mean_per_landmark = np.mean(pointwise_errors, axis=0)
    overall_mean_error = np.mean(mean_per_landmark)
    average_heatmap = total_heatmap_loss / max(total_samples, 1)
    average_prior = total_prior_loss / max(total_samples, 1)

    logger.info("========== Landmark Point Errors ==========")
    for index, label in enumerate(points):
        logger.info("%s: %.3f voxels", label, mean_per_landmark[index])
    logger.info("Overall mean error: %.3f voxels", overall_mean_error)
    logger.info("========== Loss Summary ==========")
    logger.info("BMC heatmap term: %.6f", average_heatmap)
    logger.info("BMC prior term: %.6f", average_prior)

    print("\n=== Test Summary ===")
    for index, label in enumerate(points):
        print(f"{label}: {mean_per_landmark[index]:.3f} voxels")
    print(f"Overall mean error: {overall_mean_error:.3f} voxels")
    print(f"BMC heatmap term: {average_heatmap:.6f}")
    print(f"BMC prior term: {average_prior:.6f}")


def main():
    args = parse_args()
    opt = Logger.dict_to_nonedict(Logger.parse(args))
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt["path"]["log"], "test", level=logging.INFO)
    logger = logging.getLogger("base")
    logger.info(Logger.dict2str(opt))

    points, test_set, test_loader = build_test_loader(opt)
    logger.info("Test dataset is ready.")

    device = torch.device("cuda:0" if opt["gpu_ids"] and torch.cuda.is_available() else "cpu")
    evaluate_checkpoint(opt, logger, device, points, test_set, test_loader)


if __name__ == "__main__":
    main()

