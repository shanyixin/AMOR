import os
import json
from typing import Dict, Tuple, Optional, Any

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt


# =========================================================
# Internal processing step.
# =========================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def strip_nii_name(filename: str) -> str:
    if filename.lower().endswith('.nii.gz'):
        return filename[:-7]
    return os.path.splitext(filename)[0]


def save_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def load_landmarks_from_json(json_path: str):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    landmarks = {}
    if 'markups' in data and len(data['markups']) > 0:
        for pt in data['markups'][0].get('controlPoints', []):
            label = pt.get('label', '')
            pos = pt.get('position', None)
            if label and pos is not None and len(pos) == 3:
                landmarks[label] = np.array(pos, dtype=np.float64)

    return landmarks, data


# =========================================================
# Internal processing step.
# =========================================================
def collect_scapula_fit_points(landmarks: Dict[str, np.ndarray]):
    """     bi* SCP*   gc    scapula      """
    pts = []
    used_labels = []
    for k in sorted(landmarks.keys()):
        if k == 'gc' or k.startswith('bi') or k.startswith('SCP'):
            pts.append(landmarks[k])
            used_labels.append(k)

    if len(pts) == 0:
        return None, []
    return np.stack(pts, axis=0), used_labels


def fit_plane_svd(points: np.ndarray):
    centroid = np.mean(points, axis=0)
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vh[-1]
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    return normal, centroid


def choose_inplane_axes(points: np.ndarray, centroid: np.ndarray, normal: np.ndarray):
    vecs = points - centroid
    signed_dist = vecs @ normal
    proj = points - np.outer(signed_dist, normal)
    proj_centered = proj - np.mean(proj, axis=0)

    if proj.shape[0] < 2 or np.linalg.norm(proj_centered) < 1e-8:
        tmp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(np.dot(tmp, normal)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        u = tmp - np.dot(tmp, normal) * normal
        u = u / (np.linalg.norm(u) + 1e-12)
        v = np.cross(normal, u)
        v = v / (np.linalg.norm(v) + 1e-12)
        return u, v

    _, _, vh = np.linalg.svd(proj_centered, full_matrices=False)
    u = vh[0]
    u = u - np.dot(u, normal) * normal
    if np.linalg.norm(u) < 1e-8:
        tmp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(np.dot(tmp, normal)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        u = tmp - np.dot(tmp, normal) * normal

    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(normal, u)
    v = v / (np.linalg.norm(v) + 1e-12)
    return u, v


def compute_obb_from_points(points: np.ndarray, margin_inplane_mm: float, margin_normal_mm: float):
    normal, centroid = fit_plane_svd(points)
    u, v = choose_inplane_axes(points, centroid, normal)

    rel = points - centroid
    cu = rel @ u
    cv = rel @ v
    cn = rel @ normal

    return {
        'center': centroid,
        'u': u,
        'v': v,
        'n': normal,
        'u_min': float(np.min(cu) - margin_inplane_mm),
        'u_max': float(np.max(cu) + margin_inplane_mm),
        'v_min': float(np.min(cv) - margin_inplane_mm),
        'v_max': float(np.max(cv) + margin_inplane_mm),
        'n_min': float(np.min(cn) - margin_normal_mm),
        'n_max': float(np.max(cn) + margin_normal_mm),
    }


def build_uv_frame_from_landmarks(landmarks: Dict[str, np.ndarray]):
    fit_points, used_labels = collect_scapula_fit_points(landmarks)
    if fit_points is None or len(fit_points) < 4:
        return None, None

    n, center = fit_plane_svd(fit_points)
    u, v = choose_inplane_axes(fit_points, center, n)
    frame = {'center': center, 'u': u, 'v': v, 'n': n}
    return frame, used_labels


# =========================================================
# Internal processing step.
# =========================================================
def get_direction_matrix(img: sitk.Image):
    return np.array(img.GetDirection(), dtype=np.float64).reshape(3, 3)


def indices_xyz_to_physical(img: sitk.Image, indices_xyz: np.ndarray):
    origin = np.array(img.GetOrigin(), dtype=np.float64)
    spacing = np.array(img.GetSpacing(), dtype=np.float64)
    direction = get_direction_matrix(img)
    scaled = indices_xyz * spacing
    return origin + scaled @ direction.T


def physical_to_local_coords(phys_pts: np.ndarray, frame: Dict[str, np.ndarray]):
    rel = phys_pts - frame['center'][None, :]
    cu = rel @ frame['u']
    cv = rel @ frame['v']
    ch = rel @ frame['n']
    return cu, cv, ch


# =========================================================
# Internal processing step.
# =========================================================
def build_mask_inside_obb(img: sitk.Image, obb: Dict[str, Any], use_only_foreground: bool = True, foreground_threshold: int = 0):
    arr = sitk.GetArrayFromImage(img)  # z, y, x

    if use_only_foreground:
        z_idx, y_idx, x_idx = np.where(arr > foreground_threshold)
    else:
        z_dim, y_dim, x_dim = arr.shape
        zz, yy, xx = np.meshgrid(
            np.arange(z_dim),
            np.arange(y_dim),
            np.arange(x_dim),
            indexing='ij',
        )
        z_idx = zz.ravel()
        y_idx = yy.ravel()
        x_idx = xx.ravel()

    if len(z_idx) == 0:
        inside_arr = np.zeros_like(arr, dtype=bool)
        masked_arr = np.zeros_like(arr)
        return inside_arr, masked_arr

    idx_xyz = np.stack([x_idx, y_idx, z_idx], axis=1).astype(np.float64)
    phys_pts = indices_xyz_to_physical(img, idx_xyz)
    cu, cv, cn = physical_to_local_coords(phys_pts, obb)

    inside = (
        (cu >= obb['u_min']) & (cu <= obb['u_max']) &
        (cv >= obb['v_min']) & (cv <= obb['v_max']) &
        (cn >= obb['n_min']) & (cn <= obb['n_max'])
    )

    inside_arr = np.zeros_like(arr, dtype=bool)
    inside_arr[z_idx[inside], y_idx[inside], x_idx[inside]] = True

    masked_arr = np.zeros_like(arr)
    masked_arr[inside_arr] = arr[inside_arr]
    return inside_arr, masked_arr


def crop_to_nonzero(masked_arr: np.ndarray, ref_img: sitk.Image):
    nz = np.where(masked_arr > 0)
    if len(nz[0]) == 0:
        return None, None

    z0, y0, x0 = np.min(nz[0]), np.min(nz[1]), np.min(nz[2])
    z1, y1, x1 = np.max(nz[0]), np.max(nz[1]), np.max(nz[2])

    crop_arr = masked_arr[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]
    crop_img = sitk.GetImageFromArray(crop_arr)

    origin_phys = ref_img.TransformIndexToPhysicalPoint((int(x0), int(y0), int(z0)))
    crop_img.SetSpacing(ref_img.GetSpacing())
    crop_img.SetDirection(ref_img.GetDirection())
    crop_img.SetOrigin(origin_phys)

    crop_info = {
        'crop_index_min_xyz': [int(x0), int(y0), int(z0)],
        'crop_index_max_xyz': [int(x1), int(y1), int(z1)],
        'crop_size_xyz': [int(x1 - x0 + 1), int(y1 - y0 + 1), int(z1 - z0 + 1)],
    }
    return crop_img, crop_info


# =========================================================
# Internal processing step.
# =========================================================
def _save_mid_slices_png(arr_zyx: np.ndarray, out_png: str, title: str) -> None:
    if arr_zyx.ndim != 3:
        return

    zc = arr_zyx.shape[0] // 2
    yc = arr_zyx.shape[1] // 2
    xc = arr_zyx.shape[2] // 2

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(arr_zyx[zc], cmap='gray', origin='upper')
    axes[0].set_title(f'{title} - axial(z)')
    axes[1].imshow(arr_zyx[:, yc, :], cmap='gray', origin='upper')
    axes[1].set_title(f'{title} - coronal(y)')
    axes[2].imshow(arr_zyx[:, :, xc], cmap='gray', origin='upper')
    axes[2].set_title(f'{title} - sagittal(x)')
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_png, dpi=180, bbox_inches='tight')
    plt.close(fig)


# =========================================================
# Internal processing step.
# =========================================================
def get_cropped_img(
    nii_path: str,
    json_path: str,
    out_dir: str,
    box_margin_inplane_mm: float = 6.0,
    box_margin_normal_mm: float = 2.5,
    use_only_foreground: bool = True,
    foreground_threshold: int = 0,
):
    """Step1      mask/json      OBB      crop mask """
    ensure_dir(out_dir)

    case_id = strip_nii_name(os.path.basename(nii_path))
    step_dir = os.path.join(out_dir, 'step1_crop')
    ensure_dir(step_dir)

    full_img = sitk.ReadImage(nii_path)
    full_arr = sitk.GetArrayFromImage(full_img)
    landmarks, raw_json = load_landmarks_from_json(json_path)

    fit_points, used_labels = collect_scapula_fit_points(landmarks)
    if fit_points is None or len(fit_points) < 4:
        raise RuntimeError(f'[{case_id}]            OBB    ')

    obb = compute_obb_from_points(
        fit_points,
        margin_inplane_mm=box_margin_inplane_mm,
        margin_normal_mm=box_margin_normal_mm,
    )

    _, masked_arr = build_mask_inside_obb(
        full_img,
        obb,
        use_only_foreground=use_only_foreground,
        foreground_threshold=foreground_threshold,
    )
    if np.count_nonzero(masked_arr) == 0:
        raise RuntimeError(f'[{case_id}] OBB       ')

    crop_img, crop_info = crop_to_nonzero(masked_arr, full_img)
    if crop_img is None:
        raise RuntimeError(f'[{case_id}] crop    ')

    frame, _ = build_uv_frame_from_landmarks(landmarks)
    if frame is None:
        raise RuntimeError(f'[{case_id}] uv frame      ')

    crop_path = os.path.join(step_dir, f'{case_id}_local_crop_mask.nii.gz')
    sitk.WriteImage(crop_img, crop_path, useCompression=True)

    full_png = os.path.join(step_dir, f'{case_id}_full_mask_mid_slices.png')
    crop_png = os.path.join(step_dir, f'{case_id}_crop_mask_mid_slices.png')
    _save_mid_slices_png((full_arr > foreground_threshold).astype(np.uint8), full_png, 'full mask')
    _save_mid_slices_png((sitk.GetArrayFromImage(crop_img) > foreground_threshold).astype(np.uint8), crop_png, 'crop mask')

    info_json_path = os.path.join(step_dir, f'{case_id}_step1_info.json')
    step1_info = {
        'case_id': case_id,
        'nii_path': nii_path,
        'json_path': json_path,
        'used_labels': used_labels,
        'num_landmarks': int(len(landmarks)),
        'crop_info': crop_info,
        'box_margin_inplane_mm': float(box_margin_inplane_mm),
        'box_margin_normal_mm': float(box_margin_normal_mm),
        'use_only_foreground': bool(use_only_foreground),
        'foreground_threshold': int(foreground_threshold),
        'outputs': {
            'crop_mask_nii': crop_path,
            'full_mask_mid_slices_png': full_png,
            'crop_mask_mid_slices_png': crop_png,
        },
    }
    save_json(step1_info, info_json_path)

    return {
        'case_id': case_id,
        'nii_path': nii_path,
        'json_path': json_path,
        'out_dir': out_dir,
        'full_img': full_img,
        'crop_img': crop_img,
        'crop_info': crop_info,
        'landmarks': landmarks,
        'raw_json': raw_json,
        'used_labels': used_labels,
        'obb': obb,
        'frame': frame,
        'step1_dir': step_dir,
        'step1_info_path': info_json_path,
        'step1_outputs': step1_info['outputs'],
    }

