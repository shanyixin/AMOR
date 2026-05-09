
import os
import json
from typing import Dict, Any, Optional

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

from scipy import ndimage as ndi
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from skimage import measure


# =========================================================
# Internal processing step.
# =========================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


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
def build_uv_grid_info(points_uv: np.ndarray, uv_resolution_mm: float, margin_mm: float = 0.0):
    u_min = np.min(points_uv[:, 0]) - margin_mm
    u_max = np.max(points_uv[:, 0]) + margin_mm
    v_min = np.min(points_uv[:, 1]) - margin_mm
    v_max = np.max(points_uv[:, 1]) + margin_mm

    nu = int(np.ceil((u_max - u_min) / uv_resolution_mm)) + 1
    nv = int(np.ceil((v_max - v_min) / uv_resolution_mm)) + 1

    return {
        'u_min': float(u_min),
        'u_max': float(u_max),
        'v_min': float(v_min),
        'v_max': float(v_max),
        'res': float(uv_resolution_mm),
        'nu': int(nu),
        'nv': int(nv),
    }


def keep_largest_connected_component(binary_mask: np.ndarray):
    lbl, num = ndi.label(binary_mask > 0)
    if num == 0:
        return np.zeros_like(binary_mask, dtype=np.uint8)

    areas = ndi.sum(binary_mask > 0, lbl, index=np.arange(1, num + 1))
    keep_id = int(np.argmax(areas)) + 1
    return (lbl == keep_id).astype(np.uint8)


def extract_outer_boundary_2d(binary_2d: np.ndarray):
    """                 """
    occ = binary_2d > 0
    if np.count_nonzero(occ) == 0:
        return np.zeros_like(binary_2d, dtype=np.uint8)

    bg = ~occ
    structure = ndi.generate_binary_structure(2, 1)
    lbl, num = ndi.label(bg, structure=structure)
    if num == 0:
        return np.zeros_like(binary_2d, dtype=np.uint8)

    border_labels = set()
    for arr in [lbl[0, :], lbl[-1, :], lbl[:, 0], lbl[:, -1]]:
        for x in np.unique(arr):
            if x > 0:
                border_labels.add(int(x))

    if len(border_labels) == 0:
        return np.zeros_like(binary_2d, dtype=np.uint8)

    outside_bg = np.isin(lbl, list(border_labels))
    dil_outside = ndi.binary_dilation(outside_bg, structure=structure)
    boundary = occ & dil_outside
    return boundary.astype(np.uint8)


def remove_sparse_boundary_pixels(boundary_2d, min_neighbors=2, max_iters=20):
    """
                3x3       8          
         < min_neighbors     
    
    min_neighbors=2    
    - 0      
    - 1      
    - 2          
    """
    b = (boundary_2d > 0).astype(np.uint8)
    if np.count_nonzero(b) == 0:
        return b

    for _ in range(max_iters):
        p = np.pad(b, 1, mode='constant', constant_values=0)

        n8 = (
            p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
            p[1:-1, :-2]               + p[1:-1, 2:] +
            p[2:, :-2]  + p[2:, 1:-1]  + p[2:, 2:]
        )

        remove_mask = (b > 0) & (n8 < min_neighbors)
        if not np.any(remove_mask):
            break

        b[remove_mask] = 0

    return b.astype(np.uint8)


def remove_diagonal_only_boundary_pixels(boundary_2d: np.ndarray, max_iters: int = 20):
    """
                  
    -       
    - 4      
    -   4       
    """
    b = (boundary_2d > 0).astype(np.uint8)
    if np.count_nonzero(b) == 0:
        return b

    for _ in range(max_iters):
        p = np.pad(b, 1, mode='constant', constant_values=0)

        up = p[:-2, 1:-1]
        down = p[2:, 1:-1]
        left = p[1:-1, :-2]
        right = p[1:-1, 2:]
        n4 = up + down + left + right

        ul = p[:-2, :-2]
        ur = p[:-2, 2:]
        dl = p[2:, :-2]
        dr = p[2:, 2:]
        ndiag = ul + ur + dl + dr

        remove_mask = (b > 0) & (n4 == 0) & (ndiag > 0)
        if not np.any(remove_mask):
            break
        b[remove_mask] = 0

    return b.astype(np.uint8)


def save_layer_debug_images(occ2d: np.ndarray, boundary2d: np.ndarray, k: int, out_root: str, h_value=None) -> None:
    ensure_dir(out_root)
    occ_dir = os.path.join(out_root, 'occ2d')
    boundary_dir = os.path.join(out_root, 'boundary2d')
    overlay_dir = os.path.join(out_root, 'overlay')
    ensure_dir(occ_dir)
    ensure_dir(boundary_dir)
    ensure_dir(overlay_dir)

    if h_value is None:
        stem = f'slice_{k:04d}'
        title = f'slice {k}'
    else:
        stem = f'slice_{k:04d}_h_{h_value:+08.3f}'
        title = f'slice {k}, h={h_value:.3f} mm'

    plt.figure(figsize=(6, 6))
    plt.imshow(occ2d, cmap='gray', origin='upper')
    plt.title(title + ' - occ2d')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(occ_dir, stem + '_occ.png'), dpi=180, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.imshow(boundary2d, cmap='gray', origin='upper')
    plt.title(title + ' - boundary2d')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(boundary_dir, stem + '_boundary.png'), dpi=180, bbox_inches='tight')
    plt.close()

    overlay = np.zeros((occ2d.shape[0], occ2d.shape[1], 3), dtype=np.float32)
    overlay[..., 0] = occ2d.astype(np.float32) * 0.75 + boundary2d.astype(np.float32) * 1.0
    overlay[..., 1] = occ2d.astype(np.float32) * 0.75
    overlay[..., 2] = occ2d.astype(np.float32) * 0.75
    overlay = np.clip(overlay, 0.0, 1.0)

    plt.figure(figsize=(6, 6))
    plt.imshow(overlay, origin='upper')
    plt.title(title + ' - overlay')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(overlay_dir, stem + '_overlay.png'), dpi=180, bbox_inches='tight')
    plt.close()


def save_surface_overview(surface_mask_arr: np.ndarray, out_png: str) -> None:
    zc = surface_mask_arr.shape[0] // 2
    yc = surface_mask_arr.shape[1] // 2
    xc = surface_mask_arr.shape[2] // 2

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(surface_mask_arr[zc], cmap='gray', origin='upper')
    axes[0].set_title('surface - axial(z)')
    axes[1].imshow(surface_mask_arr[:, yc, :], cmap='gray', origin='upper')
    axes[1].set_title('surface - coronal(y)')
    axes[2].imshow(surface_mask_arr[:, :, xc], cmap='gray', origin='upper')
    axes[2].set_title('surface - sagittal(x)')
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_png, dpi=180, bbox_inches='tight')
    plt.close(fig)


def save_surface_mask_nii(surface_mask_arr: np.ndarray, ref_img: sitk.Image, out_path: str) -> None:
    img = sitk.GetImageFromArray(surface_mask_arr.astype(np.uint8))
    img.CopyInformation(ref_img)
    sitk.WriteImage(img, out_path, useCompression=True)


def surface_mask_to_phys(surface_mask_arr: np.ndarray, ref_img: sitk.Image):
    z_idx, y_idx, x_idx = np.where(surface_mask_arr > 0)
    if len(z_idx) == 0:
        return None

    idx_xyz = np.stack([x_idx, y_idx, z_idx], axis=1).astype(np.float64)
    return indices_xyz_to_physical(ref_img, idx_xyz)


def extract_layered_surface(
    crop_img: sitk.Image,
    frame: Dict[str, np.ndarray],
    surface_dir: str,
    case_id: str,
    uv_resolution_mm: float = 0.5,
    h_step_mm: float = 0.5,
    keep_largest_cc_per_slice: bool = False,
    save_layer_debug: bool = True,
    layer_debug_every: int = 1,
):
    arr = sitk.GetArrayFromImage(crop_img)
    z_idx, y_idx, x_idx = np.where(arr > 0)
    if len(z_idx) == 0:
        raise RuntimeError(f'[{case_id}] crop mask         surface ')

    idx_xyz = np.stack([x_idx, y_idx, z_idx], axis=1).astype(np.float64)
    phys = indices_xyz_to_physical(crop_img, idx_xyz)
    u, v, h = physical_to_local_coords(phys, frame)

    points_uv = np.stack([u, v], axis=1)
    grid = build_uv_grid_info(points_uv, uv_resolution_mm=uv_resolution_mm, margin_mm=0.0)

    h_min = float(np.min(h))
    h_max = float(np.max(h))
    num_h = int(np.ceil((h_max - h_min) / h_step_mm)) + 1

    cols = np.round((u - grid['u_min']) / grid['res']).astype(int)
    rows = np.round((grid['v_max'] - v) / grid['res']).astype(int)
    ks = np.round((h - h_min) / h_step_mm).astype(int)

    valid = (
        (rows >= 0) & (rows < grid['nv']) &
        (cols >= 0) & (cols < grid['nu']) &
        (ks >= 0) & (ks < num_h)
    )

    x_idx = x_idx[valid]
    y_idx = y_idx[valid]
    z_idx = z_idx[valid]
    rows = rows[valid]
    cols = cols[valid]
    ks = ks[valid]

    surface_mask_arr = np.zeros_like(arr, dtype=np.uint8)
    saved_debug_layers = []

    layer_debug_dir = os.path.join(surface_dir, 'layer_debug')
    if save_layer_debug:
        ensure_dir(layer_debug_dir)

    for k in range(num_h):
        sel = (ks == k)
        if not np.any(sel):
            continue

        occ2d = np.zeros((grid['nv'], grid['nu']), dtype=np.uint8)
        occ2d[rows[sel], cols[sel]] = 1

        if keep_largest_cc_per_slice:
            occ2d = keep_largest_connected_component(occ2d)

        if np.count_nonzero(occ2d) == 0:
            continue

        boundary2d = extract_outer_boundary_2d(occ2d)
        boundary2d = remove_sparse_boundary_pixels(boundary2d, min_neighbors=2)


        rr = rows[sel]
        cc = cols[sel]
        keep_local = boundary2d[rr, cc] > 0

        x_keep = x_idx[sel][keep_local]
        y_keep = y_idx[sel][keep_local]
        z_keep = z_idx[sel][keep_local]
        surface_mask_arr[z_keep, y_keep, x_keep] = 1

        if save_layer_debug and (layer_debug_every <= 1 or (k % layer_debug_every == 0)):
            h_value = h_min + k * h_step_mm
            save_layer_debug_images(occ2d, boundary2d, k, layer_debug_dir, h_value=h_value)
            saved_debug_layers.append({
                'k': int(k),
                'h_value': float(h_value),
                'num_occ_pixels': int(np.count_nonzero(occ2d)),
                'num_boundary_pixels': int(np.count_nonzero(boundary2d)),
            })

    if np.count_nonzero(surface_mask_arr) == 0:
        raise RuntimeError(f'[{case_id}] layered surface      ')

    surface_mask_path = os.path.join(surface_dir, f'{case_id}_local_surface_mask.nii.gz')
    save_surface_mask_nii(surface_mask_arr, crop_img, surface_mask_path)

    surface_overview_png = os.path.join(surface_dir, f'{case_id}_surface_overview.png')
    save_surface_overview(surface_mask_arr, surface_overview_png)

    surface_phys = surface_mask_to_phys(surface_mask_arr, crop_img)
    if surface_phys is None or len(surface_phys) < 10:
        raise RuntimeError(f'[{case_id}] surface     ')

    surface_info = {
        'num_fg_voxels': int(len(z_idx)),
        'uv_resolution_mm': float(uv_resolution_mm),
        'h_step_mm': float(h_step_mm),
        'h_min': h_min,
        'h_max': h_max,
        'num_h_slices': int(num_h),
        'grid_nu': int(grid['nu']),
        'grid_nv': int(grid['nv']),
        'saved_debug_layers': saved_debug_layers,
        'outputs': {
            'surface_mask_nii': surface_mask_path,
            'surface_overview_png': surface_overview_png,
            'layer_debug_dir': layer_debug_dir if save_layer_debug else None,
        },
    }
    return {
        'surface_mask_arr': surface_mask_arr,
        'surface_phys': surface_phys,
        'surface_info': surface_info,
    }


# =========================================================
# Internal processing step.
# =========================================================
def project_mask_foreground_to_uv(mask_img: sitk.Image, frame: Dict[str, np.ndarray]):
    arr = sitk.GetArrayFromImage(mask_img)
    z_idx, y_idx, x_idx = np.where(arr > 0)
    if len(z_idx) == 0:
        return None

    idx_xyz = np.stack([x_idx, y_idx, z_idx], axis=1).astype(np.float64)
    phys = indices_xyz_to_physical(mask_img, idx_xyz)
    u, v, _ = physical_to_local_coords(phys, frame)
    return np.stack([u, v], axis=1)


def rasterize_uv_occupancy(points_uv: np.ndarray, grid: Dict[str, Any]):
    mask = np.zeros((grid['nv'], grid['nu']), dtype=np.uint8)
    cols = np.round((points_uv[:, 0] - grid['u_min']) / grid['res']).astype(int)
    rows = np.round((grid['v_max'] - points_uv[:, 1]) / grid['res']).astype(int)

    valid = (
        (rows >= 0) & (rows < grid['nv']) &
        (cols >= 0) & (cols < grid['nu'])
    )
    mask[rows[valid], cols[valid]] = 1
    return mask


def longest_outer_contour(binary_mask: np.ndarray):
    contours = measure.find_contours(binary_mask.astype(float), 0.5)
    if len(contours) == 0:
        return None
    lengths = [len(c) for c in contours]
    return contours[int(np.argmax(lengths))]


def build_h_interpolators(surface_phys: np.ndarray, frame: Dict[str, np.ndarray]):
    u, v, h = physical_to_local_coords(surface_phys, frame)
    pts = np.stack([u, v], axis=1)
    linear_interp = LinearNDInterpolator(pts, h, fill_value=np.nan)
    nearest_interp = NearestNDInterpolator(pts, h)
    return pts, h, linear_interp, nearest_interp


def evaluate_h_on_grid(grid: Dict[str, Any], largest_mask: np.ndarray, linear_interp, nearest_interp):
    h_grid = np.full((grid['nv'], grid['nu']), np.nan, dtype=np.float64)
    rows, cols = np.where(largest_mask > 0)
    u = grid['u_min'] + cols * grid['res']
    v = grid['v_max'] - rows * grid['res']

    hv = linear_interp(u, v)
    nan_idx = np.isnan(hv)
    if np.any(nan_idx):
        hv[nan_idx] = nearest_interp(u[nan_idx], v[nan_idx])

    h_grid[rows, cols] = hv
    return h_grid


def project_landmarks_to_uv(landmarks: Optional[Dict[str, np.ndarray]], frame: Dict[str, np.ndarray]):
    if landmarks is None:
        return None
    out = {}
    for k, p in landmarks.items():
        uu, vv, hh = physical_to_local_coords(p[None, :], frame)
        out[k] = [float(uu[0]), float(vv[0]), float(hh[0])]
    return out


def should_show_label(name: str) -> bool:
    return name == 'gc' or name.startswith('bi') or name.startswith('SCP')


def filter_display_landmarks(landmarks_uvh: Optional[Dict[str, Any]]):
    if landmarks_uvh is None:
        return None
    return {k: v for k, v in landmarks_uvh.items() if should_show_label(k)}


def plot_uvh_depth(
    h_grid: np.ndarray,
    largest_mask: np.ndarray,
    grid: Dict[str, Any],
    out_png: str,
    landmarks_uvh=None,
    gt_landmarks_uvh=None,
):
    plt.figure(figsize=(8, 7))
    show_img = h_grid.copy()
    show_img[largest_mask == 0] = np.nan

    im = plt.imshow(show_img, cmap='viridis', origin='upper')
    plt.colorbar(im, label='h (mm)')
    plt.title('u-v-h depth map')

    pred_drawn = False
    gt_drawn = False

    if landmarks_uvh is not None:
        for name, (u, v, _) in landmarks_uvh.items():
            rr, cc = uv_to_rc(u, v, grid)
            if 0 <= rr < grid['nv'] and 0 <= cc < grid['nu']:
                if not pred_drawn:
                    plt.scatter([cc], [rr], c='red', s=20, marker='o', label='Pred', zorder=3)
                    pred_drawn = True
                else:
                    plt.scatter([cc], [rr], c='red', s=20, marker='o', zorder=3)
                plt.text(cc + 0.4, rr + 0.4, name, fontsize=8, color='red')

    if gt_landmarks_uvh is not None:
        for name, (u, v, _) in gt_landmarks_uvh.items():
            rr, cc = uv_to_rc(u, v, grid)
            if 0 <= rr < grid['nv'] and 0 <= cc < grid['nu']:
                if not gt_drawn:
                    plt.scatter([cc], [rr], c='lime', s=28, marker='o', label='GT', zorder=4)
                    gt_drawn = True
                else:
                    plt.scatter([cc], [rr], c='lime', s=28, marker='o', zorder=4)
                plt.text(cc + 0.4, rr + 0.4, name, fontsize=8, color='green')

    if pred_drawn or gt_drawn:
        plt.legend(loc='best')

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_uv_mask_with_contour(uv_occ: np.ndarray, largest_mask: np.ndarray, contour_rc: np.ndarray, out_png: str):
    plt.figure(figsize=(8, 7))
    plt.imshow(uv_occ, cmap='gray', origin='upper')
    plt.title('uv occupancy and largest contour')

    if largest_mask is not None:
        rr, cc = np.where(largest_mask > 0)
        plt.scatter(cc, rr, s=1)

    if contour_rc is not None and len(contour_rc) > 0:
        plt.plot(contour_rc[:, 1], contour_rc[:, 0], color='yellow', linewidth=1.5)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def uv_to_rc(u: float, v: float, grid: Dict[str, Any]):
    col = (u - grid['u_min']) / grid['res']
    row = (grid['v_max'] - v) / grid['res']
    return row, col


# =========================================================
# Internal processing step.
# =========================================================
def get_uvh_map(
    crop_img: sitk.Image,
    frame: Dict[str, np.ndarray],
    landmarks: Dict[str, np.ndarray],
    gt_landmarks: Optional[Dict[str, np.ndarray]] = None,
    out_dir: str = '',
    case_id: str = '',
    uv_resolution_mm: float = 0.5,
    h_step_mm: float = 0.5,
    keep_largest_cc_per_slice: bool = False,
    save_layer_debug: bool = True,
    layer_debug_every: int = 1,
):
    """
        Step2 
    1)        step2   layered surface   
    2)      step3   uv-h depth map   
    3) uv-h       bi* / gc / SCP*  
    """
    step_dir = os.path.join(out_dir, 'step2_uvh_map')
    ensure_dir(step_dir)
    surface_dir = os.path.join(step_dir, 'surface_debug')
    depth_dir = os.path.join(step_dir, 'depth_map')
    ensure_dir(surface_dir)
    ensure_dir(depth_dir)

    surface_res = extract_layered_surface(
        crop_img=crop_img,
        frame=frame,
        surface_dir=surface_dir,
        case_id=case_id,
        uv_resolution_mm=uv_resolution_mm,
        h_step_mm=h_step_mm,
        keep_largest_cc_per_slice=keep_largest_cc_per_slice,
        save_layer_debug=save_layer_debug,
        layer_debug_every=layer_debug_every,
    )
    surface_phys = surface_res['surface_phys']

    proj_fg_uv = project_mask_foreground_to_uv(crop_img, frame)
    if proj_fg_uv is None or len(proj_fg_uv) == 0:
        raise RuntimeError(f'[{case_id}]        ')

    grid = build_uv_grid_info(proj_fg_uv, uv_resolution_mm=uv_resolution_mm, margin_mm=0.0)
    uv_occ = rasterize_uv_occupancy(proj_fg_uv, grid)
    largest_mask = keep_largest_connected_component(uv_occ)
    if np.count_nonzero(largest_mask) == 0:
        raise RuntimeError(f'[{case_id}] uv        ')

    contour_rc = longest_outer_contour(largest_mask)
    if contour_rc is None or len(contour_rc) < 10:
        raise RuntimeError(f'[{case_id}]        ')

    _, _, linear_interp, nearest_interp = build_h_interpolators(surface_phys, frame)
    h_grid = evaluate_h_on_grid(grid, largest_mask, linear_interp, nearest_interp)

    landmarks_uvh_all = project_landmarks_to_uv(landmarks, frame)
    gt_landmarks_uvh_all = project_landmarks_to_uv(gt_landmarks, frame)
    landmarks_uvh = filter_display_landmarks(landmarks_uvh_all)
    gt_landmarks_uvh = filter_display_landmarks(gt_landmarks_uvh_all)

    uvh_png = os.path.join(depth_dir, f'{case_id}_uvh_depth.png')
    uv_occ_png = os.path.join(depth_dir, f'{case_id}_uv_occ_and_contour.png')
    plot_uvh_depth(h_grid, largest_mask, grid, uvh_png, landmarks_uvh, gt_landmarks_uvh)
    plot_uv_mask_with_contour(uv_occ, largest_mask, contour_rc, uv_occ_png)

    step2_info = {
        'case_id': case_id,
        'surface_info': surface_res['surface_info'],
        'depth_info': {
            'grid_nu': int(grid['nu']),
            'grid_nv': int(grid['nv']),
            'uv_resolution_mm': float(uv_resolution_mm),
            'num_uv_occ_pixels': int(np.count_nonzero(uv_occ)),
            'num_largest_mask_pixels': int(np.count_nonzero(largest_mask)),
            'contour_length': int(len(contour_rc)),
            'num_pred_landmarks_displayed': int(len(landmarks_uvh)) if landmarks_uvh is not None else 0,
            'num_gt_landmarks_displayed': int(len(gt_landmarks_uvh)) if gt_landmarks_uvh is not None else 0,
            'displayed_labels_rule': 'only bi*, gc, SCP* are shown on uvh depth map',
            'outputs': {
                'uvh_depth_png': uvh_png,
                'uv_occ_and_contour_png': uv_occ_png,
            },
        },
    }
    info_json_path = os.path.join(step_dir, f'{case_id}_step2_info.json')
    save_json(step2_info, info_json_path)

    outputs = {}
    outputs.update(surface_res['surface_info'].get('outputs', {}))
    outputs.update(step2_info['depth_info']['outputs'])

    return {
        'grid': grid,
        'uv_occ': uv_occ,
        'largest_mask': largest_mask,
        'contour_rc': contour_rc,
        'h_grid': h_grid,
        'landmarks_uvh': landmarks_uvh,
        'gt_landmarks_uvh': gt_landmarks_uvh,
        'linear_interp': linear_interp,
        'nearest_interp': nearest_interp,
        'surface_phys': surface_phys,
        'surface_mask_arr': surface_res['surface_mask_arr'],
        'step2_dir': step_dir,
        'step2_info_path': info_json_path,
        'step2_outputs': outputs,
    }
