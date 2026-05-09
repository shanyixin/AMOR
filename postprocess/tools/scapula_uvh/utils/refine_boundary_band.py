import os
import csv
import json
import copy
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


# =========================================================
# Internal processing step.
# =========================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


TARGET_BI_LABELS = {f'bi{i}' for i in range(2, 10)}


def _label_is_target_bi(name: str) -> bool:
    return name in TARGET_BI_LABELS


# =========================================================
# Internal processing step.
# =========================================================
def uv_to_rc(u: float, v: float, grid: Dict[str, Any]):
    col = (u - grid['u_min']) / grid['res']
    row = (grid['v_max'] - v) / grid['res']
    return row, col


def rc_to_uv(row: float, col: float, grid: Dict[str, Any]):
    u = grid['u_min'] + col * grid['res']
    v = grid['v_max'] - row * grid['res']
    return float(u), float(v)


def local_to_physical(u: float, v: float, h: float, frame: Dict[str, np.ndarray]) -> np.ndarray:
    return frame['center'] + u * frame['u'] + v * frame['v'] + h * frame['n']


def bilinear_sample_nan(grid2d: np.ndarray, row: float, col: float):
    h, w = grid2d.shape
    if row < 0 or row > h - 1 or col < 0 or col > w - 1:
        return np.nan

    r0 = int(np.floor(row))
    c0 = int(np.floor(col))
    r1 = min(r0 + 1, h - 1)
    c1 = min(c0 + 1, w - 1)

    dr = row - r0
    dc = col - c0

    v00 = grid2d[r0, c0]
    v01 = grid2d[r0, c1]
    v10 = grid2d[r1, c0]
    v11 = grid2d[r1, c1]

    vals = np.array([v00, v01, v10, v11], dtype=np.float64)
    if np.any(np.isnan(vals)):
        return np.nan

    return (
        v00 * (1 - dr) * (1 - dc) +
        v01 * (1 - dr) * dc +
        v10 * dr * (1 - dc) +
        v11 * dr * dc
    )


def sample_h_with_fallback(h_grid: np.ndarray, row: float, col: float, fallback: Optional[float] = None) -> float:
    val = bilinear_sample_nan(h_grid, row, col)
    if np.isfinite(val):
        return float(val)

    rr = int(np.round(row))
    cc = int(np.round(col))
    h, w = h_grid.shape
    best = None
    best_d2 = None
    for rad in range(1, 5):
        r0 = max(0, rr - rad)
        r1 = min(h, rr + rad + 1)
        c0 = max(0, cc - rad)
        c1 = min(w, cc + rad + 1)
        sub = h_grid[r0:r1, c0:c1]
        ys, xs = np.where(np.isfinite(sub))
        if len(ys) == 0:
            continue
        ys = ys + r0
        xs = xs + c0
        d2 = (ys - row) ** 2 + (xs - col) ** 2
        idx = int(np.argmin(d2))
        best = float(h_grid[ys[idx], xs[idx]])
        best_d2 = float(d2[idx])
        break

    if best is not None:
        return best
    if fallback is not None:
        return float(fallback)
    return float('nan')


# =========================================================
# Internal processing step.
# =========================================================
def normalize_vec(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.zeros_like(v)
    return v / n


def build_ray_band_mask(
    gc_rc: np.ndarray,
    dir_rc: np.ndarray,
    largest_mask: np.ndarray,
    band_half_width_px: float = 1.0,
) -> Tuple[np.ndarray, float]:
    """
      gc->bi                               
          mask       
    -        t >= 0
    -          <= band_half_width_px
             t    t_end 
    """
    rr, cc = np.where(largest_mask > 0)
    if len(rr) == 0:
        return np.zeros_like(largest_mask, dtype=np.uint8), 0.0

    pts = np.stack([rr, cc], axis=1).astype(np.float64)
    rel = pts - gc_rc[None, :]
    t = rel @ dir_rc
    perp = rel - np.outer(t, dir_rc)
    d = np.linalg.norm(perp, axis=1)

    keep = (t >= 0.0) & (d <= float(band_half_width_px))
    band_mask = np.zeros_like(largest_mask, dtype=np.uint8)
    if not np.any(keep):
        return band_mask, 0.0

    sel = pts[keep].astype(int)
    band_mask[sel[:, 0], sel[:, 1]] = 1
    t_end = float(np.max(t[keep]))
    return band_mask, t_end


def select_highest_point_in_band(
    band_mask: np.ndarray,
    h_grid: np.ndarray,
    gc_rc: np.ndarray,
    dir_rc: np.ndarray,
) -> Optional[Dict[str, Any]]:
    rr, cc = np.where((band_mask > 0) & np.isfinite(h_grid))
    if len(rr) == 0:
        return None

    pts = np.stack([rr, cc], axis=1).astype(np.float64)
    rel = pts - gc_rc[None, :]
    t = rel @ dir_rc
    h_val = h_grid[rr, cc]
    score = h_val + 1e-3 * t
    idx = int(np.argmax(score))
    return {
        'row': float(rr[idx]),
        'col': float(cc[idx]),
        'h': float(h_val[idx]),
        't': float(t[idx]),
        'score': float(score[idx]),
    }


def mask_outline_from_binary(mask: np.ndarray) -> Optional[np.ndarray]:
    rr, cc = np.where(mask > 0)
    if len(rr) == 0:
        return None
    edge = mask > 0
    edge = edge ^ ndi.binary_erosion(edge)
    rr, cc = np.where(edge)
    if len(rr) == 0:
        return None
    return np.stack([rr, cc], axis=1).astype(np.float64)


def update_json_positions(raw_json: Dict[str, Any], new_positions_xyz: Dict[str, np.ndarray]) -> Dict[str, Any]:
    out = copy.deepcopy(raw_json)
    if 'markups' not in out or len(out['markups']) == 0:
        return out
    for pt in out['markups'][0].get('controlPoints', []):
        label = pt.get('label', '')
        if label in new_positions_xyz:
            pt['position'] = [float(x) for x in new_positions_xyz[label].tolist()]
    return out


# =========================================================
# Internal processing step.
# =========================================================

def plot_step3_overview(
    h_grid: np.ndarray,
    largest_mask: np.ndarray,
    grid: Dict[str, Any],
    pred_landmarks_uvh: Dict[str, List[float]],
    gt_landmarks_uvh: Optional[Dict[str, List[float]]],
    per_bi_results: List[Dict[str, Any]],
    out_png: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 8.0))

    # Internal processing step.
    show_img = np.ma.masked_where(largest_mask == 0, h_grid)
    cmap = plt.cm.get_cmap('viridis').copy()
    cmap.set_bad((1, 1, 1, 0))
    ax.imshow(show_img, cmap=cmap, origin='upper')

    pred_before_drawn = False
    pred_after_drawn = False
    gt_drawn = False
    change_line_drawn = False

    # Internal processing step.
    if gt_landmarks_uvh is not None:
        for name, (u, v, _) in gt_landmarks_uvh.items():
            rr, cc = uv_to_rc(u, v, grid)
            label_txt = str(name).upper()
            if not gt_drawn:
                ax.scatter(
                    [cc], [rr], c='lime', s=52, marker='o',
                    edgecolors='black', linewidths=0, zorder=7, label='GT'
                )
                gt_drawn = True
            else:
                ax.scatter(
                    [cc], [rr], c='lime', s=52, marker='o',
                    edgecolors='black', linewidths=0, zorder=7
                )
            # ax.text(cc + 0.5, rr - 0.5, label_txt, fontsize=8, color='lime', weight='bold', zorder=8)

    # Internal processing step.
    if pred_landmarks_uvh is not None:
        for name, (u, v, _) in pred_landmarks_uvh.items():
            rr, cc = uv_to_rc(u, v, grid)
            label_txt = str(name).upper()
            if not pred_before_drawn:
                ax.scatter(
                    [cc], [rr], c='red', s=58, marker='o',
                    edgecolors='white', linewidths=0, zorder=9, label='Pred before'
                )
                pred_before_drawn = True
            else:
                ax.scatter(
                    [cc], [rr], c='red', s=58, marker='o',
                    edgecolors='white', linewidths=0, zorder=9
                )
            # ax.text(cc + 0.5, rr + 0.6, label_txt, fontsize=8, color='red', weight='bold', zorder=10)

    # Internal processing step.
    for item in per_bi_results:
        bi_rr, bi_cc = item['bi_rc_before']
        final_rr, final_cc = item['bi_rc_after']
        label_txt = str(item['label']).upper()

        if item['changed']:
            if not change_line_drawn:
                ax.plot(
                    [bi_cc, final_cc], [bi_rr, final_rr],
                    color='magenta', linestyle='--', linewidth=1.2, zorder=5, label='Before After'
                )
                change_line_drawn = True
            else:
                ax.plot(
                    [bi_cc, final_cc], [bi_rr, final_rr],
                    color='magenta', linestyle='--', linewidth=1.2, zorder=5
                )

        if not pred_after_drawn:
            ax.scatter(
                [final_cc], [final_rr], c='magenta', s=72, marker='s',
                edgecolors='white', linewidths=0.6, zorder=11, label='Pred after'
            )
            pred_after_drawn = True
        else:
            ax.scatter(
                [final_cc], [final_rr], c='magenta', s=72, marker='s',
                edgecolors='white', linewidths=0.6, zorder=11
            )

        ax.text(final_cc + 0.5, final_rr - 0.8, label_txt, fontsize=8, color='magenta', weight='bold', zorder=12)

    ax.set_title('Step3 overview: GT / pred before / pred after', fontsize=12)
    ax.set_axis_off()
    if gt_drawn or pred_before_drawn or pred_after_drawn or change_line_drawn:
        ax.legend(loc='best', fontsize=8, framealpha=0.9)

    plt.tight_layout(pad=0.3)
    plt.savefig(out_png, dpi=220, bbox_inches='tight', pad_inches=0.05, transparent=True)
    plt.close()

def plot_single_bi_band(
    h_grid: np.ndarray,
    largest_mask: np.ndarray,
    grid: Dict[str, Any],
    gc_uvh: List[float],
    bi_uvh_before: List[float],
    candidate_boundary_rc: List[float],
    final_rc: List[float],
    band_mask: np.ndarray,
    out_png: str,
    title: str,
    gt_uvh: Optional[List[float]] = None,
    changed: bool = False,
) -> None:
    plt.figure(figsize=(7.5, 6.5))
    show_img = h_grid.copy()
    show_img[largest_mask == 0] = np.nan
    im = plt.imshow(show_img, cmap='viridis', origin='upper')
    plt.colorbar(im, label='h (mm)')
    plt.title(title)

    band_rr, band_cc = np.where(band_mask > 0)
    plt.scatter(band_cc, band_rr, s=5, c='white', alpha=0.35, label='strip band')

    gc_rr, gc_cc = uv_to_rc(gc_uvh[0], gc_uvh[1], grid)
    bi_rr, bi_cc = uv_to_rc(bi_uvh_before[0], bi_uvh_before[1], grid)
    cand_rr, cand_cc = candidate_boundary_rc
    plt.scatter([gc_cc], [gc_rr], c='cyan', s=35, marker='o', label='gc')
    plt.scatter([bi_cc], [bi_rr], c='red', s=28, marker='o', label='bi before')
    plt.plot([gc_cc, cand_cc], [gc_rr, cand_rr], color='deepskyblue', linewidth=1.3, label='gc-bi ray')
    plt.scatter([cand_cc], [cand_rr], c='yellow', s=45, marker='x', label='boundary max-h')

    if changed:
        aft_rr, aft_cc = final_rc
        plt.scatter([aft_cc], [aft_rr], c='magenta', s=40, marker='s', label='bi after')
        plt.plot([bi_cc, aft_cc], [bi_rr, aft_rr], color='magenta', linestyle='--', linewidth=1.0)

    if gt_uvh is not None:
        gt_rr, gt_cc = uv_to_rc(gt_uvh[0], gt_uvh[1], grid)
        plt.scatter([gt_cc], [gt_rr], c='lime', s=30, marker='o', label='GT')

    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


# =========================================================
# Internal processing step.
# =========================================================
def save_log_csv(per_bi_results: List[Dict[str, Any]], out_csv: str) -> None:
    fields = [
        'label', 'changed', 'reason',
        'candidate_distance_px', 'final_move_distance_px', 'move_distance_mm',
        'gt_dist_before_mm', 'gt_dist_after_mm', 'gt_delta_mm', 'gt_trend',
        'candidate_boundary_h', 'band_t_end_px', 'candidate_boundary_t_px',
        'inward_pull_px', 'settle_inward_px', 'final_h_kept_from_original', 'surface_projection_distance_mm'
    ]
    with open(out_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in per_bi_results:
            row = {k: item.get(k, None) for k in fields}
            writer.writerow(row)


# =========================================================
# Internal processing step.
# =========================================================
def refine_bi_by_gc_ray_max_h(
    grid: Dict[str, Any],
    largest_mask: np.ndarray,
    h_grid: np.ndarray,
    frame: Dict[str, np.ndarray],
    landmarks_pred_phys: Dict[str, np.ndarray],
    landmarks_pred_uvh_all: Dict[str, List[float]],
    raw_json_pred: Dict[str, Any],
    gt_landmarks_phys: Optional[Dict[str, np.ndarray]],
    surface_phys: np.ndarray,
    gt_landmarks_uvh_all: Optional[Dict[str, List[float]]],
    out_dir: str,
    case_id: str,
    band_half_width_px: float = 1.0,
    move_threshold_px: float = 6.0,
    settle_inward_px: float = 2.0,
) -> Dict[str, Any]:
    step_dir = os.path.join(out_dir, 'step3_bi_refine')
    vis_dir = os.path.join(step_dir, 'vis')
    per_bi_dir = os.path.join(step_dir, 'per_bi')
    data_dir = os.path.join(step_dir, 'data')
    ensure_dir(step_dir)
    ensure_dir(vis_dir)
    ensure_dir(per_bi_dir)
    ensure_dir(data_dir)

    if 'gc' not in landmarks_pred_uvh_all:
        raise RuntimeError(f'[{case_id}]    gc      step3 ')

    gc_uvh = landmarks_pred_uvh_all['gc']
    gc_rc = np.array(uv_to_rc(gc_uvh[0], gc_uvh[1], grid), dtype=np.float64)

    if surface_phys is None or len(surface_phys) == 0:
        raise RuntimeError(f'[{case_id}] surface_phys             ')
    surface_tree = cKDTree(surface_phys)

    new_positions_xyz: Dict[str, np.ndarray] = {}
    per_bi_results: List[Dict[str, Any]] = []

    for label in sorted(landmarks_pred_uvh_all.keys()):
        if not _label_is_target_bi(label):
            continue

        bi_uvh = landmarks_pred_uvh_all[label]
        bi_rc = np.array(uv_to_rc(bi_uvh[0], bi_uvh[1], grid), dtype=np.float64)
        dir_rc = normalize_vec(bi_rc - gc_rc)
        if np.linalg.norm(dir_rc) < 1e-8:
            continue

        band_mask, t_end = build_ray_band_mask(gc_rc, dir_rc, largest_mask, band_half_width_px=band_half_width_px)
        selected = select_highest_point_in_band(band_mask, h_grid, gc_rc, dir_rc)
        if selected is None:
            continue

        candidate_rc = np.array([selected['row'], selected['col']], dtype=np.float64)
        candidate_dist_px = float(np.linalg.norm(candidate_rc - bi_rc))
        inward_pull_px = float(np.dot(candidate_rc - bi_rc, dir_rc))

        changed = False
        reason = 'within_threshold'
        final_rc = bi_rc.copy()
        final_phys = landmarks_pred_phys[label].copy()
        final_h = bi_uvh[2]

        # Internal processing step.
        if inward_pull_px <= 0:
            reason = 'skip_inward_or_same'
        elif candidate_dist_px <= float(move_threshold_px):
            reason = 'within_threshold'
        else:
            reason = 'moved_outward'
            changed = True
            back_dir = normalize_vec(bi_rc - candidate_rc)
            final_rc = candidate_rc + float(settle_inward_px) * back_dir
            final_uv = rc_to_uv(final_rc[0], final_rc[1], grid)
            # Internal processing step.
            final_h = float(bi_uvh[2])
            provisional_phys = local_to_physical(final_uv[0], final_uv[1], final_h, frame)
            _, proj_idx = surface_tree.query(provisional_phys, k=1)
            final_phys = surface_phys[int(proj_idx)].astype(np.float64)
            new_positions_xyz[label] = final_phys

        move_distance_mm = float(np.linalg.norm(final_phys - landmarks_pred_phys[label]))
        final_move_distance_px = float(np.linalg.norm(final_rc - bi_rc)) if changed else 0.0

        gt_dist_before_mm = None
        gt_dist_after_mm = None
        gt_delta_mm = None
        gt_trend = 'NA'
        if gt_landmarks_phys is not None and label in gt_landmarks_phys:
            gt_phys = gt_landmarks_phys[label]
            gt_dist_before_mm = float(np.linalg.norm(landmarks_pred_phys[label] - gt_phys))
            gt_dist_after_mm = float(np.linalg.norm(final_phys - gt_phys))
            gt_delta_mm = float(gt_dist_after_mm - gt_dist_before_mm)
            if gt_delta_mm < -1e-6:
                gt_trend = 'decrease'
            elif gt_delta_mm > 1e-6:
                gt_trend = 'increase'
            else:
                gt_trend = 'same'

        band_outline = mask_outline_from_binary(band_mask)
        item = {
            'label': label,
            'changed': changed,
            'reason': reason,
            'bi_rc_before': [float(bi_rc[0]), float(bi_rc[1])],
            'bi_rc_after': [float(final_rc[0]), float(final_rc[1])],
            'candidate_boundary_rc': [float(candidate_rc[0]), float(candidate_rc[1])],
            'candidate_distance_px': candidate_dist_px,
            'final_move_distance_px': final_move_distance_px,
            'move_distance_mm': move_distance_mm,
            'candidate_boundary_h': float(selected['h']),
            'band_t_end_px': float(t_end),
            'candidate_boundary_t_px': float(selected['t']),
            'inward_pull_px': inward_pull_px,
            'settle_inward_px': float(settle_inward_px),
            'gt_dist_before_mm': gt_dist_before_mm,
            'gt_dist_after_mm': gt_dist_after_mm,
            'gt_delta_mm': gt_delta_mm,
            'gt_trend': gt_trend,
            'pred_xyz_before': landmarks_pred_phys[label].tolist(),
            'pred_xyz_after': final_phys.tolist(),
            'surface_projection_distance_mm': float(np.linalg.norm(local_to_physical(rc_to_uv(final_rc[0], final_rc[1], grid)[0], rc_to_uv(final_rc[0], final_rc[1], grid)[1], final_h, frame) - final_phys)) if changed else 0.0,
            'candidate_uvh': [float(rc_to_uv(candidate_rc[0], candidate_rc[1], grid)[0]), float(rc_to_uv(candidate_rc[0], candidate_rc[1], grid)[1]), float(selected['h'])],
            'final_uvh_after': [float(rc_to_uv(final_rc[0], final_rc[1], grid)[0]), float(rc_to_uv(final_rc[0], final_rc[1], grid)[1]), float(final_h)],
            'final_h_kept_from_original': float(final_h),
            'band_outline_rc': band_outline.tolist() if band_outline is not None else None,
        }
        per_bi_results.append(item)

        single_png = os.path.join(per_bi_dir, f'{case_id}_{label}_band.png')
        gt_uvh = gt_landmarks_uvh_all.get(label, None) if gt_landmarks_uvh_all is not None else None
        plot_single_bi_band(
            h_grid=h_grid,
            largest_mask=largest_mask,
            grid=grid,
            gc_uvh=gc_uvh,
            bi_uvh_before=bi_uvh,
            candidate_boundary_rc=item['candidate_boundary_rc'],
            final_rc=item['bi_rc_after'],
            band_mask=band_mask,
            out_png=single_png,
            title=f'{label}: strip max-h boundary',
            gt_uvh=gt_uvh,
            changed=changed,
        )
        item['single_vis_png'] = single_png

    updated_json = update_json_positions(raw_json_pred, new_positions_xyz)
    updated_json_path = os.path.join(data_dir, f'{case_id}_pred_refined.json')
    save_json(updated_json, updated_json_path)

    log_json_path = os.path.join(data_dir, f'{case_id}_bi_refine_log.json')
    log_csv_path = os.path.join(data_dir, f'{case_id}_bi_refine_log.csv')
    save_json({'case_id': case_id, 'per_bi_results': per_bi_results}, log_json_path)
    save_log_csv(per_bi_results, log_csv_path)

    overview_png = os.path.join(vis_dir, f'{case_id}_step3_overview.png')
    gt_subset = None
    if gt_landmarks_uvh_all is not None:
        gt_subset = {k: v for k, v in gt_landmarks_uvh_all.items() if _label_is_target_bi(k) or k == 'gc'}
    pred_subset = {k: v for k, v in landmarks_pred_uvh_all.items() if _label_is_target_bi(k) or k == 'gc'}
    plot_step3_overview(
        h_grid=h_grid,
        largest_mask=largest_mask,
        grid=grid,
        pred_landmarks_uvh=pred_subset,
        gt_landmarks_uvh=gt_subset,
        per_bi_results=per_bi_results,
        out_png=overview_png,
    )

    step3_info = {
        'case_id': case_id,
        'mode': 'only bi2-bi9; strip search ignores path gaps; if candidate is outward and farther than threshold, final point stays 2 px inside candidate along bi-candidate line',
        'band_half_width_px': float(band_half_width_px),
        'move_threshold_px': float(move_threshold_px),
        'settle_inward_px': float(settle_inward_px),
        'num_bi_points': int(len(per_bi_results)),
        'num_changed_bi_points': int(sum(int(x['changed']) for x in per_bi_results)),
        'outputs': {
            'overview_png': overview_png,
            'updated_json': updated_json_path,
            'log_json': log_json_path,
            'log_csv': log_csv_path,
            'per_bi_dir': per_bi_dir,
        },
    }
    info_json_path = os.path.join(step_dir, f'{case_id}_step3_info.json')
    save_json(step3_info, info_json_path)

    return {
        'updated_json_path': updated_json_path,
        'per_bi_results': per_bi_results,
        'step3_dir': step_dir,
        'step3_info_path': info_json_path,
        'step3_outputs': step3_info['outputs'],
    }
