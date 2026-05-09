import os
import json
import numpy as np
import SimpleITK as sitk
from skimage import measure


# ============================================================
# Internal processing step.
# ============================================================


def compute_mm_errors(real_gt_dir, real_pred_dir, output_filepath):
    if not real_gt_dir or not real_pred_dir:
        return

    details = {}
    all_errors = []
    label_errors_collection = {}

    files = [f for f in os.listdir(real_gt_dir) if f.endswith('.json')]
    for fname in files:
        pred_path = os.path.join(real_pred_dir, fname)
        if not os.path.exists(pred_path):
            continue

        with open(os.path.join(real_gt_dir, fname), 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        with open(pred_path, 'r', encoding='utf-8') as f:
            pred_data = json.load(f)

        gt_pts = gt_data['markups'][0]['controlPoints']
        pred_pts = pred_data['markups'][0]['controlPoints']

        gt_map = {p['label']: np.array(p['position'], dtype=np.float64) for p in gt_pts}
        pred_map = {p['label']: np.array(p['position'], dtype=np.float64) for p in pred_pts}

        case_errors = []
        point_errors = {}
        for label, p_gt in gt_map.items():
            if label in pred_map:
                dist = float(np.linalg.norm(p_gt - pred_map[label]))
                case_errors.append(dist)
                point_errors[label] = dist
                all_errors.append(dist)
                label_errors_collection.setdefault(label, []).append(dist)

        if case_errors:
            case_id = fname.split('_')[0] if '_' in fname else os.path.splitext(fname)[0]
            details[case_id] = {
                'mean_error': float(np.mean(case_errors)),
                'std_error': float(np.std(case_errors)),
                'point_errors': point_errors,
            }

    summary = {}
    if all_errors:
        summary['total_mean_error'] = float(np.mean(all_errors))
        summary['total_std_error'] = float(np.std(all_errors))
    summary['per_point_summary'] = {}
    for label, errs in label_errors_collection.items():
        summary['per_point_summary'][label] = {
            'mean': float(np.mean(errs)),
            'std': float(np.std(errs)),
            'count': len(errs),
        }

    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump({'summary': summary, 'details': details}, f, indent=4)
    print(f'  Refined Error report saved to: {output_filepath}')


# ============================================================
# Internal processing step.
# ============================================================


def _normalize(v, eps=1e-8):
    n = np.linalg.norm(v)
    if n < eps:
        return None
    return v / n



def _build_orthonormal_basis_perp_to_u(u):
    """
           u       u              e1, e2
    """
    u = _normalize(u)
    if u is None:
        return None, None

    if abs(u[0]) < 0.9:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    e1 = ref - np.dot(ref, u) * u
    e1 = _normalize(e1)
    if e1 is None:
        return None, None

    e2 = np.cross(u, e1)
    e2 = _normalize(e2)
    if e2 is None:
        return None, None

    return e1, e2



def fit_best_plane_through_line(sm, si, points_to_fit):
    """
           sm-si         points_to_fit              

      :
        plane_point:         sm 
        plane_normal:      
        success: bool
    """
    line_dir = si - sm
    u = _normalize(line_dir)
    if u is None:
        return sm, None, False

    e1, e2 = _build_orthonormal_basis_perp_to_u(u)
    if e1 is None or e2 is None:
        return sm, None, False

    A = np.zeros((2, 2), dtype=np.float64)
    for p in points_to_fit:
        q = p - sm
        x = np.dot(q, e1)
        y = np.dot(q, e2)
        v = np.array([x, y], dtype=np.float64).reshape(2, 1)
        A += v @ v.T

    eigvals, eigvecs = np.linalg.eigh(A)
    idx = np.argmin(eigvals)
    ab = eigvecs[:, idx]

    n = ab[0] * e1 + ab[1] * e2
    n = _normalize(n)
    if n is None:
        return sm, None, False

    return sm, n, True



def project_point_to_plane(point, plane_point, plane_normal):
    v = point - plane_point
    dist = np.dot(v, plane_normal)
    proj = point - dist * plane_normal
    return proj



def refine_scp_points_by_plane(control_points):
    """
       scapula    SCP6 / SCP12 / gc         
    -    sm, si
    -      sm-si      
    -   SCP6, SCP12, gc       
    """
    label_to_idx = {pt['label']: i for i, pt in enumerate(control_points)}
    required_labels = ['sm', 'si', 'SCP6', 'SCP12', 'gc']

    if not all(lbl in label_to_idx for lbl in required_labels):
        return control_points

    sm = np.array(control_points[label_to_idx['sm']]['position'], dtype=np.float64)
    si = np.array(control_points[label_to_idx['si']]['position'], dtype=np.float64)
    scp6 = np.array(control_points[label_to_idx['SCP6']]['position'], dtype=np.float64)
    scp12 = np.array(control_points[label_to_idx['SCP12']]['position'], dtype=np.float64)
    gc = np.array(control_points[label_to_idx['gc']]['position'], dtype=np.float64)

    plane_point, plane_normal, success = fit_best_plane_through_line(
        sm, si, [scp6, scp12, gc]
    )
    if not success:
        return control_points

    refined_targets = {
        'SCP6': project_point_to_plane(scp6, plane_point, plane_normal),
        'SCP12': project_point_to_plane(scp12, plane_point, plane_normal),
        'gc': project_point_to_plane(gc, plane_point, plane_normal),
    }

    refined_points = []
    for pt in control_points:
        new_pt = pt.copy()
        label = pt['label']
        if label in refined_targets:
            new_pt['position'] = refined_targets[label].tolist()
        refined_points.append(new_pt)

    return refined_points


# ============================================================
# Internal processing step.
# Internal processing step.
# Internal processing step.
# Internal processing step.
# Internal processing step.
# Internal processing step.
# ============================================================


def filter_seg_to_contour_boundary(seg_array):
    """
              mask      fliter_seg          

          type    
    - scapula / humerus   mask       
    -      > 0        

        z    
    1)         measure.find_contours(level=0.5)    
    2)      round      
    3)           3
    4)         fliter_seg   boundary==3   
    """
    fg = (seg_array > 0).astype(np.float32)
    if not np.any(fg):
        return None

    boundary = np.zeros_like(fg, dtype=np.uint8)
    z_dim, h_dim, w_dim = fg.shape

    for z in range(z_dim):
        slice_ = fg[z, :, :]
        contours = measure.find_contours(slice_, level=0.5)
        for contour in contours:
            contour = np.round(contour).astype(int)
            for point in contour:
                y, x = int(point[0]), int(point[1])
                if 0 <= y < h_dim and 0 <= x < w_dim:
                    boundary[z, y, x] = 3

    if not np.any(boundary == 3):
        return None
    return boundary



def build_surface_mask_from_mask_image(mask_img):
    """
      mask image             mask  

                fliter_seg    
    -         mask   3D     
    -      z     contour
    -    contour             

      :
        surface_mask_zyx: bool ndarray, shape (z, y, x)
    """
    arr = sitk.GetArrayFromImage(mask_img)  # (z, y, x)
    boundary = filter_seg_to_contour_boundary(arr)
    if boundary is None:
        return None
    return boundary == 3



def extract_surface_indices_from_mask(mask_img):
    """
        fliter_seg        boundary           
      :
        surface_indices_xyz: np.ndarray, shape (N, 3),     (x, y, z)
    """
    surface_mask_zyx = build_surface_mask_from_mask_image(mask_img)
    if surface_mask_zyx is None:
        return None

    surface_indices_zyx = np.argwhere(surface_mask_zyx)
    surface_indices_xyz = np.stack([
        surface_indices_zyx[:, 2],
        surface_indices_zyx[:, 1],
        surface_indices_zyx[:, 0],
    ], axis=1).astype(np.float64)

    return surface_indices_xyz



def project_physical_point_to_surface_like_softargmax(target_point_mm, mask_img, surface_indices_xyz):
    """
                          refine    

    1)                (x, y, z)
    2)                         
    3)                 

       
    -      spacing                  
    -     surface_indices_xyz    argwhere(boundary)    (x, y, z)    
    """
    if surface_indices_xyz is None or len(surface_indices_xyz) == 0:
        return list(target_point_mm)

    target_idx_xyz = mask_img.TransformPhysicalPointToContinuousIndex(tuple(target_point_mm))
    target_idx_xyz = np.array(target_idx_xyz, dtype=np.float64)

    distances = np.linalg.norm(surface_indices_xyz - target_idx_xyz, axis=1)
    closest_idx = int(np.argmin(distances))
    closest_surface_index_xyz = surface_indices_xyz[closest_idx]

    refined_point_mm = mask_img.TransformIndexToPhysicalPoint(
        (
            int(round(closest_surface_index_xyz[0])),
            int(round(closest_surface_index_xyz[1])),
            int(round(closest_surface_index_xyz[2])),
        )
    )
    return list(refined_point_mm)



def project_control_points_to_surface(control_points, mask_img, surface_indices_xyz):
    """
        control points       
                  find_center_softargmax refine    
    """
    refined_points = []
    for pt in control_points:
        new_pt = pt.copy()
        new_pt['position'] = project_physical_point_to_surface_like_softargmax(
            pt['position'], mask_img, surface_indices_xyz
        )
        refined_points.append(new_pt)
    return refined_points


# ============================================================
# Internal processing step.
# ============================================================


def _get_case_id_from_filename(fname):
    return fname.split('_')[0] if '_' in fname else os.path.splitext(fname)[0]



def _load_markup_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)



def _save_markup_json(data, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)



def _get_control_points_from_data(data):
    if 'markups' not in data or len(data['markups']) == 0:
        return []
    return data['markups'][0].get('controlPoints', [])



def _set_control_points_to_data(data, control_points):
    if 'markups' in data and len(data['markups']) > 0:
        data['markups'][0]['controlPoints'] = control_points


# ============================================================
# Internal processing step.
# ============================================================


def process_humerus_surface(gt_dir, pred_dir, mask_dir, output_root):
    """
    humerus      
    -                      
    -    SCP   
    """
    print('\n>>> Processing Surface Refinement for: humerus')

    output_pred_dir = os.path.join(output_root, 'pred_surface', 'humerus')
    os.makedirs(output_pred_dir, exist_ok=True)

    files = [f for f in os.listdir(pred_dir) if f.endswith('.json')]
    count = 0

    for fname in files:
        json_path = os.path.join(pred_dir, fname)
        save_path = os.path.join(output_pred_dir, fname)
        case_id = _get_case_id_from_filename(fname)
        mask_path = os.path.join(mask_dir, f'{case_id}.nrrd')

        data = _load_markup_json(json_path)
        control_points = _get_control_points_from_data(data)

        if len(control_points) == 0:
            _save_markup_json(data, save_path)
            count += 1
            continue

        if os.path.exists(mask_path):
            try:
                mask_img = sitk.ReadImage(mask_path)
                surface_indices_xyz = extract_surface_indices_from_mask(mask_img)
                if surface_indices_xyz is not None:
                    control_points = project_control_points_to_surface(
                        control_points, mask_img, surface_indices_xyz
                    )
                else:
                    print(f'  [Warn] Empty mask for {fname}, skipping surface projection.')
            except Exception as e:
                print(f'  [Error] Failed to process humerus file {fname}: {e}')
        else:
            print(f'  [Warn] Missing humerus mask for {fname}, skipping surface projection.')

        _set_control_points_to_data(data, control_points)
        _save_markup_json(data, save_path)
        count += 1

    print(f'  Refined {count} files saved to {output_pred_dir}')

    error_filename = os.path.join(os.path.dirname(output_root), 'milemeter_error_humerus_surface.json')
    print(f'  Calculating errors -> {os.path.basename(error_filename)}')
    compute_mm_errors(gt_dir, output_pred_dir, error_filename)



def process_scapula_surface(gt_dir, pred_dir, mask_dir, output_root, enable_scp_refine=True):
    """
    scapula      
    1)    SCP      enable_scp_refine    
    2)                     
    """
    print('\n>>> Processing Surface Refinement for: scapula')
    print(f'  SCP plane refine enabled: {enable_scp_refine}')

    output_pred_dir = os.path.join(output_root, 'pred_surface', 'scapula')
    os.makedirs(output_pred_dir, exist_ok=True)

    files = [f for f in os.listdir(pred_dir) if f.endswith('.json')]
    count = 0

    for fname in files:
        json_path = os.path.join(pred_dir, fname)
        save_path = os.path.join(output_pred_dir, fname)
        case_id = _get_case_id_from_filename(fname)
        mask_path = os.path.join(mask_dir, f'{case_id}.nii.gz')

        data = _load_markup_json(json_path)
        control_points = _get_control_points_from_data(data)

        if len(control_points) == 0:
            _save_markup_json(data, save_path)
            count += 1
            continue

        if enable_scp_refine:
            try:
                control_points = refine_scp_points_by_plane(control_points)
            except Exception as e:
                print(f'  [Warn] SCP plane refinement failed for {fname}: {e}')

        if os.path.exists(mask_path):
            try:
                mask_img = sitk.ReadImage(mask_path)
                surface_indices_xyz = extract_surface_indices_from_mask(mask_img)
                if surface_indices_xyz is not None:
                    control_points = project_control_points_to_surface(
                        control_points, mask_img, surface_indices_xyz
                    )
                else:
                    print(f'  [Warn] Empty mask for {fname}, skipping surface projection.')
            except Exception as e:
                print(f'  [Error] Failed to process scapula file {fname}: {e}')
        else:
            print(f'  [Warn] Missing scapula mask for {fname}, skipping surface projection.')

        _set_control_points_to_data(data, control_points)
        _save_markup_json(data, save_path)
        count += 1

    print(f'  Refined {count} files saved to {output_pred_dir}')

    error_filename = os.path.join(os.path.dirname(output_root), 'milemeter_error_scapula_surface.json')
    print(f'  Calculating errors -> {os.path.basename(error_filename)}')
    compute_mm_errors(gt_dir, output_pred_dir, error_filename)


# ============================================================
# Internal processing step.
# ============================================================


def pt_surface(pt_dir, mask_dir, enable_scapula_scp_refine=True):
    """
    Args:
        pt_dir: REALWORLD     
        mask_dir: mask          scapula / humerus
        enable_scapula_scp_refine:     scapula    SCP     
    """
    print('\n==============================================')
    print('>>> Starting Post-Processing: SCP-Plane + Surface Projection')
    print('==============================================')

    gt_scapula = os.path.join(pt_dir, 'gt', 'scapula')
    pred_scapula = os.path.join(pt_dir, 'pred', 'scapula')
    mask_scapula = os.path.join(mask_dir, 'scapula')

    gt_humerus = os.path.join(pt_dir, 'gt', 'humerus')
    pred_humerus = os.path.join(pt_dir, 'pred', 'humerus')
    mask_humerus = os.path.join(mask_dir, 'humerus')

    if os.path.exists(gt_scapula) and os.path.exists(pred_scapula) and os.path.exists(mask_scapula):
        process_scapula_surface(
            gt_dir=gt_scapula,
            pred_dir=pred_scapula,
            mask_dir=mask_scapula,
            output_root=pt_dir,
            enable_scp_refine=enable_scapula_scp_refine,
        )
    else:
        print('\n[Skip] Scapula processing skipped. (Missing one of: gt/pred/mask folders)')

    if os.path.exists(gt_humerus) and os.path.exists(pred_humerus) and os.path.exists(mask_humerus):
        process_humerus_surface(
            gt_dir=gt_humerus,
            pred_dir=pred_humerus,
            mask_dir=mask_humerus,
            output_root=pt_dir,
        )
    else:
        print('\n[Skip] Humerus processing skipped. (Missing one of: gt/pred/mask folders)')


__all__ = [
    'compute_mm_errors',
    'fit_best_plane_through_line',
    'project_point_to_plane',
    'refine_scp_points_by_plane',
    'filter_seg_to_contour_boundary',
    'build_surface_mask_from_mask_image',
    'extract_surface_indices_from_mask',
    'project_physical_point_to_surface_like_softargmax',
    'project_control_points_to_surface',
    'process_humerus_surface',
    'process_scapula_surface',
    'pt_surface',
]

