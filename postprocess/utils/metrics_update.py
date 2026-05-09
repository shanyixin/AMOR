# Internal processing step.
# Internal processing step.
# Internal processing step.
# Internal processing step.
# Internal processing step.
# Internal processing step.
# Internal processing step.
# Internal processing step.
# Internal processing step.
# Internal processing step.
# Internal processing step.
# Internal processing step.
# Internal processing step.
# Internal processing step.
# Internal processing step.
# Internal processing step.
# Internal processing step.
# Internal processing step.
# Internal processing step.
# Internal processing step.
# Internal processing step.
# Internal processing step.
# Internal processing step.
# Internal processing step.
# Internal processing step.
# 
# Internal processing step.
# Internal processing step.
# Internal processing step.

import os
import json
import numpy as np
import math
from scipy.optimize import least_squares

class GeometryUtils:
    """
            (    )
    """
    @staticmethod
    def distance(p1, p2):
        return np.linalg.norm(p1 - p2)

    @staticmethod
    def angle_between_vectors(v1, v2):
        v1_u = v1 / (np.linalg.norm(v1) + 1e-8)
        v2_u = v2 / (np.linalg.norm(v2) + 1e-8)
        dot_product = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
        return np.degrees(np.arccos(dot_product))

    @staticmethod
    def project_point_on_plane(point, plane_point, plane_normal):
        plane_normal = plane_normal / (np.linalg.norm(plane_normal) + 1e-8)
        v = point - plane_point
        dist = np.dot(v, plane_normal)
        return point - dist * plane_normal

    @staticmethod
    def project_vector_on_plane(vector, plane_normal):
        plane_normal = plane_normal / (np.linalg.norm(plane_normal) + 1e-8)
        proj = vector - np.dot(vector, plane_normal) * plane_normal
        return proj

    @staticmethod
    def fit_circle_3d(p1, p2, p3):
        x1, x2, x3 = p1, p2, p3
        v12 = x2 - x1
        v13 = x3 - x1
        normal = np.cross(v12, v13)
        normal = normal / (np.linalg.norm(normal) + 1e-8)
        
        m12 = (x1 + x2) / 2
        m13 = (x1 + x3) / 2
        
        dir12 = np.cross(v12, normal)
        dir13 = np.cross(v13, normal)
        
        A = np.vstack((dir12, -dir13)).T
        res = np.linalg.lstsq(A, m13 - m12, rcond=None)
        a = res[0][0]
        
        center = m12 + a * dir12
        radius = np.linalg.norm(center - p1)
        return center, radius, normal

    @staticmethod
    def fit_sphere(points):
        p_mean = np.mean(points, axis=0)
        def residuals(c, x):
            r = np.linalg.norm(x - c, axis=1)
            return r - np.mean(r)
        res = least_squares(residuals, p_mean, args=(points,))
        center = res.x
        radius = np.mean(np.linalg.norm(points - center, axis=1))
        return center, radius

    @staticmethod
    def polygon_area_3d(points, normal):
        z_axis = np.array([0, 0, 1])
        axis = np.cross(normal, z_axis)
        if np.linalg.norm(axis) < 1e-8:
            pts_2d = points[:, :2]
        else:
            u = np.cross(normal, np.array([1, 0, 0]))
            if np.linalg.norm(u) < 1e-5: u = np.cross(normal, np.array([0, 1, 0]))
            u = u / np.linalg.norm(u)
            v = np.cross(normal, u)
            pts_2d = np.zeros((len(points), 2))
            for i, p in enumerate(points):
                pts_2d[i, 0] = np.dot(p - points[0], u)
                pts_2d[i, 1] = np.dot(p - points[0], v)
        x = pts_2d[:, 0]
        y = pts_2d[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    
    @staticmethod
    def fit_plane_svd(points):
        """[  ] SVD       (    Sagi )"""
        if len(points) < 3: return None, None
        centroid = np.mean(points, axis=0)
        u, s, vh = np.linalg.svd(points - centroid)
        normal = vh[2, :]
        return normal / np.linalg.norm(normal), centroid

    @staticmethod
    def fit_circle_3d_robust(points):
        """[  ]         3D  (  r1)"""
        if len(points) < 3: return None, 0, None
        # Internal processing step.
        normal, centroid = GeometryUtils.fit_plane_svd(points)
        
        # Internal processing step.
        z_axis = normal
        x_axis = np.cross(normal, np.array([0,0,1]))
        if np.linalg.norm(x_axis) < 1e-6: x_axis = np.cross(normal, np.array([0,1,0]))
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        
        # Internal processing step.
        vecs = points - centroid
        u = np.dot(vecs, x_axis)
        v = np.dot(vecs, y_axis)
        
        # Internal processing step.
        def calc_R(xc, yc):
            return np.sqrt((u-xc)**2 + (v-yc)**2)

        def f_2(c):
            Ri = calc_R(*c)
            return Ri - Ri.mean()

        center_estimate = np.mean(u), np.mean(v)
        try:
            from scipy.optimize import least_squares # Internal processing step.
            center_2d = least_squares(f_2, center_estimate).x
            u_c, v_c = center_2d
            radius = calc_R(u_c, v_c).mean()
            center_3d = centroid + u_c * x_axis + v_c * y_axis
            return center_3d, radius, normal
        except:
            return None, 0, None

def get_plane_normal(p1, p2, p3):
    v1 = p2 - p1
    v2 = p3 - p1
    n = np.cross(v1, v2)
    return n / (np.linalg.norm(n) + 1e-8)

def calculate_single_case_metrics(scapula_pts, humerus_pts=None):
    """
               
    """
    metrics = {}
    
    def p(label):
        if label not in scapula_pts: return None
        return np.array(scapula_pts[label])
    
    def ph(label):
        if humerus_pts is None or label not in humerus_pts: return None
        return np.array(humerus_pts[label])

    pts_labels = ['bi2', 'bi3', 'bi4', 'bi5', 'bi6', 'bi9', 'bi12', 'ss', 'sm', 'si', 'gc', 'c', 'AA', 'AP', 'SCP6', 'SCP12']
    P = {l: p(l) for l in pts_labels}
    
    # Internal processing step.
    if P['si'] is not None and P['ss'] is not None:
        metrics['Scapular_Length'] = GeometryUtils.distance(P['si'], P['ss'])
    
    if P['sm'] is not None and P['bi6'] is not None:
        metrics['Scapular_Width'] = GeometryUtils.distance(P['sm'], P['bi6'])
        
    if P['bi6'] is not None and P['bi12'] is not None:
        metrics['Scapula_Height'] = GeometryUtils.distance(P['bi6'], P['bi12'])
        
    if P['bi3'] is not None and P['bi9'] is not None:
        metrics['Scapula_Width'] = GeometryUtils.distance(P['bi3'], P['bi9'])
        
    if all(P[k] is not None for k in ['bi6', 'bi12', 'si']):
        vec1 = P['bi6'] - P['bi12']
        vec2 = P['si'] - P['bi12']
        metrics['GPA'] = GeometryUtils.angle_between_vectors(vec1, vec2)
        
    if all(P[k] is not None for k in ['sm', 'gc', 'bi9', 'bi3']):
        metrics['Scapula_Version'] = GeometryUtils.angle_between_vectors(P['sm'] - P['gc'], P['bi9'] - P['bi3'])
        
    if all(P[k] is not None for k in ['bi2', 'bi6', 'bi9']):
        norm_sagi_alpha = get_plane_normal(P['bi2'], P['bi6'], P['bi9'])
        if all(P[k] is not None for k in ['AP', 'gc', 'c']):
            v_AP_gc = P['AP'] - P['gc']
            v_c_gc = P['c'] - P['gc']
            v_AP_proj = GeometryUtils.project_vector_on_plane(v_AP_gc, norm_sagi_alpha)
            v_c_proj = GeometryUtils.project_vector_on_plane(v_c_gc, norm_sagi_alpha)
            metrics['Alpha_Angle'] = GeometryUtils.angle_between_vectors(v_AP_proj, v_c_proj)
            
    if all(P[k] is not None for k in ['bi3', 'bi6', 'bi9']):
        norm_sagl = get_plane_normal(P['bi3'], P['bi6'], P['bi9'])
        if all(P[k] is not None for k in ['AP', 'gc', 'si']):
            v_AP_gc = P['AP'] - P['gc']
            v_si_gc = P['si'] - P['gc']
            v_AP_proj = GeometryUtils.project_vector_on_plane(v_AP_gc, norm_sagl)
            v_si_proj = GeometryUtils.project_vector_on_plane(v_si_gc, norm_sagl)
            metrics['Beta_Angle'] = GeometryUtils.angle_between_vectors(v_AP_proj, v_si_proj)

    if all(P[k] is not None for k in ['gc', 'sm', 'si']):
        norm_cora = get_plane_normal(P['gc'], P['sm'], P['si'])
        if all(P[k] is not None for k in ['bi6', 'bi12']):
            v_sm_gc = P['sm'] - P['gc'] 
            v_l1 = P['bi6'] - P['bi12'] 
            v_l1_proj = GeometryUtils.project_vector_on_plane(v_l1, norm_cora)
            metrics['Scapula_Inclination'] = GeometryUtils.angle_between_vectors(v_sm_gc, v_l1_proj)

        if P['AA'] is not None and all(P[k] is not None for k in ['bi6', 'bi12']):
            v_l16 = P['AA'] - P['bi6'] 
            v_l1 = P['bi6'] - P['bi12']
            v_l16_proj = GeometryUtils.project_vector_on_plane(v_l16, norm_cora)
            v_l1_proj = GeometryUtils.project_vector_on_plane(v_l1, norm_cora)
            metrics['CSA'] = GeometryUtils.angle_between_vectors(v_l16_proj, v_l1_proj)

    # Internal processing step.
    
    sagi_normal_new = None
    fit_points_new = []
    for k_pt, v_pt in P.items():
        if (k_pt.startswith('bi') or k_pt.startswith('SCP')) and v_pt is not None:
            fit_points_new.append(v_pt)
    sagi_normal_new, _ = GeometryUtils.fit_plane_svd(np.array(fit_points_new))
    
    # Internal processing step.
    if all(P[k] is not None for k in ['bi2', 'bi9', 'bi6']):
        
        rim_points_list = []
        for i in range(2, 9):
            l = f"bi{i}"
            if l in P and P[l] is not None: 
                rim_points_list.append(P[l])
        center_O, r1, _ = GeometryUtils.fit_circle_3d_robust(np.array(rim_points_list))
        area1 = np.pi * (r1 ** 2)
        
        # Internal processing step.
        metrics['r1'] = r1
        metrics['area1'] = area1
        
        if all(P[k] is not None for k in ['bi3', 'bi4', 'bi5', 'gc']):
            
            def get_proj_len(p_start, p_end, plane_n):
                v = p_end - p_start
                v_proj = GeometryUtils.project_vector_on_plane(v, plane_n)
                return np.linalg.norm(v_proj)
            
            l11 = get_proj_len(P['bi3'], P['gc'], sagi_normal_new)
            l12 = get_proj_len(P['bi4'], P['gc'], sagi_normal_new)
            l13 = get_proj_len(P['bi5'], P['gc'], sagi_normal_new)
            
            # Internal processing step.
            metrics['l11'] = l11
            metrics['l12'] = l12
            metrics['l13'] = l13
            
            metrics['Bone_Defect_Linear'] = max((r1 - l11)/(2*r1), (r1 - l12)/(2*r1), (r1 - l13)/(2*r1))
            
            poly_points = [P['bi2'], P['bi3'], P['bi4'], P['bi5'], P['bi6'], center_O]
            poly_proj = [GeometryUtils.project_point_on_plane(pt, P['bi2'], sagi_normal_new) for pt in poly_points]
            poly_proj_np = np.array(poly_proj)
            
            area2 = GeometryUtils.polygon_area_3d(poly_proj_np, sagi_normal_new)
            
            # Internal processing step.
            metrics['area2'] = area2
            
            metrics['Bone_Defect_Area'] = ( (2/3)*area1 + area2 ) / area1

    # Internal processing step.
    sphere_points_list = []
    if P['gc'] is not None:
        sphere_points_list.append(P['gc'])
    if P['SCP6'] is not None:
        sphere_points_list.append(P['SCP6'])
    for i in range(2, 10):
        l = f"bi{i}"
        if l in P and P[l] is not None:
            sphere_points_list.append(P[l])
            
    r2 = None
    # Internal processing step.
    if len(sphere_points_list) >= 4:
        _, r2 = GeometryUtils.fit_sphere(np.array(sphere_points_list))
    
    if r2 is not None:
        metrics['r2'] = r2
        
        # Internal processing step.
        if all(k in P and P[k] is not None for k in ['bi3', 'bi9', 'gc', 'sm']):
            # Internal processing step.
            l4 = GeometryUtils.distance(P['bi3'], P['bi9'])
            
            # Internal processing step.
            v_l4 = P['bi9'] - P['bi3']
            v_l3 = P['gc'] - P['sm']
            beta_sr = np.radians(GeometryUtils.angle_between_vectors(v_l4, v_l3))
            
            # Internal processing step.
            denom = np.sqrt(abs(4 * r2**2 - l4**2))
            if denom != 0:
                term1 = np.arctan(l4 / denom)
                metrics['Stability_Ratio'] = term1 + beta_sr - np.pi/2

    # Internal processing step.
    if humerus_pts is not None:
        h_labels = ['HS', 'HI', 'HA', 'HP', 'HC']
        if all(ph(l) is not None for l in h_labels):
            h_points = np.array([ph(l) for l in h_labels])
            center_S, radius_S = GeometryUtils.fit_sphere(h_points)
            
            if all(P[k] is not None for k in ['bi2', 'bi6', 'bi9', 'gc']):
                norm_sagi = get_plane_normal(P['bi2'], P['bi6'], P['bi9'])
                dist_S_sagi = np.dot(center_S - P['gc'], norm_sagi)
                metrics['Humeral_Center_Offset_Sagi'] = dist_S_sagi

            if P['AA'] is not None:
                dist_S_AA = GeometryUtils.distance(center_S, P['AA'])
                metrics['AHD'] = dist_S_AA - radius_S

    # Internal processing step.
    if all(P[k] is not None for k in ['sm', 'gc', 'ss', 'si', 'bi2', 'bi6', 'bi9']):
        norm_sagi = get_plane_normal(P['bi2'], P['bi6'], P['bi9'])
        
        vec_line = P['gc'] - P['sm']
        len_line = np.linalg.norm(vec_line)
        vec_line_u = vec_line / (len_line + 1e-8)
        
        best_angle = 180
        best_p = P['sm']
        
        for t in np.linspace(0, len_line, 100):
            pt = P['sm'] + t * vec_line_u
            curr_n = get_plane_normal(P['ss'], pt, P['si'])
            ang = GeometryUtils.angle_between_vectors(curr_n, norm_sagi)
            if ang > 90: ang = 180 - ang 
            
            if ang < best_angle:
                best_angle = ang
                best_p = pt
        
        vec_pss = P['ss'] - best_p
        vec_psi = P['si'] - best_p
        metrics['SCA'] = GeometryUtils.angle_between_vectors(vec_pss, vec_psi)

    return metrics

def calculate_ClinicalMetrics_error(realworld_dir, output_base, exclude_case_prefixes=None):
    """
       :    RealWorld JSON          
    """
    gt_scapula_dir = os.path.join(realworld_dir, 'gt', 'scapula')
    pred_scapula_dir = os.path.join(realworld_dir, 'pred_surface', 'scapula') if os.path.exists(os.path.join(realworld_dir, 'pred_surface', 'scapula')) else os.path.join(realworld_dir, 'pred', 'scapula')
    gt_humerus_dir = os.path.join(realworld_dir, 'gt', 'humerus')
    pred_humerus_dir = os.path.join(realworld_dir, 'pred_surface', 'humerus') if os.path.exists(os.path.join(realworld_dir, 'pred_surface', 'humerus')) else os.path.join(realworld_dir, 'pred', 'humerus')
    
    output_path = os.path.join(output_base, 'clinical_metrics_results.json')
    
    if not os.path.exists(output_base):
        os.makedirs(output_base)

    def load_points(json_path):
        if not os.path.exists(json_path): return None
        with open(json_path, 'r') as f: data = json.load(f)
        pts = {}
        if 'markups' in data:
            for p in data['markups'][0]['controlPoints']:
                pts[p['label']] = np.array(p['position'])
        return pts

    files = [f for f in os.listdir(gt_scapula_dir) if f.endswith('.json')]
    case_ids = []
    if exclude_case_prefixes is None:
        exclude_case_prefixes = ["a24", "a26", "a99", "b350"]
    exclude_case_prefixes = [str(x).lower() for x in exclude_case_prefixes]

    def is_excluded_case(case_id):
        case_id_lower = str(case_id).lower()
        return any(case_id_lower.startswith(prefix) for prefix in exclude_case_prefixes)
    
    for f in files:
        if "_" in f: case_ids.append(f.split('_')[0])
        else: case_ids.append(os.path.splitext(f)[0])
    
    results = {"summary": {}, "details": {}}
    metric_errors = {}
    
    print(f"Starting Clinical Metrics Calculation for {len(case_ids)} cases...")

    excluded_cases = []
    for case_id in case_ids:
        if is_excluded_case(case_id):
            excluded_cases.append(case_id)
            continue
        g_name = f"{case_id}_scapula.json"
        if not os.path.exists(os.path.join(gt_scapula_dir, g_name)):
             g_name = f"{case_id}.json"
        
        h_name = f"{case_id}.json"
        
        gt_g_pts = load_points(os.path.join(gt_scapula_dir, g_name))
        gt_h_pts = load_points(os.path.join(gt_humerus_dir, h_name))
        pred_g_pts = load_points(os.path.join(pred_scapula_dir, g_name))
        pred_h_pts = load_points(os.path.join(pred_humerus_dir, h_name))
        
        if gt_g_pts is None or pred_g_pts is None:
            continue
            
        m_gt = calculate_single_case_metrics(gt_g_pts, gt_h_pts)
        m_pred = calculate_single_case_metrics(pred_g_pts, pred_h_pts)
        
        case_res = {}
        for key in m_gt:
            if key in m_pred:
                val_gt = m_gt[key]
                val_pred = m_pred[key]
                
                if np.isnan(val_gt) or np.isnan(val_pred):
                    continue
                    
                abs_err = abs(val_gt - val_pred)
                rel_err = abs_err / (abs(val_gt) + 1e-6)
                
                # Internal processing step.
                denom = (abs(val_gt) + abs(val_pred)) / 2.0
                smape = abs_err / (denom + 1e-8)
                
                case_res[key] = {
                    "gt": float(val_gt),
                    "pred": float(val_pred),
                    "abs_error": float(abs_err),
                    "rel_error": float(rel_err),
                    "smape": float(smape)
                }
                
                if key not in metric_errors: metric_errors[key] = {"abs": [], "rel": [], "smape": []}
                metric_errors[key]["abs"].append(abs_err)
                metric_errors[key]["rel"].append(rel_err)
                metric_errors[key]["smape"].append(smape)
                
        results["details"][case_id] = case_res

    print("\n--- Clinical Metrics Summary ---")
    
    def print_metric_summary(key, errs):
        mean_abs = np.mean(errs["abs"])
        std_abs = np.std(errs["abs"])
        mean_rel = np.mean(errs["rel"])
        mean_smape = np.mean(errs["smape"])
        
        results["summary"][key] = {
            "mean_abs_error": float(mean_abs),
            "std_abs_error": float(std_abs),
            "mean_rel_error": float(mean_rel),
            "mean_smape": float(mean_smape)
        }
        print(f"{key:<30}: MAE={mean_abs:.4f} {std_abs:.4f}, MRE={mean_rel*100:.2f}%, SMAPE={mean_smape*100:.2f}%")

    # Internal processing step.
    special_keys = ['r1', 'r2', 'area1', 'area2', 'l11', 'l12', 'l13']
    
    # Internal processing step.
    for key, errs in metric_errors.items():
        if key not in special_keys:
            print_metric_summary(key, errs)
            
    # Internal processing step.
    print("-" * 80)
    for key in special_keys:
        if key in metric_errors:
            print_metric_summary(key, metric_errors[key])
        
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved clinical metrics to {output_path}")
