import os
import json
import numpy as np
import pandas as pd
from ast import literal_eval
import SimpleITK as sitk
import scipy

def calculate_milemeter_error(scapula_gt, scapula_pred, humerus_gt, humerus_pred, info_path, output_dir):
    """
                ID          
    1.    info         'id',  'left'  (  ),  'crop'  (  ) 
    2.           256x256x256              (Real World) 
    3.    Real World   JSON    
    4.             
    """
    
    # Internal processing step.
    if not os.path.exists(info_path):
        print(f"Error: Info file not found at {info_path}")
        return

    # Internal processing step.
    if info_path.endswith('.csv'):
        info_df = pd.read_csv(info_path)
    else:
        info_df = pd.read_excel(info_path)
    
    # Internal processing step.
    columns = info_df.columns
    
    # Internal processing step.
    id_col = "name"
    if id_col not in columns:
        print("Error: Column 'name' not found in info file.")
        return

    # Internal processing step.
    flip_col = next((c for c in columns if "flip" in c.lower()), None)
    if not flip_col:
        print("Error: Could not find a column containing 'left' for flip info.")
        return

    # Internal processing step.
    crop_col = next((c for c in columns if "crop" in c.lower()), None)
    if not crop_col:
        print("Error: Could not find a column containing 'crop' for crop info.")
        return

    try:
        info_dict = info_df.set_index(id_col).to_dict('index')
    except Exception as e:
        print(f"Critical Error creating info dictionary: {e}")
        return

    # exclude_ids = ['a25', 'a76', 'a77', 'b103', 'b144', 'b313', 'b350', 'b64', 'b74']
    exclude_ids = []
    
      
    def parse_tuple(val):
        """           """
        if isinstance(val, str):
            try:
                return literal_eval(val)
            except:
                return val
        return val

    
    # Internal processing step.
    def inverse_transform_point(points, case_info):
        """
                  
        Reverse Scale -> Reverse Flip -> Reverse Normalize
        """
        try:
            # Internal processing step.
            origin = np.array(parse_tuple(case_info.get('img_origin', case_info.get('origin'))))
            spacing = np.array(parse_tuple(case_info.get('img_spacing', case_info.get('spacing'))))
            shape_val = case_info.get('size') if 'size' in case_info else case_info.get('img_size') 
            shape = np.array(parse_tuple(shape_val))
            
            is_right = bool(case_info[flip_col]) 
            is_cropped = bool(case_info[crop_col]) 
            
        except Exception as e:
            print(f"Error parsing metadata: {e}")
            return points

        target_size = (256, 256, 256)
        
        # Scale Factors
        scale_factors = [
            target_size[0] / shape[0],
            target_size[1] / shape[1],
            target_size[2] / shape[2]
        ]
        
        # Crop Logic
        if is_cropped:
            width = 256 
            crop_width = width * 2 // 3 
            new_scale_factor_width = scale_factors[1] * (width / crop_width)
            scale_factors[0] = new_scale_factor_width
            
        scale_factors = np.array(scale_factors)
        
        restored_points = []
        for pt in points:
            new_pt = pt.copy()
            pos = np.array(pt['position'])
            
            # Step 1: Un-scale
            pos = pos / scale_factors
            
            # Step 2: Un-flip
            if is_right:
                pos[0] = 512 - pos[0]
            
            # Step 3: Un-normalize
            pos = pos * spacing + origin
            
            new_pt['position'] = pos.tolist()
            restored_points.append(new_pt)
            
        return restored_points

    # Internal processing step.
    def process_directory(input_dir, part_name, sub_type):
        if not input_dir or not os.path.exists(input_dir):
            return None
        
        save_dir = os.path.join(output_dir, "REALWORLD", sub_type, part_name)
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Restoring coordinates for {part_name} ({sub_type})...")
        
        count = 0
        for fname in os.listdir(input_dir):
            if not fname.endswith('.json'):
                continue
            case_id = fname.split('.')[0]
            
            if case_id.split('d')[0] in exclude_ids:
                continue
            
            with open(os.path.join(input_dir, fname), 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if "markups" in data and len(data["markups"]) > 0:
                control_points = data["markups"][0].get("controlPoints", [])
                real_points = inverse_transform_point(control_points, info_dict[case_id])
                data["markups"][0]["controlPoints"] = real_points
                
                save_path = os.path.join(save_dir, fname)
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4)
                count += 1
                
        print(f"  Saved {count} files to {save_dir}")
        return save_dir

    # Internal processing step.
    def compute_mm_errors(real_gt_dir, real_pred_dir, output_filename):
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
            
            with open(os.path.join(real_gt_dir, fname), 'r', encoding='utf-8') as f: gt_data = json.load(f)
            with open(pred_path, 'r', encoding='utf-8') as f: pred_data = json.load(f)
            
            gt_pts = gt_data['markups'][0]['controlPoints']
            pred_pts = pred_data['markups'][0]['controlPoints']
            
            gt_map = {p['label']: np.array(p['position']) for p in gt_pts}
            pred_map = {p['label']: np.array(p['position']) for p in pred_pts}
            
            case_errors = []
            point_errors = {}
            
            for label, p_gt in gt_map.items():
                if label in pred_map:
                    p_pred = pred_map[label]
                    dist = float(np.linalg.norm(p_gt - p_pred))
                    case_errors.append(dist)
                    point_errors[label] = dist
                    all_errors.append(dist)
                    
                    if label not in label_errors_collection:
                        label_errors_collection[label] = []
                    label_errors_collection[label].append(dist)
            
            if case_errors:
                case_id = fname.split('_')[0]
                details[case_id] = {
                    "mean_error": float(np.mean(case_errors)),
                    "std_error": float(np.std(case_errors)),
                    "point_errors": point_errors
                }
        
        summary = {}
        if all_errors:
            summary["total_mean_error"] = float(np.mean(all_errors))
            summary["total_std_error"] = float(np.std(all_errors))
            summary["total_points"] = len(all_errors)
        else:
            summary["total_mean_error"] = 0.0
            summary["total_std_error"] = 0.0
            
        summary["per_point_summary"] = {}
        for label, errs in label_errors_collection.items():
            summary["per_point_summary"][label] = {
                "mean": float(np.mean(errs)),
                "std": float(np.std(errs)),
                "count": len(errs)
            }
            
        final_output = os.path.join(output_dir, output_filename)
        with open(final_output, 'w', encoding='utf-8') as f:
            json.dump({"summary": summary, "details": details}, f, indent=4)
        print(f"Error report saved to: {final_output}")

    # Internal processing step.
    # if seg and os.path.exists(seg):
    #     separate_and_save_segmentation(seg, output_dir)
    
    real_scapula_gt_dir = process_directory(scapula_gt, "scapula", "gt")
    real_scapula_pred_dir = process_directory(scapula_pred, "scapula", "pred")
    
    if real_scapula_gt_dir and real_scapula_pred_dir:
        compute_mm_errors(real_scapula_gt_dir, real_scapula_pred_dir, "milemeter_error_scapula.json")
    
    real_humerus_gt_dir = process_directory(humerus_gt, "humerus", "gt")
    real_humerus_pred_dir = process_directory(humerus_pred, "humerus", "pred")
    
    if real_humerus_gt_dir and real_humerus_pred_dir:
        compute_mm_errors(real_humerus_gt_dir, real_humerus_pred_dir, "milemeter_error_humerus.json")
