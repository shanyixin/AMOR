import os
import json
import numpy as np
import glob

def load_markups_map(json_path):
    """
       Slicer Markup JSON           { "label_name": [x, y, z] }
           label        
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Internal processing step.
        points_map = {}
        if "markups" in data and len(data["markups"]) > 0:
            control_points = data["markups"][0].get("controlPoints", [])
            for pt in control_points:
                label = pt.get("label")
                pos = pt.get("position")
                if label and pos:
                    points_map[label] = np.array(pos)
        return points_map
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return {}

def process_part_metrics(gt_dir, pred_dir, output_path):
    """
             Scapula   Humerus           
    """
    if not gt_dir or not os.path.exists(gt_dir):
        return

    # Internal processing step.
    gt_files = [f for f in os.listdir(gt_dir) if f.endswith('.json')]
    if not gt_files:
        print(f"No JSON files found in {gt_dir}")
        return

    # Internal processing step.
    # Internal processing step.
    details = {}
    # Internal processing step.
    # Internal processing step.
    label_errors_collection = {} 
    # Internal processing step.
    all_errors = []

    print(f"Processing {len(gt_files)} files for output: {os.path.basename(output_path)}")

    for file_name in gt_files:
        gt_path = os.path.join(gt_dir, file_name)
        pred_path = os.path.join(pred_dir, file_name)
        
        case_id = os.path.splitext(file_name)[0]

        if not os.path.exists(pred_path):
            print(f"  [Warning] Prediction file missing for case: {case_id}")
            continue

        # Internal processing step.
        gt_map = load_markups_map(gt_path)
        pred_map = load_markups_map(pred_path)

        case_point_errors = {}
        current_case_errors_list = []

        # Internal processing step.
        for label, p_gt_raw in gt_map.items():
            if label in pred_map:
                p_pred = pred_map[label]

                # Internal processing step.
                distance = float(np.linalg.norm(p_gt_raw - p_pred))

                # Internal processing step.
                case_point_errors[label] = distance
                current_case_errors_list.append(distance)
                all_errors.append(distance)

                # Internal processing step.
                if label not in label_errors_collection:
                    label_errors_collection[label] = []
                label_errors_collection[label].append(distance)
            else:
                # Internal processing step.
                print(f"  [Warning] Case {case_id}: Label '{label}' not found in prediction.")

        # Internal processing step.
        if current_case_errors_list:
            case_mean = float(np.mean(current_case_errors_list))
            case_std = float(np.std(current_case_errors_list))
        else:
            case_mean = 0.0
            case_std = 0.0

        details[case_id] = {
            "case_total_mean": case_mean,
            "case_total_std": case_std,
            "point_errors": case_point_errors
        }

    # Internal processing step.
    summary = {}
    
    # Internal processing step.
    if all_errors:
        summary["total_mean_error"] = float(np.mean(all_errors))
        summary["total_std_error"] = float(np.std(all_errors))
    else:
        summary["total_mean_error"] = 0.0
        summary["total_std_error"] = 0.0

    # Internal processing step.
    summary["per_point_summary"] = {}
    for label, errs in label_errors_collection.items():
        summary["per_point_summary"][label] = {
            "mean": float(np.mean(errs)),
            "std": float(np.std(errs)),
            "count": len(errs)
        }

    # Internal processing step.
    final_results = {
        "summary": summary,
        "details": details
    }

    # Internal processing step.
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4)
    
    print(f"Saved results to {output_path}")
    print(f"  Total Mean: {summary.get('total_mean_error', 0):.4f}   {summary.get('total_std_error', 0):.4f}")


def calculate_voxel_error(scapula_gt, scapula_pred, humerus_gt, humerus_pred, output):
    """
                 
         Scapula   Humerus                      
    """
    if not os.path.exists(output):
        os.makedirs(output)

    # Internal processing step.
    if scapula_gt and scapula_pred and os.path.exists(scapula_gt):
        scapula_output_file = os.path.join(output, "voxel_error_scapula.json")
        print("\n--- Calculating Scapula Voxel Errors ---")
        process_part_metrics(scapula_gt, scapula_pred, scapula_output_file)

    # Internal processing step.
    if humerus_gt and humerus_pred and os.path.exists(humerus_gt):
        humerus_output_file = os.path.join(output, "voxel_error_humerus.json")
        print("\n--- Calculating Humerus Voxel Errors ---")
        process_part_metrics(humerus_gt, humerus_pred, humerus_output_file)
