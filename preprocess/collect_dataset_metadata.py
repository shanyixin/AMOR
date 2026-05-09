"""
    1.                           spacing origin orientation                     
"""

import os
import json
import pandas as pd
import openpyxl
import SimpleITK as sitk

def save_json(data, path):
    """    JSON    """
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Saved JSON: {path}")
    except Exception as e:
        print(f"Error saving {path}: {e}")

def get_nrrd_info(nrrd_path):
    try:
        image = sitk.ReadImage(nrrd_path)
        size = image.GetSize()  # Internal processing step.
        spacing = image.GetSpacing()  # Internal processing step.
        origin = image.GetOrigin()
        orientation = image.GetDirection()
        return size, spacing, origin, orientation
    except Exception as e:
        print(f"Error reading {nrrd_path}: {e}")
        return None, None, None, None

def get_landmarks(json_path, type):
    """
    Args:
        json_path
        type: 0 - scapula; 1 - humerus
    Returns:

    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            points = data.get("markups", [])[0].get("controlPoints", [])
            points_num = 0
            points_list = []
            for point in points:
                label = point.get("label", "")
                position = point.get("position", None)
                if position and isinstance(position, list) and len(position)==3:
                    points_num += 1
                    points_list.append({"label":label, "position":position})              
            return points_num, points_list
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return None

def process_dataset(root_folder, output_excel):
    wb = openpyxl.load_workbook(output_excel)  # Internal processing step.
    ws = wb.active
    
    headers = ["img_size", "seg_size", "img_origin", "seg_origin",
            "img_spacing", "seg_spacing", "img_orientation", "seg_orientation",
            "points_num_numerus", "points_num_scapula"]
    
    for i in range(1, 106):  # Internal processing step.
        folder_path = os.path.join(root_folder, str(i))
        if not os.path.exists(folder_path):
            print(f"Skipping missing folder: {folder_path}")
            continue
        
        img_path = os.path.join(folder_path, "data", "img.nrrd")
        seg_path = os.path.join(folder_path, "data", "seg_bone.nrrd")
        scapula_path = os.path.join(folder_path, "scapula.mrk.json")
        if not os.path.exists(scapula_path):
                scapula_path = os.path.join(folder_path, "gleniod.mrk.json")
        humerus_path = os.path.join(folder_path, "humerus.mrk.json")
        
        img_size, img_spacing, img_origin, img_orientation = get_nrrd_info(img_path) if os.path.exists(img_path) else (None, None, None, None)
        seg_size, seg_spacing, seg_origin, seg_orientation = get_nrrd_info(seg_path) if os.path.exists(seg_path) else (None, None, None, None)
        scapula_landmarks_num, scapula_landmarks_dict  = get_landmarks(scapula_path, 0) if os.path.exists(scapula_path) else None
        humerus_landmarks_num, humerus_landmarks_dict = get_landmarks(humerus_path, 1) if os.path.exists(humerus_path) else None
        
        data_folder = os.path.join(folder_path, "data")
        # Internal processing step.
        if scapula_landmarks_dict is not None:
            save_json(scapula_landmarks_dict, os.path.join(data_folder, "scapula.json"))
        if humerus_landmarks_dict is not None:
            save_json(humerus_landmarks_dict, os.path.join(data_folder, "humerus.json"))
        
        row_index = i + 1
        
        # Internal processing step.
        col_index = 4  # Internal processing step.
        for value in [img_size, seg_size, img_origin, seg_origin,
                      img_spacing, seg_spacing, img_orientation, seg_orientation,
                      scapula_landmarks_num, humerus_landmarks_num]:
            ws.cell(row=row_index, column=col_index, value=str(value))
            col_index += 1
        
    wb.save(output_excel)
    print(f"       {output_excel}")

if __name__ == "__main__":
    root_dir = r"<DATA_ROOT>"
    output_file = r"<DATA_ROOT>"
    process_dataset(root_dir, output_file)


