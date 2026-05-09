"""
    3.                    (origin/spacing/orientation )         (0-1)
"""
import openpyxl
import os
from openpyxl.utils import column_index_from_string
import SimpleITK as sitk
import numpy as np
import json
import scipy

def process_dataset(col, input_data, input_info, output_img, output_seg, output_scapula, output_humerus):
    """
    Args:
        col (_type_):                  id   
        input_data (_type_):         
        input_info (_type_):       
        output_img (_type_): image    
        output_seg (_type_): seg    
        output_scapula (_type_): points_scapula    
        output_humerus (_type_): points_humerus    
    """
    seg = os.path.exists(output_seg)
    scapula = os.path.exists(output_scapula)
    humerus = os.path.exists(output_humerus)
    
    wb = openpyxl.load_workbook(input_info)
    sheet = wb.active
    
    total_rows = 0
    for row in sheet.iter_rows(min_col=1, max_col=1, min_row=2, values_only=True):
        if row[0] is not None:
            total_rows += 1
    
    col_index = column_index_from_string(col)
    origin_col_index = column_index_from_string("F")
    spacing_col_index = column_index_from_string("H")
    left_right_col_index = column_index_from_string("B")
    crop_needed_col_index = column_index_from_string("C")
    shape_col_index = column_index_from_string("D")
    for row_idx in range(2, total_rows + 2):
        cell_value = sheet.cell(row=row_idx, column=col_index).value
        if cell_value == 1:
            print(f"Processing row {row_idx} (col '{col}' value is {cell_value})...")
            origin_str = sheet.cell(row=row_idx, column=origin_col_index).value
            spacing_str = sheet.cell(row=row_idx, column=spacing_col_index).value
            shape_str = sheet.cell(row=row_idx, column=shape_col_index).value
            is_right = sheet.cell(row=row_idx, column=left_right_col_index).value # Internal processing step.
            is_cropped = sheet.cell(row=row_idx, column=crop_needed_col_index).value # Internal processing step.
            origin = eval(origin_str)  
            spacing = eval(spacing_str)
            shape = eval(shape_str)
            # Internal processing step.
            scale_factor = 512 / shape[2]
            process_img_or_seg(input_data, output_img, row_idx, origin, spacing, is_right, is_cropped, scale_factor, 0)
            if seg:
                process_img_or_seg(input_data, output_seg, row_idx, origin, spacing, is_right, is_cropped, scale_factor, 1)
            if scapula:
                process_points(input_data, output_scapula, row_idx, origin, spacing, is_right, is_cropped, scale_factor, 0)
            if humerus:
                process_points(input_data, output_humerus, row_idx, origin, spacing, is_right, is_cropped, scale_factor, 1)
            
        else:
            print(f"Skipping row {row_idx} (col '{col}' value is {cell_value})")


def process_img_or_seg(input_data, output_data, row_idx, origin, spacing, is_right, is_cropped, scale_factor, is_seg):
    """    
       img   seg 
    1.    nrrd   
    2.   is_right   True           
    3.    origin   (0,0,0) spacing   (1,1,1)
    4.              
    5.        
    """
    # Internal processing step.
    img_type = "seg_bone.nrrd" if is_seg else "img.nrrd"
    img_path = os.path.join(input_data, str(row_idx - 1), "data", img_type)
    output_path = os.path.join(output_data, f"{row_idx - 1}.nrrd")

    if not os.path.exists(img_path):
        print(f"   {img_path}         ")
        return

    # Internal processing step.
    img = sitk.ReadImage(img_path)
    img_array = sitk.GetArrayFromImage(img)  # Internal processing step.
    metadata = img.GetMetaDataKeys()  # Internal processing step.

    # Internal processing step.
    if is_right:
        img_array = np.flip(img_array, axis=2)  # Internal processing step.

    if is_cropped:
        ...
        # Internal processing step.

    # Internal processing step.
    if not is_seg:
        img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min())

    # Internal processing step.
    img_array = scipy.ndimage.zoom(img_array, (scale_factor, 1, 1), order=1)  # Internal processing step.

    # Internal processing step.
    new_img = sitk.GetImageFromArray(img_array)

    # Internal processing step.
    for key in metadata:
        new_img.SetMetaData(key, img.GetMetaData(key))

    # Internal processing step.
    new_img.SetOrigin((0.0, 0.0, 0.0))
    new_img.SetSpacing((1.0, 1.0, 1.0))

    # Internal processing step.
    sitk.WriteImage(new_img, output_path)
    print(f"    {output_path}, change_left_right={is_right}")


def process_points(input_data, output_data, row_idx, origin, spacing, is_right, is_cropped, scale_factor, is_humerus):
    """
       points_scapula   points_humerus 
    1.    JSON   
    2.    origin   spacing       
    3.   is_right   True    X          
    4.     output_data          3D Slicer      Fiducial JSON

    Args:
        input_data (str):          
        output_data (str):        
        row_idx (int):       
        origin (tuple):    origin
        spacing (tuple):    spacing
        is_right (bool):         
        is_cropped (bool):              
        is_humerus (bool):      humerus        scapula 
    """
    # Internal processing step.
    points_type = "humerus.json" if is_humerus else "scapula.json"
    points_path = os.path.join(input_data, str(row_idx - 1), "data", points_type)
    output_path = os.path.join(output_data, f"{row_idx - 1}.json")
    output_slicer_path = os.path.join(output_data, "slicer", f"{row_idx - 1}_slicer.json")
    
    if not os.path.exists(points_path):
        print(f"   {points_path}         ")
        return

    # Internal processing step.
    with open(points_path, "r") as f:
        points_data = json.load(f)

    slicer_format = {
        "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
        "markups": [
            {
                "type": "Fiducial",
                "controlPoints": [],
                "measurements": [],
                "display": {
                    "visibility": False,
                    "opacity": 1.0,
                    "color": [0.4, 1.0, 1.0],
                    "selectedColor": [1.0, 0.5000076295109483, 0.5000076295109483],
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
                    "snapMode": "toVisibleSurface"
                }
            }
        ]
    }

    # Internal processing step.
    for idx, point in enumerate(points_data):
        # Internal processing step.
        pos = np.array(point["position"])  # pos = [x, y, z]

        # Internal processing step.
        pos = (pos - np.array(origin)) / np.array(spacing)

        # Internal processing step.
        if is_right:
            pos[0] = 512 - pos[0]  # Internal processing step.

        if is_cropped:
            ...  # Internal processing step.
        
        pos[2] = pos[2] * scale_factor  # Internal processing step.
        
        # Internal processing step.
        point["position"] = pos.tolist()

        # Internal processing step.
        slicer_format["markups"][0]["controlPoints"].append({
            "id": str(idx + 1),
            "label": point["label"],  
            "position": pos.tolist(),
            "selected": True,
            "locked": False,
            "visibility": True
        })

    # Internal processing step.
    with open(output_path, "w") as f:
        json.dump(points_data, f, indent=4)
    
    # Internal processing step.
    with open(output_slicer_path, "w") as f:
        json.dump(slicer_format, f, indent=4)

    print(f"        : {output_path}   {output_slicer_path}, change_left_right={is_right}")
    
    
    




if __name__ == '__main__':
    input_data = r"<DATA_ROOT>"
    input_info = r"<DATA_ROOT>"
    output_img = r"<DATA_ROOT>"
    output_seg = r"<DATA_ROOT>"
    output_scapula = r"<DATA_ROOT>"
    output_humerus = r"<DATA_ROOT>"
    
    process_dataset("O", input_data, input_info, output_img, output_seg, output_scapula, output_humerus)
        



