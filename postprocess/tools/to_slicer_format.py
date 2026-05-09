import os
import json
import shutil
from pathlib import Path


def is_slicer_markup(data):
    """
            Slicer Markups JSON   
    """
    return isinstance(data, dict) and "markups" in data


def build_slicer_markup(points):
    """
           :
    [
        {"label": "ss", "position": [x, y, z]},
        ...
    ]
       Slicer Markups JSON   
    """
    control_points = []
    for i, pt in enumerate(points, start=1):
        label = pt.get("label", f"point_{i}")
        position = pt.get("position", [0.0, 0.0, 0.0])

        # Internal processing step.
        if not isinstance(position, list) or len(position) != 3:
            raise ValueError(f"  {label}   position   : {position}")

        control_points.append({
            "id": f"point_{i}",
            "label": label,
            "position": [float(position[0]), float(position[1]), float(position[2])],
            "selected": True,
            "locked": False,
            "visibility": True
        })

    slicer_data = {
        "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
        "markups": [
            {
                "type": "Fiducial",
                "coordinateSystem": "LPS",
                "locked": False,
                "labelFormat": "%N-%d",
                "controlPoints": control_points,
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
                    "snapMode": "toVisibleSurface"
                }
            }
        ]
    }
    return slicer_data


def convert_one_json(input_json_path, output_json_path=None, overwrite=True):
    """
         json   
    -       Slicer         
    -               Slicer   
    """
    input_json_path = Path(input_json_path)

    if output_json_path is None:
        output_json_path = input_json_path
    output_json_path = Path(output_json_path)

    if (not overwrite) and output_json_path.exists():
        print(f"[  ]    : {output_json_path}")
        return

    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if is_slicer_markup(data):
        # Internal processing step.
        if input_json_path.resolve() != output_json_path.resolve():
            shutil.copy2(input_json_path, output_json_path)
        # Internal processing step.
        return

    if not isinstance(data, list):
        raise ValueError(f"        Slicer            : {input_json_path}")

    slicer_data = build_slicer_markup(data)

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(slicer_data, f, ensure_ascii=False, indent=4)

    print(f"[    ] {input_json_path} -> {output_json_path}")


def convert_folder(input_dir, output_dir=None, overwrite=True, recursive=False):
    """
                json
           
    """
    input_dir = Path(input_dir)
    if input_dir is None:
        return
    if output_dir is None:
        output_dir = input_dir / "slicer_format"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if recursive:
        json_files = list(input_dir.rglob("*.json"))
    else:
        json_files = list(input_dir.glob("*.json"))

    for json_file in json_files:
        if recursive:
            rel_path = json_file.relative_to(input_dir)
            out_file = output_dir / rel_path
        else:
            out_file = output_dir / json_file.name

        out_file.parent.mkdir(parents=True, exist_ok=True)
        convert_one_json(json_file, out_file, overwrite=overwrite)


if __name__ == "__main__":
    # =========================
    # Internal processing step.
    # =========================
    # input_json = r"<LOCAL_WORKSPACE>\Desktop\tmp\shoulder_joint\post-preprocess\data\GT\scapula\a2d2c1.json"
    # output_json = r"<LOCAL_WORKSPACE>\Desktop\tmp\shoulder_joint\post-preprocess\data\GT\scapula_slicer\a2d2c1.json"
    # convert_one_json(input_json, output_json, overwrite=True)

    # =========================
    # Internal processing step.
    # =========================
    input_dir = r"<LOCAL_WORKSPACE>\Desktop\tmp\shoulder_joint\post-preprocess\data\GT\Ex\humerus"
    output_dir = r"<LOCAL_WORKSPACE>\Desktop\tmp\shoulder_joint\post-preprocess\data\GT\Ex\humerus"
    convert_folder(input_dir, output_dir, overwrite=True, recursive=False)
