import os
import json
import shutil
from typing import Optional, Dict, Any

from utils.crop_scapula_volume import get_cropped_img, load_landmarks_from_json
from utils.build_uvh_map import get_uvh_map
from utils.refine_boundary_band import refine_bi_by_gc_ray_max_h

RAW_DIR = r"<LOCAL_WORKSPACE>\Desktop\tmp\shoulder_joint\post-preprocess\tools\scapula_uvh\test\raw\scapula_mask"
JSON_DIR = r"<LOCAL_WORKSPACE>\Desktop\tmp\shoulder_joint\post-preprocess\tools\scapula_uvh\test\raw\scapula_pred"
GT_DIR = r"<LOCAL_WORKSPACE>\Desktop\tmp\shoulder_joint\post-preprocess\tools\scapula_uvh\test\raw\scapula_gt"
OUT_DIR = r"<LOCAL_WORKSPACE>\Desktop\tmp\shoulder_joint\post-preprocess\tools\scapula_uvh\test\uvh_result"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)




def reset_dir(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def collect_summary_outputs(out_dir: str) -> Dict[str, Any]:
    """
         
    - OUT_DIR/scapula/:      step3        json
    - OUT_DIR/vis/:      step3_overview.png
    """
    scapula_dir = os.path.join(out_dir, 'scapula')
    vis_dir = os.path.join(out_dir, 'vis')
    reset_dir(scapula_dir)
    reset_dir(vis_dir)

    num_json = 0
    num_vis = 0

    for name in sorted(os.listdir(out_dir)):
        case_dir = os.path.join(out_dir, name)
        if not os.path.isdir(case_dir):
            continue
        if name in {'scapula', 'vis'}:
            continue

        refined_json = os.path.join(case_dir, 'step3_bi_refine', 'data', f'{name}_pred_refined.json')
        overview_png = os.path.join(case_dir, 'step3_bi_refine', 'vis', f'{name}_step3_overview.png')

        if os.path.exists(refined_json):
            dst_json = os.path.join(scapula_dir, f'{name}.json')
            shutil.copy2(refined_json, dst_json)
            num_json += 1

        if os.path.exists(overview_png):
            dst_png = os.path.join(vis_dir, f'{name}_step3_overview.png')
            shutil.copy2(overview_png, dst_png)
            num_vis += 1

    summary = {
        'scapula_dir': scapula_dir,
        'vis_dir': vis_dir,
        'num_refined_json_copied': int(num_json),
        'num_overview_png_copied': int(num_vis),
    }
    summary_json = os.path.join(out_dir, 'summary_collection.json')
    save_json(summary, summary_json)
    return summary

def strip_nii_name(filename: str) -> str:
    if filename.lower().endswith('.nii.gz'):
        return filename[:-7]
    return os.path.splitext(filename)[0]


def find_matching_json(case_id: str, json_dir: Optional[str]) -> Optional[str]:
    if json_dir is None:
        return None
    cands = [f'{case_id}_scapula.json', f'{case_id}.json']
    for name in cands:
        p = os.path.join(json_dir, name)
        if os.path.exists(p):
            return p
    return None


def save_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def process_one_case_uvh(
    nii_path,
    json_path,
    out_dir,
    gt_json_path=None,
    box_margin_inplane_mm=6.0,
    box_margin_normal_mm=2.5,
    surface_h_step_mm=0.5,
    uv_resolution_mm=0.5,
    use_only_foreground=True,
    foreground_threshold=0,
    save_layer_debug=True,
    layer_debug_every=1,
    band_half_width_px=1.0,
    move_threshold_px=4.0,
    settle_inward_px=2.0,
):
    case_id = strip_nii_name(os.path.basename(nii_path))
    case_out_dir = os.path.join(out_dir, case_id)
    ensure_dir(case_out_dir)

    step1 = get_cropped_img(
        nii_path=nii_path,
        json_path=json_path,
        out_dir=case_out_dir,
        box_margin_inplane_mm=box_margin_inplane_mm,
        box_margin_normal_mm=box_margin_normal_mm,
        use_only_foreground=use_only_foreground,
        foreground_threshold=foreground_threshold,
    )

    gt_landmarks = None
    if gt_json_path is not None and os.path.exists(gt_json_path):
        gt_landmarks, _ = load_landmarks_from_json(gt_json_path)

    step2 = get_uvh_map(
        crop_img=step1['crop_img'],
        frame=step1['frame'],
        landmarks=step1['landmarks'],
        gt_landmarks=gt_landmarks,
        out_dir=case_out_dir,
        case_id=case_id,
        uv_resolution_mm=uv_resolution_mm,
        h_step_mm=surface_h_step_mm,
        keep_largest_cc_per_slice=False,
        save_layer_debug=save_layer_debug,
        layer_debug_every=layer_debug_every,
    )

    step3 = refine_bi_by_gc_ray_max_h(
        grid=step2['grid'],
        largest_mask=step2['largest_mask'],
        h_grid=step2['h_grid'],
        frame=step1['frame'],
        landmarks_pred_phys=step1['landmarks'],
        landmarks_pred_uvh_all=step2['landmarks_uvh'],
        raw_json_pred=step1['raw_json'],
        gt_landmarks_phys=gt_landmarks,
        surface_phys=step2['surface_phys'],
        gt_landmarks_uvh_all=step2['gt_landmarks_uvh'],
        out_dir=case_out_dir,
        case_id=case_id,
        band_half_width_px=band_half_width_px,
        move_threshold_px=move_threshold_px,
        settle_inward_px=settle_inward_px,
    )

    process_info = {
        'case_id': case_id,
        'inputs': {'nii_path': nii_path, 'json_path': json_path, 'gt_json_path': gt_json_path},
        'params': {
            'box_margin_inplane_mm': float(box_margin_inplane_mm),
            'box_margin_normal_mm': float(box_margin_normal_mm),
            'uv_resolution_mm': float(uv_resolution_mm),
            'use_only_foreground': bool(use_only_foreground),
            'foreground_threshold': int(foreground_threshold),
            'band_half_width_px': float(band_half_width_px),
            'move_threshold_px': float(move_threshold_px),
            'settle_inward_px': float(settle_inward_px),
        },
        'step_info_paths': {
            'step1': step1['step1_info_path'],
            'step2': step2['step2_info_path'],
            'step3': step3['step3_info_path'],
        },
        'outputs': {**step1['step1_outputs'], **step2['step2_outputs'], **step3['step3_outputs']},
    }
    info_json_path = os.path.join(case_out_dir, f'{case_id}_process_info.json')
    save_json(process_info, info_json_path)

    print(f'[{case_id}] OK')
    return True


def process_folder_uvh(mask_dir, json_dir, out_dir, gt_dir=None, only_case_id=None, **kwargs):
    ensure_dir(out_dir)
    nii_files = [f for f in os.listdir(mask_dir) if f.lower().endswith('.nii.gz')]
    print(f'Found {len(nii_files)} nii.gz files.')

    ok = 0
    fail = 0
    for nii_name in nii_files:
        case_id = strip_nii_name(nii_name)
        if only_case_id is not None and case_id != only_case_id:
            continue

        nii_path = os.path.join(mask_dir, nii_name)
        json_path = find_matching_json(case_id, json_dir)
        gt_json_path = find_matching_json(case_id, gt_dir) if gt_dir is not None else None
        if json_path is None:
            print(f'[{case_id}] skip: no matched json')
            fail += 1
            continue

        print(f'Processing case: {case_id}')
        try:
            success = process_one_case_uvh(
                nii_path=nii_path,
                json_path=json_path,
                out_dir=out_dir,
                gt_json_path=gt_json_path,
                **kwargs,
            )
            ok += int(bool(success))
            fail += int(not bool(success))
        except Exception as e:
            print(f'[{case_id}] error: {e}')
            fail += 1

    print('\n==============================')
    print(f'Done. OK={ok}, Fail/Skip={fail}')
    print('==============================')

    summary = collect_summary_outputs(out_dir)
    print(f"Summary collected: json={summary['num_refined_json_copied']}, vis={summary['num_overview_png_copied']}")


if __name__ == '__main__':
    process_folder_uvh(
        mask_dir=RAW_DIR,
        json_dir=JSON_DIR,
        out_dir=OUT_DIR,
        gt_dir=GT_DIR,
        only_case_id=None,
        # only_case_id='a26d2c1',
    )

