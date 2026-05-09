import os
import json
import numpy as np


NME_NORM = 256.0


def error_summary(Dir):
    """
                Dir/error_summary.json

         
    1)    / scapula / humerus   MRE(mean/std)
    2)    / scapula / humerus   NME(mean/std)     error / 256   
    3)    SDR(%)    1mm~12mm
    4)     landmark          MRE / NME(mean/std)   SDR(%)    1mm~12mm
    """

    landmark_order = [
        "ss", "sm", "si", "AA", "AP", "C", "gc", "SCP6", "SCP12",
        "bi2", "bi3", "bi4", "bi5", "bi6", "bi7", "bi8", "bi9", "bi12",
        "HC", "HI", "HS", "HP", "HA", "HBGS", "HBGI"
    ]
    thresholds = list(range(1, 13))

    def _first_existing(paths):
        for p in paths:
            if p and os.path.exists(p):
                return p
        return None

    scapula_error_path = _first_existing([
        os.path.join(Dir, 'milemeter_error_scapula_surface.json'),
        os.path.join(Dir, 'milemeter_error_scapula.json'),
        os.path.join(Dir, 'milimeter_error_scapula_surface.json'),
        os.path.join(Dir, 'milimeter_error_scapula.json'),
    ])

    humerus_error_path = _first_existing([
        os.path.join(Dir, 'milemeter_error_humerus_surface.json'),
        os.path.join(Dir, 'milemeter_error_humerus.json'),
        os.path.join(Dir, 'milimeter_error_humerus_surface.json'),
        os.path.join(Dir, 'milimeter_error_humerus.json'),
    ])

    if scapula_error_path is None and humerus_error_path is None:
        raise FileNotFoundError(f'No millimeter error json found in {Dir}')

    def _load_json(path):
        if path is None or not os.path.exists(path):
            return None
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    scapula_error = _load_json(scapula_error_path)
    humerus_error = _load_json(humerus_error_path)

    data_by_part = {'scapula': {}, 'humerus': {}}
    all_errors = []
    all_nmes = []

    def _part_from_label(lbl):
        return 'scapula' if lbl in landmark_order[:18] else 'humerus'

    def _ingest_error_json(part_name, err_json):
        if err_json is None:
            return
        details = err_json.get('details', {})
        for _, case_info in details.items():
            for lbl, err in case_info.get('point_errors', {}).items():
                err = float(err)
                nme = err / NME_NORM
                data_by_part[part_name].setdefault(lbl, {'errors': [], 'nmes': []})
                data_by_part[part_name][lbl]['errors'].append(err)
                data_by_part[part_name][lbl]['nmes'].append(nme)
                all_errors.append(err)
                all_nmes.append(nme)

    _ingest_error_json('scapula', scapula_error)
    _ingest_error_json('humerus', humerus_error)

    def _mean_std(vals):
        vals = np.asarray(vals, dtype=np.float64)
        if vals.size == 0:
            return {'mean': None, 'std': None, 'count': 0}
        return {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
            'count': int(vals.size)
        }

    def _sdr(vals, thresholds_mm):
        vals = np.asarray(vals, dtype=np.float64)
        out = {}
        if vals.size == 0:
            for t in thresholds_mm:
                out[f'{t}mm'] = None
            return out
        for t in thresholds_mm:
            out[f'{t}mm'] = float(np.mean(vals <= t) * 100.0)
        return out

    def _merge_errors_for_part(part_name):
        merged_e = []
        merged_n = []
        for _, bucket in data_by_part[part_name].items():
            merged_e.extend(bucket['errors'])
            merged_n.extend(bucket['nmes'])
        return merged_e, merged_n

    scapula_errors_all, scapula_nmes_all = _merge_errors_for_part('scapula')
    humerus_errors_all, humerus_nmes_all = _merge_errors_for_part('humerus')

    summary = {
        'overall': {
            'MRE': _mean_std(all_errors),
            'NME': _mean_std(all_nmes),
            'SDR_percent': _sdr(all_errors, thresholds),
        },
        'scapula': {
            'MRE': _mean_std(scapula_errors_all),
            'NME': _mean_std(scapula_nmes_all),
        },
        'humerus': {
            'MRE': _mean_std(humerus_errors_all),
            'NME': _mean_std(humerus_nmes_all),
        },
        'per_landmark': {}
    }

    for lbl in landmark_order:
        part_name = _part_from_label(lbl)
        bucket = data_by_part[part_name].get(lbl, {'errors': [], 'nmes': []})
        summary['per_landmark'][lbl] = {
            'part': part_name,
            'MRE': _mean_std(bucket['errors']),
            'NME': _mean_std(bucket['nmes']),
            'SDR_percent': _sdr(bucket['errors'], thresholds),
        }

    result = {
        'normalization': {
            'name': 'fixed_256',
            'description': 'NME = Euclidean error / 256'
        },
        'source_files': {
            'scapula_error': scapula_error_path,
            'humerus_error': humerus_error_path,
        },
        'landmark_order': landmark_order,
        'thresholds_mm': thresholds,
        'summary': summary,
    }

    out_path = os.path.join(Dir, 'error_summary.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f'Error summary saved to: {out_path}')
    return result

