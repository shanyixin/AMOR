"""Run post-processing metrics for shoulder landmark predictions."""

import argparse
import os
from pathlib import Path

from postprocess.tools.to_slicer_format import convert_folder
from postprocess.utils.error_summary import error_summary
from postprocess.utils.metrics_update import calculate_ClinicalMetrics_error
from postprocess.utils.mm import calculate_milemeter_error
from postprocess.utils.post import pt_surface
from postprocess.utils.voxel import calculate_voxel_error


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate voxel, millimeter, and clinical metrics.")
    parser.add_argument("--gt-root", required=True, help="Ground-truth root containing In/ and Ex/ folders.")
    parser.add_argument("--work-root", required=True, help="Prediction work root containing In/ and Ex/ folders.")
    parser.add_argument("--info", required=True, help="CSV/XLSX metadata file used for physical-space recovery.")
    parser.add_argument("--mask-root", required=True, help="Mask root containing In/ and Ex/ folders.")
    parser.add_argument("--sites", nargs="+", default=["In", "Ex"], help="Dataset sites to process.")
    parser.add_argument("--skip-millimeter", action="store_true", help="Skip physical-space error calculation.")
    parser.add_argument("--skip-clinical", action="store_true", help="Skip clinical metric calculation.")
    parser.add_argument("--skip-scapula", action="store_true", help="Skip scapula landmarks.")
    parser.add_argument("--skip-humerus", action="store_true", help="Skip humerus landmarks.")
    parser.add_argument("--skip-summary", action="store_true", help="Skip final error summary.")
    return parser.parse_args()


def process_site(args, site):
    gt_dir = Path(args.gt_root) / site
    work_dir = Path(args.work_root) / site
    mask_dir = Path(args.mask_root) / site
    pred_dir = work_dir / "pred"
    output_dir = work_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    scapula_gt_dir = gt_dir / "scapula" if not args.skip_scapula else None
    scapula_pred_dir = pred_dir / "scapula" if not args.skip_scapula else None
    humerus_gt_dir = gt_dir / "humerus" if not args.skip_humerus else None
    humerus_pred_dir = pred_dir / "humerus" if not args.skip_humerus else None

    for folder in (scapula_pred_dir, humerus_pred_dir):
        if folder is not None and folder.exists():
            convert_folder(str(folder), str(folder), overwrite=True, recursive=False)

    calculate_voxel_error(
        str(scapula_gt_dir) if scapula_gt_dir else None,
        str(scapula_pred_dir) if scapula_pred_dir else None,
        str(humerus_gt_dir) if humerus_gt_dir else None,
        str(humerus_pred_dir) if humerus_pred_dir else None,
        str(output_dir),
    )

    if not args.skip_millimeter:
        calculate_milemeter_error(
            str(scapula_gt_dir) if scapula_gt_dir else None,
            str(scapula_pred_dir) if scapula_pred_dir else None,
            str(humerus_gt_dir) if humerus_gt_dir else None,
            str(humerus_pred_dir) if humerus_pred_dir else None,
            args.info,
            str(output_dir),
        )
        realworld_dir = output_dir / "REALWORLD"
        pt_surface(str(realworld_dir), str(mask_dir))

        if not args.skip_clinical:
            calculate_ClinicalMetrics_error(str(realworld_dir), str(output_dir))

    if not args.skip_summary:
        error_summary(str(output_dir))


def main():
    args = parse_args()
    for site in args.sites:
        process_site(args, site)


if __name__ == "__main__":
    main()

