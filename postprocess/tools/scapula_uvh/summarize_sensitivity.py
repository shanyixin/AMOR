import os
import json
import re
import pandas as pd


SENS_DIR = r"<LOCAL_WORKSPACE>\Desktop\tmp\shoulder_joint\post-preprocess\tools\scapula_uvh\test\uvh_result\sensitivity"


def safe_get(d, keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def parse_param_from_folder(folder_name):
    """
         
    mt_2_bw_0p5
    mt_6_bw_1p0
    """
    m = re.match(r"mt_(.*?)_bw_(.*)", folder_name)
    if m is None:
        return None, None

    mt = m.group(1).replace("p", ".")
    bw = m.group(2).replace("p", ".")

    try:
        mt = float(mt)
    except Exception:
        mt = None

    try:
        bw = float(bw)
    except Exception:
        bw = None

    return mt, bw


def collect_one_folder(folder_path, folder_name):
    stat_json = os.path.join(folder_path, "optimization_statistics.json")
    if not os.path.exists(stat_json):
        return None

    with open(stat_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    move_threshold_px, band_half_width_px = parse_param_from_folder(folder_name)

    total = data.get("total_summary", data.get("summary", data))

    row = {
        "folder": folder_name,
        "move_threshold_px": move_threshold_px,
        "band_half_width_px": band_half_width_px,

        "num_cases": safe_get(total, ["num_cases"]),
        "num_cases_with_changed_bi": safe_get(total, ["num_cases_with_changed_bi"]),

        "num_bi_points": safe_get(total, ["num_bi_points"]),
        "num_changed_bi_points": safe_get(total, ["num_changed_bi_points"]),
        "changed_bi_point_ratio": safe_get(total, ["changed_bi_point_ratio"]),

        "changed_points_num_improved": safe_get(total, ["changed_points_num_improved"]),
        "changed_points_num_worse": safe_get(total, ["changed_points_num_worse"]),
        "changed_points_num_same": safe_get(total, ["changed_points_num_same"]),
        "changed_points_improved_ratio": safe_get(total, ["changed_points_improved_ratio"]),

        "all25_before_mean_mm": safe_get(total, ["all25_before_mean_mm"]),
        "all25_after_mean_mm": safe_get(total, ["all25_after_mean_mm"]),
        "all25_improve_mean_mm": safe_get(total, ["all25_improve_mean_mm"]),
        "all25_total_improve_mm": safe_get(total, ["all25_total_improve_mm"]),

        "bi8_before_mean_mm": safe_get(total, ["bi8_before_mean_mm"]),
        "bi8_after_mean_mm": safe_get(total, ["bi8_after_mean_mm"]),
        "bi8_improve_mean_mm": safe_get(total, ["bi8_improve_mean_mm"]),
        "bi8_total_improve_mm": safe_get(total, ["bi8_total_improve_mm"]),

        "changed_bi_before_mean_mm": safe_get(total, ["changed_bi_before_mean_mm"]),
        "changed_bi_after_mean_mm": safe_get(total, ["changed_bi_after_mean_mm"]),
        "changed_bi_improve_mean_mm": safe_get(total, ["changed_bi_improve_mean_mm"]),
        "changed_bi_total_improve_mm": safe_get(total, ["changed_bi_total_improve_mm"]),
    }

    # Internal processing step.
    if row["num_cases"] is None:
        row["num_cases"] = safe_get(total, ["case_count"])
    if row["num_changed_bi_points"] is None:
        row["num_changed_bi_points"] = safe_get(total, ["total_changed_bi_points"])

    return row


def main():
    rows = []

    for name in sorted(os.listdir(SENS_DIR)):
        folder_path = os.path.join(SENS_DIR, name)
        if not os.path.isdir(folder_path):
            continue

        row = collect_one_folder(folder_path, name)
        if row is not None:
            rows.append(row)

    if len(rows) == 0:
        print("      optimization_statistics.json")
        return

    df = pd.DataFrame(rows)

    sort_cols = []
    if "band_half_width_px" in df.columns:
        sort_cols.append("band_half_width_px")
    if "move_threshold_px" in df.columns:
        sort_cols.append("move_threshold_px")

    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    out_xlsx = os.path.join(SENS_DIR, "sensitivity_overall_compare.xlsx")

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="overall_compare")

    print(f"        {out_xlsx}")
    print(df)


if __name__ == "__main__":
    main()
