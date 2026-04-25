#!/usr/bin/env python3
"""
Phase 3: Data Transformation

Reads the cleaned user metrics CSV from Phase 2 and applies feature engineering:
  - Normalises the `engagement_events` column to the [0, 1] range using
    min-max scaling computed over the full dataset (single pass).
  - Appends the new column `engagement_events_normalized` to each row.

Input :
  results/phase2/combined_user_metrics_clean.csv   (from Phase 2)

Output:
  results/phase3/combined_user_metrics_transformed.csv
"""
import argparse
import csv
import time
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import numpy as np
import utils_eda as eda_lib


def safe_float(value) -> Optional[float]:
    """Convert a raw CSV string value to float, returning None on failure."""
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_engagement_events(rows: List[Dict[str, str]]) -> Dict[str, float]:
    """
    Compute min / max / mean / std of `engagement_events` across all rows.

    These statistics are used for min-max normalization so that the result
    always falls within [0, 1].  Returns safe fallback values when the column
    is entirely missing or empty.
    """
    values = []
    for row in rows:
        val = safe_float(row.get("engagement_events"))
        if val is not None:
            values.append(val)

    if not values:
        # No valid values – return neutral fallback
        return {"min": 0.0, "max": 1.0, "mean": 0.5, "std": 0.0}

    arr = np.array(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Data Transformation")
    parser.add_argument("--combined-input", type=Path, required=True,
                        help="Path to combined_user_metrics_clean.csv from Phase 2")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory to write transformed CSV")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Limit number of input rows for testing")
    args, _ = parser.parse_known_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading data from {args.combined_input}")

    rows, columns = eda_lib.load_combined_csv(args.combined_input, args.max_rows)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Computing normalization parameters...")
    normalization = normalize_engagement_events(rows)

    output_csv = args.output_dir / "combined_user_metrics_transformed.csv"
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Writing transformed CSV to {output_csv}")

    min_val = normalization["min"]
    max_val = normalization["max"]

    # Add the normalized column to the schema if not already present
    if "engagement_events_normalized" not in columns:
        columns.append("engagement_events_normalized")

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for row in rows:
            e_raw = safe_float(row.get("engagement_events"))
            if e_raw is None:
                # Missing value: leave the normalized column blank
                row["engagement_events_normalized"] = ""
            else:
                # Min-max normalization clamped to [0, 1]
                if max_val <= min_val:
                    e_norm = 0.0
                else:
                    e_norm = (e_raw - min_val) / (max_val - min_val)
                    e_norm = max(0.0, min(1.0, e_norm))
                row["engagement_events_normalized"] = str(round(e_norm, 6))
            writer.writerow(row)

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Phase 3 Transformation completed.")


if __name__ == "__main__":
    main()
