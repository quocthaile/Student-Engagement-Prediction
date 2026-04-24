"""
Phase 1: Exploratory Data Analysis (EDA)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import csv
import json
import sqlite3
from datetime import datetime, timezone
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import *
import numpy as np


def now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def log(message: str) -> None:
    print(f"[{now_text()}] {message}")

def resolve_path_arg(path_value: Path, project_root: Path, default_base: Path) -> Path:
    if path_value.is_absolute():
        return path_value.resolve()
    if path_value.parent == Path("."):
        return (default_base / path_value).resolve()
    return (project_root / path_value).resolve()

def run_command(command: List[str], cwd: Path, label: str) -> None:
    log(f"Running {label}")
    proc = subprocess.run(command, cwd=str(cwd), check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {proc.returncode}")

"""
Phase 1B: EDA and Data Cleaning for Combined Metrics.

Aligned with data preparation workflow:
1. Exploratory Data Analysis (EDA)
   - Descriptive statistics
   - Data visualization
   - Detect outliers and anomalies
   - Statistical analysis

2. Data Cleaning
   - Handle missing values (missing data)
   - Handle duplicates/inconsistencies
   - Remove/fix erroneous data
   - Standardize format

3. Data Transformation
   - Normalize engagement_events
   - Create derived features
   - Feature scaling if needed

Outputs (all written to experiment/results/):
- phase1_eda_report.txt           – descriptive statistics, outlier and quality report
- combined_user_metrics_clean.csv – rows after missing/invalid removal
- engagement_events_normalized.csv – min-max normalised engagement scores
"""





@dataclass
class Phase1bConfig:
    combined_csv: Path
    output_dir: Path
    output_clean_csv: Path
    output_report_txt: Path
    output_normalization_csv: Path
    missing_threshold: float = 0.3
    outlier_iqr_multiplier: float = 1.5
    log_every: int = 100000
    max_rows: Optional[int] = None






def safe_float(value) -> Optional[float]:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value) -> Optional[int]:
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def load_combined_csv(path: Path, max_rows: Optional[int]) -> Tuple[List[Dict[str, str]], List[str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        columns = list(reader.fieldnames or [])
        for idx, row in enumerate(reader, start=1):
            if max_rows is not None and idx > max_rows:
                break
            rows.append(row)
            if idx % 100000 == 0:
                log(f"Load progress: {idx:,} rows")

    return rows, columns


def compute_descriptive_stats(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, float]]:
    """Compute descriptive statistics for numeric columns."""
    numeric_cols = [
        "num_courses",
        "problem_total",
        "problem_accuracy",
        "avg_attempts",
        "avg_score",
        "video_sessions",
        "video_count",
        "segment_count",
        "watched_seconds",
        "watched_hours",
        "avg_speed",
        "reply_count",
        "comment_count",
        "forum_total",
        "engagement_events",
    ]

    stats: Dict[str, Dict[str, float]] = {}

    for col in numeric_cols:
        values = []
        missing_count = 0

        for row in rows:
            val = safe_float(row.get(col))
            if val is None:
                missing_count += 1
            else:
                values.append(val)

        if not values:
            stats[col] = {
                "count": 0,
                "missing": missing_count,
                "missing_pct": 100.0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "q25": 0.0,
                "median": 0.0,
                "q75": 0.0,
                "max": 0.0,
            }
            continue

        arr = np.array(values, dtype=np.float64)
        missing_pct = (missing_count / len(rows)) * 100.0

        stats[col] = {
            "count": len(values),
            "missing": missing_count,
            "missing_pct": missing_pct,
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "q25": float(np.quantile(arr, 0.25)),
            "median": float(np.quantile(arr, 0.50)),
            "q75": float(np.quantile(arr, 0.75)),
            "max": float(np.max(arr)),
        }

    return stats


def detect_outliers_iqr(
    rows: List[Dict[str, str]], col: str, multiplier: float = 1.5
) -> Tuple[int, float, float]:
    """Detect outliers using IQR method."""
    values = []
    for row in rows:
        val = safe_float(row.get(col))
        if val is not None:
            values.append(val)

    if not values:
        return 0, 0.0, 0.0

    arr = np.array(values, dtype=np.float64)
    q1 = float(np.quantile(arr, 0.25))
    q3 = float(np.quantile(arr, 0.75))
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    outlier_count = 0
    for val in arr:
        if val < lower_bound or val > upper_bound:
            outlier_count += 1

    outlier_pct = (outlier_count / len(arr)) * 100.0 if len(arr) > 0 else 0.0
    return outlier_count, outlier_pct, lower_bound, upper_bound


def normalize_engagement_events(rows: List[Dict[str, str]]) -> Dict[str, float]:
    """Compute normalization parameters for engagement_events."""
    values = []
    for row in rows:
        val = safe_float(row.get("engagement_events"))
        if val is not None:
            values.append(val)

    if not values:
        return {"min": 0.0, "max": 1.0, "mean": 0.5, "std": 0.0}

    arr = np.array(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def clean_rows(
    rows: List[Dict[str, str]],
    columns: List[str],
    normalization: Dict[str, float],
    missing_threshold: float,
) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    """Clean rows by removing/fixing problematic data."""
    numeric_cols = [
        "num_courses",
        "problem_total",
        "problem_accuracy",
        "avg_attempts",
        "avg_score",
        "video_sessions",
        "video_count",
        "segment_count",
        "watched_seconds",
        "watched_hours",
        "avg_speed",
        "reply_count",
        "comment_count",
        "forum_total",
        "engagement_events",
    ]

    cleaned_rows: List[Dict[str, str]] = []
    stats = {
        "total": len(rows),
        "removed_too_many_missing": 0,
        "removed_invalid": 0,
        "kept": 0,
    }

    for row in rows:
        missing_count = 0
        for col in numeric_cols:
            if col in columns and (row.get(col) is None or str(row.get(col)).strip() == ""):
                missing_count += 1

        missing_ratio = missing_count / len(numeric_cols)
        if missing_ratio > missing_threshold:
            stats["removed_too_many_missing"] += 1
            continue

        is_valid = True
        for col in numeric_cols:
            if col in columns:
                val = safe_float(row.get(col))
                if val is not None and (np.isnan(val) or np.isinf(val)):
                    is_valid = False
                    break

        if not is_valid:
            stats["removed_invalid"] += 1
            continue

        cleaned_rows.append(row)
        stats["kept"] += 1

    return cleaned_rows, stats


def write_normalized_csv(
    rows: List[Dict[str, str]],
    output_path: Path,
    normalization: Dict[str, float],
) -> None:
    """Write normalized engagement_events to CSV."""
    min_val = normalization["min"]
    max_val = normalization["max"]

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "engagement_events", "engagement_events_normalized"])

        for row in rows:
            user_id = row.get("user_id", "")
            e_raw = safe_float(row.get("engagement_events"))

            if e_raw is None:
                continue

            if max_val <= min_val:
                e_norm = 0.0
            else:
                e_norm = (e_raw - min_val) / (max_val - min_val)
                e_norm = max(0.0, min(1.0, e_norm))

            writer.writerow([user_id, round(e_raw, 6), round(e_norm, 6)])


def write_clean_csv(
    rows: List[Dict[str, str]],
    columns: List[str],
    output_path: Path,
) -> None:
    """Write cleaned data to CSV."""
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    output_path: Path,
    original_rows: int,
    cleaned_rows: int,
    stats_before: Dict[str, Dict[str, float]],
    stats_after: Dict[str, Dict[str, float]],
    normalization: Dict[str, float],
    cleaning_stats: Dict[str, int],
    outlier_detection: Dict[str, Tuple[int, float, float, float]],
    elapsed: float,
) -> None:
    """Write comprehensive EDA report."""
    with output_path.open("w", encoding="utf-8") as f:
        f.write("Phase 1B - EDA and Data Cleaning Report\n")
        f.write("=" * 100 + "\n")
        f.write(f"Generated at                 : {now_text()}\n")
        f.write(f"Elapsed (seconds)            : {elapsed:.2f}\n")

        f.write("\n1. DATA OVERVIEW\n")
        f.write("-" * 100 + "\n")
        f.write(f"Original rows                : {original_rows:,}\n")
        f.write(f"After cleaning               : {cleaned_rows:,}\n")
        f.write(f"Rows removed (missing data)  : {cleaning_stats['removed_too_many_missing']:,}\n")
        f.write(f"Rows removed (invalid)       : {cleaning_stats['removed_invalid']:,}\n")
        f.write(f"Retention rate               : {(cleaned_rows / original_rows * 100):.2f}%\n")

        f.write("\n2. DESCRIPTIVE STATISTICS (BEFORE CLEANING)\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Column':<20} {'Count':>10} {'Missing':>10} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}\n")
        for col in sorted(stats_before.keys()):
            stat = stats_before[col]
            f.write(
                f"{col:<20} {int(stat['count']):>10,} {int(stat['missing']):>10,} "
                f"{stat['mean']:>12.4f} {stat['std']:>12.4f} {stat['min']:>12.4f} {stat['max']:>12.4f}\n"
            )

        f.write("\n3. DESCRIPTIVE STATISTICS (AFTER CLEANING)\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Column':<20} {'Count':>10} {'Missing':>10} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}\n")
        for col in sorted(stats_after.keys()):
            stat = stats_after[col]
            f.write(
                f"{col:<20} {int(stat['count']):>10,} {int(stat['missing']):>10,} "
                f"{stat['mean']:>12.4f} {stat['std']:>12.4f} {stat['min']:>12.4f} {stat['max']:>12.4f}\n"
            )

        f.write("\n4. OUTLIER DETECTION (IQR Method, Multiplier=1.5)\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Column':<20} {'Outliers':>10} {'%':>8} {'Lower Bound':>15} {'Upper Bound':>15}\n")
        for col in sorted(outlier_detection.keys()):
            count, pct, lower, upper = outlier_detection[col]
            f.write(
                f"{col:<20} {count:>10,} {pct:>7.2f}% {lower:>15.4f} {upper:>15.4f}\n"
            )

        f.write("\n5. ENGAGEMENT_EVENTS NORMALIZATION\n")
        f.write("-" * 100 + "\n")
        f.write(f"Raw min value                : {normalization['min']:.6f}\n")
        f.write(f"Raw max value                : {normalization['max']:.6f}\n")
        f.write(f"Raw mean                     : {normalization['mean']:.6f}\n")
        f.write(f"Raw std                      : {normalization['std']:.6f}\n")
        f.write(f"Normalized range             : [0.0, 1.0]\n")
        f.write("Formula: (E - min) / (max - min), clipped to [0, 1]\n")

        f.write("\n6. DATA QUALITY SUMMARY\n")
        f.write("-" * 100 + "\n")
        f.write("Data Cleaning Steps:\n")
        f.write(f"- Removed rows with >30% missing numeric values: {cleaning_stats['removed_too_many_missing']:,}\n")
        f.write(f"- Removed rows with invalid values (NaN/Inf): {cleaning_stats['removed_invalid']:,}\n")
        f.write(f"- Final clean dataset: {cleaned_rows:,} rows\n")

        f.write("\nData Quality Checks:\n")
        for col in sorted(outlier_detection.keys()):
            count, pct, _, _ = outlier_detection[col]
            if pct > 5.0:
                f.write(f"⚠️  {col}: {pct:.2f}% outliers detected (>5% threshold)\n")
            elif pct > 1.0:
                f.write(f"ℹ️  {col}: {pct:.2f}% outliers detected (acceptable)\n")


def run_eda_and_cleaning(
    combined_csv: Path,
    output_dir: Path,
    missing_threshold: float = 0.3,
    outlier_iqr_multiplier: float = 1.5,
    log_every: int = 200000,
    max_rows: Optional[int] = None,
) -> None:
    """Run full EDA + cleaning pipeline and write all outputs to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    output_report = output_dir / "phase1_eda_report.txt"
    output_clean_csv = output_dir / "combined_user_metrics_clean.csv"
    output_norm_csv = output_dir / "engagement_events_normalized.csv"

    cfg = Phase1bConfig(
        combined_csv=combined_csv,
        output_dir=output_dir,
        output_clean_csv=output_clean_csv,
        output_report_txt=output_report,
        output_normalization_csv=output_norm_csv,
        missing_threshold=missing_threshold,
        outlier_iqr_multiplier=outlier_iqr_multiplier,
        log_every=log_every,
        max_rows=max_rows,
    )

    started = time.time()

    if not cfg.combined_csv.exists():
        raise FileNotFoundError(f"Input file not found: {cfg.combined_csv}")

    log(f"Loading data from: {cfg.combined_csv}")
    rows, columns = load_combined_csv(cfg.combined_csv, cfg.max_rows)
    log(f"Loaded {len(rows):,} rows, {len(columns)} columns")

    log("Computing descriptive statistics (before cleaning)...")
    stats_before = compute_descriptive_stats(rows)

    log("Computing normalization parameters...")
    normalization = normalize_engagement_events(rows)

    log("Detecting outliers (IQR method)...")
    numeric_cols = [
        "num_courses", "problem_total", "problem_accuracy", "avg_attempts",
        "avg_score", "video_sessions", "video_count", "segment_count",
        "watched_seconds", "watched_hours", "avg_speed",
        "reply_count", "comment_count", "forum_total", "engagement_events",
    ]
    outlier_detection: Dict[str, Tuple[int, float, float, float]] = {}
    for col in numeric_cols:
        result = detect_outliers_iqr(rows, col, cfg.outlier_iqr_multiplier)
        outlier_detection[col] = result

    log("Cleaning data...")
    cleaned_rows, cleaning_stats = clean_rows(rows, columns, normalization, cfg.missing_threshold)
    log(f"Cleaned: {cleaning_stats['kept']:,} rows kept, "
        f"{cleaning_stats['removed_too_many_missing']:,} removed (missing), "
        f"{cleaning_stats['removed_invalid']:,} removed (invalid)")

    log("Computing descriptive statistics (after cleaning)...")
    stats_after = compute_descriptive_stats(cleaned_rows)

    log(f"Writing clean CSV to: {cfg.output_clean_csv}")
    write_clean_csv(cleaned_rows, columns, cfg.output_clean_csv)

    log(f"Writing normalized engagement CSV to: {cfg.output_normalization_csv}")
    write_normalized_csv(cleaned_rows, cfg.output_normalization_csv, normalization)

    elapsed = time.time() - started
    log(f"Writing EDA report to: {cfg.output_report_txt}")
    write_report(
        output_path=cfg.output_report_txt,
        original_rows=len(rows),
        cleaned_rows=len(cleaned_rows),
        stats_before=stats_before,
        stats_after=stats_after,
        normalization=normalization,
        cleaning_stats=cleaning_stats,
        outlier_detection=outlier_detection,
        elapsed=elapsed,
    )
    log(f"EDA complete. Report saved to: {cfg.output_report_txt}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 1: Exploratory Data Analysis (EDA).")
    # Output directory defaults to experiment/results/ (sibling of this script)
    default_results = Path(__file__).resolve().parent / "results"
    parser.add_argument("--results-dir", type=Path, default=default_results)
    parser.add_argument("--combined-input", type=Path, default=Path("combined_user_metrics.csv"))
    parser.add_argument("--missing-threshold", type=float, default=0.3)
    parser.add_argument("--outlier-iqr-multiplier", type=float, default=1.5)
    parser.add_argument("--top-users", type=int, default=100)
    parser.add_argument("--min-school-size", type=int, default=20)
    parser.add_argument("--top-schools", type=int, default=30)
    parser.add_argument("--log-every", type=int, default=200000)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    # results_dir resolves to experiment/results/ by default
    results_dir = resolve_path_arg(args.results_dir, project_root, project_root)
    combined_csv = resolve_path_arg(args.combined_input, project_root, results_dir)

    try:
        started = time.time()
        log("Starting Phase 1: Exploratory Data Analysis")
        log(f"Output directory: {results_dir}")

        run_eda_and_cleaning(
            combined_csv=combined_csv,
            output_dir=results_dir,
            missing_threshold=args.missing_threshold,
            outlier_iqr_multiplier=args.outlier_iqr_multiplier,
            log_every=max(1, args.log_every),
            max_rows=args.max_rows,
        )

        log(f"Phase 1 completed in {time.time() - started:.2f}s")
        return 0
    except Exception as exc:
        log(f"FAILED: {exc}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
