# Student Engagement Prediction on MOOCCubeX

> Course: DS317 – Data Mining | University of Information Technology (UIT)

A data mining project that predicts **student engagement levels** (Low / Medium / High) on the [MOOCCubeX](https://github.com/THU-KEG/MOOCCube) platform using behavior features extracted from large-scale MOOC interaction logs, then trains supervised classifiers to automatically label new learners.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Pipeline Architecture](#pipeline-architecture)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Running the Pipeline](#running-the-pipeline)
- [Results](#results)

---

## Project Overview

The project follows a full end-to-end knowledge discovery pipeline:

1. **EDA** – Explore distributions, missing values, outliers, and school-level statistics.
2. **Data Cleaning** – Normalize Chinese school names to English, filter noisy records.
3. **Data Transformation** – Aggregate raw event logs into per-user behavioral features (video watch time, problem accuracy, forum participation, etc.).
4. **Data Labeling** – Apply K-Means clustering to unsupervisedly group users into 3 engagement tiers and assign ground-truth labels.
5. **Data Splitting** – Stratified train/validation/test splits with optional SMOTE oversampling for class imbalance.
6. **Model Training** – Train Logistic Regression, Random Forest, and Gradient Boosting classifiers with cross-validation.
7. **Model Evaluation** – Compute Macro F1, Weighted F1, AUC-ROC, and per-class precision/recall.
8. **Model Interpretability** – SHAP-style global and local feature importance analysis.

---

## Dataset

The project uses the **MOOCCubeX** dataset (not included in this repository due to size).

| File | Description |
|---|---|
| `user.json` | User profiles (id, school, gender, year of birth, enrolled courses) |
| `user-problem.json` | Problem submissions (correctness, attempts, score, timestamp) |
| `user-video.json` | Video watching sessions (segments, speed, watched duration) |
| `reply.json` | Forum reply activity |
| `comment.json` | Forum comment activity |

**Download:** https://github.com/THU-KEG/MOOCCube

Place the downloaded files in the `D:/MOOCCubeX_dataset/` directory (or pass a custom `--dataset-dir` flag).

Alternatively, if you already have the pre-joined flat table, place it at:
```
experiment/results/combined_all_data.parquet
```
and use the `--combined-parquet` flag to skip JSON streaming (see [Running the Pipeline](#running-the-pipeline)).

---

## Pipeline Architecture

```
MOOCCubeX Dataset (JSON)  ──or──  combined_all_data.parquet
          │                                │
          ▼                                ▼
  Phase 2: Data Cleaning       Phase 3: Data Transformation
          │                                │
          └──────────────┬─────────────────┘
                         ▼
               Phase 1: EDA (Exploration)
                         │
                         ▼
            Phase 4: Data Labeling (K-Means)
                         │
                         ▼
            Phase 5: Data Splitting (Train/Val/Test)
                         │
                         ▼
            Phase 6: Model Training
                         │
                         ▼
            Phase 7: Model Evaluation
                         │
                         ▼
            Phase 8: Model Interpretability
```

---

## Repository Structure

```
project/
├── experiment/                    # Core experiment scripts
│   ├── phase_1_eda.py             # Phase 1: Exploratory Data Analysis
│   ├── phase_2_data_cleaning.py   # Phase 2: Data Cleaning (school name translation)
│   ├── phase_3_data_transformation.py  # Phase 3: Feature engineering from raw logs
│   ├── phase_4_data_labeling.py   # Phase 4: K-Means clustering & engagement labeling
│   ├── phase_5_data_splitting.py  # Phase 5: Train/Valid/Test split + SMOTE
│   ├── phase_6_model_training.py  # Phase 6: Supervised model training
│   ├── phase_7_model_evaluation.py     # Phase 7: Evaluation metrics & reports
│   ├── phase_8_model_interpretability.py # Phase 8: Feature importance & SHAP
│   ├── run_experiment_stages.py   # Orchestrator – run any phase or full pipeline
│   ├── results/                   # Output artifacts (CSVs, reports, models)
│   └── command.txt                # Quick command reference
├── config/                        # Configuration files
├── docs/                          # Additional documentation
├── examples/                      # Example notebooks / demos
├── final/                         # Final report and presentation
├── reports/                       # Auto-generated experiment reports
└── README.md
```

---

## Getting Started

### 1. Install dependencies

```bash
pip install -r dataset/requirements.txt
```

### 2. Configure dataset path

By default, the pipeline looks for the MOOCCubeX dataset at `D:/MOOCCubeX_dataset/`.  
You can override this with `--dataset-dir <path>`.

---

## Running the Pipeline

All commands are run from the **project root** directory.  
See `experiment/command.txt` for a full command reference.

### Option A – From raw JSON dataset

Run the full pipeline end-to-end:

```bash
python experiment/run_experiment_stages.py --phase all
```

Or run individual phases in order:

```bash
# 1. Data Cleaning (translate school names)
python experiment/run_experiment_stages.py --phase 2

# 2. Data Transformation (extract user-level features from raw logs)
python experiment/run_experiment_stages.py --phase 3

# 3. EDA (plots and statistics)
python experiment/run_experiment_stages.py --phase 1

# 4. Data Labeling (K-Means clustering)
python experiment/run_experiment_stages.py --phase 4

# 5. Data Splitting (train/val/test + SMOTE)
python experiment/run_experiment_stages.py --phase 5

# 6. Model Training
python experiment/run_experiment_stages.py --phase 6

# 7. Model Evaluation
python experiment/run_experiment_stages.py --phase 7

# 8. Model Interpretability
python experiment/run_experiment_stages.py --phase 8
```

### Option B – From pre-joined Parquet file (fast)

If you already have `combined_all_data.parquet`, Phase 3 will aggregate features directly from it — skipping JSON streaming entirely:

```bash
# Full pipeline
python experiment/run_experiment_stages.py --phase all \
    --combined-parquet experiment/results/combined_all_data.parquet

# Only Phase 3 (feature extraction from parquet)
python experiment/run_experiment_stages.py --phase 3 \
    --combined-parquet experiment/results/combined_all_data.parquet
```

### Common flags

| Flag | Default | Description |
|---|---|---|
| `--phase` | `all` | Phase to run: `1`–`8` or `all` |
| `--dataset-dir` | `D:/MOOCCubeX_dataset` | Path to raw JSON dataset |
| `--results-dir` | `results` | Output directory for artifacts |
| `--combined-parquet` | *(none)* | Path to pre-joined parquet file (skips JSON streaming in Phase 3) |
| `--clusters` | `3` | Number of K-Means clusters for labeling |
| `--split-strategy` | `stratified` | Split strategy: `stratified`, `group`, `temporal`, `hybrid` |
| `--imbalance-method` | `random_oversample` | Imbalance handling: `none`, `random_oversample`, `smote` |
| `--phase6-models` | `logistic,random_forest,hist_gb` | Models to train in Phase 6 |
| `--seed` | `42` | Random seed for reproducibility |
| `--max-rows` | *(none)* | Limit rows for quick smoke-tests |

---

## Results

Experiment outputs are saved to `experiment/results/`:

| File | Description |
|---|---|
| `combined_user_metrics.csv` | Per-user aggregated behavioral features |
| `step5_standard_labels_kmeans.csv` | K-Means engagement labels (Low/Medium/High) |
| `stage3_train_modeling.csv` | Final training set |
| `stage3_valid.csv` | Validation set |
| `stage3_test.csv` | Test set |
| `phase6_*.pkl` | Trained model artifacts |
| `phase7_evaluation_report.txt` | Evaluation metrics summary |
| `phase8_interpretability_report.txt` | Feature importance report |
