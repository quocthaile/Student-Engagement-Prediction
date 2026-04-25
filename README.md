# Student Engagement Prediction on MOOCCubeX

> **Môn học:** DS317 – Khai phá Dữ liệu | Trường Đại học Công nghệ Thông tin (UIT)
> **Loại:** Đồ án thực hành + Đồ án môn học

Dự án khai phá dữ liệu dự đoán **mức độ tham gia học tập** (Low / Medium / High) của học sinh trên nền tảng [MOOCCubeX](https://github.com/THU-KEG/MOOCCube) sử dụng các đặc trưng hành vi được trích xuất từ log tương tác quy mô lớn, sau đó huấn luyện bộ phân loại có giám sát để gán nhãn tự động cho học sinh mới.

---

## 📋 Mục lục

- [Yêu cầu môn học](#yêu-cầu-môn-học)
- [Tổng quan dự án](#tổng-quan-dự-án)
- [Dataset](#dataset)
- [Kiến trúc Pipeline](#kiến-trúc-pipeline)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Cài đặt & Chạy](#cài-đặt--chạy)
- [Chạy trên Kaggle](#chạy-trên-kaggle)
- [Kết quả](#kết-quả)

---

## 🎓 Yêu cầu môn học

### Thực hành

| STT | Hạng mục | Trạng thái | Vị trí |
|-----|----------|------------|--------|
| 1 | Thuyết minh đề tài (docx, pptx) | 📝 | `docs/` |
| 2 | Báo cáo phân tích bộ dữ liệu (docx) | 📝 | `docs/` |
| 3 | Bộ dữ liệu sau khi tiền xử lý | ✅ | `experiment/results/phase1/combined_user_metrics.csv` |
| 4 | Video thuyết trình (tất cả thành viên, bật camera) | 📝 | — |

**Đánh giá thực hành:**
- **What** (3đ): Mô tả bài toán, dữ liệu, đặc trưng
- **Why** (3đ): Lý do chọn phương pháp, phân tích EDA
- **How** (4đ): Quy trình xử lý, kết quả thực nghiệm

### Đồ án môn học

| STT | Hạng mục | Trạng thái | Vị trí |
|-----|----------|------------|--------|
| 1 | Báo cáo đồ án (docx, pptx) | 📝 | `final/` |
| 2 | Bộ dữ liệu thực nghiệm | ✅ | `experiment/results/` |
| 3 | Toàn bộ source code | ✅ | `experiment/` |
| 4 | Video thuyết trình (tất cả thành viên, bật camera) | 📝 | — |

**Đánh giá đồ án:**
- **Data Quality** (4đ): Chất lượng dữ liệu, quy trình làm sạch & kỹ thuật đặc trưng
- **Machine Learning & Framework** (3đ): Lựa chọn mô hình, huấn luyện, đánh giá
- **Experiment & Demo** (3đ): Kết quả thực nghiệm, demo và khả năng trình bày

---

## Tổng quan dự án

Pipeline tuân theo quy trình khai phá tri thức đầy đủ từ đầu đến cuối:

1. **Data Preparation** – Dịch tên trường học (Trung → Anh), gộp log hành vi học tập từ 5 nguồn, thực hiện EDA ban đầu.
2. **Data Cleaning** – Loại bỏ bản ghi thiếu quá nhiều giá trị hoặc chứa giá trị NaN/Inf.
3. **Data Transformation** – Chuẩn hóa `engagement_events` sang khoảng [0, 1] bằng min-max scaling.
4. **Data Labeling** – Áp dụng K-Means (k=3) để phân nhóm không giám sát và gán nhãn mức độ tham gia học tập.
5. **Data Splitting** – Chia tập train/validation/test theo chiến lược stratified (70/15/15).
6. **Model Training** – Huấn luyện Logistic Regression, Random Forest, HistGradientBoosting với cross-validation.
7. **Model Evaluation** – Tính Macro F1, Weighted F1, AUC-ROC và precision/recall theo từng class.
8. **Model Interpretability** – Phân tích SHAP global, class-level và local (per-sample).

---

## Dataset

Dự án sử dụng tập dữ liệu **MOOCCubeX** (không được đưa vào repository do kích thước lớn).

| File | Mô tả |
|------|-------|
| `user.json` | Hồ sơ học sinh (id, trường, giới tính, năm sinh, khóa học đã đăng ký) |
| `user-problem.json` | Lịch sử làm bài tập (đúng/sai, số lần thử, điểm, thời gian) |
| `user-video.json` | Phiên xem video (đoạn đã xem, tốc độ, thời lượng) |
| `reply.json` | Hoạt động trả lời trong forum |
| `comment.json` | Hoạt động bình luận trong forum |

**Tải về:** https://github.com/THU-KEG/MOOCCube  
**Kaggle:** https://www.kaggle.com/datasets/thiuyn/mooccubexdataset

Đặt các file đã tải vào thư mục `D:/MOOCCubeX_dataset/` hoặc truyền `--dataset-dir <path>` khi chạy.

---

## Kiến trúc Pipeline

```
MOOCCubeX Dataset (JSON)
         │
         ▼
Phase 1: Data Preparation
  ├── 1/3 Translation  (Dịch tên trường)
  ├── 2/3 Combine      (Gộp log từ 5 file)
  └── 3/3 EDA          (Phân tích & biểu đồ)
         │
         ▼
Phase 2: Data Cleaning
  └── Lọc bản ghi thiếu / không hợp lệ
         │
         ▼
Phase 3: Data Transformation
  └── Min-max normalization engagement_events
         │
         ▼
Phase 4: Data Labeling
  ├── 4.1 K-Means Engagement Report
  ├── 4.2 Standard Labels (Q25/Q75)
  └── 4.3 Temporal Dynamics
         │
         ▼
Phase 5: Data Splitting
  └── Stratified 70% train / 15% valid / 15% test
         │
         ▼
Phase 6: Model Training
  └── LR / RF / HistGB / LightGBM / XGBoost
         │
         ▼
Phase 7: Model Evaluation
  └── Macro F1, AUC-ROC, Recall Low class
         │
         ▼
Phase 8: Model Interpretability
  └── SHAP global / class / local
```

---

## Cấu trúc thư mục

```
project/
├── experiment/                              # Toàn bộ source code pipeline
│   ├── run_experiment_stages.py             # Orchestrator – điều phối tất cả phase
│   ├── phase_1_data_preparation.py          # Phase 1: Translate + Combine + EDA
│   ├── phase_2_data_cleaning.py             # Phase 2: Làm sạch dữ liệu
│   ├── phase_3_data_transformation.py       # Phase 3: Feature engineering
│   ├── phase_4_data_labeling.py             # Phase 4: Gán nhãn K-Means
│   ├── phase_5_data_splitting.py            # Phase 5: Chia tập train/valid/test
│   ├── phase_6_model_training.py            # Phase 6: Huấn luyện mô hình
│   ├── phase_7_model_evaluation.py          # Phase 7: Đánh giá mô hình
│   ├── phase_8_model_interpretability.py    # Phase 8: Giải thích mô hình (SHAP)
│   ├── utils_eda.py                         # Hàm tiện ích EDA dùng chung
│   ├── dataset-metadata.json                # Kaggle dataset config
│   └── results/                             # Output artifacts (không commit)
│       ├── phase1/
│       ├── phase2/
│       └── ...
├── config/
│   └── kaggle_update_code.txt               # Lệnh upload code lên Kaggle
├── docs/                                    # Tài liệu thực hành
├── final/                                   # Báo cáo đồ án cuối kỳ
└── README.md
```

---

## Cài đặt & Chạy

### 1. Cài đặt thư viện

```bash
pip install pandas numpy scikit-learn matplotlib seaborn lightgbm xgboost
```

### 2. Chạy toàn bộ pipeline

```bash
# Từ thư mục gốc project
python experiment/run_experiment_stages.py --phase all
```

### 3. Chạy từng phase

```bash
python experiment/run_experiment_stages.py --phase 1   # Data Preparation
python experiment/run_experiment_stages.py --phase 2   # Data Cleaning
python experiment/run_experiment_stages.py --phase 3   # Data Transformation
python experiment/run_experiment_stages.py --phase 4   # Data Labeling
python experiment/run_experiment_stages.py --phase 5   # Data Splitting
python experiment/run_experiment_stages.py --phase 6   # Model Training
python experiment/run_experiment_stages.py --phase 7   # Model Evaluation
python experiment/run_experiment_stages.py --phase 8   # Model Interpretability
```

### Tham số phổ biến

| Flag | Mặc định | Mô tả |
|------|----------|-------|
| `--phase` | `all` | Phase cần chạy: `1`–`8` hoặc `all` |
| `--dataset-dir` | `D:/MOOCCubeX_dataset` | Đường dẫn đến dataset gốc |
| `--results-dir` | `experiment/results` | Thư mục lưu kết quả |
| `--max-rows` | *(none)* | Giới hạn số dòng để test nhanh |
| `--seed` | `42` | Random seed để tái lập kết quả |

---

## Chạy trên Kaggle

Dataset code đã được upload tại: https://www.kaggle.com/datasets/thaile2024/experiment

```python
import os, sys, subprocess

CODE_DIR    = "/kaggle/input/experiment"
DATASET_DIR = "/kaggle/input/mooccubexdataset/MOOCCubeXData/MOOCCubeXData"
RESULTS_DIR = "/kaggle/working/results"

os.makedirs(RESULTS_DIR, exist_ok=True)

result = subprocess.run([
    sys.executable, f"{CODE_DIR}/run_experiment_stages.py",
    "--phase", "all",
    "--dataset-dir", DATASET_DIR,
    "--results-dir", RESULTS_DIR,
], capture_output=True, text=True)

print(result.stdout)
if result.returncode != 0:
    print("ERROR:", result.stderr)
```

**Cập nhật code lên Kaggle (sau khi sửa local):**
```bash
kaggle datasets version -p "experiment" -m "Update: mô tả thay đổi"
```

---

## Kết quả

Các output được lưu trong `experiment/results/`:

| File | Mô tả |
|------|-------|
| `phase1/combined_user_metrics.csv` | Đặc trưng hành vi theo từng học sinh |
| `phase1/phase1_eda_report.txt` | Báo cáo thống kê EDA |
| `phase2/combined_user_metrics_clean.csv` | Dữ liệu sau làm sạch |
| `phase3/combined_user_metrics_transformed.csv` | Dữ liệu sau feature engineering |
| `phase4/phase4_2_standard_labels_kmeans.csv` | Nhãn engagement (Low/Medium/High) |
| `phase5/phase5_train/valid/test.csv` | Tập train, validation, test |
| `phase6/phase6_best_model.pkl` | Model tốt nhất (serialized) |
| `phase6/phase6_model_comparison.csv` | Bảng so sánh hiệu năng các model |
| `phase7/phase7_evaluation_report.txt` | Báo cáo đánh giá model |
| `phase7/final_summary_report.txt` | Báo cáo tổng hợp toàn pipeline |
| `phase8/phase8_global_contributions.csv` | SHAP feature importance toàn cục |
