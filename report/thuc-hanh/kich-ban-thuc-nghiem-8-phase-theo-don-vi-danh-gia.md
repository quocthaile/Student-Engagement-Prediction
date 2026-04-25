# Kich ban thuc nghiem 8 phase theo don vi danh gia

## 1) Dinh nghia danh gia can dung trong de tai

### Dinh nghia A: Danh gia theo tung user-course
- Don vi mau: 1 cap `(user_id, course_id)`.
- Mot user co the xuat hien nhieu dong (moi khoa hoc 1 dong).
- Y nghia metric: do kha nang phan loai muc do tham gia trong tung khoa hoc cu the.
- Dung khi muc tieu la can thiep theo mon/khoa hoc.

### Dinh nghia B: Danh gia theo user tong hop toan bo khoa hoc
- Don vi mau: 1 `user_id` duy nhat, dac trung da tong hop qua tat ca khoa hoc.
- Moi user chi xuat hien 1 dong.
- Y nghia metric: do kha nang phan loai ho so hoc tap tong the cua nguoi hoc.
- Dung khi muc tieu la phan tang user toan cuc va uu tien ho tro chung.

## 2) Chot pham vi cho pipeline hien tai

Pipeline 8 phase hien tai cua du an dang van hanh theo **Dinh nghia B (user tong hop)**.

Bang chung tu code va du lieu:
- Aggregate du lieu theo `user_id`.
- Feature/model khong su dung `course_id`.
- Du lieu train/valid/test va bao cao phase 6/7 danh gia tren tap user tong hop.

Do do, toan bo kich ban ben duoi duoc viet theo user-level (tong hop).

## 3) Kich ban thuc nghiem 8 phase (phien ban chuan dang chay)

## Phase 1 - Data Preparation
- Muc tieu: dich/chu an hoa thong tin user va tong hop su kien tho thanh user metrics.
- Input chinh: `user.json`, `user-problem.json`, `user-video.json`, `reply.json`, `comment.json`.
- Output chinh:
  - `results/phase1/user_school_en.json`
  - `results/phase1/combined_user_metrics.csv`
  - `results/phase1/step2_user_week_activity.csv`
- Don vi du lieu sau phase: 1 dong = 1 user.

## Phase 2 - Data Cleaning
- Muc tieu: xu ly missing/noise/outlier co kiem soat.
- Input: `results/phase1/combined_user_metrics.csv`.
- Output: `results/phase2/combined_user_metrics_clean.csv`.
- Don vi du lieu: van la user-level.

## Phase 3 - Data Transformation
- Muc tieu: bien doi/scale dac trung cho downstream labeling va modeling.
- Input: `results/phase2/combined_user_metrics_clean.csv`.
- Output: `results/phase3/combined_user_metrics_transformed.csv`.
- Don vi du lieu: user-level.

## Phase 4 - Data Labeling
- Muc tieu: tao nhan `Low/Medium/High` (KMeans + chuan hoa engagement) va bao cao chat luong nhan.
- Input:
  - `results/phase3/combined_user_metrics_transformed.csv`
  - `results/phase1/step2_user_week_activity.csv`
- Output: `results/phase4/phase4_2_standard_labels_kmeans.csv`.
- Don vi danh gia: nhan gan cho tung user (khong tach theo course).

## Phase 5 - Data Splitting
- Muc tieu: chia train/valid/test va xu ly mat can bang.
- Input:
  - `results/phase4/phase4_2_standard_labels_kmeans.csv`
  - `results/phase3/combined_user_metrics_transformed.csv`
- Output:
  - `results/phase5/phase5_train_modeling.csv`
  - `results/phase5/phase5_valid.csv`
  - `results/phase5/phase5_test.csv`
- Luu y danh gia:
  - Chien luoc `group` dung `group_column=user_id` de tranh leakage theo user.
  - Van la bai toan phan loai user tong hop.

## Phase 6 - Model Training
- Muc tieu: train nhieu mo hinh va chon mo hinh tot nhat tren validation.
- Input: cac file split tu Phase 5.
- Output chinh:
  - `results/phase6/phase6_model_comparison.csv`
  - `results/phase6/phase6_classification_metrics.csv`
  - `results/phase6/phase6_confusion_matrix.csv`
  - `results/phase6/phase6_best_model.pkl`
  - `results/phase6/phase6_best_model_predictions.csv`
- Metric cot loi: `macro_f1`, `weighted_f1`, `accuracy`, `roc_auc_ovr_macro`, `recall_low`.
- Don vi danh gia: prediction cho tung user.

## Phase 7 - Model Evaluation
- Muc tieu: tong hop metric cho mo hinh duoc chon va check threshold thuc te.
- Input: artifacts tu phase 6 va bao cao trung gian.
- Output chinh:
  - `results/phase7/phase7_model_selection_summary.csv`
  - `results/phase7/phase7_metric_checks.csv`
  - `results/phase7/final_summary_report.txt`
- Don vi danh gia: hieu nang tong the tren tap user valid/test.

## Phase 8 - Model Interpretability
- Muc tieu: giai thich dac trung quan trong toan cuc/lop/mau.
- Input:
  - `results/phase6/phase6_best_model.pkl`
  - `results/phase6/phase6_best_model_predictions.csv`
  - `results/phase5/phase5_test.csv`
  - `results/phase7/final_summary_report.txt`
- Output: bo file `results/phase8/phase8_*` va tong ket giai thich.
- Don vi giai thich: user-level prediction.

## 4) Dinh nghia metric theo 2 don vi danh gia

### Neu danh gia theo user-course (Dinh nghia A)
- Macro-F1: trung binh F1 theo lop tren tap mau `(user, course)`.
- Co the bao cao them:
  - Macro-F1 theo tung `course_id`.
  - Weighted trung binh giua cac course (theo so mau course).

### Neu danh gia theo user tong hop (Dinh nghia B - dang dung)
- Macro-F1: trung binh F1 theo lop tren tap user.
- AUC/Recall Low: danh gia kha nang nhan dien user nguy co thap tham gia tren toan bo hanh vi hoc tap.

## 5) Kich ban chay de tai (de xuat de dua vao bao cao)

- Kich ban chinh (bat buoc):
  - Su dung Dinh nghia B voi pipeline 8 phase hien tai.
  - Chon model theo metric `macro_f1` tren validation.
  - Bao cao test: `macro_f1`, `weighted_f1`, `accuracy`, `roc_auc_ovr_macro`, `recall_low`.

- Kich ban mo rong (neu con thoi gian):
  - Tai cau truc du lieu sang Dinh nghia A (them truc `course_id`).
  - Chay lai phase 1-8 voi don vi `(user_id, course_id)`.
  - So sanh B vs A tren cung bo metric de danh gia loi ich theo mon hoc.

## 6) Lenh chay pipeline 8 phase hien tai

```bash
python experiment/run_experiment_stages.py --phase all
```

Tuy chon thuong dung:

```bash
python experiment/run_experiment_stages.py --phase all --split-strategy group --seed 42
```

```bash
python experiment/run_experiment_stages.py --phase 6 --phase6-primary-metric macro_f1
```
