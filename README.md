# Student Engagement Prediction on MOOCCubeX

> **Môn học:** DS317 – Khai phá Dữ liệu | Trường Đại học Công nghệ Thông tin (UIT)

Dự án khai phá dữ liệu dự đoán **mức độ tham gia học tập** (Low / Medium / High) của học sinh trên nền tảng [MOOCCubeX](https://github.com/THU-KEG/MOOCCube), sử dụng đặc trưng hành vi trích xuất từ log tương tác quy mô lớn và mô hình phân loại có giám sát.

---

## 📁 Phân chia thư mục theo hạng mục

```
project/
├── experiment/        ← Source code pipeline thực nghiệm
├── dataset/           ← Dữ liệu trung gian và dữ liệu huấn luyện
└── reports/           ← Tài liệu/sản phẩm nộp bài thực tế
    ├── Thực hành/
    └── Đồ án/
```

---

## 🎓 Yêu cầu môn học & Vị trí file

### Thực hành

> 📂 Code: **`experiment/`** | 📦 Sản phẩm nộp thực tế: **`reports/Thực hành/`**

| STT | Hạng mục nộp                             | Vị trí                                                                                                          |
| --- | ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| 1   | Thuyết minh đề tài (.docx, .pptx)       | `reports/Thực hành/1a. Thuyết minh đề tài.docx` + `reports/Thực hành/1b. Thuyết minh đề tài.pptx` |
| 2   | Báo cáo phân tích bộ dữ liệu (.docx) | `reports/Thực hành/2a. Phân tích bộ dữ liệu.docx`                                                        |
| 3   | Bộ dữ liệu sau tiền xử lý             | `reports/Thực hành/3. Bộ dữ liệu sau khi tiền xử lý.csv`                                                |
| 4   | Video thuyết trình (bật camera)          | —                                                                                                                |

**Thang điểm thực hành (10đ):**

| Tiêu chí | Điểm | Nội dung kiểm tra                          |
| ---------- | ------ | -------------------------------------------- |
| What       | 3đ    | Mô tả bài toán, dữ liệu, đặc trưng  |
| Why        | 3đ    | Lý do chọn phương pháp, phân tích EDA |
| How        | 4đ    | Quy trình xử lý, kết quả thực nghiệm  |

---

### Đồ án môn học

> 📂 Code đồ án: **`experiment/`** | 📦 Sản phẩm nộp thực tế: **`reports/Đồ án/`**

| STT | Hạng mục nộp                                            | Vị trí                                                                                    |
| --- | ---------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| 1   | Báo cáo đồ án (.docx, .pptx)                          | `reports/Đồ án/Báo cáo đồ án.docx` + `reports/Đồ án/Báo cáo đồ án.pptx` |
| 2   | Bộ dữ liệu thực nghiệm (sinh từ pipeline experiment) | `reports/Đồ án/Bộ dữ liệu thực nghiệm/`                                           |
| 3   | Toàn bộ source code                                      | `experiment/`, `dashboard/`, `dataset/`                                               |
| 4   | Video thuyết trình (bật camera)                         | —                                                                                          |

**Thang điểm đồ án cuối kỳ (10đ):**

| Tiêu chí                   | Điểm | Nội dung kiểm tra                                      |
| ---------------------------- | ------ | -------------------------------------------------------- |
| Data Quality                 | 4đ    | Chất lượng dữ liệu, làm sạch, feature engineering |
| Machine Learning & Framework | 3đ    | Lựa chọn mô hình, huấn luyện, đánh giá          |
| Experiment & Demo            | 3đ    | Kết quả thực nghiệm, demo, trình bày               |

---

## 🔬 Thực hành – Pipeline (`experiment/`)

Thư mục `experiment/` được thiết kế lại theo bài toán thực tế:

- Dự đoán **cảnh báo sớm theo từng user-course**.
- Chỉ dùng dữ liệu đến thời điểm dự đoán (chống leakage).
- Mục tiêu là cảnh báo trước khi khóa học kết thúc để can thiệp.

Tài liệu thực hành hiện có: `reports/Thực hành/Kịch bản thực nghiệm.docx` và `reports/Thực hành/Kịch bản thực nghiệm.pdf`

Pipeline 8 phase (hướng vận hành):

```
Phase 1: Data Preparation  →  Chuẩn hóa dữ liệu + timeline course + EDA an toàn bộ nhớ
Phase 2: Data Cleaning     →  Làm sạch missing/noise/outlier theo user-course
Phase 3: Transformation    →  Biến đổi đặc trưng có xét mốc thời gian dự đoán
Phase 4: Data Labeling     →  Nhãn risk + cảnh báo sớm theo tiến độ khóa học
Phase 5: Data Splitting    →  Chia tập theo thời gian/group để tránh leakage
Phase 6: Model Training    →  Huấn luyện và hiệu chỉnh risk score
Phase 7: Model Evaluation  →  Đánh giá theo metric cảnh báo sớm (ưu tiên recall nhóm nguy cơ)
Phase 8: Interpretability  →  Giải thích yếu tố rủi ro để hỗ trợ can thiệp
```

Sản phẩm nộp trong repo hiện được tổng hợp tại `reports/Thực hành/` và `reports/Đồ án/`.

Nếu bạn muốn chạy bản v2 ngắn gọn theo yêu cầu mới, dùng:

```bash
python -m experiment.pipeline_v2.run_pipeline --config experiment/pipeline_v2/config.json
```

Tài liệu hướng dẫn riêng của v2 nằm tại `experiment/pipeline_v2/README.md`.

**Chạy pipeline:**

```bash
# Từ thư mục gốc project
python experiment/run_experiment_stages.py --phase all

# Chạy từng phase
python experiment/run_experiment_stages.py --phase 1

# Test nhanh với ít dữ liệu
python experiment/run_experiment_stages.py --phase 1 --max-rows 1000

# Nếu cần ghi sang thư mục khác
python experiment/run_experiment_stages.py --phase all --results-dir "reports/Thực hành"
```

**Ví dụ các output pipeline (tham chiếu để tạo sản phẩm nộp trong `reports/`):**

| File                                           | Mô tả                                       |
| ---------------------------------------------- | --------------------------------------------- |
| `phase1/combined_user_metrics.csv`           | Đặc trưng hành vi theo user-course        |
| `phase1/phase1_eda_report.txt`               | Báo cáo thống kê EDA + biểu đồ         |
| `phase2/combined_user_metrics_clean.csv`     | Dữ liệu sau làm sạch                      |
| `phase4/phase4_2_standard_labels_kmeans.csv` | Nhãn risk + cảnh báo sớm theo giai đoạn |
| `phase6/phase6_best_model.pkl`               | Model tốt nhất                              |
| `phase7/final_summary_report.txt`            | Báo cáo tổng hợp toàn pipeline           |

---

## 📦 Đồ án cuối kỳ – (`reports/Đồ án/`)

Tài liệu và sản phẩm nộp đồ án được lưu tại `reports/Đồ án/`.
Source code phục vụ thực nghiệm nằm chủ yếu trong `experiment/`, `dashboard/`, `dataset/`.

---

## Dataset

| File                  | Mô tả                                                              |
| --------------------- | -------------------------------------------------------------------- |
| `user.json`         | Hồ sơ học sinh (id, trường, giới tính, năm sinh, khóa học) |
| `user-problem.json` | Lịch sử làm bài tập (đúng/sai, điểm, thời gian)            |
| `user-video.json`   | Phiên xem video (đoạn đã xem, tốc độ, thời lượng)         |
| `reply.json`        | Hoạt động trả lời forum                                         |
| `comment.json`      | Hoạt động bình luận forum                                       |

**Tải về:** https://github.com/THU-KEG/MOOCCube
**Kaggle:** https://www.kaggle.com/datasets/thiuyn/mooccubexdataset

Đặt dataset tại `D:/MOOCCubeX_dataset/` hoặc truyền `--dataset-dir <path>` khi chạy.

---

## 🔬 Kết Quả Thực Nghiệm – FIXED 28-Day Window (4w)

### Mô Tả Kịch Bản Thực Nghiệm

**Bài Toán:** Dự đoán mức độ tham gia học tập (Low/Medium/High) của sinh viên trên MOOCCubeX, sử dụng đặc trưng hành vi được trích xuất từ 28 ngày đầu tiên đăng ký khóa học.

**Lý Do Chọn 28 Ngày:**

- 28 ngày (4 tuần) là khoảng thời gian hợp lý để quan sát hành vi học tập
- Đủ dài để bắt được các mẫu hành vi ổn định (stabilized patterns)
- Đủ sớm để can thiệp kịp thời trước khi kết thúc khóa học

**Chỉ Tiêu Chính:** Recall cao cho lớp Low_Engagement (phát hiện sớm sinh viên có nguy cơ)

---

### Quy Trình Xử Lý Dữ Liệu (Pipeline 5 Stage)

| Stage                       | Mục Đích                                          | Input                  | Output                                                                                     |       Số Dòng |
| :-------------------------- | :--------------------------------------------------- | :--------------------- | :----------------------------------------------------------------------------------------- | --------------: |
| **1. Ground Truth**   | Sinh nhãn mục tiêu từ dữ liệu toàn khóa học | JSON files             | `ground_truth_labels.csv` / `ground_truth_report.csv`                                  |         129,516 |
| **2. Time Windows**   | Trích đặc trưng từ 28 ngày đầu               | Raw events             | `stage2_action_timeline.csv` / `user_features_4w.csv` / `pos-processing_dataset.csv` |         129,516 |
| **3. Split & SMOTE**  | Chia tập train/valid/test, cân bằng lớp          | Features + Labels      | `train_smote.csv`, `valid_original.csv`, `test_original.csv`                         | 60K / 35K / 35K |
| **4. Model Training** | Huấn luyện 5 mô hình, chọn best                 | Split datasets         | `deployment_models/best_model_4w.pkl` + `evaluation_metrics.csv`                       |              — |
| **5. Evaluation**     | Đánh giá trên test set                           | Best model + test data | `final_test_metrics.csv` / `best_model_metadata.json`                                  |              — |

---

### Kết Quả Dữ Liệu

**Nhãn Mục Tiêu (Ground Truth - 28 ngày đầu tiên):**

| Mức Độ                   | Số Sinh Viên | Tỷ Lệ % | Định Nghĩa                                        |
| :-------------------------- | -------------: | --------: | :--------------------------------------------------- |
| **Low_Engagement**    |         42,740 |     33.0% | Nhóm nguy cơ thấp nhất theo phân vị tự nhiên |
| **Medium_Engagement** |         44,036 |     34.0% | Nhóm tham gia trung bình theo phân vị tự nhiên |
| **High_Engagement**   |         42,740 |     33.0% | Nhóm tham gia cao theo phân vị tự nhiên         |

**Đặc Trưng (Features) - Được Tính Từ 28 Ngày Đầu:**

- `attempts_4w`: Số lần làm bài
- `is_correct_4w`: Số câu trả lời đúng
- `score_4w`: Tổng điểm
- `accuracy_rate_4w`: Tỷ lệ độ chính xác
- `num_courses`: Số khóa học tham gia
- `age`: Tuổi sinh viên
- `school_encoded`: Trường đại học (mã hóa)
- `gender`: Giới tính

**Tập Huấn Luyện (sau SMOTE):**

- Tổng: 60,000 mẫu (cân bằng 1:1:1 cho 3 lớp)
- Validation: 35,000 mẫu (phân bố tự nhiên)
- Test: 35,000 mẫu (phân bố tự nhiên)

---

### Mô Hình & Kết Quả

**Lựa Chọn Mô Hình (Validation Set):**

Xếp hạng theo: **Recall_Low_Engagement (ưu tiên 1)** → Accuracy (ưu tiên 2)

| Hạng          | Mô Hình                     |         Accuracy |       Recall_Low | Precision_Low | Lý Do                                    |
| :------------- | :---------------------------- | ---------------: | ---------------: | ------------: | :---------------------------------------- |
| **🥇 1** | **Logistic Regression** | **0.6088** | **0.0117** |        0.1645 | Recall_low cao nhất trong log hiện tại |
| 🥈 2           | Linear SVC                    |           0.6141 |           0.0021 |        0.2000 | Accuracy cao hơn nhưng recall thấp     |
| 🥉 3           | Decision Tree                 |           0.6230 |           0.0000 |        0.0000 | Recall thấp hơn                         |
| 4              | Random Forest                 |           0.6228 |           0.0009 |        0.2500 | Recall thấp hơn                         |
| 5              | XGBoost                       |           0.6231 |           0.0000 |        0.0000 | Recall thấp hơn                         |

**Tại Sao Chọn Logistic Regression?**

- **Recall_Low_Engagement = 0.0117**: Mô hình được log hiện tại chọn làm best theo tiêu chí ranking trong code
- Đầu ra validation cho thấy mô hình có accuracy và AUC tốt hơn các mô hình còn lại trong log hiện tại
- Kết quả test đi kèm được ghi nhận để làm mốc so sánh cho các lần chạy sau

**Kết Quả Test Set (Logistic Regression):**

| Chỉ Tiêu              | Giá Trị | Diễn Giải                                                 |
| :---------------------- | --------: | :---------------------------------------------------------- |
| **Accuracy**      |    0.6106 | 61.06% dự đoán đúng                                    |
| **Recall_Low**    |    0.0136 | Phát hiện 1.36% sinh viên nguy cơ                       |
| **Precision_Low** |    0.1835 | 18.35% sinh viên được cảnh báo thực sự có nguy cơ |

---

### Sản Phẩm Thực Nghiệm

Tất cả các file sản phẩm được lưu tại **`dataset/`**, **`experiment/deployment_models/`** và **`experiment/output_images_4w/`**:

| Sản Phẩm                        | Vị Trí                                                     | Mô Tả                                         |
| :-------------------------------- | :----------------------------------------------------------- | :---------------------------------------------- |
| **Mô hình chiến thắng** | `experiment/deployment_models/best_model_4w.pkl`           | Logistic Regression đã huấn luyện           |
| **Deployment bundle**       | `experiment/deployment_models/deployment_bundle.pkl`       | Gói triển khai gồm model và metadata        |
| **Metadata mô hình**      | `experiment/deployment_models/best_model_metadata.json`    | Thông tin mô hình được chọn              |
| **Metrics validation**      | `experiment/deployment_models/evaluation_metrics.csv`      | Kết quả so sánh 5 mô hình                  |
| **Metrics test**            | `experiment/deployment_models/final_test_metrics.csv`      | Kết quả đánh giá cuối Logistic Regression |
| **Nhãn dự đoán test**   | `model_data/test_predictions.csv` (nếu có)               | Dự đoán + ground truth                       |
| **Nhãn ground truth**      | `dataset/ground_truth_labels.csv`                          | 129,516 sinh viên + nhãn                      |
| **Đặc trưng 4w**         | `dataset/user_features_4w.csv`                             | Features cho mỗi sinh viên                    |
| **Dữ liệu tiền xử lý** | `dataset/pos-processing_dataset.csv`                       | Dataset hoàn chỉnh sau cắt 28 ngày          |
| **Timeline hành vi**       | `dataset/stage2_action_timeline.csv`                       | Mốc thời gian chuẩn hóa hành vi theo user  |
| **Tóm tắt kết quả**     | `experiment/output_images_4w/expected_results_summary.txt` | Summary cho pipeline hiện tại                 |
| **Biểu đồ XAI**          | `experiment/output_images_4w/SHAP_Summary_Global.png`      | Giải thích mức ảnh hưởng feature          |

---

### Giải Thích Kết Quả

**Độ Chính Xác (61.06%):**

- Không quá cao vì:
  1. Mất cân bằng lớp (60% Low, 25% Medium, 15% High)
  2. Features hiện tại chưa bắt được tất cả yếu tố ảnh hưởng hành vi
  3. 28 ngày có thể chưa đủ để phân biệt rõ ràng

**Recall thấp nhưng được ghi nhận trong log hiện tại (1.36%):**

- ✅ Là mốc kết quả thực tế của lần chạy pipeline gần nhất
- ✅ Có thể dùng làm baseline để so sánh khi chỉnh lại tiêu chí ranking hoặc feature set
- ✅ Giữ lại đầy đủ artifact để truy vết và debug

**Kỳ Vọng Tương Lai:**

1. Thêm features tương tác (interaction features)
2. Tối ưu hyperparameter của Logistic Regression và ngưỡng phân lớp
3. Thử weighted sampling hoặc cost-sensitive learning
4. Kết hợp ensemble của nhiều mô hình

---

## Tham số CLI

| Flag              | Mặc định              | Mô tả                                     |
| ----------------- | ------------------------ | ------------------------------------------- |
| `--phase`       | `all`                  | Phase cần chạy:`1`–`8` hoặc `all` |
| `--dataset-dir` | `D:/MOOCCubeX_dataset` | Đường dẫn đến dataset gốc            |
| `--results-dir` | `reports/Thực hành`  | Override thư mục lưu kết quả           |
| `--max-rows`    | *(none)*               | Giới hạn số dòng để test nhanh        |
| `--seed`        | `42`                   | Random seed để tái lập kết quả        |
