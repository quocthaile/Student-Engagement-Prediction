# Báo Cáo Giai Đoạn 3-4: Chuẩn Bị Dữ Liệu và Huấn Luyện Mô Hình

## Giai Đoạn 3: Chia Tách Dữ Liệu (Data Splitting)

### Mục Tiêu
Chia dữ liệu thành ba tập con **độc lập** để:
- **Train**: Huấn luyện mô hình
- **Validation**: Tinh chỉnh ngưỡng (threshold) và chọn mô hình tốt nhất
- **Test**: Đánh giá hiệu năng trên dữ liệu hoàn toàn chưa nhìn thấy

### Phương Pháp Chia Tách

Dữ liệu gốc sau preprocessing được chia theo tỷ lệ **80:10:10**:

| Tập Dữ Liệu | Số Mẫu | Tỷ Lệ | Phân Bố Nhãn |
|---|---|---|---|
| **Train** | 105,684 | 80% | 1:1:1 (cân bằng) |
| **Validation** | 12,952 | 10% | Phân bố tự nhiên (~33:34:33) |
| **Test** | 12,952 | 10% | Phân bố tự nhiên (~33:34:33) |

**Nguyên Tắc Quan Trọng:**
1. Chia theo phân tầng (stratified split) dựa vào nhãn đích (target_label)
2. **Không rò rỉ dữ liệu (No Data Leakage)**: Validation và Test không bao giờ tham gia vào quá trình cân bằng dữ liệu
3. Cân bằng dữ liệu chỉ áp dụng cho **Train set** nếu cần thiết (tùy chọn)

### Cân Bằng Dữ Liệu (Optional)

**Vai Trò của SMOTE:**
- SMOTE (Synthetic Minority Oversampling Technique) là một kỹ thuật cân bằng dữ liệu không cân bằng
- Trong pipeline này, **SMOTE là tùy chọn**, có thể bật/tắt qua `config.py` (biến `ENABLE_SMOTE`)
- Nếu áp dụng, SMOTE chỉ được sử dụng trên **Train set** để tạo các mẫu tổng hợp cho các lớp thiểu số
- Validation và Test được giữ nguyên bản, phản ánh phân bố tự nhiên của dữ liệu thực tế

**Khi sử dụng SMOTE:**
- Train được cân bằng thành tỷ lệ 1:1:1 (mỗi lớp có ~35,228 mẫu)
- Validation và Test giữ nguyên (không thay đổi)

**Khi không sử dụng SMOTE:**
- Tất cả ba tập được giữ nguyên bản với phân bố tự nhiên

### Đặc Trưng Huấn Luyện

Mô hình được huấn luyện trên 12 đặc trưng:
1. **school_encoded** - Mã hóa trường học
2. **seq** - Thứ tự bài học đã hoàn thành
3. **speed** - Tốc độ hoàn thành bài tập
4. **rep_counts** - Số lần gửi lại
5. **cmt_counts** - Số lần nhận xét
6. **age** - Tuổi sinh viên
7. **gender_encoded** - Giới tính (mã hóa)
8. **num_courses** - Số khóa học đăng ký
9. **attempts_4w** - Số lần cố gắng trong 4 tuần
10. **is_correct_4w** - Tỷ lệ trả lời đúng
11. **score_4w** - Điểm số trung bình
12. **accuracy_rate_4w** - Tỷ lệ chính xác

### Kết Quả Giai Đoạn 3

- ✅ Dữ liệu được chia lọc sạch, không rò rỉ
- ✅ Validation/Test được giữ nguyên bản phân bố tự nhiên
- ✅ Train được cân bằng (tuỳ chọn) mà không ảnh hưởng đến evaluation sets
- ✅ Sản phẩm đầu ra: `train_smote.csv`, `valid_original.csv`, `test_original.csv`

---

## Giai Đoạn 4: Huấn Luyện Mô Hình và Chọn Lựa

### Mục Tiêu

1. Huấn luyện nhiều mô hình đại diện khác nhau
2. Tinh chỉnh ngưỡng quyết định trên **Validation set** duy nhất
3. Chọn mô hình tốt nhất dựa trên tiêu chí cân bằng
4. Đánh giá mô hình được chọn trên **Test set** để báo cáo kết quả cuối cùng

### Chiến Lược Chọn Mô Hình

#### Định Nghĩa Vấn Đề
Nhiệm vụ là dự báo học sinh có **độ tham gia thấp** (Low_Engagement) để có thể can thiệp sớm. Mô hình cần:
- Phát hiện được phần lớn học sinh độ tham gia thấp (Recall cao)
- Không dự báo sai quá nhiều học sinh thường lượng (giữ Precision hợp lý)
- Duy trì độ chính xác chung (Accuracy) ở mức sử dụng được

#### Chiến Lược Điểm Số (Validation-Based Selection)

Để chọn mô hình, ta dùng một **chiến lược điểm số nhiều tiêu chí** áp dụng trên Validation set:

**Bước 1: Cổng Tối Thiểu (Gate)**
- Chỉ xét những mô hình có `Recall_Low_Engagement ≥ 0.60`
- Mô hình phải phát hiện ít nhất 60% học sinh độ tham gia thấp

**Bước 2: Vùng Recall Thực Tế**
- Ưu tiên các mô hình có `Recall_Low_Engagement` trong dải [0.65, 0.85]
- Tránh các mô hình có Recall quá cao (> 0.90) hoặc quá thấp (< 0.60)
- Lý do: Recall = 1.0 có nghĩa dự báo tất cả học sinh là Low_Engagement → không hữu ích
- Recall quá thấp bỏ sót quá nhiều học sinh cần can thiệp

**Bước 3: Tối Ưu Hóa Thứ Cấp**
Trong nhóm mô hình đã qua cổng và đúng vùng Recall, ưu tiên theo thứ tự:
1. **Accuracy** - Độ chính xác chung (phần trăm dự báo đúng)
2. **Balanced_Accuracy** - Độ chính xác cân bằng (trung bình Recall trên từng lớp)
3. **Precision_Low_Engagement** - Độ chính xác của dự báo Low_Engagement
4. **F1_Low_Engagement** - Cân bằng Precision và Recall cho lớp Low_Engagement
5. **AUC_ROC_OVR** - Diện tích dưới đường cong ROC (One-vs-Rest)
6. **F1_Macro** - F1-score trung bình trên tất cả lớp

#### Tinh Chỉnh Ngưỡng (Threshold Tuning)

Mô hình phân loại thô (untrained) dự báo xác suất cho ba lớp. Để dự báo lớp Low_Engagement:

1. **Lấy điểm xác suất** cho lớp Low_Engagement từ mô hình
2. **Thử nhiều ngưỡng** (0.0 → 1.0) trên Validation set
3. **Tính Recall, Precision, F1** cho mỗi ngưỡng
4. **Chọn ngưỡng tốt nhất** là ngưỡng cho kết quả Recall thực tế và Accuracy cao nhất

**Ví Dụ Minh Họa:**
- Ngưỡng 0.05 → Dự báo ~95% là Low_Engagement → Recall = 1.0, Accuracy = 0.39 ❌ (Quá cao)
- Ngưỡng 0.45 → Dự báo ~54% là Low_Engagement → Recall = 0.74, Accuracy = 0.58 ✅ (Cân bằng)

### Các Mô Hình Ứng Viên (11 mô hình)

#### Nhóm 1: Logistic Regression (4 mô hình)
- **C=1.0 (L2)**: Regularization = 1.0, không cân bằng lớp
- **C=1.0 Balanced**: Regularization = 1.0, cân bằng lớp
- **C=0.3 Balanced**: Regularization = 0.3, cân bằng lớp (yếu hơn)
- **C=3.0 Balanced**: Regularization = 3.0, cân bằng lớp (mạnh hơn)

#### Nhóm 2: Linear SVC (2 mô hình)
- **Linear SVC**: Máy vector hỗ trợ tuyến tính, không cân bằng
- **Linear SVC Balanced**: Với cân bằng lớp

#### Nhóm 3: Decision Tree (2 mô hình)
- **Decision Tree**: max_depth=5, không cân bằng
- **Decision Tree Balanced**: max_depth=5, cân bằng lớp

#### Nhóm 4: Random Forest (2 mô hình)
- **Random Forest**: n_estimators=200, max_depth=10, cân bằng lớp
- **Random Forest Deep Balanced**: n_estimators=300, max_depth=None, cân bằng lớp (phức tạp hơn)

#### Nhóm 5: XGBoost (1 mô hình)
- **XGBoost**: max_depth=5, n_estimators=200, learning_rate=0.08 (gradient boosting)

### Kết Quả Chọn Mô Hình

#### Mô Hình Được Chọn: **Random Forest**

**Thông Số Mô Hình:**
- `n_estimators`: 200 (số cây quyết định)
- `max_depth`: 10 (độ sâu tối đa của mỗi cây)
- `class_weight`: "balanced" (cân bằng trọng số lớp)
- `criterion`: "gini" (hàm mất mát)
- `random_state`: 42 (tái tạo lại)

**Ngưỡng Quyết Định:**
- `Low_Threshold`: 0.45 (dự báo Low_Engagement nếu xác suất > 0.45)

**Lý Do Lựa Chọn:**
1. Đạt Recall_Low = 0.7361 (trên Validation) → trong dải lý tưởng [0.65, 0.85]
2. Accuracy cao: 0.5752 (Validation), 0.578 (Test) → cân bằng tốt
3. AUC_ROC = 0.7933 (Validation), 0.7945 (Test) → phân biệt tốt hai lớp
4. Không bị overfitting trên Validation (Validation metrics ≈ Test metrics)

#### Hiệu Năng Trên Validation Set

| Chỉ Số | Giá Trị |
|---|---|
| **Accuracy** | 0.5752 |
| **Balanced Accuracy** | 0.5809 |
| **Precision_Low** | 0.4513 |
| **Recall_Low** | 0.7361 |
| **F1_Low** | 0.5595 |
| **F1_Weighted** | 0.4696 |
| **F1_Macro** | 0.4741 |
| **AUC_ROC_OVR** | 0.7933 |

#### Hiệu Năng Trên Test Set (Đánh Giá Cuối Cùng)

| Chỉ Số | Giá Trị |
|---|---|
| **Accuracy** | 0.578 |
| **Balanced Accuracy** | 0.5837 |
| **Precision_Low** | 0.451 |
| **Recall_Low** | 0.7417 |
| **F1_Low** | 0.5609 |
| **F1_Weighted** | 0.4732 |
| **F1_Macro** | 0.4777 |
| **AUC_ROC_OVR** | 0.7945 |

**Nhận Xét:**
- Test metrics rất gần với Validation metrics → mô hình tổng quát hóa tốt
- Recall_Low = 74.17% → phát hiện được ~3/4 học sinh độ tham gia thấp
- Precision_Low = 45.1% → trong những dự báo Low_Engagement, ~45% thực sự thấp
- Accuracy chung = 57.8% → mô hình đúng hơn 1/2 lần cho tất cả 3 lớp

---

## Bảng Xếp Hạng Hiệu Năng Các Thuật Toán (Trên Test Set)

| Thứ Tự | Mô Hình | Validation Accuracy | Test Accuracy | Validation Recall_Low | Test Recall_Low | Validation F1_Low | Test F1_Low | Ngưỡng | Ghi Chú |
|---|---|---|---|---|---|---|---|---|---|
| **🥇 1** | **Random Forest** | **0.5752** | **0.578** | **0.7361** | **0.7417** | **0.5595** | **0.5609** | 0.45 | **Được chọn: Cân bằng tốt giữa Recall và Accuracy** |
| 🥈 2 | Random Forest Deep Balanced | 0.575 | - | 0.752 | - | 0.5643 | - | 0.35 | Recall cao hơn nhưng Accuracy hơi thấp |
| 🥉 3 | XGBoost | 0.5751 | - | 0.7501 | - | 0.564 | - | 0.35 | Hiệu năng tương đương, AUC cao |
| 4 | Decision Tree | 0.5749 | - | 0.7475 | - | 0.5627 | - | 0.3 | Recall cao, có nguy cơ overfitting |
| 5 | Decision Tree Balanced | 0.5749 | - | 0.7475 | - | 0.5627 | - | 0.3 | Hiệu năng tương đương với Decision Tree |
| 6 | Logistic Regression C1 | 0.5622 | - | 0.7625 | - | 0.5589 | - | 0.4 | Recall cao nhất, nhưng Accuracy thấp |
| 7 | Logistic Regression C0.3 Balanced | 0.5608 | - | 0.7625 | - | 0.5574 | - | 0.4 | Recall cao, Accuracy thấp |
| 8 | Logistic Regression C1 Balanced | 0.5608 | - | 0.7623 | - | 0.5574 | - | 0.4 | Tương đương C0.3 Balanced |
| 9 | Logistic Regression C3 Balanced | 0.5608 | - | 0.7623 | - | 0.5574 | - | 0.4 | Tương đương các C1 Balanced |
| 10 | Linear SVC | 0.555 | - | 0.6998 | - | 0.5471 | - | -0.18 | Recall thấp, không đạt mục tiêu tốt |
| 11 | Linear SVC Balanced | 0.555 | - | 0.6998 | - | 0.5471 | - | -0.18 | Recall thấp, không đạt mục tiêu tốt |

### Phân Tích Chi Tiết

**Nhóm Hiệu Năng Cao (Accuracy ≥ 0.575):**
- Random Forest, Random Forest Deep Balanced, XGBoost, Decision Trees
- Những mô hình này đạt được Recall_Low trong dải [0.74, 0.76] và Accuracy [0.575, 0.5752]
- Khuyến nghị: Random Forest được chọn vì đạt Accuracy cao nhất (0.5752) trong nhóm này

**Nhóm Trung Bình (Accuracy 0.56-0.57):**
- Các biến thể Logistic Regression
- Recall_Low cao (0.76), nhưng Accuracy thấp hơn (0.56)
- Có nguy cơ đánh giá sai các lớp Medium/High Engagement

**Nhóm Thấp (Accuracy < 0.56):**
- Linear SVC
- Recall_Low quá thấp (0.70), không đạt mục tiêu phát hiện học sinh
- Không khuyến nghị để production

### Kết Luận Giai Đoạn 4

✅ **Hoàn thành thành công:**
- 11 mô hình được huấn luyện trên Train set
- Các ngưỡng tối ưu được tìm thấy trên Validation set
- Random Forest được chọn với chiến lược đa tiêu chí cân bằng
- Hiệu năng Test: **Accuracy 57.8%, Recall_Low 74.17%, AUC 0.7945**
- Mô hình sẵn sàng để triển khai hoặc tiếp tục cải thiện

---

## Tóm Tắt Đường Ống (Pipeline Summary)

| Giai Đoạn | Hoạt Động | Kết Quả |
|---|---|---|
| **Giai Đoạn 3** | Chia dữ liệu 80:10:10; cân bằng Train (tùy chọn) | Train: 105,684; Valid: 12,952; Test: 12,952 |
| **Giai Đoạn 4** | Huấn luyện 11 mô hình; tinh chỉnh ngưỡng trên Valid; chọn Random Forest | Test Accuracy: 57.8%, Recall_Low: 74.17% |
| **Giai Đoạn 5** | Giải thích mô hình (XAI) và báo cáo kết quả | Danh sách đặc trưng quan trọng và độ ảnh hưởng |

---

## Tài Liệu Tham Khảo

- Tệp cấu hình: `experiment/config.py`
- Script Giai Đoạn 3: `experiment/stage_3_split_and_smote.py`
- Script Giai Đoạn 4: `experiment/stage_4_model_training_eval.py`
- Kết quả đầu ra: `experiment/deployment_models/`
