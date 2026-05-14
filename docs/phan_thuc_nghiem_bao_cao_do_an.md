# Phần Thực Nghiệm Báo Cáo Đồ Án

Tài liệu này viết lại phần thực nghiệm của báo cáo đồ án dựa trên lần chạy pipeline gần nhất, với dữ liệu log ghi nhận ở các giai đoạn Stage 1 đến Stage 4.

Mục tiêu của phần thực nghiệm là trình bày rõ:

- quy trình tạo nhãn mục tiêu từ dữ liệu toàn khóa;
- cách trích xuất đặc trưng từ 28 ngày đầu để tránh leak dữ liệu;
- cách chia dữ liệu và cân bằng lớp;
- quy trình huấn luyện, chọn mô hình và đánh giá cuối cùng.

---

## 1. Thiết lập thực nghiệm

Bài toán thực nghiệm là dự đoán mức độ tham gia học tập của sinh viên theo ba mức:

- `Low_Engagement`
- `Medium_Engagement`
- `High_Engagement`

Pipeline hiện tại được thiết kế theo hướng cảnh báo sớm. Vì vậy, dữ liệu dùng để huấn luyện mô hình chỉ lấy từ 28 ngày đầu tiên kể từ khi người học bắt đầu tham gia, trong khi nhãn mục tiêu vẫn được sinh từ toàn bộ quá trình học của sinh viên.

Lần chạy pipeline gần nhất cho thấy:

- số bản ghi hành vi thô: `4,438,340`;
- số người học duy nhất: `129,516`;
- dữ liệu đầu ra ở Stage 1 và Stage 2 đều giữ nguyên số người học, nhưng được biến đổi thành các bảng đặc trưng và nhãn phù hợp cho huấn luyện.

---

## 2. Giai đoạn 1: Sinh nhãn mục tiêu từ toàn khóa

Ở Stage 1, hệ thống đọc dữ liệu hành vi thô từ file parquet và tổng hợp theo từng `user_id` để sinh nhãn mục tiêu cho toàn bộ khóa học.

### 2.1. Dữ liệu đầu vào

Nguồn dữ liệu đầu vào là file:

- `dataset/combined_all_data.parquet`

Các cột chính được sử dụng trong bước này gồm:

- `user_id`
- `attempts`
- `is_correct`
- `score`
- `create_time_x`
- `create_time_y`

### 2.2. Kết quả tổng hợp

Sau khi group theo `user_id`, dữ liệu được rút gọn từ `4,438,340` bản ghi sự kiện xuống còn `129,516` sinh viên, tương ứng trung bình khoảng `34.3` bản ghi cho mỗi người học.

Các chỉ số tổng hợp chính ở Stage 1 gồm:

- tổng `attempts`: `2,613,465`;
- tổng `is_correct`: `1,812,892`;
- `avg_score`: `0.0039`;
- `video_clicks`: `4,059,617`;
- `forum_posts`: `4,254,109`.

### 2.3. Nhãn mục tiêu

Stage 1 áp dụng phân vị tự nhiên để gán nhãn, với các mốc phân vị `0.33` và `0.67`. Cách làm này giúp phân phối nhãn thực tế cân bằng hơn thay vì ép tỷ lệ cố định `1:1:1` một cách cơ học.

Phân phối nhãn thu được như sau:

| Nhãn | Số lượng | Tỷ lệ |
| :--- | ---: | ---: |
| `Low_Engagement` | `42,740` | `33.0%` |
| `Medium_Engagement` | `44,036` | `34.0%` |
| `High_Engagement` | `42,740` | `33.0%` |

File đầu ra chính của Stage 1:

- `dataset/ground_truth_labels.csv`
- `dataset/ground_truth_report.csv`

Ngoài ra, Stage 1 còn lưu snapshot đặc trưng tổng hợp tại:

- `dataset/user_features_aggregated.csv`

---

## 3. Giai đoạn 2: Trích xuất đặc trưng theo cửa sổ 28 ngày

Stage 2 có nhiệm vụ cắt dữ liệu về 28 ngày đầu tiên kể từ thời điểm bắt đầu học của mỗi sinh viên, sau đó tổng hợp đặc trưng hành vi theo cửa sổ thời gian cố định `4w`.

### 3.1. Mục tiêu của Stage 2

Mục tiêu chính của Stage 2 là chặn leak dữ liệu. Nếu dùng hành vi ở toàn bộ khóa học để dự đoán nhãn toàn khóa, mô hình có thể vô tình học thông tin từ tương lai. Do đó, chỉ 28 ngày đầu tiên được giữ lại cho phần đặc trưng đầu vào của mô hình.

### 3.2. Kiểm tra và chuẩn hóa thời gian

Theo log, pipeline đã:

- chuẩn hóa mốc thời gian hành vi;
- xác định thời điểm bắt đầu học cho từng người học;
- tính số ngày kể từ lúc bắt đầu học cho từng hành vi;
- đo độ dài hành vi tối đa theo từng user để hỗ trợ relative windows.

Do file parquet không có cột `enroll_time`, pipeline đã dùng fallback bằng thời điểm hành vi sớm nhất của từng user để xác định mốc bắt đầu học.

Thống kê `days_since_enroll` sau khi chuẩn hóa:

- min: `0.0`
- max: `421.0`
- mean: `83.77`

### 3.3. Chặn leak dữ liệu bằng cửa sổ 28 ngày

Sau khi chuẩn hóa timeline, Stage 2 chỉ giữ lại hành vi trong `28` ngày đầu tiên.

Kết quả sau khi cắt thời gian:

- số dòng còn lại: `1,231,213`;
- số người học: `129,516`.

### 3.4. Đặc trưng được trích xuất

Stage 2 tổng hợp đặc trưng cho từng sinh viên từ 28 ngày đầu, sau đó xuất file tiền xử lý cho các stage sau.

Các đặc trưng chính ghi nhận trong log gồm:

- `attempts_4w`
- `is_correct_4w`
- `score_4w`
- `accuracy_rate_4w`

Kết quả đầu ra quan trọng:

- `dataset/stage2_action_timeline.csv`
- `dataset/user_features_4w.csv`
- `dataset/pos-processing_dataset.csv`
- `dataset/time_window_comparison.csv`
- `dataset/user_features_and_wes.csv`

---

## 4. Giai đoạn 3: Chia tập và cân bằng lớp

Stage 3 chịu trách nhiệm chia dữ liệu thành train, validation và test, sau đó cân bằng tập huấn luyện bằng SMOTE/SMOTENC.

### 4.1. Dữ liệu đầu vào

Tập dữ liệu đầu vào của Stage 3 có:

- `129,516` dòng;
- các cột chính gồm `school`, `school_encoded`, `stage2_action_time`, `attempts_4w`, `is_correct_4w`, `score_4w`, `accuracy_rate_4w`, `num_courses`, `seq`, `speed`, `rep_counts`, `cmt_counts`, `year_of_birth`, `gender`, `target_label`.

### 4.2. Chia dữ liệu

Dữ liệu được chia theo tỷ lệ xấp xỉ `8:1:1`:

- Train: `103,612` mẫu;
- Validation: `12,952` mẫu;
- Test: `12,952` mẫu.

Phân phối nhãn của validation và test giữ nguyên phân phối tự nhiên, không bị can thiệp bởi SMOTE.

### 4.3. Cân bằng tập train

Stage 3 áp dụng SMOTENC trên tập train để xử lý cả đặc trưng số và đặc trưng phân loại.

Kết quả sau SMOTE:

- số mẫu train sau cân bằng: `105,684`;
- phân phối nhãn train sau SMOTE: `35,228` mẫu cho mỗi lớp.

### 4.4. File đầu ra

Các file chính được lưu lại gồm:

- `dataset/model_data/train_smote.csv`
- `dataset/model_data/valid_original.csv`
- `dataset/model_data/test_original.csv`
- `experiment/deployment_models/label_encoder.pkl`

---

## 5. Giai đoạn 4: Huấn luyện, chọn mô hình và đánh giá

Stage 4 là phần trung tâm của thực nghiệm. Ở giai đoạn này, nhiều mô hình được huấn luyện trên tập train đã cân bằng, sau đó được đánh giá trên validation để chọn mô hình và ngưỡng dự đoán tốt nhất cho lớp `Low_Engagement`.

### 5.1. Mục tiêu chọn mô hình

Pipeline hiện tại ưu tiên bài toán cảnh báo sớm, nên tiêu chí chọn model không chỉ dựa trên accuracy. Các chỉ tiêu quan trọng được xem xét bao gồm:

- `Recall_Low_Engagement`
- `Precision_Low_Engagement`
- `F1_Low_Engagement`
- `Accuracy`
- `Balanced_Accuracy`
- `AUC_ROC_OVR`
- `F1_Macro`

### 5.2. Các mô hình được thử nghiệm

Theo log, Stage 4 huấn luyện và so sánh các mô hình sau:

- Logistic Regression C1
- Logistic Regression C0.3 Balanced
- Logistic Regression C1 Balanced
- Logistic Regression C3 Balanced
- Linear SVC
- Linear SVC Balanced
- Decision Tree
- Decision Tree Balanced
- Random Forest
- Random Forest Deep Balanced
- XGBoost

### 5.3. Kết quả trên validation

Kết quả validation cho thấy các mô hình có hiệu năng khá sát nhau, nhưng `Random Forest` là mô hình được chọn theo policy hiện tại.

Kết quả nổi bật của mô hình được chọn trên validation:

- Accuracy: `0.5752`
- Balanced Accuracy: `0.5809`
- F1 Macro: `0.4741`
- F1 Weighted: `0.4696`
- Recall `Low_Engagement`: `0.7361`
- Precision `Low_Engagement`: `0.4513`
- F1 `Low_Engagement`: `0.5595`
- AUC ROC OVR: `0.7933`
- Low threshold: `0.45`
- Decision policy: `low_threshold_predict_proba`

### 5.4. Kết quả trên test

Sau khi khóa mô hình và threshold từ validation, Stage 4 đánh giá một lần duy nhất trên tập test giữ lại độc lập.

Kết quả test của mô hình cuối cùng:

- Accuracy: `0.5780`
- Balanced Accuracy: `0.5837`
- F1 Macro: `0.4777`
- F1 Weighted: `0.4732`
- Precision Macro: `0.5709`
- Recall Macro: `0.5837`
- Precision `Low_Engagement`: `0.4510`
- Recall `Low_Engagement`: `0.7417`
- F1 `Low_Engagement`: `0.5609`
- AUC ROC OVR: `0.7945`
- Predicted Low Rate: `0.5427`
- Actual Low Rate: `0.33`

### 5.5. Diễn giải kết quả

Kết quả trên cho thấy mô hình đã phát hiện được phần lớn sinh viên thuộc nhóm nguy cơ thấp. Recall của `Low_Engagement` đạt khoảng `0.74`, nghĩa là mô hình bắt được phần lớn trường hợp cần cảnh báo.

Đồng thời, accuracy toàn cục đạt khoảng `0.5780`, cao hơn so với cấu hình trước đó khi threshold bị đẩy quá thấp và mô hình dự đoán Low quá nhiều.

Điều này cho thấy:

- SMOTE có vai trò hỗ trợ cân bằng tập train;
- threshold tuning ảnh hưởng rất lớn đến hiệu năng cuối;
- validation là tập quyết định mô hình, còn test chỉ dùng để benchmark cuối cùng.

### 5.6. Mô hình triển khai

Mô hình được chọn cuối cùng trong lần chạy này là **Random Forest**.

Các artifact triển khai chính gồm:

- `experiment/deployment_models/best_model_4w.pkl`
- `experiment/deployment_models/deployment_bundle.pkl`
- `experiment/deployment_models/best_model_metadata.json`
- `experiment/deployment_models/evaluation_metrics.csv`
- `experiment/deployment_models/validation_summary_table.csv`
- `experiment/deployment_models/final_test_metrics.csv`

Ngoài ra, Stage 4 còn sinh các báo cáo và hình ảnh phục vụ phân tích:

- `experiment/deployment_models/best_model_valid_classification_report.csv`
- `experiment/deployment_models/best_model_test_classification_report.csv`
- `experiment/output_images_4w/CM_TEST_Random_Forest.png`

---

## 6. Tóm tắt kết quả thực nghiệm

Có thể tóm tắt phần thực nghiệm như sau:

1. Dữ liệu gốc có hơn `4.4` triệu sự kiện và `129,516` người học.
2. Stage 1 sinh nhãn toàn khóa với phân phối gần cân bằng giữa ba lớp.
3. Stage 2 chỉ giữ hành vi trong `28` ngày đầu để tránh leak dữ liệu.
4. Stage 3 chia dữ liệu theo tỷ lệ `8:1:1` và cân bằng tập train bằng SMOTENC.
5. Stage 4 huấn luyện nhiều mô hình, chọn `Random Forest` theo validation, rồi đánh giá cuối cùng trên test.
6. Mô hình cuối đạt `Accuracy = 0.5780`, `Balanced Accuracy = 0.5837`, `Recall_Low_Engagement = 0.7417` trên test.

Kết quả này phù hợp với mục tiêu cảnh báo sớm: mô hình không chỉ nhắm tới accuracy, mà ưu tiên phát hiện phần lớn sinh viên có nguy cơ thấp trong khi vẫn giữ mức sai lệch ở ngưỡng chấp nhận được.

---

## 7. Ghi chú sử dụng trong báo cáo

Nếu cần đưa nội dung này vào báo cáo đồ án, có thể nhấn mạnh ba ý chính:

- Stage 1 tạo nhãn mục tiêu từ toàn khóa học;
- Stage 2 chỉ dùng 28 ngày đầu để xây dựng đặc trưng đầu vào;
- Stage 4 chọn mô hình và threshold dựa trên validation, sau đó mới đánh giá test một lần duy nhất.

Như vậy, quy trình thực nghiệm đảm bảo cả tính thực tế của bài toán cảnh báo sớm lẫn tính khách quan của quá trình đánh giá.