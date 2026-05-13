# Giải thích kết quả Stage 4

Tài liệu này dùng để diễn giải kết quả của Stage 4 theo văn phong báo cáo đồ án. Nội dung tập trung vào:

- ý nghĩa tên các mô hình như `C1`, `C0.3 Balanced`
- cách đọc bảng validation và test
- vì sao Stage 4 chọn `Random Forest`
- vì sao recall của lớp `Low_Engagement` không còn bị ép lên 1.0

## 1. Ý nghĩa các tên mô hình

Trong Stage 4, nhiều mô hình được thử với các biến thể khác nhau. Tên mô hình thường được ghép từ 3 phần: loại thuật toán, tham số cấu hình, và chiến lược cân bằng lớp.

### 1.1 Logistic Regression C1, C0.3, C3

`C` là tham số điều chuẩn của Logistic Regression. Đây là nghịch đảo của độ mạnh regularization:

- `C` lớn hơn: regularization yếu hơn, mô hình linh hoạt hơn
- `C` nhỏ hơn: regularization mạnh hơn, mô hình bị “siết” nhiều hơn để tránh overfit

Giải thích từng biến thể:

- `Logistic Regression C1`: Logistic Regression với `C = 1.0`
- `Logistic Regression C0.3`: Logistic Regression với `C = 0.3`, regularization mạnh hơn
- `Logistic Regression C3`: Logistic Regression với `C = 3.0`, regularization yếu hơn

### 1.2 Balanced

Từ `Balanced` nghĩa là mô hình dùng chiến lược cân bằng lớp bằng `class_weight='balanced'`.

Ý nghĩa:

- các lớp hiếm sẽ được gán trọng số lớn hơn trong quá trình học
- mô hình được khuyến khích quan tâm hơn tới lớp khó phát hiện, đặc biệt là `Low_Engagement`
- mục tiêu là giảm tình trạng mô hình chỉ đoán tốt lớp phổ biến mà bỏ qua lớp nguy cơ

Ví dụ:

- `Logistic Regression C1 Balanced`: Logistic Regression với `C = 1.0` và trọng số lớp cân bằng
- `Linear SVC Balanced`: Linear SVC có `class_weight='balanced'`
- `Decision Tree Balanced`: Decision Tree có trọng số lớp cân bằng
- `Random Forest Deep Balanced`: Random Forest cấu hình sâu hơn, dùng chiến lược cân bằng lớp

### 1.3 One-vs-Rest trong Logistic Regression

Trong bài toán 3 lớp, Logistic Regression được bọc bởi `OneVsRestClassifier`.

Điều này có nghĩa là:

- mô hình sẽ học 3 bộ phân loại nhị phân riêng
- mỗi bộ phân loại tách một lớp khỏi hai lớp còn lại
- khi dự đoán, mô hình chọn lớp có xác suất/điểm cao nhất

### 1.4 Các mô hình khác

- `Linear SVC`: SVM tuyến tính, không dùng xác suất trực tiếp mà dùng decision function
- `Decision Tree`: cây quyết định cơ bản
- `Random Forest`: tập hợp nhiều cây quyết định, thường ổn định hơn cây đơn
- `XGBoost`: mô hình boosting mạnh, thường cho kết quả tốt trên dữ liệu bảng

## 2. Ý nghĩa các cột trong bảng kết quả

### 2.1 Validation

Validation là tập dùng để:

- so sánh các mô hình ứng viên
- chọn threshold cho lớp `Low_Engagement`
- quyết định model tốt nhất trước khi chạm vào test

Các cột chính:

- `Accuracy`: tỷ lệ dự đoán đúng toàn bộ mẫu
- `Balanced Accuracy`: trung bình recall của các lớp, hữu ích khi dữ liệu lệch lớp
- `F1 Macro`: F1 trung bình theo từng lớp, mỗi lớp có trọng số như nhau
- `F1 Weighted`: F1 có tính đến số lượng mẫu của từng lớp
- `Recall Low_Engagement`: tỷ lệ phát hiện đúng mẫu nguy cơ
- `Precision Low_Engagement`: trong các mẫu bị cảnh báo Low, có bao nhiêu mẫu là đúng
- `F1 Low_Engagement`: chỉ số cân bằng giữa precision và recall của lớp Low
- `Predicted Low Rate`: tỷ lệ mẫu mà mô hình gán nhãn Low
- `Actual Low Rate`: tỷ lệ Low thực tế trong tập dữ liệu

### 2.2 Test

Test là tập giữ lại đến cuối cùng để đánh giá benchmark.

Nguyên tắc:

- không dùng test để chọn mô hình
- không dùng test để tune threshold
- chỉ đánh giá một lần sau khi đã khóa model và policy từ validation

## 3. Diễn giải kết quả hiện tại

Sau khi chỉnh policy threshold để tránh dự đoán Low quá mức, Stage 4 hiện tại chọn `Random Forest` làm model cuối cùng.

Kết quả benchmark gần nhất:

- Validation accuracy khoảng `0.5752`
- Test accuracy khoảng `0.5780`
- Recall của `Low_Engagement` trên test khoảng `0.7417`
- Precision của `Low_Engagement` trên test khoảng `0.4510`

### 3.1 Vì sao trước đây accuracy thấp nhưng recall Low gần 1.0?

Trước khi điều chỉnh policy threshold, mô hình được phép đẩy quá nhiều mẫu sang nhãn `Low_Engagement`.

Hệ quả:

- recall của lớp Low tăng rất cao, thậm chí tiến gần 1.0
- nhưng số lượng dự đoán Low bị phóng đại
- các lớp còn lại bị nhầm sang Low nhiều hơn
- accuracy tổng thể giảm mạnh

Đây là một dạng đánh đổi không phù hợp nếu muốn báo cáo kết quả thực tế hơn.

### 3.2 Vì sao chọn Random Forest?

`Random Forest` được chọn vì trên validation nó cho cân bằng tốt giữa:

- accuracy
- balanced accuracy
- recall của lớp Low ở mức chấp nhận được
- precision của lớp Low ổn định hơn

So với các mô hình khác:

- Logistic Regression đạt recall Low cao nhưng accuracy thấp hơn
- XGBoost và Random Forest khá gần nhau, nhưng Random Forest được chọn theo ranking validation của lần chạy này
- Linear SVC có recall thấp hơn và không cho lợi thế rõ ràng hơn về accuracy

### 3.3 Ý nghĩa thực tế của recall 0.7417

Recall `0.7417` nghĩa là:

- trong 100 sinh viên thực sự thuộc nhóm nguy cơ thấp `Low_Engagement`
- mô hình phát hiện đúng khoảng 74 sinh viên
- còn khoảng 26 sinh viên bị bỏ sót

Đây là mức thực tế hơn so với việc ép recall lên gần 100%, vì nếu recall quá cao bằng cách dự đoán Low quá nhiều thì precision và accuracy sẽ giảm mạnh.

## 4. Kết luận ngắn để đưa vào báo cáo

Có thể viết trong báo cáo như sau:

> Stage 4 sử dụng tập validation để so sánh các mô hình và điều chỉnh threshold cho lớp `Low_Engagement`. Sau khi hiệu chỉnh policy theo hướng thực tế hơn, mô hình Random Forest được chọn là mô hình cuối cùng vì đạt sự cân bằng tốt giữa accuracy và khả năng phát hiện nhóm nguy cơ. Trên tập test, mô hình đạt accuracy khoảng 0.5780 và recall cho lớp `Low_Engagement` khoảng 0.7417, cho thấy mô hình có khả năng phát hiện phần lớn sinh viên nguy cơ mà không làm giả tăng recall bằng cách dự đoán Low quá mức.

## 5. Ghi chú khi đưa vào DOCX

- Nếu muốn báo cáo ngắn, giữ lại mục 1, mục 3 và mục 4.
- Nếu muốn giải thích kỹ hơn cho hội đồng, nên giữ cả mục 2 để họ hiểu cách đọc số liệu.
- Trong phần kết quả, nên nhấn mạnh rằng validation dùng để chọn model, còn test chỉ dùng để benchmark cuối cùng.
