# Nội dung cập nhật báo cáo đồ án

Tài liệu này là bản thảo để thay thế hoặc chỉnh sửa phần báo cáo từ **Giai đoạn 4: Huấn luyện và tối ưu mô hình (Supervised Learning)** trở đi.

Mục tiêu của bản chỉnh sửa này là:

- giải thích rõ vai trò của `train`, `valid`, `test`
- giải thích vì sao luồng hiện tại **không cần xem SMOTE là điều kiện bắt buộc**
- mô tả cách Stage 4 chọn mô hình và threshold dựa trên validation
- trình bày kết quả theo đúng văn phong báo cáo đồ án

---

## Giai đoạn 4: Huấn luyện và tối ưu mô hình (Supervised Learning)

### 4.1. Mục tiêu của giai đoạn

Giai đoạn 4 được thiết kế nhằm huấn luyện các mô hình phân loại có giám sát để dự đoán mức độ tham gia học tập của sinh viên, với trọng tâm là phát hiện sớm nhóm `Low_Engagement`.

Trong bài toán này, độ chính xác tổng thể không phải là mục tiêu duy nhất. Mô hình cần đảm bảo:

- phát hiện được phần lớn sinh viên thuộc nhóm nguy cơ thấp;
- không tạo quá nhiều cảnh báo giả;
- có thể tổng quát hóa trên dữ liệu kiểm tra chưa từng dùng trong quá trình chọn mô hình.

Vì vậy, giai đoạn này sử dụng ba tập dữ liệu với vai trò khác nhau:

- **Train**: dùng để fit mô hình;
- **Validation**: dùng để chọn mô hình và tối ưu ngưỡng dự đoán;
- **Test**: chỉ dùng một lần ở cuối để đánh giá khách quan.

### 4.2. Có cần SMOTE không?

Trong luồng hiện tại, **SMOTE không phải là điều kiện bắt buộc**, mà là một kỹ thuật hỗ trợ cân bằng dữ liệu train.

Lý do:

1. **SMOTE chỉ được áp dụng trên train**, không áp dụng cho validation và test.
2. Validation và test phải giữ **phân phối thật** để phản ánh đúng khả năng tổng quát hóa.
3. Trong cấu hình hiện tại, train đã được cân bằng bằng SMOTE/SMOTENC, nên mô hình không còn học trên dữ liệu lệch lớp quá mạnh.
4. Tuy nhiên, nếu chỉ dùng SMOTE mà không tối ưu threshold và tiêu chí chọn mô hình, mô hình vẫn có thể dự đoán lớp `Low_Engagement` quá nhiều, làm accuracy giảm.

Kết luận thực nghiệm:

- **SMOTE hữu ích ở mức tiền xử lý train**, vì giúp mô hình quan tâm hơn đến lớp thiểu số.
- **SMOTE không đủ để giải quyết toàn bộ bài toán** nếu ngưỡng dự đoán và tiêu chí chọn mô hình vẫn chưa phù hợp.
- Với bài toán hiện tại, yếu tố quan trọng hơn SMOTE là **cách chọn threshold và cách xếp hạng mô hình trên validation**.

Nói ngắn gọn, có thể trình bày trong báo cáo như sau:

> SMOTE được sử dụng như một kỹ thuật hỗ trợ trên tập train để giảm mất cân bằng lớp, nhưng không phải là yếu tố quyết định duy nhất. Validation và test vẫn giữ phân phối gốc để đảm bảo đánh giá khách quan. Trong thực nghiệm này, hiệu quả cuối cùng phụ thuộc nhiều hơn vào threshold tuning và tiêu chí chọn mô hình trên validation.

### 4.3. Các mô hình được huấn luyện

Stage 4 thử nhiều mô hình phân loại khác nhau để tìm mô hình phù hợp nhất cho bài toán cảnh báo sớm:

- Logistic Regression với các mức điều chuẩn khác nhau: `C1`, `C0.3`, `C3`
- Logistic Regression có cân bằng lớp: `Balanced`
- Linear SVC
- Decision Tree
- Random Forest
- XGBoost

Ý nghĩa của các ký hiệu:

- `C1`, `C0.3`, `C3`: tham số điều chuẩn của Logistic Regression; `C` càng nhỏ thì regularization càng mạnh.
- `Balanced`: mô hình có `class_weight='balanced'`, tức là lớp hiếm được gán trọng số lớn hơn.
- `Random Forest Deep Balanced`: Random Forest với cấu hình sâu hơn và chiến lược cân bằng lớp.

### 4.4. Cách chọn mô hình và threshold

Mỗi mô hình sau khi huấn luyện trên train sẽ được đánh giá trên validation.

Stage 4 không chỉ so sánh mô hình theo dự đoán mặc định, mà còn quét ngưỡng dự đoán cho lớp `Low_Engagement` để tìm quyết định thực tế hơn.

Quy trình chọn lựa gồm:

1. Huấn luyện mô hình trên train.
2. Tính điểm/xác suất trên validation.
3. Thử nhiều ngưỡng dự đoán cho lớp `Low_Engagement`.
4. Chọn ngưỡng và mô hình tốt nhất theo thứ tự ưu tiên.

Trong bản điều chỉnh hiện tại, tiêu chí chọn mô hình được đặt theo hướng thực tế hơn:

- đảm bảo recall của `Low_Engagement` không quá thấp;
- tránh trường hợp dự đoán `Low` quá nhiều làm accuracy giảm mạnh;
- ưu tiên accuracy và balanced accuracy khi recall đã đạt mức chấp nhận được.

### 4.5. Kết quả huấn luyện và chọn mô hình

Kết quả validation gần nhất cho thấy mô hình tốt nhất là **Random Forest**.

Kết quả trên validation:

- Accuracy: khoảng `0.5752`
- Balanced Accuracy: khoảng `0.5809`
- Recall `Low_Engagement`: khoảng `0.7361`
- Precision `Low_Engagement`: khoảng `0.4513`
- F1 `Low_Engagement`: khoảng `0.5595`

Kết quả trên test:

- Accuracy: khoảng `0.5780`
- Balanced Accuracy: khoảng `0.5837`
- Recall `Low_Engagement`: khoảng `0.7417`
- Precision `Low_Engagement`: khoảng `0.4510`
- F1 `Low_Engagement`: khoảng `0.5609`

### 4.6. Giải thích vì sao accuracy trước đây thấp

Ở phiên bản trước, threshold của lớp `Low_Engagement` bị chọn quá thấp. Điều này làm cho mô hình dự đoán Low rất nhiều, dẫn tới:

- recall của Low tăng gần 1.0;
- nhưng nhiều mẫu thuộc lớp Medium/High bị gán nhầm sang Low;
- accuracy tổng thể giảm mạnh.

Đây là lý do vì sao accuracy có thể thấp dù recall Low nhìn có vẻ rất cao. Trong bài toán cảnh báo sớm, recall cao là quan trọng, nhưng nếu recall bị đẩy lên bằng cách dự đoán quá nhiều nhãn Low thì mô hình không còn thực tế.

### 4.7. Ý nghĩa của kết quả mới

Sau khi điều chỉnh threshold và cách chọn mô hình, recall của lớp `Low_Engagement` đã giảm từ mức gần 1.0 xuống khoảng `0.74`.

Điều này có ý nghĩa:

- mô hình vẫn phát hiện được phần lớn sinh viên nguy cơ;
- số lượng cảnh báo giả giảm so với trước;
- accuracy và balanced accuracy tăng lên rõ rệt;
- kết quả phản ánh thực tế hơn cho mục tiêu cảnh báo sớm.

### 4.8. Kết luận giai đoạn 4

Stage 4 cho thấy rằng trong bài toán dự đoán mức độ tham gia học tập, không nên chỉ tối ưu riêng recall của lớp nguy cơ. Nếu threshold bị đẩy quá thấp, mô hình có thể dự đoán Low quá nhiều và làm accuracy giảm mạnh.

Trong cấu hình hiện tại:

- SMOTE được dùng như một công cụ hỗ trợ trên train;
- validation mới là nơi chọn mô hình và threshold;
- test chỉ dùng để benchmark cuối cùng;
- Random Forest là mô hình được chọn cho triển khai.

---

## Giai đoạn 5: Giải thích mô hình và phân tích kết quả

Sau khi chọn được mô hình cuối cùng, giai đoạn tiếp theo là giải thích kết quả dự đoán và phân tích những yếu tố ảnh hưởng đến mức độ tham gia học tập.

### 5.1. Mục tiêu

Giai đoạn này nhằm:

- minh họa mô hình đã chọn hoạt động ra sao;
- phân tích những đặc trưng quan trọng nhất;
- giải thích dự đoán cho toàn bộ tập test và một số trường hợp cụ thể.

### 5.2. Kết quả đầu ra

Các artifact chính của giai đoạn này gồm:

- `expected_results_summary.txt`
- `feature_importance_values.csv`
- biểu đồ feature importance
- biểu đồ SHAP toàn cục
- biểu đồ SHAP cục bộ cho một số sinh viên điển hình

### 5.3. Cách đọc kết quả giải thích

- Feature importance cho biết biến nào có ảnh hưởng lớn nhất đến quyết định của mô hình.
- SHAP cho biết mỗi đặc trưng đóng góp theo hướng tăng hay giảm xác suất dự đoán một lớp cụ thể.

Trong bối cảnh cảnh báo sớm, phần giải thích này giúp xác định:

- sinh viên nào đang có dấu hiệu nguy cơ;
- đặc trưng nào làm mô hình nhận diện họ là nhóm `Low_Engagement`;
- yếu tố nào nên được ưu tiên theo dõi để can thiệp sớm.

---

## Phần kết luận chung của báo cáo

Quy trình thực nghiệm hiện tại đã tách rõ vai trò của train, validation và test.

- Train được dùng để học mô hình, có thể áp dụng SMOTE để giảm mất cân bằng lớp.
- Validation được dùng để chọn mô hình và tối ưu threshold.
- Test chỉ dùng một lần ở cuối để đánh giá khách quan.

Với bài toán cảnh báo sớm sinh viên có nguy cơ, kết quả thực nghiệm cho thấy việc cân bằng train bằng SMOTE chỉ là một phần của giải pháp. Hiệu quả cuối cùng phụ thuộc nhiều hơn vào cách chọn threshold và tiêu chí xếp hạng mô hình trên validation.

Do đó, nếu viết lại báo cáo, nên nhấn mạnh rằng:

1. SMOTE là công cụ hỗ trợ trên train, không phải điều kiện bắt buộc để thành công.
2. Validation mới là tập quyết định mô hình và threshold.
3. Test chỉ dùng để benchmark cuối cùng.
4. Mô hình Random Forest hiện là mô hình cuối cùng được chọn trong run gần nhất.
