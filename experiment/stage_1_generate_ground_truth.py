import gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from config import (
    DEFAULT_OBSERVATION_DAYS,
    GROUND_TRUTH_FILE,
    GROUND_TRUTH_REPORT_FILE,
    GROUND_TRUTH_WEIGHTS,
    LABEL_PERCENTILES,
    PRIMARY_KEY,
    RANDOM_STATE,
    RAW_DATA_PARQUET,
    RAW_REQUIRED_COLUMNS_STEP1,
    LABELING_STRATEGY,
    USE_ENTROPY_WEIGHT_METHOD,
    LAPLACE_SMOOTHING_ALPHA,
    SCALER_TYPE,
    USER_FEATURES_AGGREGATED_FILE,
)


def apply_laplace_smoothing(accuracy_series: pd.Series, attempts_series: pd.Series, 
                            alpha: float = 1.0) -> pd.Series:
    """
    Áp dụng Laplace Smoothing cho accuracy_rate để tránh ảo hóa khi attempts quá thấp.
    
    Công thức: smoothed_accuracy = (is_correct + alpha) / (attempts + 2*alpha)
    
    Args:
        accuracy_series: pd.Series accuracy_rate thô (is_correct / attempts)
        attempts_series: pd.Series số lần thử (attempts)
        alpha: hệ số smoothing (mặc định 1.0 = Laplace smoothing chuẩn)
    
    Returns:
        pd.Series accuracy_rate sau smoothing
    
    Ví dụ:
        - Học viên A: attempts=1, is_correct=1 → thô = 1.0, sau smoothing ≈ 0.67
        - Học viên B: attempts=100, is_correct=95 → thô = 0.95, sau smoothing ≈ 0.94
    """
    is_correct = accuracy_series * attempts_series
    smoothed = (is_correct + alpha) / (attempts_series + 2 * alpha)
    return smoothed.fillna(0)


def entropy_weight_method(df_features: pd.DataFrame, metric_columns: list) -> dict:
    """
    Tính trọng số tự động bằng Entropy Weight Method (EWM) - khoa học hơn hardcode.
    
    Quy trình:
    1. Chuẩn hóa từng chỉ số về [0, 1]
    2. Tính entropy của từng chỉ số
    3. Tính độ phân biệt (divergence) = 1 - entropy
    4. Chuẩn hóa trọng số sao cho tổng = 1.0
    
    Args:
        df_features: DataFrame chứa các chỉ số
        metric_columns: list tên các cột để tính trọng số
    
    Returns:
        dict trọng số {metric: weight}
    
    Ưu điểm:
    - Không phụ thuộc ý kiến chủ quan
    - Độ phân biệt cao (entropy thấp) → trọng số cao
    - Có thể reproduce khi dữ liệu thay đổi
    """
    # Chuẩn hóa về [0, 1]
    df_norm = df_features[metric_columns].copy()
    for col in metric_columns:
        col_min, col_max = df_norm[col].min(), df_norm[col].max()
        if col_max > col_min:
            df_norm[col] = (df_norm[col] - col_min) / (col_max - col_min)
        else:
            df_norm[col] = 0.5  # Nếu tất cả giá trị bằng nhau, gán 0.5
    
    # Tính entropy cho từng metric
    entropies = {}
    for col in metric_columns:
        # Tính proportion của từng giá trị
        # Bin vào 10 khoảng để tránh quá chi tiết
        hist, _ = np.histogram(df_norm[col], bins=10, range=(0, 1))
        hist = hist / hist.sum()
        hist = hist[hist > 0]  # Loại bỏ 0 để tính log
        
        entropy = -np.sum(hist * np.log2(hist))
        entropies[col] = entropy
    
    # Tính trọng số từ entropy
    max_entropy = np.log2(10)  # log(số bin)
    weights_raw = {}
    for col in metric_columns:
        divergence = 1 - entropies[col] / max_entropy if max_entropy > 0 else 0
        weights_raw[col] = max(divergence, 0.01)  # Tối thiểu 0.01
    
    # Chuẩn hóa sao cho tổng = 1.0
    total_weight = sum(weights_raw.values())
    weights_normalized = {col: w / total_weight for col, w in weights_raw.items()}
    
    return weights_normalized


def main():
    print("=" * 80)
    print("STEP 1: GENERATE GROUND TRUTH LABELS (FULL COURSE BEHAVIOR)")
    print("=" * 80)

    if not RAW_DATA_PARQUET.exists():
        raise FileNotFoundError(f"Input parquet not found: {RAW_DATA_PARQUET}")

    print(f"[1/4] Đang tải dữ liệu hành vi thô từ: {RAW_DATA_PARQUET}")
    df = pd.read_parquet(RAW_DATA_PARQUET, columns=RAW_REQUIRED_COLUMNS_STEP1)
    raw_row_count = len(df)
    raw_user_count = df[PRIMARY_KEY].nunique()
    print(f"      -> Đã tải {raw_row_count:,} records (mỗi record = 1 submission/event)")
    print(f"      -> Từ {raw_user_count:,} người học (unique user_id)")
    print(f"      -> Cột sử dụng: {', '.join(RAW_REQUIRED_COLUMNS_STEP1)}")
    print()
    print("      GIẢI THÍCH LEAK DATA CONTROL:")
    print(f"         Dữ liệu thô: ~{raw_row_count:,} records từ tất cả thời gian")
    print(f"         → Tất cả {raw_user_count:,} người học")
    print(f"         → Sẽ cắt xuống chỉ giữ hành vi trong 28 ngày đầu...")

    if PRIMARY_KEY not in df.columns:
        raise KeyError(f"Missing primary key column: {PRIMARY_KEY}")

    for col in ["attempts", "is_correct", "score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["video_clicks"] = df["create_time_x"].notna().astype(int)
    df["forum_posts"] = df["create_time_y"].notna().astype(int)

    print("[2/4] Đang tổng hợp hành vi toàn khóa theo từng sinh viên")
    print(f"      -> Dữ liệu thô: {raw_row_count:,} records từ {raw_user_count:,} người học")
    print()
    print("      GIẢI THÍCH GROUPBY (aggregation):")
    print("         Mỗi record = 1 submission/event của 1 user")
    print("         Groupby user_id = Gom tất cả events của mỗi user lại thành 1 dòng")
    print("         Kết quả: 1 dòng per user với các chỉ số tổng hợp (tổng attempts, etc.)")
    print()
    
    user_features = (
        df.groupby("user_id")
        .agg(
            attempts=("attempts", "sum"),
            is_correct=("is_correct", "sum"),
            avg_score=("score", "mean"),
            total_study_time=("video_clicks", "sum"),
            total_forum_activity=("forum_posts", "sum"),
        )
        .reset_index()
    )
    print(f"      -> Sau groupby: {len(user_features):,} sinh viên (= unique user_id)")
    print(f"      -> Giảm từ {raw_row_count:,} records → {len(user_features):,} users")
    print(f"         Tỷ lệ: trung bình {raw_row_count / len(user_features):.1f} records/user")
    print()
    print(
        "      -> Thống kê tổng hợp: "
        f"attempts={user_features['attempts'].sum():,.0f}, "
        f"is_correct={user_features['is_correct'].sum():,.0f}, "
        f"avg_score={user_features['avg_score'].mean():.4f}, "
        f"video_clicks={user_features['total_study_time'].sum():,.0f}, "
        f"forum_posts={user_features['total_forum_activity'].sum():,.0f}"
    )

    # Export aggregated per-user features so intermediate artifact is persisted
    print("[2.2/4] Đang lưu snapshot aggregated user features (Stage1)...")
    user_features.to_csv(USER_FEATURES_AGGREGATED_FILE, index=False, encoding="utf-8-sig")
    print(f"      -> Đã lưu user features aggregated: {USER_FEATURES_AGGREGATED_FILE}")
    # Sau khi đã lưu lên đĩa, giải phóng bộ nhớ RAM
    del df
    gc.collect()

    print("[2.5/4] Đang áp dụng Laplace Smoothing cho accuracy_rate...")
    user_features["accuracy_rate_raw"] = (
        user_features["is_correct"] / user_features["attempts"].replace(0, np.nan)
    ).fillna(0)
    user_features["accuracy_rate"] = apply_laplace_smoothing(
        user_features["accuracy_rate_raw"], 
        user_features["attempts"],
        alpha=LAPLACE_SMOOTHING_ALPHA
    )
    print(
        f"      -> accuracy_rate trước smoothing (alpha={LAPLACE_SMOOTHING_ALPHA}): "
        f"min={user_features['accuracy_rate_raw'].min():.4f}, "
        f"mean={user_features['accuracy_rate_raw'].mean():.4f}, "
        f"max={user_features['accuracy_rate_raw'].max():.4f}"
    )
    print(
        f"      -> accuracy_rate sau smoothing: "
        f"min={user_features['accuracy_rate'].min():.4f}, "
        f"mean={user_features['accuracy_rate'].mean():.4f}, "
        f"max={user_features['accuracy_rate'].max():.4f}"
    )
    
    model_columns = list(GROUND_TRUTH_WEIGHTS.keys())
    user_features[model_columns] = (
        user_features[model_columns].replace([np.inf, -np.inf], np.nan).fillna(0)
    )

    print("[3/4] Đang chuẩn hóa các chỉ số bằng RobustScaler và tính trọng số...")
    print(f"      -> Trọng số mặc định từ config: {GROUND_TRUTH_WEIGHTS}")
    print(f"      -> USE_ENTROPY_WEIGHT_METHOD = {USE_ENTROPY_WEIGHT_METHOD}")
    print(f"      -> SCALER_TYPE = {SCALER_TYPE}")
    print()
    
    # Chọn phương pháp tính trọng số
    if USE_ENTROPY_WEIGHT_METHOD:
        computed_weights = entropy_weight_method(user_features, list(GROUND_TRUTH_WEIGHTS.keys()))
        print(f"      -> Trọng số tính bằng EWM (Entropy Weight Method): {computed_weights}")
        print("         (Tự động từ entropy của từng chỉ số, không bias)")
        weights_to_use = computed_weights
    else:
        print("      -> Sử dụng trọng số từ config (hardcode)")
        weights_to_use = GROUND_TRUTH_WEIGHTS
    
    print()
    
    # Chọn Scaler
    if SCALER_TYPE.lower() == "robust":
        scaler = RobustScaler()
        print("      -> Chuẩn hóa bằng RobustScaler (chống outlier tốt hơn MinMaxScaler)")
    else:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        print("      -> Chuẩn hóa bằng MinMaxScaler")
    
    model_columns = list(GROUND_TRUTH_WEIGHTS.keys())
    scaled = pd.DataFrame(
        scaler.fit_transform(user_features[model_columns]),
        columns=model_columns,
        index=user_features.index,
    )
    
    # Tính WES dùng trọng số đã chọn
    user_features["weighted_score"] = sum(
        scaled[col] * weights_to_use[col] for col in model_columns
    )
    print(
        "      -> WES sau chuẩn hóa: "
        f"min={user_features['weighted_score'].min():.4f}, "
        f"mean={user_features['weighted_score'].mean():.4f}, "
        f"max={user_features['weighted_score'].max():.4f}"
    )

    print("[4/4] Đang chia nhãn theo phân vị tự nhiên để tránh tỷ lệ 1:1:1")
    label_order = ["Low_Engagement", "Medium_Engagement", "High_Engagement"]
    rank_scores = user_features["weighted_score"].rank(method="first", ascending=True)
    print(
        "      -> Mốc phân vị: "
        f"{LABEL_PERCENTILES[0]} / {LABEL_PERCENTILES[1]}"
    )

    if LABELING_STRATEGY == "quantile_rank":
        user_features["target_label"] = pd.qcut(
            rank_scores,
            q=[0.0, float(LABEL_PERCENTILES[0]), float(LABEL_PERCENTILES[1]), 1.0],
            labels=label_order,
        )
    else:
        user_features["target_label"] = pd.qcut(
            rank_scores,
            q=[0.0, float(LABEL_PERCENTILES[0]), float(LABEL_PERCENTILES[1]), 1.0],
            labels=label_order,
        )
    user_features["target_label"] = user_features["target_label"].astype(str)
    print("      -> Đã gán nhãn xong, đang tổng hợp báo cáo phân phối nhãn")

    report = pd.DataFrame(
        {
            "metric": [
                "labeling_strategy",
                "raw_rows",
                "raw_users",
                "observation_days_hint",
                "labeled_users",
                "low_engagement_users",
                "medium_engagement_users",
                "high_engagement_users",
            ],
            "value": [
                LABELING_STRATEGY,
                raw_row_count,
                raw_user_count,
                DEFAULT_OBSERVATION_DAYS,
                len(user_features),
                int((user_features["target_label"] == "Low_Engagement").sum()),
                int((user_features["target_label"] == "Medium_Engagement").sum()),
                int((user_features["target_label"] == "High_Engagement").sum()),
            ],
        }
    )

    user_features[["user_id", "target_label"]].to_csv(
        GROUND_TRUTH_FILE, index=False, encoding="utf-8-sig"
    )
    report.to_csv(GROUND_TRUTH_REPORT_FILE, index=False, encoding="utf-8-sig")
    print(f"\nĐã lưu file nhãn mục tiêu: {GROUND_TRUTH_FILE}")
    print(f"Đã lưu report stage 1: {GROUND_TRUTH_REPORT_FILE}")
    
    print("\nPhân phối nhãn thực tế:")
    distribution = (
        user_features["target_label"]
        .value_counts()
        .reindex(label_order)
        .fillna(0)
        .astype(int)
    )
    distribution_df = pd.DataFrame({
        "target_label": distribution.index,
        "count": distribution.values,
        "percentage": (distribution / distribution.sum() * 100).round(2),
    })
    print(distribution_df.to_string(index=False))
    
    print()
    print("=" * 80)
    print("TIẾP THEO - CHỐT CHẶN LEAK DATA TRONG STAGE 2:")
    print("=" * 80)
    print("Stage 1 vừa tạo Ground Truth Labels từ toàn bộ khóa học (~130K users)")
    print()
    print("Stage 2 sẽ:")
    print("  1. Cắt dữ liệu thô xuống chỉ 28 ngày đầu (từ ~4M records → ~1.2M records)")
    print("  2. Tính time-window features từ 1.2M records")
    print("  3. Groupby user_id để được ~130K users với features mới")
    print("  4. Ghép nhãn từ Stage 1 để tạo dữ liệu train")
    print()
    print("Lý do cắt 28 ngày:")
    print("   - Tránh leak data khi huấn luyện model (tránh dùng info từ tương lai)")
    print("   - Mô phỏng cảnh báo sớm (chỉ dùng 28 ngày đầu để predict toàn khóa)")

if __name__ == "__main__":
    main()
