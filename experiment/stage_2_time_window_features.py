import gc
import logging
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from sklearn.preprocessing import OrdinalEncoder

from config import (
    DEFAULT_OBSERVATION_DAYS,
    FEATURES_COMPAT_FILE,
    FEATURES_WINDOW_FILE,
    GROUND_TRUTH_FILE,
    PRIMARY_KEY,
    RAW_DATA_PARQUET,
    STAGE2_TIMELINE_FILE,
    WINDOW_SUFFIX,
    TIME_WINDOW_COMPARE_SUMMARY_FILE,
    TIME_WINDOW_MODE,
    PREPROCESSING_DATASET_FILE,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

OBSERVATION_DAYS = DEFAULT_OBSERVATION_DAYS
WINDOW_MODE = str(TIME_WINDOW_MODE).strip().lower()

TARGET_LABEL_ORDER = {
    "Low_Engagement": 0,
    "Medium_Engagement": 1,
    "High_Engagement": 2,
}

RAW_REQUIRED_COLUMNS_STEP2 = [
    'user_id', 'enroll_time', 'submit_time', 'create_time_x', 'create_time_y',
    'school', 'year_of_birth', 'gender', 'num_courses', 'attempts', 'is_correct', 'score'
]

def parse_datetime(series: pd.Series) -> pd.Series:
    extracted = series.astype(str).str.extract(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")[0]
    return pd.to_datetime(extracted, errors="coerce")

def load_data() -> pd.DataFrame:
    if not RAW_DATA_PARQUET.exists():
        logger.error(f"Không tìm thấy file dữ liệu gốc: {RAW_DATA_PARQUET}")
        raise FileNotFoundError(f"Input parquet not found: {RAW_DATA_PARQUET}")
        
    if not GROUND_TRUTH_FILE.exists():
        logger.error(f"Không tìm thấy file Nhãn. Bạn phải chạy Bước 1 trước: {GROUND_TRUTH_FILE}")
        raise FileNotFoundError(f"Run step 1 first: {GROUND_TRUTH_FILE}")

    logger.info("Đang kiểm tra schema (cấu trúc) của file Parquet để xác định cột có thể dùng...")
    available_columns = set(pq.ParquetFile(RAW_DATA_PARQUET).schema.names)
    selected_columns = [col for col in RAW_REQUIRED_COLUMNS_STEP2 if col in available_columns]
    logger.info(f"-> Sẽ tải {len(selected_columns)} cột: {', '.join(selected_columns)}")
    
    missing_columns = sorted(set(RAW_REQUIRED_COLUMNS_STEP2) - available_columns)
    if missing_columns:
        logger.warning(f"File Parquet bị thiếu các cột: {', '.join(missing_columns)}")

    logger.info(f"Đang tải dữ liệu thô từ: {RAW_DATA_PARQUET}")
    df = pd.read_parquet(RAW_DATA_PARQUET, columns=selected_columns)
    logger.info(f"-> Đã tải thành công: {len(df):,} dòng, {df['user_id'].nunique():,} người học")
    return df

def build_action_timeline(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Đang chuẩn hóa các mốc thời gian hành động để xác định timeline học tập...")
    for col in ["submit_time", "create_time_x", "create_time_y"]:
        if col not in df.columns:
            df[col] = pd.NaT

    df["action_time"] = df["submit_time"].combine_first(df["create_time_x"]).combine_first(df["create_time_y"])
    df["action_time"] = parse_datetime(df["action_time"])

    logger.info("Đang xác định thời điểm bắt đầu học (Enroll Time) cho từng người học...")
    if "enroll_time" in df.columns:
        df["enroll_time"] = parse_datetime(df["enroll_time"])
        if df["enroll_time"].isna().all():
            logger.warning(
                "Cột enroll_time có trong schema nhưng không parse được giá trị nào; "
                "sẽ fallback bằng action_time nhỏ nhất của từng user."
            )
            df["enroll_time"] = df.groupby("user_id")["action_time"].transform("min")
        else:
            missing_enroll = int(df["enroll_time"].isna().sum())
            if missing_enroll > 0:
                logger.warning(
                    f"Cột enroll_time có {missing_enroll:,} giá trị thiếu sau khi parse; "
                    "các dòng đó sẽ dùng action_time để tính days_since_enroll."
                )
    else:
        logger.warning(
            "File Parquet bị thiếu cột enroll_time; fallback bằng action_time nhỏ nhất của từng user."
        )
        df["enroll_time"] = df.groupby("user_id")["action_time"].transform("min")

    logger.info("Đang tính số ngày kể từ lúc bắt đầu học cho từng hành vi...")
    df["days_since_enroll"] = (df["action_time"] - df["enroll_time"]).dt.days
    df["days_since_enroll"] = pd.to_numeric(df["days_since_enroll"], errors="coerce")
    df["days_since_enroll"] = df["days_since_enroll"].clip(lower=0)

    logger.info("Đang đo độ dài hành vi tối đa theo từng user để hỗ trợ relative windows...")
    max_days_per_user = df.groupby(PRIMARY_KEY)["days_since_enroll"].max().fillna(0)
    max_days_per_user = max_days_per_user.clip(lower=1)
    df["max_days_per_user"] = df[PRIMARY_KEY].map(max_days_per_user)
    logger.info(
        f"-> days_since_enroll: min={df['days_since_enroll'].min()}, max={df['days_since_enroll'].max()}, "
        f"mean={df['days_since_enroll'].mean():.2f}"
    )
    return df


def build_fixed_window(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"CHỐT CHẶN LEAK DATA (fixed): chỉ giữ hành vi trong {OBSERVATION_DAYS} ngày đầu...")
    within_window = (df["days_since_enroll"] <= OBSERVATION_DAYS) | df["days_since_enroll"].isna()
    df_window = df.loc[within_window].copy()
    df_window["window_type"] = "fixed"
    df_window["window_value"] = OBSERVATION_DAYS

    logger.info(
        f"-> Dữ liệu sau khi cắt thời gian: {len(df_window):,} dòng, "
        f"{df_window[PRIMARY_KEY].nunique():,} người học"
    )
    return df_window


def build_relative_window(df: pd.DataFrame, fraction: float) -> pd.DataFrame:
    pct_text = int(round(float(fraction) * 100))
    logger.info(f"CHỐT CHẶN LEAK DATA (relative): chỉ giữ {pct_text}% đầu timeline của từng user...")

    per_row_limit = (df["max_days_per_user"] * float(fraction)).round().astype(int).clip(lower=1)
    within_window = (df["days_since_enroll"] <= per_row_limit) | df["days_since_enroll"].isna()
    df_window = df.loc[within_window].copy()
    df_window["window_type"] = "relative"
    df_window["window_value"] = pct_text

    logger.info(
        f"-> Dữ liệu sau khi cắt relative {pct_text}%: {len(df_window):,} dòng, "
        f"{df_window[PRIMARY_KEY].nunique():,} người học"
    )
    return df_window

def extract_features(df_window: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"Đang tổng hợp đặc trưng khởi đầu cho cửa sổ {WINDOW_SUFFIX}...")

    attempts_col = f"attempts_{WINDOW_SUFFIX}"
    correct_col = f"is_correct_{WINDOW_SUFFIX}"
    score_col = f"score_{WINDOW_SUFFIX}"
    accuracy_col = f"accuracy_rate_{WINDOW_SUFFIX}"
    
    for col in ["attempts", "is_correct", "score", "num_courses"]:
        if col not in df_window.columns:
            df_window[col] = 0
        df_window[col] = pd.to_numeric(df_window[col], errors="coerce").fillna(0)
        
    for col in ["school", "year_of_birth", "gender"]:
        if col not in df_window.columns:
            df_window[col] = np.nan

    features = (
        df_window.groupby(PRIMARY_KEY)
        .agg(
            school=("school", "first"),
            year_of_birth=("year_of_birth", "first"),
            gender=("gender", "first"),
            num_courses=("num_courses", "first"),
            **{
                attempts_col: ("attempts", "sum"),
                correct_col: ("is_correct", "sum"),
                score_col: ("score", "mean"),
            },
        )
        .reset_index()
    )
    
    features[accuracy_col] = (
        features[correct_col] / features[attempts_col].replace(0, np.nan)
    ).fillna(0)

    logger.info(
        f"-> Đã tổng hợp đặc trưng cho {len(features):,} sinh viên; "
        f"các cột chính: {attempts_col}, {correct_col}, {score_col}, {accuracy_col}"
    )
    return features


def finalize_with_labels(features: pd.DataFrame) -> pd.DataFrame:
    labels = pd.read_csv(GROUND_TRUTH_FILE)
    return features.merge(labels, on="user_id", how="inner")


def calculate_additional_features(df_window: pd.DataFrame) -> pd.DataFrame:
    """Tính toán các feature bổ sung: seq, speed, rep_counts, cmt_counts"""
    additional = (
        df_window.groupby(PRIMARY_KEY)
        .agg(
            seq=("days_since_enroll", "max"),  # tổng thời gian học (max days)
            speed=("score", "mean"),           # tốc độ = điểm trung bình
            rep_counts=("attempts", "count"),   # số lần phản hồi = số submissions
            cmt_counts=("is_correct", "sum"),   # số lần bình luận = số correct
        )
        .reset_index()
    )
    return additional


def build_preprocessing_dataset(df_timeline: pd.DataFrame, features: pd.DataFrame, 
                               df_window: pd.DataFrame, final_df: pd.DataFrame) -> pd.DataFrame:
    """Tạo comprehensive pre-processing dataset với toàn bộ cột"""
    # Lấy raw features từ df_window (attempts, is_correct, score trước groupby)
    raw_features = (
        df_window.groupby(PRIMARY_KEY)
        .agg(
            attempts_raw=("attempts", "sum"),
            is_correct_raw=("is_correct", "sum"),
            score_raw=("score", "mean"),
        )
        .reset_index()
    )
    
    # Timeline summary
    timeline_summary = (
        df_timeline.groupby(PRIMARY_KEY)
        .agg(
            submit_time=("submit_time", "first"),
            create_time_x=("create_time_x", "first"),
            create_time_y=("create_time_y", "first"),
            action_time=("action_time", "min"),
            enroll_time=("enroll_time", "first"),
            days_since_enroll=("days_since_enroll", "max"),
            max_days_per_user=("max_days_per_user", "first"),
        )
        .reset_index()
    )
    
    # Additional features
    additional = calculate_additional_features(df_window)
    
    # Merge all pieces
    preprocessing = final_df.copy()
    preprocessing = preprocessing.merge(raw_features, on=PRIMARY_KEY, how="left")
    preprocessing = preprocessing.merge(timeline_summary, on=PRIMARY_KEY, how="left")
    preprocessing = preprocessing.merge(additional, on=PRIMARY_KEY, how="left")

    # Remove school from the comprehensive file per request
    preprocessing = preprocessing.drop(columns=["school"], errors="ignore")

    def _coerce_numeric_for_corr(series: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(series):
            return pd.to_numeric(series, errors="coerce")

        if pd.api.types.is_datetime64_any_dtype(series):
            return series.astype("int64").astype(float)

        parsed_dt = pd.to_datetime(series, errors="coerce")
        if parsed_dt.notna().any():
            return parsed_dt.astype("int64").astype(float)

        encoded = pd.Series(pd.factorize(series.astype(str).fillna("Unknown"))[0], index=series.index, dtype=float)
        encoded.replace(-1, np.nan, inplace=True)
        return encoded

    def _sort_columns_by_target_corr(df: pd.DataFrame) -> pd.DataFrame:
        if "target_label" not in df.columns:
            return df

        target_numeric = df["target_label"].map(TARGET_LABEL_ORDER)
        if target_numeric.isna().all():
            target_numeric = pd.Series(pd.factorize(df["target_label"].astype(str))[0], index=df.index, dtype=float)

        corr_scores: dict[str, float] = {}
        for col in df.columns:
            if col in {PRIMARY_KEY, "target_label"}:
                continue

            candidate = _coerce_numeric_for_corr(df[col])
            score = candidate.corr(target_numeric)
            corr_scores[col] = abs(float(score)) if pd.notna(score) else -1.0

        ordered_columns = [PRIMARY_KEY]
        ordered_columns.extend(sorted(
            [col for col in df.columns if col not in {PRIMARY_KEY, "target_label"}],
            key=lambda col: (-corr_scores.get(col, -1.0), col),
        ))
        if "target_label" in df.columns:
            ordered_columns.append("target_label")
        return df[ordered_columns]
    
    return _sort_columns_by_target_corr(preprocessing)


def add_action_time_to_features(features: pd.DataFrame, df_timeline: pd.DataFrame) -> pd.DataFrame:
    """Thêm action_time (cột thời gian từ timeline) vào features"""
    action_time_data = df_timeline.groupby(PRIMARY_KEY)["action_time"].min().reset_index()
    action_time_data = action_time_data.rename(columns={"action_time": "stage2_action_time"})
    features = features.merge(action_time_data, on=PRIMARY_KEY, how="left")
    return features


def encode_school_column(features: pd.DataFrame) -> pd.DataFrame:
    """Mã hóa cột school thành school_encoded"""
    school_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    features["school"] = features["school"].fillna("Unknown").astype(str).str.strip()
    features["school_encoded"] = school_encoder.fit_transform(features[["school"]])
    return features


def reorganize_columns(final_df: pd.DataFrame) -> pd.DataFrame:
    """Sắp xếp lại các cột theo thứ tự: user_id, school, school_encoded, stage2_action_time, features..., target_label"""
    feature_cols = [
        f"attempts_{WINDOW_SUFFIX}",
        f"is_correct_{WINDOW_SUFFIX}",
        f"score_{WINDOW_SUFFIX}",
        f"accuracy_rate_{WINDOW_SUFFIX}",
        "num_courses",
        "year_of_birth",
        "gender",
    ]
    
    cols = [PRIMARY_KEY]
    if "school" in final_df.columns:
        cols.append("school")
    if "school_encoded" in final_df.columns:
        cols.append("school_encoded")
    if "stage2_action_time" in final_df.columns:
        cols.append("stage2_action_time")
    
    cols.extend([c for c in feature_cols if c in final_df.columns])
    
    if "target_label" in final_df.columns:
        cols.append("target_label")
    
    return final_df[cols]


def export_window_dataset(final_df: pd.DataFrame, mode: str, window_value: int) -> str:
    if mode == "fixed":
        output_path = FEATURES_WINDOW_FILE
        final_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        final_df.to_csv(FEATURES_COMPAT_FILE, index=False, encoding="utf-8-sig")
        return str(output_path)

    output_path = Path(str(RELATIVE_FEATURES_OUTPUT_PATTERN).format(pct=window_value))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    # Relative branch: mặc định chọn 50% để làm file tương thích cho Stage 3 nếu user muốn chạy tiếp.
    if int(window_value) == 50:
        final_df.to_csv(FEATURES_COMPAT_FILE, index=False, encoding="utf-8-sig")

    return str(output_path)


def summarize_outputs(items: list[dict]) -> None:
    summary = pd.DataFrame(items)
    summary.to_csv(TIME_WINDOW_COMPARE_SUMMARY_FILE, index=False, encoding="utf-8-sig")
    logger.info(f"Đã lưu bảng so sánh time-window tại: {TIME_WINDOW_COMPARE_SUMMARY_FILE}")

def main():
    print("=" * 80)
    print(f"STEP 2: EXTRACT EARLY TIME-WINDOW FEATURES ({WINDOW_MODE.upper()})")
    print("=" * 80)

    try:
        logger.info(f"Cấu hình hiện tại: window_mode={WINDOW_MODE}, observation_days={OBSERVATION_DAYS}")
        df = load_data()
        df = build_action_timeline(df)
        # Export cleaned action timeline to disk so intermediate snapshot is available
        try:
            logger.info("[0.5] Lưu snapshot timeline đã chuẩn hóa (Stage2)...")
            STAGE2_TIMELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(STAGE2_TIMELINE_FILE, index=False, encoding="utf-8-sig")
            logger.info(f"-> Đã lưu timeline snapshot: {STAGE2_TIMELINE_FILE}")
        except Exception:
            logger.exception("Không lưu được timeline snapshot")
        logger.info("Đang dọn dẹp bộ nhớ RAM sau khi dựng timeline gốc...")
        summary_rows = []

        if WINDOW_MODE == "relative":
            try:
                from config import RELATIVE_WINDOW_FRACTIONS
            except ImportError:
                RELATIVE_WINDOW_FRACTIONS = [0.5]
            
            for frac in RELATIVE_WINDOW_FRACTIONS:
                pct_value = int(round(float(frac) * 100))
                df_window = build_relative_window(df, float(frac))
                features = extract_features(df_window)
                features = add_action_time_to_features(features, df)
                features = encode_school_column(features)
                final_df = finalize_with_labels(features)
                final_df = reorganize_columns(final_df)
                output_path = export_window_dataset(final_df, mode="relative", window_value=pct_value)
                logger.info(f"-> Đã xuất file đặc trưng {pct_value}% tại: {output_path}")
                summary_rows.append(
                    {
                        "window_mode": "relative",
                        "window_value": pct_value,
                        "num_rows": int(len(df_window)),
                        "num_users": int(final_df[PRIMARY_KEY].nunique()),
                        "output_file": output_path,
                    }
                )
        else:
            df_window = build_fixed_window(df)
            features = extract_features(df_window)
            features = add_action_time_to_features(features, df)
            features = encode_school_column(features)
            final_df = finalize_with_labels(features)
            final_df = reorganize_columns(final_df)
            output_path = export_window_dataset(final_df, mode="fixed", window_value=OBSERVATION_DAYS)
            logger.info(f"-> Đã xuất file đặc trưng fixed tại: {output_path}")
            
            # Build và save comprehensive pre-processing dataset
            preprocessing_df = build_preprocessing_dataset(df, features, df_window, final_df)
            PREPROCESSING_DATASET_FILE.parent.mkdir(parents=True, exist_ok=True)
            preprocessing_df.to_csv(PREPROCESSING_DATASET_FILE, index=False, encoding="utf-8-sig")
            logger.info(f"-> Đã lưu pre-processing dataset tại: {PREPROCESSING_DATASET_FILE}")
            
            summary_rows.append(
                {
                    "window_mode": "fixed",
                    "window_value": int(OBSERVATION_DAYS),
                    "num_rows": int(len(df_window)),
                    "num_users": int(final_df[PRIMARY_KEY].nunique()),
                    "output_file": output_path,
                }
            )

        summarize_outputs(summary_rows)
        del df
        gc.collect()

        print("=" * 80)
        logger.info("HOÀN TẤT GIAI ĐOẠN 2.")
        logger.info(f"File tương thích cho Stage 3: {FEATURES_COMPAT_FILE}")
        logger.info(f"Số cấu hình time-window đã xuất: {len(summary_rows)}")
        print("=" * 80)

    except Exception as e:
        logger.exception("Đã xảy ra lỗi nghiêm trọng trong quá trình xử lý:")

if __name__ == "__main__":
    main()
