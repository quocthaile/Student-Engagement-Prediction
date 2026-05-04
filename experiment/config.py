from pathlib import Path
import json

# ============================================================================
# CONFIGURATION GUIDE
# ============================================================================
# Làm việc với cấu hình linh hoạt:
#
# 1. ĐỔI CHIẾN LƯỢC TRAIN CLASS RATIOS (Phân bố lớp trong train):
#    - Mở config.py, tìm TRAIN_CLASS_RATIOS
#    - Chọn 1 trong các chiến lược preset hoặc gõ tùy chỉnh
#    - VD dùng Balanced 3-way qua run_pipeline:
#      $ python run_pipeline.py --param TRAIN_CLASS_RATIOS='{"Low_Engagement":1.0,"Medium_Engagement":1.0,"High_Engagement":1.0}'
#
# 2. ĐỔI LABEL PERCENTILES (Mốc chia nhãn khi tạo ground truth):
#    - Mở config.py, tìm LABEL_PERCENTILES
#    - Chọn preset hoặc gõ tuple (p1, p2)
#    - VD dùng Aggressive low detection qua run_pipeline:
#      $ python run_pipeline.py --param LABEL_PERCENTILES='[0.70, 0.90]'
#
# 3. LỰA CHỌN PRESET CÓ SẴN:
#    - Import từ config: from config import get_label_percentile_presets, get_train_ratio_presets
#    - Xem các lựa chọn: get_label_percentile_presets(), get_train_ratio_presets()
#

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATASET_DIR = PROJECT_DIR / "dataset"
WINDOW_SUFFIX = "4w"

# Input
# Use combined_all_data.parquet located inside the project's `dataset/` folder
RAW_DATA_PARQUET = DATASET_DIR / "combined_all_data.parquet"
RANDOM_STATE = 42
PRIMARY_KEY = "user_id"

# Time windows suggested for early warning / time series modeling
TIME_WINDOWS_DAYS = [14, 21, 28]
DEFAULT_OBSERVATION_DAYS = 28
# FIXED MODE: Use 28-day fixed observation window (no relative/percentage-based windows)
TIME_WINDOW_MODE = "fixed"

MODEL_DATA_DIR = DATASET_DIR / "model_data"
FEATURES_WINDOW_FILE = DATASET_DIR / f"user_features_{WINDOW_SUFFIX}.csv"
FEATURES_COMPAT_FILE = DATASET_DIR / "user_features_and_wes.csv"
RELATIVE_FEATURES_OUTPUT_PATTERN = DATASET_DIR / "user_features_relative_{pct}.csv"
TIME_WINDOW_COMPARE_SUMMARY_FILE = DATASET_DIR / "time_window_comparison.csv"
EXPERIMENTAL_DATASET_FILE = DATASET_DIR / "experimental_dataset.csv"
TRAIN_FILE = MODEL_DATA_DIR / "train_smote.csv"
VALID_FILE = MODEL_DATA_DIR / "valid_original.csv"
# Preprocessed dataset placed at top-level dataset folder per user request
PREPROCESSING_DATASET_FILE = DATASET_DIR / "pos-processing_dataset.csv"
TEST_FILE = MODEL_DATA_DIR / "test_original.csv"
MODEL_OUT_DIR = BASE_DIR / "deployment_models"
IMAGE_OUT_DIR = BASE_DIR / f"output_images_{WINDOW_SUFFIX}"
MODEL_BUNDLE_FILE = MODEL_OUT_DIR / "deployment_bundle.pkl"
CONFIG_SEARCH_RESULTS_FILE = MODEL_OUT_DIR / "config_search_results.csv"

# Backward compatibility alias
FULL_PREPROCESSED_FILE = PREPROCESSING_DATASET_FILE

# Stage outputs
GROUND_TRUTH_FILE = DATASET_DIR / "ground_truth_labels.csv"
GROUND_TRUTH_REPORT_FILE = DATASET_DIR / "ground_truth_report.csv"
# Exported intermediate artifacts
USER_FEATURES_AGGREGATED_FILE = DATASET_DIR / "user_features_aggregated.csv"
STAGE2_TIMELINE_FILE = DATASET_DIR / "stage2_action_timeline.csv"

# ============================================================================
# LABELING & TRAINING DATA CONTROLS
# ============================================================================

LABELING_STRATEGY = "quantile_rank"

# LABEL_PERCENTILES: Tỷ lệ chia nhãn khi gán ground truth (Low_Engagement | Medium | High)
# 
# Các lựa chọn preset:
#   (0.60, 0.85) -> 60% Low, 25% Medium, 15% High    [Khuyến nghị: cân bằng tốt cho early warning]
#   (0.50, 0.80) -> 50% Low, 30% Medium, 20% High    [Balanced: Cân bằng giữa 3 lớp]
#   (0.70, 0.90) -> 70% Low, 20% Medium, 10% High    [Aggressive: Tập trung cao vào Low]
#   (0.55, 0.82) -> 55% Low, 27% Medium, 18% High    [Tối ưu: Dựa trên nghiên cứu early warning]
#
# Cách sử dụng: Đổi thành bất kỳ tuple (p1, p2) nào, trong đó 0 < p1 < p2 < 1
LABEL_PERCENTILES = (0.33, 0.67)

MAX_TRAIN_SAMPLES_PER_CLASS = None
TRAIN_TARGET_TOTAL_SAMPLES = None

# TRAIN_CLASS_RATIOS: Tỷ lệ lớp khi cân bằng dữ liệu train
#
# Các lựa chọn preset:
#
#   Chiến lược 1 - ƯU TIÊN PHÁT HIỆN LOW (hiện tại):
#   {
#       "Low_Engagement": 4.8,
#       "Medium_Engagement": 1.8,
#       "High_Engagement": 3.5,
#   }
#   -> Low:Medium:High = 4.8:1.8:3.5 ≈ 48:18:34 (nếu TRAIN_TARGET_TOTAL_SAMPLES=60000)
#      Cách dùng: Cảnh báo sớm, muốn Recall cao cho Low Engagement
#
#   Chiến lược 2 - CÂN BẰNG 3 LỚP:
#   {
#       "Low_Engagement": 1.0,
#       "Medium_Engagement": 1.0,
#       "High_Engagement": 1.0,
#   }
#   -> Low:Medium:High = 1:1:1 (balanced 3 ways)
#      Cách dùng: Khi muốn Model phân loại tổng quát cho cả 3 lớp
#
#   Chiến lược 3 - DÙNG PHÂN PHỐI THỰC TẾ:
#   {
#       "Low_Engagement": 0.60,
#       "Medium_Engagement": 0.25,
#       "High_Engagement": 0.15,
#   }
#   -> Giữ tỷ lệ gốc từ dữ liệu (60:25:15)
#      Cách dùng: Muốn train phản ánh phân phối tự nhiên
#
#   Chiến lược 4 - ƯU TIÊN LOW & MEDIUM:
#   {
#       "Low_Engagement": 3.0,
#       "Medium_Engagement": 2.5,
#       "High_Engagement": 1.0,
#   }
#   -> Low:Medium:High = 3:2.5:1 (ưu tiên phát hiện Low & Medium)
#
# Cách sử dụng: Chọn 1 chiến lược hoặc gõ dict tùy chỉnh, sau đó override qua run_pipeline
TRAIN_CLASS_RATIOS = {
    "Low_Engagement": 1.0,
    "Medium_Engagement": 1.0,
    "High_Engagement": 1.0,
  }

# ENABLE_SMOTE: Áp dụng SMOTE trên training data sau khi split
# - True: Tạo thêm minority class samples (data augmentation)
# - False: Giữ đúng tỷ lệ cân bằng, bỏ SMOTE (khuyến nghị khi TRAIN_CLASS_RATIOS đã 1:1:1)
ENABLE_SMOTE = True


# Runtime overrides file (written by run_pipeline for interactive runs)
RUNTIME_OVERRIDES_FILE = BASE_DIR / "runtime_overrides.json"


def _apply_runtime_overrides():
    if not RUNTIME_OVERRIDES_FILE.exists():
        return
    try:
        with open(RUNTIME_OVERRIDES_FILE, "r", encoding="utf-8") as f:
            overrides = json.load(f)
    except Exception:
        return

    # Apply simple overrides: if key exists in module globals, replace it
    gl = globals()
    for k, v in overrides.items():
        if k in gl:
            gl[k] = v


_apply_runtime_overrides()

# ============================================================================
# HELPER FUNCTIONS FOR FLEXIBLE CONFIGURATION
# ============================================================================

def get_label_percentile_presets():
    """Trả về các preset phân vị tối ưu cho từng trường hợp sử dụng."""
    return {
        "early_warning": (0.60, 0.85),       # Ưu tiên phát hiện Low (mặc định)
        "balanced_3way": (0.33, 0.67),       # Cân bằng 3 lớp
        "aggressive_low": (0.70, 0.90),      # Tập trung cao vào Low Engagement
        "optimized": (0.55, 0.82),           # Tối ưu dựa trên nghiên cứu
        "conservative": (0.50, 0.80),        # Cân bằng hơn nhưng vẫn ưu tiên Low
    }

def get_train_ratio_presets():
    """Trả về các preset tỷ lệ lớp cho các chiến lược khác nhau."""
    return {
        "early_warning": {
            "Low_Engagement": 4.8,
            "Medium_Engagement": 1.8,
            "High_Engagement": 3.5,
        },
        "balanced_3way": {
            "Low_Engagement": 1.0,
            "Medium_Engagement": 1.0,
            "High_Engagement": 1.0,
        },
        "natural_distribution": {
            "Low_Engagement": 0.60,
            "Medium_Engagement": 0.25,
            "High_Engagement": 0.15,
        },
        "low_and_medium": {
            "Low_Engagement": 3.0,
            "Medium_Engagement": 2.5,
            "High_Engagement": 1.0,
        },
    }

def get_pipeline_search_presets():
    """Trả về các tổ hợp config candidate để run_pipeline tự thử và chọn best model."""
    return {
        "balanced_3way_minmax": {
            "LABEL_PERCENTILES": (0.60, 0.85),
            "TRAIN_CLASS_RATIOS": {
                "Low_Engagement": 1.0,
                "Medium_Engagement": 1.0,
                "High_Engagement": 1.0,
            },
            "USE_ENTROPY_WEIGHT_METHOD": True,
            "LAPLACE_SMOOTHING_ALPHA": 1.0,
            "SCALER_TYPE": "minmax",
        },
        "early_warning_robust": {
            "LABEL_PERCENTILES": (0.60, 0.85),
            "TRAIN_CLASS_RATIOS": {
                "Low_Engagement": 4.8,
                "Medium_Engagement": 1.8,
                "High_Engagement": 3.5,
            },
            "USE_ENTROPY_WEIGHT_METHOD": True,
            "LAPLACE_SMOOTHING_ALPHA": 1.0,
            "SCALER_TYPE": "robust",
        },
        "balanced_3way_robust": {
            "LABEL_PERCENTILES": (0.50, 0.80),
            "TRAIN_CLASS_RATIOS": {
                "Low_Engagement": 1.0,
                "Medium_Engagement": 1.0,
                "High_Engagement": 1.0,
            },
            "USE_ENTROPY_WEIGHT_METHOD": True,
            "LAPLACE_SMOOTHING_ALPHA": 1.0,
            "SCALER_TYPE": "robust",
        },
        "aggressive_low_robust": {
            "LABEL_PERCENTILES": (0.70, 0.90),
            "TRAIN_CLASS_RATIOS": {
                "Low_Engagement": 3.0,
                "Medium_Engagement": 2.5,
                "High_Engagement": 1.0,
            },
            "USE_ENTROPY_WEIGHT_METHOD": True,
            "LAPLACE_SMOOTHING_ALPHA": 1.0,
            "SCALER_TYPE": "robust",
        },
        "natural_distribution_robust": {
            "LABEL_PERCENTILES": (0.55, 0.82),
            "TRAIN_CLASS_RATIOS": {
                "Low_Engagement": 0.60,
                "Medium_Engagement": 0.25,
                "High_Engagement": 0.15,
            },
            "USE_ENTROPY_WEIGHT_METHOD": True,
            "LAPLACE_SMOOTHING_ALPHA": 1.0,
            "SCALER_TYPE": "robust",
        },
    }

def compute_ratios_from_distribution(actual_counts: dict) -> dict:
    """
    Tính tỷ lệ lớp từ phân phối thực tế.
    
    Args:
        actual_counts: dict với keys là tên lớp, values là số lượng mẫu
                      Ví dụ: {"Low_Engagement": 77710, "Medium_Engagement": 32378, ...}
    
    Returns:
        dict tỷ lệ chuẩn hóa (tổng = 1.0 nếu thực hiện normalization)
    """
    total = sum(actual_counts.values())
    return {label: count / total for label, count in actual_counts.items()}

def suggest_optimal_percentiles(label_distribution: dict) -> tuple:
    """
    Gợi ý percentiles tối ưu dựa trên phân phối nhãn thực tế.
    
    Args:
        label_distribution: dict phân phối nhãn, ví dụ từ ground_truth_labels.csv value_counts()
    
    Returns:
        tuple (p1, p2) gợi ý cho LABEL_PERCENTILES
    """
    # Nếu Low_Engagement < 50% trong dữ liệu thực tế, sử dụng 60:25:15
    # Nếu Low_Engagement > 70%, sử dụng 70:20:10 (aggressive)
    # Ngược lại: 55:27:18 (balanced)
    
    total = sum(label_distribution.values())
    low_pct = label_distribution.get("Low_Engagement", 0) / total if total > 0 else 0
    
    if low_pct < 0.50:
        return (0.60, 0.85)  # Ưu tiên phát hiện Low (mặc định)
    elif low_pct > 0.70:
        return (0.70, 0.90)  # Aggressive: tập trung cao vào Low
    else:
        return (0.55, 0.82)  # Balanced: cân bằng giữa các lớp

GROUND_TRUTH_WEIGHTS = {
    "total_study_time": 0.35,
    "avg_score": 0.30,
    "accuracy_rate": 0.20,
    "attempts": 0.10,
    "total_forum_activity": 0.05,
}

# CHỌN PHƯƠNG PHÁP TÍNH TRỌNG SỐ:
# USE_ENTROPY_WEIGHT_METHOD = True  → Tính tự động bằng EWM (khoa học, không bias)
# USE_ENTROPY_WEIGHT_METHOD = False → Dùng hardcode GROUND_TRUTH_WEIGHTS (tuỳ chỉnh thủ công)
USE_ENTROPY_WEIGHT_METHOD = True

# LAPLACE SMOOTHING cho accuracy_rate:
# Tránh ảo hóa khi attempts quá thấp
# Công thức: (is_correct + alpha) / (attempts + 2*alpha)
# alpha = 1.0 là Laplace smoothing chuẩn; có thể tuỳ chỉnh
LAPLACE_SMOOTHING_ALPHA = 1.0

# Cấu hình Scaler để chuẩn hóa features:
# - "robust": RobustScaler (chống outlier tốt, mặc định mới)
# - "minmax": MinMaxScaler (lựa chọn cũ, nhạy hơn với outlier)
SCALER_TYPE = "robust"

# Cấu hình auto-search cho run_pipeline
# - "accuracy_recall": ưu tiên đồng thời Accuracy và Recall_Low_Engagement
# - "recall_first": ưu tiên Recall_Low_Engagement trước
# - "balanced": cân bằng F1/Balanced Accuracy
PIPELINE_AUTO_SELECT_MODE = "accuracy_recall"
PIPELINE_AUTO_SELECT_CANDIDATES = [
    "balanced_3way_minmax",
    "early_warning_robust",
    "balanced_3way_robust",
    "aggressive_low_robust",
    "natural_distribution_robust",
]

# Required raw columns for stage 1
RAW_REQUIRED_COLUMNS_STEP1 = [
    "user_id",
    "attempts",
    "is_correct",
    "score",
    "create_time_x",
    "create_time_y",
]
