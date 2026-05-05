import pandas as pd
import numpy as np
import sys
from pathlib import Path
from imblearn.over_sampling import SMOTE, SMOTENC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


from config import (
    FEATURES_COMPAT_FILE,
    PRIMARY_KEY,
    RANDOM_STATE,
    MODEL_DATA_DIR,
    MODEL_OUT_DIR,
    TRAIN_FILE,
    VALID_FILE,
    TEST_FILE,
    WINDOW_SUFFIX,
    ENABLE_SMOTE,
)

FEATURES_AND_LABELS_FILE = FEATURES_COMPAT_FILE
TARGET_LABELS_ORDER = ["Low_Engagement", "Medium_Engagement", "High_Engagement"]

# Features để sử dụng cho train/valid/test
TRAINING_FEATURES = [
    "school_encoded",
    "seq",
    "speed",
    "rep_counts",
    "cmt_counts",
    "age",
    "gender_encoded",
    "num_courses",
    f"attempts_{WINDOW_SUFFIX}",
    f"is_correct_{WINDOW_SUFFIX}",
    f"score_{WINDOW_SUFFIX}",
    f"accuracy_rate_{WINDOW_SUFFIX}",
]

STATIC_COLUMNS = [
    "school_encoded",
    "seq",
    "speed",
    "rep_counts",
    "cmt_counts",
    "age",
    "gender_encoded",
    "num_courses",
]

DYNAMIC_COLUMNS = [
    f"attempts_{WINDOW_SUFFIX}",
    f"is_correct_{WINDOW_SUFFIX}",
    f"score_{WINDOW_SUFFIX}",
    f"accuracy_rate_{WINDOW_SUFFIX}",
]

RAW_METADATA_COLUMNS = [
    "school",
    "year_of_birth",
    "gender",
]

CATEGORICAL_FEATURES = [
    "school_encoded",
    "gender_encoded",
]


def prepare_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    for column in TRAINING_FEATURES:
        if column not in prepared.columns:
            prepared[column] = np.nan

    for column in TRAINING_FEATURES:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    for column in CATEGORICAL_FEATURES:
        if column in prepared.columns:
            prepared[column] = prepared[column].fillna(-1).astype(int)

    numeric_columns = [column for column in TRAINING_FEATURES if column not in CATEGORICAL_FEATURES]
    for column in numeric_columns:
        series = prepared[column]
        fill_value = series.median() if series.notna().any() else 0
        if pd.isna(fill_value):
            fill_value = 0
        prepared[column] = series.fillna(fill_value)

    return prepared


def reorder_output_columns(frame: pd.DataFrame, include_metadata: bool = False) -> pd.DataFrame:
    ordered_columns = [
        *[column for column in STATIC_COLUMNS if column in frame.columns],
    ]

    if include_metadata:
        ordered_columns.extend([column for column in RAW_METADATA_COLUMNS if column in frame.columns])

    ordered_columns.extend([column for column in DYNAMIC_COLUMNS if column in frame.columns])

    if "target_label" in frame.columns:
        ordered_columns.append("target_label")

    remaining_columns = [column for column in frame.columns if column not in ordered_columns]
    ordered_columns.extend(remaining_columns)
    return frame[ordered_columns]


def main():
    print("=" * 80)
    smote_desc = "+ SMOTE" if ENABLE_SMOTE else "(SMOTE disabled)"
    print(f"STEP 3: STRATIFIED SPLITTING (8:1:1) {smote_desc}")
    print("=" * 80)

    try:
        MODEL_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        if not FEATURES_AND_LABELS_FILE.exists():
            raise FileNotFoundError(f"Missing features file: {FEATURES_AND_LABELS_FILE}")

        # Load data
        df = pd.read_csv(FEATURES_AND_LABELS_FILE)
        print(f"Đã tải dữ liệu: {len(df):,} dòng")
        print(f"Cột: {list(df.columns)}")

        # Ensure target_label exists
        if "target_label" not in df.columns:
            raise KeyError("Missing target_label column")

        # Static features: age, gender_encoded
        current_year = 2024
        if "year_of_birth" in df.columns and "age" not in df.columns:
            df["age"] = (current_year - pd.to_numeric(df["year_of_birth"], errors="coerce")).clip(lower=10, upper=100)

        if "gender" in df.columns and "gender_encoded" not in df.columns:
            gender_map = {"male": 0, "female": 1}
            df["gender_encoded"] = (
                df["gender"]
                .map(gender_map)
                .fillna(pd.to_numeric(df["gender"], errors="coerce"))
                .fillna(-1)
                .astype(int)
            )

        if "school_encoded" in df.columns:
            df["school_encoded"] = pd.to_numeric(df["school_encoded"], errors="coerce").fillna(-1).astype(int)
        if "age" in df.columns:
            df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(df["age"].median())
        if "gender_encoded" in df.columns:
            df["gender_encoded"] = pd.to_numeric(df["gender_encoded"], errors="coerce").fillna(-1).astype(int)
        
        # Check required training features
        missing = [c for c in TRAINING_FEATURES if c not in df.columns]
        if missing:
            raise KeyError(f"Missing training features: {missing}")
        
        print(f"Label distribution: {df['target_label'].value_counts().to_dict()}")

        # Stratified Split: 8:1:1 (train 80%, temp 20% → valid 50%, test 50%)
        df_train, df_temp = train_test_split(
            df,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=df["target_label"],
        )
        df_valid, df_test = train_test_split(
            df_temp,
            test_size=0.5,
            random_state=RANDOM_STATE,
            stratify=df_temp["target_label"],
        )
        
        print(f" Split: Train {len(df_train)}, Valid {len(df_valid)}, Test {len(df_test)}")

        # Encode labels
        label_encoder = LabelEncoder()
        label_encoder.fit(TARGET_LABELS_ORDER)
        
        y_train = label_encoder.transform(df_train["target_label"])
        y_valid = label_encoder.transform(df_valid["target_label"])
        y_test = label_encoder.transform(df_test["target_label"])

        # Prepare feature matrices
        X_train = prepare_feature_frame(df_train[TRAINING_FEATURES])
        X_valid = prepare_feature_frame(df_valid[TRAINING_FEATURES])
        X_test = prepare_feature_frame(df_test[TRAINING_FEATURES])

        # Apply SMOTE on training data only (if enabled)
        if ENABLE_SMOTE:
            print("Áp dụng SMOTE trên training data...")
            categorical_indices = [
                idx for idx, col in enumerate(TRAINING_FEATURES) if col in CATEGORICAL_FEATURES
            ]
            if categorical_indices:
                print("  -> Dùng SMOTENC cho đặc trưng phân loại (school/gender)")
                smote = SMOTENC(categorical_features=categorical_indices, random_state=RANDOM_STATE, k_neighbors=5)
                X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
            else:
                smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
                X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
            print(f"  Train sau SMOTE: {len(X_train_smote)} mẫu")
            print(f"  Label distribution: {np.unique(y_train_smote, return_counts=True)}")
        else:
            print("ℹ SMOTE bị tắt (ENABLE_SMOTE=False), giữ nguyên tỷ lệ cân bằng train")
            X_train_smote = X_train
            y_train_smote = y_train
            print(f"  Train giữ nguyên: {len(X_train_smote)} mẫu")
            print(f"  Label distribution: {np.unique(y_train_smote, return_counts=True)}")

        # Save datasets
        train_final = pd.DataFrame(X_train_smote, columns=TRAINING_FEATURES)
        train_final["target_label"] = label_encoder.inverse_transform(y_train_smote)
        train_final = reorder_output_columns(train_final, include_metadata=False)
        train_final.to_csv(TRAIN_FILE, index=False, encoding="utf-8-sig")
        print(f"Lưu train: {TRAIN_FILE}")

        valid_final = pd.DataFrame(X_valid, columns=TRAINING_FEATURES)
        for col in RAW_METADATA_COLUMNS:
            if col in df_valid.columns:
                valid_final[col] = df_valid[col].values
        valid_final["target_label"] = df_valid["target_label"].values
        valid_final = reorder_output_columns(valid_final, include_metadata=True)
        valid_final.to_csv(VALID_FILE, index=False, encoding="utf-8-sig")
        print(f"Lưu valid: {VALID_FILE}")

        test_final = pd.DataFrame(X_test, columns=TRAINING_FEATURES)
        for col in RAW_METADATA_COLUMNS:
            if col in df_test.columns:
                test_final[col] = df_test[col].values
        test_final["target_label"] = df_test["target_label"].values
        test_final = reorder_output_columns(test_final, include_metadata=True)
        test_final.to_csv(TEST_FILE, index=False, encoding="utf-8-sig")
        print(f"Lưu test: {TEST_FILE}")

        MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(label_encoder, MODEL_OUT_DIR / "label_encoder.pkl")
        print(f"Lưu label encoder: {MODEL_OUT_DIR / 'label_encoder.pkl'}")

        print("=" * 80)
        print("HOÀN TẤT Stage 3: Split stratified + SMOTE")
        print("=" * 80)

    except Exception as e:
        print(f"Lỗi: {e}")
        raise


if __name__ == "__main__":
    main()
