import pandas as pd
import numpy as np
import sys
from pathlib import Path
from imblearn.over_sampling import SMOTE
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
    f"attempts_{WINDOW_SUFFIX}",
    f"is_correct_{WINDOW_SUFFIX}",
    f"score_{WINDOW_SUFFIX}",
    f"accuracy_rate_{WINDOW_SUFFIX}",
    "num_courses",
]


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
        print(f"✓ Đã tải dữ liệu: {len(df):,} dòng")
        print(f"  Cột: {list(df.columns)}")

        # Ensure target_label exists
        if "target_label" not in df.columns:
            raise KeyError("Missing target_label column")
        
        # Check required training features
        missing = [c for c in TRAINING_FEATURES if c not in df.columns]
        if missing:
            raise KeyError(f"Missing training features: {missing}")
        
        print(f"✓ Label distribution: {df['target_label'].value_counts().to_dict()}")

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
        
        print(f"✓ Split: Train {len(df_train)}, Valid {len(df_valid)}, Test {len(df_test)}")

        # Encode labels
        label_encoder = LabelEncoder()
        label_encoder.fit(TARGET_LABELS_ORDER)
        
        y_train = label_encoder.transform(df_train["target_label"])
        y_valid = label_encoder.transform(df_valid["target_label"])
        y_test = label_encoder.transform(df_test["target_label"])

        # Prepare feature matrices
        X_train = df_train[TRAINING_FEATURES].copy()
        X_valid = df_valid[TRAINING_FEATURES].copy()
        X_test = df_test[TRAINING_FEATURES].copy()

        # Apply SMOTE on training data only (if enabled)
        if ENABLE_SMOTE:
            print("✓ Áp dụng SMOTE trên training data...")
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
        train_final.to_csv(TRAIN_FILE, index=False, encoding="utf-8-sig")
        print(f"✓ Lưu train: {TRAIN_FILE}")

        valid_final = pd.DataFrame(X_valid, columns=TRAINING_FEATURES)
        valid_final["target_label"] = df_valid["target_label"].values
        valid_final.to_csv(VALID_FILE, index=False, encoding="utf-8-sig")
        print(f"✓ Lưu valid: {VALID_FILE}")

        test_final = pd.DataFrame(X_test, columns=TRAINING_FEATURES)
        test_final["target_label"] = df_test["target_label"].values
        test_final.to_csv(TEST_FILE, index=False, encoding="utf-8-sig")
        print(f"✓ Lưu test: {TEST_FILE}")

        MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(label_encoder, MODEL_OUT_DIR / "label_encoder.pkl")
        print(f"✓ Lưu label encoder: {MODEL_OUT_DIR / 'label_encoder.pkl'}")

        print("=" * 80)
        print("HOÀN TẤT Stage 3: Split stratified + SMOTE")
        print("=" * 80)

    except Exception as e:
        print(f"❌ Lỗi: {e}")
        raise


if __name__ == "__main__":
    main()
