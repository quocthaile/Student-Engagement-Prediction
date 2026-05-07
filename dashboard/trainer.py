import pandas as pd
import numpy as np
import joblib
import json
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_model(model_type='svc'):
    print(f"🚀 Bắt đầu huấn luyện mô hình: {model_type}...")
    
    # Path configuration
    BASE_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = BASE_DIR.parent
    MODEL_DATA_DIR = PROJECT_ROOT / "dataset" / "model_data"
    
    TRAIN_PATH = MODEL_DATA_DIR / "train_smote.csv"
    VALID_PATH = MODEL_DATA_DIR / "valid_original.csv"
    TEST_PATH = MODEL_DATA_DIR / "test_original.csv"
    
    MODEL_DIR = BASE_DIR / "models"
    MODEL_DIR.mkdir(exist_ok=True)
    
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        print(f"❌ Không tìm thấy các file train/test tại: {MODEL_DATA_DIR}")
        return False

    # 1. Load data
    print("✓ Đang nạp dữ liệu từ train_smote.csv và test.csv...")
    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)
    
    # 2. Define features and target
    # Bộ 6 đặc trưng có sẵn trong file dữ liệu đã xử lý
    feature_cols = [
        'school_encoded', 'seq', 'speed', 'rep_counts', 'cmt_counts', 
        'age', 'gender_encoded', 'num_courses', 'attempts_4w', 
        'is_correct_4w', 'score_4w', 'accuracy_rate_4w'
    ]
    target_col = 'target_label'
    
    X_train = df_train[feature_cols]
    y_train = df_train[target_col]
    X_test = df_test[feature_cols]
    y_test = df_test[target_col]
    
    # 3. Preprocessing
    print("✓ Đang tiền xử lý (Imputing & Scaling)...")
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    # Fit imputer and scaler on training data
    X_train_imputed = imputer.fit_transform(X_train)
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    
    # Transform test data
    X_test_imputed = imputer.transform(X_test)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    # Encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    # 5. Initialize & Train Model
    print(f"   -> Đang huấn luyện {model_type} trên {len(X_train)} mẫu...")
    if model_type == 'rf':
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        display_name = "Random Forest"
    elif model_type == 'lr':
        model = LogisticRegression(max_iter=1000, random_state=42)
        display_name = "Logistic Regression"
    elif model_type == 'dt':
        model = DecisionTreeClassifier(random_state=42)
        display_name = "Decision Tree"
    else:
        model = LinearSVC(dual=False, random_state=42, max_iter=2000)
        display_name = "Linear SVC"


    model.fit(X_train_scaled, y_train_encoded)
    
    # 6. Evaluate
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test_encoded, y_pred)
    report = classification_report(y_test_encoded, y_pred, target_names=le.classes_, output_dict=True)
    
    print(f"✅ Huấn luyện hoàn tất! Accuracy: {acc:.4f}")
    
    # 7. Save Artifacts
    # Try to load school_encoder from experiment directory if it exists
    EXPERIMENT_MODELS_DIR = PROJECT_ROOT / "experiment" / "deployment_models"
    school_encoder_file = EXPERIMENT_MODELS_DIR / "school_encoder.pkl"
    school_encoder = None
    if school_encoder_file.exists():
        try:
            school_encoder = joblib.load(school_encoder_file)
            print(f"✓ Đã nạp school_encoder từ {school_encoder_file}")
        except Exception as e:
            print(f"⚠️ Không thể nạp school_encoder: {e}")

    bundle = {
        'model': model,
        'scaler': scaler,
        'imputer': imputer,
        'label_encoder': le,
        'school_encoder': school_encoder,
        'feature_columns': feature_cols,
        'target_labels': list(le.classes_),
        'model_type': model_type
    }
    
    joblib.dump(bundle, MODEL_DIR / "deployment_bundle.pkl")
    
    # Save metadata for Dashboard
    # 1. Feature Importance
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
    else:
        importance = np.zeros(len(feature_cols))

    feature_importance = [
        {"feature": f, "importance": float(i)} 
        for f, i in zip(feature_cols, importance)
    ]
    feature_importance = sorted(feature_importance, key=lambda x: x['importance'], reverse=True)

    # 2. Confusion Matrix
    cm = confusion_matrix(y_test_encoded, y_pred)
    
    # 3. Label Distributions (Real data for charts)
    true_dist = pd.Series(y_test).value_counts(normalize=True).to_dict()
    pred_dist = pd.Series(le.inverse_transform(y_pred)).value_counts(normalize=True).to_dict()
    
    # 4. Data Stats
    data_stats = {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "feature_count": len(feature_cols)
    }

    metadata = {
        "model_name": display_name,
        "model_type": model_type,
        "accuracy": acc,
        "test_metrics": {
            "Accuracy": acc,
            "Recall_Low_Engagement": report.get("Low_Engagement", {}).get("recall", 0),
            "Precision_Low_Engagement": report.get("Low_Engagement", {}).get("precision", 0),
            "F1_Weighted": report.get("weighted avg", {}).get("f1-score", 0),
            "Recall_High": report.get("High_Engagement", {}).get("recall", 0),
            "Recall_Medium": report.get("Medium_Engagement", {}).get("recall", 0)
        },
        "class_metrics": {
            "Low": report.get("Low_Engagement", {}),
            "Medium": report.get("Medium_Engagement", {}),
            "High": report.get("High_Engagement", {})
        },
        "distributions": {
            "true": {k: float(v) for k, v in true_dist.items()},
            "pred": {k: float(v) for k, v in pred_dist.items()}
        },
        "data_stats": data_stats,
        "confusion_matrix": cm.tolist(),
        "feature_importance": feature_importance,
        "feature_columns": feature_cols,
        "target_names": list(le.classes_)
    }
    
    with open(MODEL_DIR / "metadata.json", "w", encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    # 8. Update benchmarks
    benchmark_file = MODEL_DIR / "benchmarks.json"
    benchmarks = {}
    if benchmark_file.exists():
        try:
            with open(benchmark_file, 'r', encoding='utf-8') as f:
                benchmarks = json.load(f)
        except:
            pass
    
    benchmarks[model_type] = {
        "name": display_name,
        "accuracy": acc,
        "f1": report.get("weighted avg", {}).get("f1-score", 0)
    }
    
    with open(benchmark_file, 'w', encoding='utf-8') as f:
        json.dump(benchmarks, f, indent=2)
    
    # 9. Optimized Sample Generation (Vectorized)
    print("✓ Đang tạo dữ liệu mẫu cho dashboard...")
    
    # Create base dataframe for samples
    df_samples = df_test.copy()
    df_samples['y_pred_label'] = le.inverse_transform(y_pred)
    df_samples['row_id'] = df_samples.index
    
    # Handle school name transformation efficiently if encoder exists
    if school_encoder:
        try:
            # Get unique encoded values in test set to minimize inverse_transform calls
            unique_codes = df_samples['school_encoded'].unique()
            code_to_name = {
                code: str(school_encoder.inverse_transform([[code]])[0][0])
                for code in unique_codes
            }
            df_samples['school_name'] = df_samples['school_encoded'].map(code_to_name)
        except Exception as e:
            print(f"⚠️ Error in school name mapping: {e}")
            df_samples['school_name'] = "N/A"
    else:
        df_samples['school_name'] = "N/A"

    # Convert to list of dicts for the JSON file (still needed for now, but we'll paginate in app.py)
    # We'll structure it to be easier for pagination
    # Note: We only keep relevant columns to reduce file size
    # Note: We keep relevant columns to reduce file size, including raw info for "accurate" detail view
    cols_to_keep = ['row_id', 'target_label', 'y_pred_label', 'school_name', 'school', 'year_of_birth', 'gender'] + feature_cols
    test_samples = df_samples[cols_to_keep].rename(columns={'target_label': 'y_true_label'}).to_dict(orient='records')
    
    with open(MODEL_DIR / "sample_predictions.json", "w", encoding='utf-8') as f:
        json.dump(test_samples, f) # No indent to save space

    return True


if __name__ == "__main__":
    import sys
    m_type = sys.argv[1] if len(sys.argv) > 1 else 'svc'
    train_model(m_type)
