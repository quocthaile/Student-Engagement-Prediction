import json
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import (
    IMAGE_OUT_DIR,
    MODEL_BUNDLE_FILE,
    MODEL_DATA_DIR,
    MODEL_OUT_DIR,
    TEST_FILE,
    TRAIN_FILE,
    VALID_FILE,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

LABEL_ENCODER_FILE = MODEL_OUT_DIR / "label_encoder.pkl"
SCHOOL_ENCODER_FILE = MODEL_OUT_DIR / "school_encoder.pkl"
IMPUTER_FILE = MODEL_OUT_DIR / "imputer.pkl"
SCALER_FILE = MODEL_OUT_DIR / "scaler.pkl"

BEST_MODEL_FILE = MODEL_OUT_DIR / "best_model_4w.pkl"
METRICS_FILE = MODEL_OUT_DIR / "evaluation_metrics.csv"
MODEL_META_FILE = MODEL_OUT_DIR / "best_model_metadata.json"

RANDOM_STATE = 42
TARGET_RISK_CLASS = "Low_Engagement"
TARGET_LABELS_ORDER = ["Low_Engagement", "Medium_Engagement", "High_Engagement"]


def export_prediction_details(
    output_path: Path,
    y_true: pd.Series,
    y_pred: np.ndarray,
    label_encoder,
    model_name: str,
    split_name: str,
) -> None:
    detail_df = pd.DataFrame(
        {
            "row_id": np.arange(len(y_true)),
            "model_name": model_name,
            "split": split_name,
            "y_true": y_true.to_numpy(),
            "y_pred": y_pred,
            "y_true_label": label_encoder.inverse_transform(pd.Series(y_true).astype(int)),
            "y_pred_label": label_encoder.inverse_transform(pd.Series(y_pred).astype(int)),
        }
    )
    detail_df.to_csv(output_path, index=False, encoding="utf-8-sig")

def load_data_and_artifacts() -> tuple:
    if not TRAIN_FILE.exists() or not VALID_FILE.exists() or not TEST_FILE.exists():
        logger.error("Không tìm thấy dữ liệu. Vui lòng chạy Giai đoạn 3 trước.")
        raise FileNotFoundError("Run step 3 first.")

    logger.info("Đang nạp train/valid/test và LabelEncoder từ stage 3...")
    train_df = pd.read_csv(TRAIN_FILE)
    valid_df = pd.read_csv(VALID_FILE)
    test_df = pd.read_csv(TEST_FILE)
    if LABEL_ENCODER_FILE.exists():
        label_encoder = joblib.load(LABEL_ENCODER_FILE)
    else:
        logger.warning(
            f"Không tìm thấy {LABEL_ENCODER_FILE}; sẽ tạo lại LabelEncoder theo thứ tự nhãn chuẩn."
        )
        label_encoder = LabelEncoder()
        label_encoder.fit(TARGET_LABELS_ORDER)
        LABEL_ENCODER_FILE.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(label_encoder, LABEL_ENCODER_FILE)
    logger.info(
        f"-> Kích thước dữ liệu: train={len(train_df):,}, valid={len(valid_df):,}, test={len(test_df):,}; "
        f"số lớp={len(label_encoder.classes_)}"
    )

    # Compute 'age' if missing from Stage 3 output
    current_year = 2024
    for df in [train_df, valid_df, test_df]:
        if 'year_of_birth' in df.columns and 'age' not in df.columns:
            df['age'] = (current_year - df['year_of_birth']).clip(lower=10, upper=100)
            logger.info("   -> Tính toán cột 'age' từ 'year_of_birth'")
    
    def encode_target(series: pd.Series) -> pd.Series:
        try:
            return pd.Series(label_encoder.transform(series.astype(str)), index=series.index)
        except Exception:
            return series.astype(int)

    feature_columns = [c for c in train_df.columns if c != "target_label"]
    X_train = train_df[feature_columns].copy()
    y_train = encode_target(train_df["target_label"])
    X_valid = valid_df.drop(columns=["target_label"]).reindex(columns=feature_columns).copy()
    X_valid = X_valid.fillna(0)
    y_valid = encode_target(valid_df["target_label"])
    X_test = test_df.drop(columns=["target_label"]).reindex(columns=feature_columns).copy()
    X_test = X_test.fillna(0)
    y_test = encode_target(test_df["target_label"])

    return X_train, y_train, X_valid, y_valid, X_test, y_test, label_encoder

def safe_name(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_")

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, label_encoder) -> tuple:
    y_pred = model.predict(X_test)
    labels_list = list(range(len(label_encoder.classes_)))
    try:
        low_idx = int(np.where(label_encoder.classes_ == TARGET_RISK_CLASS)[0][0])
    except IndexError:
        logger.warning(f"Không tìm thấy nhãn {TARGET_RISK_CLASS}, gán mặc định Index=0")
        low_idx = 0

    y_score = None
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)

    auc_roc = np.nan
    if y_score is not None:
        try:
            auc_roc = roc_auc_score(y_test, y_score, multi_class="ovr", labels=labels_list)
        except ValueError:
            auc_roc = np.nan

    precisions = precision_score(y_test, y_pred, labels=labels_list, average=None, zero_division=0)
    recalls = recall_score(y_test, y_pred, labels=labels_list, average=None, zero_division=0)
    f1s = f1_score(y_test, y_pred, labels=labels_list, average=None, zero_division=0)
    report = classification_report(
        y_test,
        y_pred,
        labels=labels_list,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Balanced_Accuracy": round(balanced_accuracy_score(y_test, y_pred), 4),
        "F1_Macro": round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "F1_Weighted": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "Precision_Macro": round(report["macro avg"]["precision"], 4),
        "Recall_Macro": round(report["macro avg"]["recall"], 4),
        f"Precision_{TARGET_RISK_CLASS}": round(precisions[low_idx], 4),
        f"Recall_{TARGET_RISK_CLASS}": round(recalls[low_idx], 4),
        f"F1_{TARGET_RISK_CLASS}": round(f1s[low_idx], 4),
        "AUC_ROC_OVR": round(float(auc_roc), 4) if not np.isnan(auc_roc) else "N/A",
    }
    return metrics, y_pred


def build_classification_report_df(y_true, y_pred, label_encoder) -> pd.DataFrame:
    labels_list = list(range(len(label_encoder.classes_)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels_list,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report).T.reset_index().rename(columns={"index": "label"})

def main():
    print("=" * 80)
    print("STEP 4: MODEL TRAINING, BENCHMARKING AND DEPLOYMENT EXPORT")
    print("=" * 80)

    try:
        IMAGE_OUT_DIR.mkdir(parents=True, exist_ok=True)
        MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Đầu vào huấn luyện: {TRAIN_FILE}, {VALID_FILE}, {TEST_FILE}")

        X_train, y_train, X_valid, y_valid, X_test, y_test, label_encoder = load_data_and_artifacts()

        logger.info("[1/4] Đang khởi tạo các thuật toán Học máy...")
        models = {
            "Logistic Regression": OneVsRestClassifier(
                LogisticRegression(
                    solver="liblinear",
                    max_iter=300,
                    random_state=RANDOM_STATE,
                )
            ),
            "Linear SVC": LinearSVC(dual=False, random_state=RANDOM_STATE),
            "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
            "Random Forest": RandomForestClassifier(
                n_estimators=200, max_depth=10, class_weight="balanced", 
                random_state=RANDOM_STATE, n_jobs=-1
            ),
            "XGBoost": XGBClassifier(
                objective="multi:softprob", eval_metric="mlogloss",
                max_depth=5, n_estimators=200, learning_rate=0.08,
                random_state=RANDOM_STATE, n_jobs=-1
            ),
        }

        logger.info("[2/4] Bắt đầu quá trình Huấn luyện và Benchmark...")
        results = []
        fitted_models = {}
        
        for name, model in models.items():
            logger.info(f"   -> Đang huấn luyện: {name} trên {len(X_train):,} mẫu, {X_train.shape[1]} đặc trưng")
            model.fit(X_train, y_train)
            fitted_models[name] = model
            
            valid_metrics, y_valid_pred = evaluate_model(model, X_valid, y_valid, label_encoder)
            valid_metrics["Model"] = name
            results.append(valid_metrics)
            export_prediction_details(
                MODEL_OUT_DIR / f"{safe_name(name)}_valid_predictions.csv",
                y_valid,
                y_valid_pred,
                label_encoder,
                name,
                "valid",
            )
            logger.info(
                f"      -> Kết quả valid của {name}: "
                f"Recall_low={valid_metrics.get(f'Recall_{TARGET_RISK_CLASS}', 'N/A')}, "
                f"F1_macro={valid_metrics.get('F1_Macro', 'N/A')}, "
                f"Balanced_acc={valid_metrics.get('Balanced_Accuracy', 'N/A')}"
            )

        metrics_df = pd.DataFrame(results)
        cols = [
            "Model",
            "Accuracy",
            "Balanced_Accuracy",
            "F1_Macro",
            "F1_Weighted",
            "Recall_Macro",
            f"Recall_{TARGET_RISK_CLASS}",
            f"Precision_{TARGET_RISK_CLASS}",
            f"F1_{TARGET_RISK_CLASS}",
            "AUC_ROC_OVR",
        ]
        metrics_df = metrics_df[cols]
        metrics_df.to_csv(METRICS_FILE, index=False, encoding="utf-8-sig")
        
        print("\n" + "=" * 90)
        print("BẢNG XẾP HẠNG THUẬT TOÁN (Tối ưu cho Cảnh báo sớm)")
        print("=" * 90)
        print(metrics_df.to_string(index=False))
        print("=" * 90 + "\n")

        ranked = metrics_df.sort_values(
            by=[f"Recall_{TARGET_RISK_CLASS}", "F1_Macro", "Balanced_Accuracy", "Accuracy"],
            ascending=False,
        )
        best_model_name = ranked.iloc[0]["Model"]
        best_model = fitted_models[best_model_name]
        logger.info(f"Thuật toán chiến thắng: {best_model_name}")
        logger.info(f"-> Hàng top 1 sau xếp hạng: {ranked.iloc[0].to_dict()}")

        logger.info("[3/4] Đang chấm điểm final trên test cho mô hình đã chọn...")
        test_metrics, test_pred = evaluate_model(best_model, X_test, y_test, label_encoder)
        test_metrics["Model"] = best_model_name
        test_metrics_df = pd.DataFrame([test_metrics])
        test_metrics_df.to_csv(MODEL_OUT_DIR / "final_test_metrics.csv", index=False, encoding="utf-8-sig")
        export_prediction_details(
            MODEL_OUT_DIR / f"{safe_name(best_model_name)}_test_predictions.csv",
            y_test,
            test_pred,
            label_encoder,
            best_model_name,
            "test",
        )
        logger.info(f"   -> Test metrics: {test_metrics}")

        valid_pred = best_model.predict(X_valid)
        valid_report_df = build_classification_report_df(y_valid, valid_pred, label_encoder)
        valid_report_df.to_csv(MODEL_OUT_DIR / "best_model_valid_classification_report.csv", index=False, encoding="utf-8-sig")

        test_report_df = build_classification_report_df(y_test, test_pred, label_encoder)
        test_report_df.to_csv(MODEL_OUT_DIR / "best_model_test_classification_report.csv", index=False, encoding="utf-8-sig")

        cm = confusion_matrix(y_test, test_pred, labels=list(range(len(label_encoder.classes_))))
        plt.figure(figsize=(7, 5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_
        )
        plt.title(f"Ma trận Nhầm lẫn Test - {best_model_name}")
        plt.xlabel("Dự đoán")
        plt.ylabel("Thực tế")
        plt.tight_layout()
        plt.savefig(IMAGE_OUT_DIR / f"CM_TEST_{safe_name(best_model_name)}.png", dpi=200)
        plt.close()

        model_metadata = {
            "model_name": best_model_name,
            "model_class": best_model.__class__.__name__,
            "model_params": best_model.get_params(),
            "validation_metrics": ranked.iloc[0].to_dict(),
            "test_metrics": test_metrics,
            "target_labels": label_encoder.classes_.tolist(),
        }
        with open(MODEL_META_FILE, "w", encoding="utf-8") as f:
            json.dump(model_metadata, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"   -> Đã lưu báo cáo validation/test report và metadata mô hình")

        logger.info("[4/4] Đang đóng gói toàn bộ pipeline để chuẩn bị triển khai...")
        deployment_bundle = {
            "model": best_model,
            "model_name": best_model_name,
            "label_encoder": label_encoder,
            "school_encoder": joblib.load(SCHOOL_ENCODER_FILE) if SCHOOL_ENCODER_FILE.exists() else None,
            "imputer": joblib.load(IMPUTER_FILE) if IMPUTER_FILE.exists() else None,
            "scaler": joblib.load(SCALER_FILE) if SCALER_FILE.exists() else None,
            "feature_columns": list(X_train.columns),
            "target_labels": label_encoder.classes_.tolist()
        }

        joblib.dump(best_model, BEST_MODEL_FILE)
        joblib.dump(deployment_bundle, MODEL_BUNDLE_FILE)
        logger.info(f"   -> Đã lưu best model: {BEST_MODEL_FILE}")
        logger.info(f"   -> Đã lưu deployment bundle: {MODEL_BUNDLE_FILE}")

        logger.info(f"HOÀN TẤT GIAI ĐOẠN 4.")
        logger.info(f"Bundle triển khai lưu tại: {MODEL_BUNDLE_FILE}")
        logger.info(f"Metadata mô hình lưu tại: {MODEL_META_FILE}")

    except Exception as e:
        logger.exception("Đã xảy ra lỗi nghiêm trọng ở Giai đoạn 4:")

if __name__ == "__main__":
    main()
