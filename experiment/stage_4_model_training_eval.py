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
TRAINING_LOG_FILE = MODEL_OUT_DIR / "stage_4_training_eval.log"

RANDOM_STATE = 42
TARGET_RISK_CLASS = "Low_Engagement"
TARGET_LABELS_ORDER = ["Low_Engagement", "Medium_Engagement", "High_Engagement"]
LOW_RECALL_TARGET = 0.60
SELECTION_OBJECTIVE = (
    "Early-warning model selection: use validation only, first require useful "
    "Low_Engagement recall, then prefer fewer false alarms via Low precision."
)
SELECTION_METRIC_ORDER = [
    f"Recall_{TARGET_RISK_CLASS} >= {LOW_RECALL_TARGET}",
    f"Precision_{TARGET_RISK_CLASS}",
    f"F2_{TARGET_RISK_CLASS}",
    f"Recall_{TARGET_RISK_CLASS}",
    "F1_Macro",
    "Balanced_Accuracy",
    "Accuracy",
]


def setup_file_logging() -> None:
    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    if any(isinstance(handler, logging.FileHandler) for handler in logger.handlers):
        return
    file_handler = logging.FileHandler(TRAINING_LOG_FILE, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s - [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(file_handler)


def format_label_distribution(series: pd.Series) -> str:
    counts = series.value_counts(dropna=False)
    total = counts.sum()
    parts = []
    for label, count in counts.items():
        pct = (count / total * 100) if total else 0
        parts.append(f"{label}: {count:,} ({pct:.2f}%)")
    return "; ".join(parts)


def log_dataset_frame(name: str, path: Path, frame: pd.DataFrame, role: str) -> None:
    logger.info("-" * 88)
    logger.info(f"{name} dataset")
    logger.info(f"  Role: {role}")
    logger.info(f"  File: {path.resolve()}")
    logger.info(f"  Shape: rows={len(frame):,}, columns={len(frame.columns):,}")
    if "target_label" in frame.columns:
        logger.info(f"  Label distribution: {format_label_distribution(frame['target_label'])}")


def log_selection_context() -> None:
    logger.info("=" * 88)
    logger.info("MODEL SELECTION CONTEXT")
    logger.info(f"Problem context: student early-warning / risk detection")
    logger.info(f"Target risk class: {TARGET_RISK_CLASS}")
    logger.info(f"Validation-only selection: yes; test is held out for final reporting only")
    logger.info(f"Objective: {SELECTION_OBJECTIVE}")
    logger.info("Metric priority:")
    for idx, metric_name in enumerate(SELECTION_METRIC_ORDER, start=1):
        logger.info(f"  {idx}. {metric_name}")
    logger.info("Rationale: Accuracy alone is not suitable here because missing Low_Engagement students is more costly than lowering overall accuracy.")
    logger.info("=" * 88)


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
        logger.error("Missing model data files. Run stage 3 before stage 4.")
        raise FileNotFoundError("Run step 3 first.")

    logger.info("Loading train/valid/test and LabelEncoder from stage 3...")
    train_df = pd.read_csv(TRAIN_FILE)
    valid_df = pd.read_csv(VALID_FILE)
    test_df = pd.read_csv(TEST_FILE)
    log_dataset_frame(
        "TRAIN",
        TRAIN_FILE,
        train_df,
        "Fit candidate models only. This split may be resampled by SMOTE/SMOTENC.",
    )
    log_dataset_frame(
        "VALID",
        VALID_FILE,
        valid_df,
        "Tune model/hyperparameters/Low_Engagement threshold and select the best candidate.",
    )
    log_dataset_frame(
        "TEST",
        TEST_FILE,
        test_df,
        "Final held-out evaluation after the validation policy has selected the model.",
    )
    if LABEL_ENCODER_FILE.exists():
        label_encoder = joblib.load(LABEL_ENCODER_FILE)
    else:
        logger.warning(
            f"Missing {LABEL_ENCODER_FILE}; rebuilding LabelEncoder with the standard label order."
        )
        label_encoder = LabelEncoder()
        label_encoder.fit(TARGET_LABELS_ORDER)
        LABEL_ENCODER_FILE.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(label_encoder, LABEL_ENCODER_FILE)
    logger.info(
        f"-> Dataset sizes: train={len(train_df):,}, valid={len(valid_df):,}, test={len(test_df):,}; "
        f"num_classes={len(label_encoder.classes_)}"
    )

    # Compute 'age' if missing from Stage 3 output
    current_year = 2024
    for df in [train_df, valid_df, test_df]:
        if 'year_of_birth' in df.columns and 'age' not in df.columns:
            df['age'] = (current_year - df['year_of_birth']).clip(lower=10, upper=100)
            logger.info("   -> Computed 'age' from 'year_of_birth'")
    
    def encode_target(series: pd.Series) -> pd.Series:
        try:
            return pd.Series(label_encoder.transform(series.astype(str)), index=series.index)
        except Exception:
            return series.astype(int)

    feature_columns = [c for c in train_df.columns if c != "target_label"]
    logger.info(f"Training feature columns ({len(feature_columns)}): {feature_columns}")
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
        logger.warning(f"Missing label {TARGET_RISK_CLASS}; using class index 0")
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


def get_low_class_index(label_encoder) -> int:
    matches = np.where(label_encoder.classes_ == TARGET_RISK_CLASS)[0]
    if len(matches) == 0:
        logger.warning(f"Missing target risk class {TARGET_RISK_CLASS}; using class index 0")
        return 0
    return int(matches[0])


def get_decision_scores(model, X: pd.DataFrame) -> tuple[np.ndarray | None, str]:
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X), "predict_proba"
        except Exception:
            return None, "none"
    if hasattr(model, "decision_function"):
        try:
            scores = np.asarray(model.decision_function(X))
            if scores.ndim == 1:
                scores = np.vstack([-scores, scores]).T
            return scores, "decision_function"
        except Exception:
            return None, "none"
    return None, "none"


def predict_with_policy(model, X: pd.DataFrame, low_idx: int, low_threshold: float | None = None) -> np.ndarray:
    y_pred = np.asarray(model.predict(X)).copy()
    if low_threshold is None:
        return y_pred

    scores, _ = get_decision_scores(model, X)
    if scores is None or np.asarray(scores).ndim != 2 or scores.shape[1] <= low_idx:
        return y_pred

    low_scores = np.asarray(scores)[:, low_idx]
    y_pred[low_scores >= low_threshold] = low_idx
    return y_pred


def build_low_threshold_candidates(model, X_valid: pd.DataFrame, low_idx: int) -> tuple[list[float | None], str]:
    scores, score_source = get_decision_scores(model, X_valid)
    if scores is None or np.asarray(scores).ndim != 2 or scores.shape[1] <= low_idx:
        return [None], score_source

    low_scores = np.asarray(scores)[:, low_idx]
    if score_source == "predict_proba":
        thresholds = np.round(np.arange(0.05, 0.96, 0.05), 4).tolist()
    else:
        thresholds = np.unique(np.quantile(low_scores, np.linspace(0.05, 0.95, 19))).round(6).tolist()
    return [None, *[float(t) for t in thresholds]], score_source


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray, label_encoder, y_score=None) -> dict:
    labels_list = list(range(len(label_encoder.classes_)))
    low_idx = get_low_class_index(label_encoder)

    auc_roc = np.nan
    if y_score is not None:
        try:
            auc_roc = roc_auc_score(y_true, y_score, multi_class="ovr", labels=labels_list)
        except ValueError:
            auc_roc = np.nan

    precisions = precision_score(y_true, y_pred, labels=labels_list, average=None, zero_division=0)
    recalls = recall_score(y_true, y_pred, labels=labels_list, average=None, zero_division=0)
    f1s = f1_score(y_true, y_pred, labels=labels_list, average=None, zero_division=0)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels_list,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )
    precision_low = precisions[low_idx]
    recall_low = recalls[low_idx]
    f2_low = (
        (5 * precision_low * recall_low) / ((4 * precision_low) + recall_low)
        if ((4 * precision_low) + recall_low) > 0
        else 0
    )

    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Balanced_Accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "F1_Macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "F1_Weighted": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "Precision_Macro": round(report["macro avg"]["precision"], 4),
        "Recall_Macro": round(report["macro avg"]["recall"], 4),
        f"Precision_{TARGET_RISK_CLASS}": round(precision_low, 4),
        f"Recall_{TARGET_RISK_CLASS}": round(recall_low, 4),
        f"F1_{TARGET_RISK_CLASS}": round(f1s[low_idx], 4),
        f"F2_{TARGET_RISK_CLASS}": round(f2_low, 4),
        "AUC_ROC_OVR": round(float(auc_roc), 4) if not np.isnan(auc_roc) else "N/A",
    }


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, label_encoder, low_threshold=None) -> tuple:
    low_idx = get_low_class_index(label_encoder)
    y_pred = predict_with_policy(model, X_test, low_idx, low_threshold)
    y_score, _ = get_decision_scores(model, X_test)
    metrics = evaluate_predictions(y_test, y_pred, label_encoder, y_score)
    return metrics, y_pred


def selection_score(metrics: dict) -> tuple:
    recall_low = float(metrics.get(f"Recall_{TARGET_RISK_CLASS}", 0) or 0)
    precision_low = float(metrics.get(f"Precision_{TARGET_RISK_CLASS}", 0) or 0)
    f2_low = float(metrics.get(f"F2_{TARGET_RISK_CLASS}", 0) or 0)
    f1_macro = float(metrics.get("F1_Macro", 0) or 0)
    balanced_acc = float(metrics.get("Balanced_Accuracy", 0) or 0)
    accuracy = float(metrics.get("Accuracy", 0) or 0)
    meets_recall_target = int(recall_low >= LOW_RECALL_TARGET)
    return (meets_recall_target, precision_low, f2_low, recall_low, f1_macro, balanced_acc, accuracy)


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
        logger.info(f"Training inputs: {TRAIN_FILE}, {VALID_FILE}, {TEST_FILE}")

        setup_file_logging()
        logger.info("=" * 88)
        logger.info("STEP 4 START: MODEL TRAINING, VALIDATION TUNING, FINAL TEST EVALUATION")
        logger.info(f"Detailed log file: {TRAINING_LOG_FILE.resolve()}")
        logger.info(f"Configured input files: train={TRAIN_FILE}, valid={VALID_FILE}, test={TEST_FILE}")
        log_selection_context()

        X_train, y_train, X_valid, y_valid, X_test, y_test, label_encoder = load_data_and_artifacts()

        logger.info("[1/4] Initializing candidate ML models...")
        models = {
            "Logistic Regression C1": OneVsRestClassifier(
                LogisticRegression(
                    C=1.0,
                    solver="liblinear",
                    max_iter=300,
                    random_state=RANDOM_STATE,
                )
            ),
            "Logistic Regression C0.3 Balanced": OneVsRestClassifier(
                LogisticRegression(
                    C=0.3,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=500,
                    random_state=RANDOM_STATE,
                )
            ),
            "Logistic Regression C1 Balanced": OneVsRestClassifier(
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=500,
                    random_state=RANDOM_STATE,
                )
            ),
            "Logistic Regression C3 Balanced": OneVsRestClassifier(
                LogisticRegression(
                    C=3.0,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=500,
                    random_state=RANDOM_STATE,
                )
            ),
            "Linear SVC": LinearSVC(dual=False, random_state=RANDOM_STATE),
            "Linear SVC Balanced": LinearSVC(dual=False, class_weight="balanced", random_state=RANDOM_STATE),
            "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
            "Decision Tree Balanced": DecisionTreeClassifier(
                max_depth=5,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=200, max_depth=10, class_weight="balanced", 
                random_state=RANDOM_STATE, n_jobs=-1
            ),
            "Random Forest Deep Balanced": RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "XGBoost": XGBClassifier(
                objective="multi:softprob", eval_metric="mlogloss",
                max_depth=5, n_estimators=200, learning_rate=0.08,
                random_state=RANDOM_STATE, n_jobs=-1
            ),
        }

        logger.info("[2/4] Training candidates and benchmarking on validation...")
        logger.info(f"Candidate models to train/evaluate on validation: {list(models.keys())}")
        logger.info(
            "Threshold tuning: for each candidate, Low_Engagement threshold is searched on VALID only; "
            "TEST remains untouched until the winning validation policy is fixed."
        )
        results = []
        fitted_models = {}
        validation_policies = {}
        low_idx = get_low_class_index(label_encoder)
        
        for name, model in models.items():
            logger.info(f"   -> Training {name} on {len(X_train):,} rows and {X_train.shape[1]} features")
            model.fit(X_train, y_train)
            fitted_models[name] = model
            
            thresholds, score_source = build_low_threshold_candidates(model, X_valid, low_idx)
            logger.info(
                f"      Validation tuning setup: score_source={score_source}, "
                f"candidate_thresholds={len(thresholds)}, target_metric={TARGET_RISK_CLASS}"
            )
            valid_metrics = None
            y_valid_pred = None
            best_threshold = None
            for threshold in thresholds:
                candidate_metrics, candidate_pred = evaluate_model(
                    model,
                    X_valid,
                    y_valid,
                    label_encoder,
                    low_threshold=threshold,
                )
                if valid_metrics is None or selection_score(candidate_metrics) > selection_score(valid_metrics):
                    valid_metrics = candidate_metrics
                    y_valid_pred = candidate_pred
                    best_threshold = threshold

            decision_policy = "default_argmax" if best_threshold is None else f"low_threshold_{score_source}"
            valid_metrics["Model"] = name
            valid_metrics["Decision_Policy"] = decision_policy
            valid_metrics["Low_Threshold"] = best_threshold if best_threshold is not None else "default"
            valid_metrics["Score_Source"] = score_source
            valid_metrics["Selection_Score"] = selection_score(valid_metrics)
            results.append(valid_metrics)
            validation_policies[name] = {
                "low_threshold": best_threshold,
                "decision_policy": decision_policy,
                "score_source": score_source,
                "selection_score": selection_score(valid_metrics),
            }
            export_prediction_details(
                MODEL_OUT_DIR / f"{safe_name(name)}_valid_predictions.csv",
                y_valid,
                y_valid_pred,
                label_encoder,
                name,
                "valid",
            )
            logger.info(
                f"      -> Validation result for {name}: "
                f"Recall_low={valid_metrics.get(f'Recall_{TARGET_RISK_CLASS}', 'N/A')}, "
                f"F2_low={valid_metrics.get(f'F2_{TARGET_RISK_CLASS}', 'N/A')}, "
                f"threshold={valid_metrics.get('Low_Threshold', 'default')}, "
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
            f"F2_{TARGET_RISK_CLASS}",
            "Decision_Policy",
            "Low_Threshold",
            "Score_Source",
            "Selection_Score",
            "AUC_ROC_OVR",
        ]
        metrics_df = metrics_df[cols]
        metrics_df.to_csv(METRICS_FILE, index=False, encoding="utf-8-sig")
        logger.info(f"Saved validation comparison metrics: {METRICS_FILE.resolve()}")
        
        print("\n" + "=" * 90)
        print("MODEL RANKING TABLE (early-warning optimized)")
        print("=" * 90)
        print(metrics_df.to_string(index=False))
        print("=" * 90 + "\n")

        ranked_records = sorted(
            metrics_df.to_dict(orient="records"),
            key=lambda row: row["Selection_Score"],
            reverse=True,
        )
        ranked = pd.DataFrame(ranked_records)
        best_model_name = ranked_records[0]["Model"]
        best_model = fitted_models[best_model_name]
        best_policy = validation_policies[best_model_name]
        logger.info(
            "Selected from VALID only: "
            f"model={best_model_name}, policy={best_policy['decision_policy']}, "
            f"low_threshold={best_policy['low_threshold']}, score_source={best_policy['score_source']}"
        )
        logger.info(
            "Selected validation metrics: "
            f"Recall_Low={ranked.iloc[0].get(f'Recall_{TARGET_RISK_CLASS}')}, "
            f"Precision_Low={ranked.iloc[0].get(f'Precision_{TARGET_RISK_CLASS}')}, "
            f"F2_Low={ranked.iloc[0].get(f'F2_{TARGET_RISK_CLASS}')}, "
            f"F1_Macro={ranked.iloc[0].get('F1_Macro')}, "
            f"Balanced_Accuracy={ranked.iloc[0].get('Balanced_Accuracy')}, "
            f"Accuracy={ranked.iloc[0].get('Accuracy')}"
        )
        logger.info(f"Winning algorithm: {best_model_name}")
        logger.info(f"Top-ranked validation candidate: {ranked.iloc[0].to_dict()}")

        logger.info("[3/4] Running final held-out test evaluation for the selected model...")
        test_metrics, test_pred = evaluate_model(
            best_model,
            X_test,
            y_test,
            label_encoder,
            low_threshold=best_policy["low_threshold"],
        )
        test_metrics["Model"] = best_model_name
        test_metrics["Decision_Policy"] = best_policy["decision_policy"]
        test_metrics["Low_Threshold"] = best_policy["low_threshold"] if best_policy["low_threshold"] is not None else "default"
        test_metrics["Score_Source"] = best_policy["score_source"]
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
        logger.info(
            "FINAL TEST RESULT (held-out, not used for selection): "
            f"model={best_model_name}, "
            f"Accuracy={test_metrics.get('Accuracy')}, "
            f"Balanced_Accuracy={test_metrics.get('Balanced_Accuracy')}, "
            f"F1_Macro={test_metrics.get('F1_Macro')}, "
            f"Precision_Low={test_metrics.get(f'Precision_{TARGET_RISK_CLASS}')}, "
            f"Recall_Low={test_metrics.get(f'Recall_{TARGET_RISK_CLASS}')}, "
            f"F2_Low={test_metrics.get(f'F2_{TARGET_RISK_CLASS}')}, "
            f"AUC_ROC_OVR={test_metrics.get('AUC_ROC_OVR')}"
        )

        valid_pred = predict_with_policy(
            best_model,
            X_valid,
            low_idx,
            best_policy["low_threshold"],
        )
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
        plt.title(f"Test Confusion Matrix - {best_model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(IMAGE_OUT_DIR / f"CM_TEST_{safe_name(best_model_name)}.png", dpi=200)
        plt.close()

        model_metadata = {
            "model_name": best_model_name,
            "model_class": best_model.__class__.__name__,
            "model_params": best_model.get_params(),
            "validation_metrics": ranked.iloc[0].to_dict(),
            "test_metrics": test_metrics,
            "selection_policy": {
                "source_split": "valid",
                "objective": "Prefer configs meeting Low_Engagement recall target, then maximize Precision_Low, F2_Low, Recall_Low, F1_Macro, Balanced_Accuracy, Accuracy.",
                "low_recall_target": LOW_RECALL_TARGET,
                "decision_policy": best_policy["decision_policy"],
                "low_threshold": best_policy["low_threshold"],
                "score_source": best_policy["score_source"],
                "selection_score": best_policy["selection_score"],
            },
            "target_labels": label_encoder.classes_.tolist(),
        }
        with open(MODEL_META_FILE, "w", encoding="utf-8") as f:
            json.dump(model_metadata, f, ensure_ascii=False, indent=2, default=str)
        logger.info("   -> Saved validation/test reports and model metadata")

        logger.info("[4/4] Packaging deployment bundle...")
        deployment_bundle = {
            "model": best_model,
            "model_name": best_model_name,
            "label_encoder": label_encoder,
            "school_encoder": joblib.load(SCHOOL_ENCODER_FILE) if SCHOOL_ENCODER_FILE.exists() else None,
            "imputer": joblib.load(IMPUTER_FILE) if IMPUTER_FILE.exists() else None,
            "scaler": joblib.load(SCALER_FILE) if SCALER_FILE.exists() else None,
            "feature_columns": list(X_train.columns),
            "target_labels": label_encoder.classes_.tolist(),
            "low_class_index": low_idx,
            "low_threshold": best_policy["low_threshold"],
            "decision_policy": best_policy["decision_policy"],
            "score_source": best_policy["score_source"],
        }

        joblib.dump(best_model, BEST_MODEL_FILE)
        joblib.dump(deployment_bundle, MODEL_BUNDLE_FILE)
        logger.info(f"   -> Saved best model: {BEST_MODEL_FILE}")
        logger.info(f"   -> Saved deployment bundle: {MODEL_BUNDLE_FILE}")

        logger.info("STAGE 4 COMPLETED.")
        logger.info(f"Deployment bundle saved at: {MODEL_BUNDLE_FILE}")
        logger.info(f"Model metadata saved at: {MODEL_META_FILE}")

        logger.info("Output artifacts:")
        logger.info(f"  Validation comparison: {METRICS_FILE.resolve()}")
        logger.info(f"  Final test metrics: {(MODEL_OUT_DIR / 'final_test_metrics.csv').resolve()}")
        logger.info(f"  Best-model valid report: {(MODEL_OUT_DIR / 'best_model_valid_classification_report.csv').resolve()}")
        logger.info(f"  Best-model test report: {(MODEL_OUT_DIR / 'best_model_test_classification_report.csv').resolve()}")
        logger.info(f"  Test confusion matrix: {(IMAGE_OUT_DIR / f'CM_TEST_{safe_name(best_model_name)}.png').resolve()}")
        logger.info(f"  Stage 4 log: {TRAINING_LOG_FILE.resolve()}")

    except Exception as e:
        logger.exception("Fatal error in stage 4:")

if __name__ == "__main__":
    main()
