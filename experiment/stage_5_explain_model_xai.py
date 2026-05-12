import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score, precision_score, recall_score

from config import (
    IMAGE_OUT_DIR,
    MODEL_BUNDLE_FILE,
    TEST_FILE,
    TIME_WINDOW_MODE,
    DEFAULT_OBSERVATION_DAYS,
    LABEL_PERCENTILES,
    TRAIN_CLASS_RATIOS,
    ENABLE_SMOTE,
)


def get_decision_scores(model, X: pd.DataFrame):
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)
        except Exception:
            return None
    if hasattr(model, "decision_function"):
        try:
            scores = np.asarray(model.decision_function(X))
            if scores.ndim == 1:
                scores = np.vstack([-scores, scores]).T
            return scores
        except Exception:
            return None
    return None


def predict_with_bundle_policy(model, X: pd.DataFrame, bundle: dict) -> np.ndarray:
    y_pred = np.asarray(model.predict(X)).copy()
    low_threshold = bundle.get("low_threshold")
    low_idx = bundle.get("low_class_index")
    if low_threshold is None or low_idx is None:
        return y_pred

    scores = get_decision_scores(model, X)
    if scores is None or np.asarray(scores).ndim != 2 or scores.shape[1] <= int(low_idx):
        return y_pred

    y_pred[np.asarray(scores)[:, int(low_idx)] >= float(low_threshold)] = int(low_idx)
    return y_pred


def main():
    print("=" * 80)
    print("STEP 5: EXPECTED RESULTS, PROPOSED MODEL, AND XAI")
    print("=" * 80)

    if not MODEL_BUNDLE_FILE.exists():
        raise FileNotFoundError(f"Run step 4 first: {MODEL_BUNDLE_FILE}")

    IMAGE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(MODEL_BUNDLE_FILE)
    model = bundle["model"]
    label_encoder = bundle["label_encoder"]
    print(f"Loaded bundle: {bundle['model_name']} ({model.__class__.__name__})")
    print(f"Bundle feature count: {len(bundle['feature_columns'])}")

    test_df = pd.read_csv(TEST_FILE)
    
    # Get all features expected by the model
    expected_features = bundle["feature_columns"]
    available_features = [col for col in expected_features if col in test_df.columns]
    missing_features = [col for col in expected_features if col not in test_df.columns]
    
    print(f"Available features in test file: {available_features}")
    if missing_features:
        print(f"WARNING: Missing features will be filled with 0: {missing_features}")
    
    # Build X_test with all expected features
    X_test = test_df[available_features].copy()
    
    # Add missing features with 0 values
    for feat in missing_features:
        X_test[feat] = 0.0
    
    # Reorder to match expected order
    X_test = X_test[expected_features]
    
    # Handle target_label - could be string labels or numeric
    target_col = test_df["target_label"]
    try:
        # Try to encode if string labels
        y_test = label_encoder.transform(target_col.astype(str))
    except (ValueError, AttributeError):
        # If already numeric or transform fails, try direct int conversion
        try:
            y_test = target_col.astype(int)
        except (ValueError, TypeError):
            print(f"WARNING: Could not convert target_label. Using values as-is.")
            y_test = target_col
    
    X_sample = X_test.sample(n=min(1000, len(X_test)), random_state=42)
    print(f"Test set shape: {X_test.shape}; sampled {len(X_sample)} rows for SHAP")

    print(f"Proposed deployment model: {bundle['model_name']}")
    print(f"Deployment artifact: {MODEL_BUNDLE_FILE}")
    print("Expected labels:", ", ".join(bundle["target_labels"]))

    y_pred = predict_with_bundle_policy(model, X_test, bundle)
    labels_list = list(range(len(label_encoder.classes_)))
    target_label = "Low_Engagement"
    low_idx = int(np.where(label_encoder.classes_ == target_label)[0][0])
    report = classification_report(
        y_test,
        y_pred,
        labels=labels_list,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)

    summary_lines = [
        f"Model: {bundle['model_name']}",
        f"Window mode: {TIME_WINDOW_MODE}, observation_days={DEFAULT_OBSERVATION_DAYS}",
        f"Label percentiles: {LABEL_PERCENTILES}",
        f"Train class ratios: {TRAIN_CLASS_RATIOS}",
        f"Enable SMOTE: {ENABLE_SMOTE}",
        f"Decision policy: {bundle.get('decision_policy', 'default_argmax')}",
        f"Low threshold: {bundle.get('low_threshold', 'default')}",
        f"Accuracy: {accuracy:.4f}",
        f"Balanced accuracy: {balanced_acc:.4f}",
        f"Macro F1: {f1_score(y_test, y_pred, average='macro', zero_division=0):.4f}",
        f"Weighted F1: {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}",
        f"Low-class precision: {precision_score(y_test, y_pred, labels=labels_list, average=None, zero_division=0)[low_idx]:.4f}",
        f"Low-class recall: {recall_score(y_test, y_pred, labels=labels_list, average=None, zero_division=0)[low_idx]:.4f}",
        f"Low-class F1: {f1_score(y_test, y_pred, labels=labels_list, average=None, zero_division=0)[low_idx]:.4f}",
    ]
    summary_path = IMAGE_OUT_DIR / "expected_results_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
        f.write("\n\nClassification report:\n")
        f.write(pd.DataFrame(report).T.to_string())
    print(f"Saved expected-results summary: {summary_path}")
    print(f"   -> Summary metrics: accuracy={accuracy:.4f}, balanced={balanced_acc:.4f}")

    importances = None
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=available_features)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        if coef.ndim == 2:
            coef = np.mean(np.abs(coef), axis=0)
        else:
            coef = np.abs(coef)
        importances = pd.Series(coef, index=expected_features)
    else:
        try:
            perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, scoring="f1_macro")
            importances = pd.Series(perm.importances_mean, index=expected_features)
        except Exception:
            importances = None

    if importances is not None:
        importances = importances.sort_values()
        plt.figure(figsize=(8, 5))
        importances.plot(kind="barh")
        plt.title(f"Feature Importance - {bundle['model_name']}")
        plt.tight_layout()
        fi_path = IMAGE_OUT_DIR / f"FI_{bundle['model_name'].replace(' ', '_')}.png"
        plt.savefig(fi_path, dpi=200)
        plt.close()
        importances.to_csv(IMAGE_OUT_DIR / "feature_importance_values.csv", header=["importance"], encoding="utf-8-sig")
        print(f"Saved feature importance: {fi_path}")
        print(f"   -> Top important features: {', '.join(importances.tail(5).index.tolist())}")
    else:
        print("Feature importance could not be derived for the selected model.")

    try:
        explainer = shap.Explainer(lambda x: model.predict_proba(x), shap.sample(X_sample, min(100, len(X_sample))))
        shap_values = explainer(X_sample)
        values_array = np.asarray(shap_values.values)
        if values_array.ndim == 3:
            target_shap = values_array[:, :, low_idx]
        else:
            target_shap = values_array

        shap.summary_plot(target_shap, X_sample, show=False)
        shap_summary_path = IMAGE_OUT_DIR / "SHAP_Summary_Global.png"
        plt.savefig(shap_summary_path, dpi=250, bbox_inches="tight")
        plt.close()
        print(f"Saved SHAP summary: {shap_summary_path}")
    except Exception as exc:
        print(f"SHAP summary skipped: {exc}")
        shap_summary_path = None

    if hasattr(model, "predict"):
        low_rows = np.where(y_pred == low_idx)[0]
        if len(low_rows) == 0:
            print("No low engagement predictions found for SHAP analysis")
            return

        row_idx = int(low_rows[0])
        print(f"Generating local SHAP explanation for row index: {row_idx}")
        try:
            local_explainer = shap.Explainer(lambda x: model.predict_proba(x), shap.sample(X_sample, min(100, len(X_sample))))
            single_explanation = local_explainer(X_test.iloc[[row_idx]])
            local_values = np.asarray(single_explanation.values)
            local_base = np.asarray(single_explanation.base_values)
            if local_values.ndim == 3:
                values = local_values[0, :, low_idx]
                base_value = local_base[0, low_idx]
            else:
                values = local_values[0]
                base_value = local_base[0] if local_base.ndim else local_base

            explanation = shap.Explanation(
                values=values,
                base_values=base_value,
                data=X_test.iloc[row_idx].values,
                feature_names=expected_features,
            )
            shap.plots.waterfall(explanation, show=False)
            local_path = IMAGE_OUT_DIR / f"SHAP_Local_Student_{row_idx}.png"
            plt.savefig(local_path, dpi=250, bbox_inches="tight")
            plt.close()
            print(f"Saved local SHAP explanation: {local_path}")
        except Exception as exc:
            print(f"Local SHAP explanation skipped: {exc}")


if __name__ == "__main__":
    main()
